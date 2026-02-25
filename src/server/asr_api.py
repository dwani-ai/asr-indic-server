from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import RedirectResponse, JSONResponse
import os
import torch
import torchaudio
from transformers import AutoModel
import argparse
import uvicorn
from pydantic import BaseModel
from typing import List

# Initialize the FastAPI app
app = FastAPI(title="Indic ASR API", version="1.0.0")

class TranscriptionResponse(BaseModel):
    text: str

# Load from local path in Docker (/app/hf_models) to avoid HF token; fall back to HF for local runs
MODEL_PATH = os.environ.get("ASR_MODEL_PATH", "/app/hf_models")
_use_local = os.path.isdir(MODEL_PATH)
if not _use_local and "ASR_MODEL_PATH" in os.environ:
    raise RuntimeError(f"ASR_MODEL_PATH={MODEL_PATH} is not an existing directory")
_model_path = MODEL_PATH if _use_local else "ai4bharat/indic-conformer-600m-multilingual"
_load_kwargs = {"trust_remote_code": True}
if _use_local:
    _load_kwargs["local_files_only"] = True
    os.environ["HF_HUB_OFFLINE"] = "1"  # prevent any hub access when using baked-in model
else:
    _load_kwargs["revision"] = "e9b71b369c048e2c6b634d4c131061c34e441179"

model = AutoModel.from_pretrained(_model_path, **_load_kwargs)

# Language mapping (removed duplicate)
model_language = {
            "kannada": "kn", "hindi": "hi", "malayalam": "ml", "assamese": "as", "bengali": "bn",
            "bodo": "brx", "dogri": "doi", "gujarati": "gu", "kashmiri": "ks", "konkani": "kok",
            "maithili": "mai", "manipuri": "mni", "marathi": "mr", "nepali": "ne", "odia": "or",
            "punjabi": "pa", "sanskrit": "sa", "santali": "sat", "sindhi": "sd", "tamil": "ta",
            "telugu": "te", "urdu": "ur"
        }

def preprocess_audio(wav, sr):
    """Preprocess audio to 16kHz mono"""
    # Convert to mono
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    
    # Resample if necessary
    target_sr = 16000
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)
    
    return wav

@app.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(..., media_type="audio/*"), 
    language: str = Query(..., enum=list(model_language.keys()))
):
    """
    Transcribe audio file in Indic languages using CTC decoder
    """
    try:
        # Validate file
        if not file.content_type or not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Load and preprocess audio
        wav, sr = torchaudio.load(file.file)
        wav = preprocess_audio(wav, sr)
        
        # Map language name to model code
        lang_code = model_language[language]
        
        # Perform ASR with CTC decoding
        transcription_ctc = model(wav, lang_code, "ctc")
        
        return JSONResponse(content={"text": transcription_ctc})        
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/")
async def root():
    """Redirect to API docs"""
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": "loaded"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Indic ASR FastAPI server")
    parser.add_argument("--port", type=int, default=10803, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    args = parser.parse_args()
    
    print("Starting Indic ASR server...")
    print(f"Languages supported: {list(model_language.keys())}")
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        log_level="info"
    )
