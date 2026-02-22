from fastapi import FastAPI, UploadFile
import torch
import torchaudio
from transformers import AutoModel
import argparse
import uvicorn
from pydantic import BaseModel
from pydub import AudioSegment
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import RedirectResponse, JSONResponse
from typing import List

# Initialize the FastAPI app
app = FastAPI()
class TranscriptionResponse(BaseModel):
    text: str

# Load the model
model = AutoModel.from_pretrained(
    "ai4bharat/indic-conformer-600m-multilingual",
    revision="e9b71b369c048e2c6b634d4c131061c34e441179",
    trust_remote_code=True
)

model_language = {
            "kannada": "kn", "hindi": "hi", "malayalam": "ml", "assamese": "as", "bengali": "bn",
            "bodo": "brx", "dogri": "doi", "gujarati": "gu", "kashmiri": "ks", "konkani": "kok",
            "maithili": "mai", "manipuri": "mni", "marathi": "mr", "nepali": "ne", "odia": "or",
            "punjabi": "pa", "sanskrit": "sa", "santali": "sat", "sindhi": "sd", "tamil": "ta",
            "telugu": "te", "urdu": "ur"
        }
@app.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...), language: str = Query(..., enum=list(model_language.keys()))):
    # Load the uploaded audio file
    wav, sr = torchaudio.load(file.file)
    wav = torch.mean(wav, dim=0, keepdim=True)

    # Resample if necessary
    target_sample_rate = 16000  # Expected sample rate
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
        wav = resampler(wav)

    # Perform ASR with CTC decoding
    #transcription_ctc = model(wav, "kn", "ctc")
    print(language)
    # Perform ASR with RNNT decoding
    transcription_rnnt = model(wav, language, "rnnt")

    return JSONResponse(content={"text": transcription_rnnt})


@app.get("/")
async def home():
    return RedirectResponse(url="/docs")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI server for ASR.")
    parser.add_argument("--port", type=int, default=8888, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on.")
    parser.add_argument("--device", type=str, default="cuda", help="Device type to run the model on (cuda or cpu).")
    args = parser.parse_args()
    
    
    uvicorn.run(app, host=args.host, port=args.port)