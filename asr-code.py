from transformers import AutoModel
import torchaudio
import torch

# Load fixed model version
model = AutoModel.from_pretrained(
    "ai4bharat/indic-conformer-600m-multilingual",
    revision="e9b71b369c048e2c6b634d4c131061c34e441179",
    trust_remote_code=True
)

# Languages and audio file mapping
languages = {
    "kn": "samples/kannada_sample_1.wav",
    "hi": "samples/hindi_sample.wav",
    "mr": "samples/marathi_sample.wav",
    "ta": "samples/tamil_sample.wav",
    "te": "samples/telugu_sample.wav"
}

# Common audio preprocessing function
def load_and_preprocess(path):
    wav, sr = torchaudio.load(path)
    wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    return wav

# Process each language with both decoders
for lang_code, audio_path in languages.items():
    print(f"Process Language - {lang_code.upper()} ---------")

    wav = load_and_preprocess(audio_path)

    print("CTC:", model(wav, lang_code, "ctc"))
    print("RNNT:", model(wav, lang_code, "rnnt"))
    print("-----")
