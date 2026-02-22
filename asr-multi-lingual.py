from transformers import AutoModel
import torchaudio
import torch

# Load fixed model version
model = AutoModel.from_pretrained(
    "ai4bharat/indic-conformer-600m-multilingual",
    revision="e9b71b369c048e2c6b634d4c131061c34e441179",
    trust_remote_code=True
)

# Load & preprocess audio (unchanged)
wav, sr = torchaudio.load("samples/kannada_sample_1.wav")
wav = torch.mean(wav, dim=0, keepdim=True)

if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    wav = resampler(wav)

# ASR with both decoders
print("CTC:", model(wav, "kn", "ctc"))
print("RNNT:", model(wav, "kn", "rnnt"))
