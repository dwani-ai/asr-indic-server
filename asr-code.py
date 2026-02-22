from transformers import AutoModel
import torchaudio
import torch

# Load fixed model version
model = AutoModel.from_pretrained(
    "ai4bharat/indic-conformer-600m-multilingual",
    revision="e9b71b369c048e2c6b634d4c131061c34e441179",
    trust_remote_code=True
)



"""
Process Language - Kannada
"""

print("Process Language - Kannada----------")
# Load & preprocess audio (unchanged)
wav, sr = torchaudio.load("samples/kannada_sample_1.wav")
wav = torch.mean(wav, dim=0, keepdim=True)

if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    wav = resampler(wav)

# ASR with both decoders
print("CTC:", model(wav, "kn", "ctc"))
print("RNNT:", model(wav, "kn", "rnnt"))

print("-----")
"""
Process Language - Hindi
"""

print("Process Language - Hindi")
# Load & preprocess audio (unchanged)
wav, sr = torchaudio.load("samples/hindi_sample.wav")
wav = torch.mean(wav, dim=0, keepdim=True)

if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    wav = resampler(wav)

# ASR with both decoders
print("CTC:", model(wav, "hi", "ctc"))
print("RNNT:", model(wav, "hi", "rnnt"))

print("-----")


"""
Process Language - Marathi
"""
print("Process Language - Marathi----")
# Load & preprocess audio (unchanged)
wav, sr = torchaudio.load("samples/marathi_sample.wav")
wav = torch.mean(wav, dim=0, keepdim=True)

if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    wav = resampler(wav)

# ASR with both decoders
print("CTC:", model(wav, "mr", "ctc"))
print("RNNT:", model(wav, "mr", "rnnt"))


print("-----")



"""
Process Language - Tamil
"""

print("Process Language - Tamil ----")
# Load & preprocess audio (unchanged)
wav, sr = torchaudio.load("samples/tamil_sample.wav")
wav = torch.mean(wav, dim=0, keepdim=True)

if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    wav = resampler(wav)

# ASR with both decoders
print("CTC:", model(wav, "ta", "ctc"))
print("RNNT:", model(wav, "ta", "rnnt"))


print("-----")




"""
Process Language - Telugu
"""

print("Process Language - Telugu ---")
# Load & preprocess audio (unchanged)
wav, sr = torchaudio.load("samples/telugu_sample.wav")
wav = torch.mean(wav, dim=0, keepdim=True)

if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    wav = resampler(wav)

# ASR with both decoders
print("CTC:", model(wav, "te", "ctc"))
print("RNNT:", model(wav, "te", "rnnt"))


print("-----")

