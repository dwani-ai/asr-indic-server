# ASR Indic Server


## Overview
Automatic Speech Recognition (ASR) for Indian languages using IndicConformer models. 

Currently verified for kannada, hindi, tamil, telugu, marathi .

Try the web demo at - [https://workshop.dwani.ai](https://workshop.dwani.ai) with Transcription page


## Table of Contents
- [Getting Started](#getting-started-development)
  - [For Development (Local)](#for-development-local)
    - [Prerequisites](#prerequisites-1)
    - [Steps](#steps-1)
- [Downloading Translation Models](#downloading-translation-models)
  - [Kannada](#kannada)
  - [Other Languages](#other-languages)
    - [Malayalam](#malayalam)
    - [Hindi](#hindi)
- [Running with FastAPI Server](#running-with-fastapi-server)
- [Evaluating Results](#evaluating-results)
  - [Kannada Transcription Examples](#kannada-transcription-examples)
    - [Sample 1: kannada_sample_1.wav](#sample-1-kannada_sample_1wav)
    - [Sample 3 - Song - 4 minutes](#sample-3---song---4-minutes)
  [Building Docker Image](#building-docker-image)
  - [Run the Docker Image](#run-the-docker-image)
- [Troubleshooting](#troubleshooting)
- [References](#references)


## Getting Started - Development

### For Development (Local)
- **Prerequisites**: Python 3.10 (compatibility verified)
- **Steps**:
  1. **Create a virtual environment**:
  ```bash
  python -m venv venv
  ```
  2. **Activate the virtual environment**:
  ```bash
  source venv/bin/activate
  ```
  On Windows, use:
  ```bash
  venv\Scripts\activate
  ```
  3. **Install dependencies**:
      ```bash
      pip install -r requirements.txt
      ```
      ```bash
      sudo apt install ffmpeg
      ```
## Downloading Translation Models
Models can be downloaded from AI4Bharat's HuggingFace repository:

### For Multi-lingual language supported model
```bash
hf download ai4bharat/indic-conformer-600m-multilingual --local-dir ./hf_models
```

### Sample Code
### For all languages
```python
from transformers import AutoModel
import torchaudio
import torch

# Load the model
model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)

# Load an audio file
wav, sr = torchaudio.load("kannada_sample_1.wav")
wav = torch.mean(wav, dim=0, keepdim=True)

target_sample_rate = 16000  # Expected sample rate
if sr != target_sample_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
    wav = resampler(wav)

# Perform ASR with CTC decoding
transcription_ctc = model(wav, "kn", "ctc")
print("CTC Transcription:", transcription_ctc)

# Perform ASR with RNNT decoding
transcription_rnnt = model(wav, "kn", "rnnt")
print("RNNT Transcription:", transcription_rnnt)

```

- Run the Code
  ```bash
  python asr-code.py
  ```


### Alternative examples for Development


#### For Server Development

#### Running with FastAPI Server
Run the server using FastAPI with the multilingual model
- for GPU
  ```bash
  python src/server/asr_api.py --port 10803 --host 0.0.0.0
  ```

#### Evaluating Results for FastApi Server
You can evaluate the ASR transcription results using `curl` commands. 
### Kannada Transcription Examples

#### Sample 1: kannada_sample_1.wav
- **Audio File**: [samples/kannada_sample_1.wav](samples/kannada_sample_1.wav)
- **Command**:
```bash
curl -X 'POST' 'http://localhost:10803/transcribe/language=kannada' -H 'accept: application/json'   -H 'Content-Type: multipa'Content-Type  multipart/form-data' -F 'file=@samples/kannada_sample_1.wav;type=audio/x-wav'
```
- **Expected Output**:
```ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು```
Translation: "What is the capital of Karnataka"


#### Sample 2 - Song - 4 minutes
- [YT Video- Navaduva Nudiye](https://www.youtube.com/watch?v=LuZzhMN8ndQ)
- **Audio File**: [samples/kannada_sample_3.wav](samples/kannada_sample_3.wav)
- **Command**:
```bash
curl -X 'POST' \
'http://localhost:10803/transcribe/language=kannada' \
-H 'accept: application/json' \
-H 'Content-Type: multipart/form-data' \
-F 'file=@samples/kannada_sample_3.wav;type=audio/x-wav'
```
- **Expected Output**: [kannada_sample_3_out.md](docs/kannada_sample_3_out.md)


**Note**: The ASR does not provide sentence breaks or punctuation (e.g., question marks).


## Troubleshooting
- **Transcription errors**: Verify the audio file is in WAV format, mono, and sampled at 16kHz. Adjust using:
```bash
ffmpeg -i sample_audio.wav -ac 1 -ar 16000 sample_audio_infer_ready.wav -y
```
- **Model not found**: Download the required models using the `huggingface-cli download` commands above.
- **Port conflicts**: Ensure port 10803 is free when running the FastAPI server.


## Demo Video

Watch a quick demo of our project in action! Click the image below to view the video on YouTube.

<a href="https://youtu.be/F0Mo0zjyysM" target="_blank">
  <img src="https://img.youtube.com/vi/F0Mo0zjyysM/0.jpg" alt="Watch the video">
</a>


## Contributing

We welcome contributions! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to this project.

Also you can join the [discord group](https://discord.gg/WZMCerEZ2P) to collaborate


## References
- [AI4Bharat IndicConformerASR GitHub Repository](https://github.com/AI4Bharat/IndicConformerASR)
- [Nemo - AI4Bharat](https://github.com/AI4Bharat/NeMo)
- [IndicConformer Collection on HuggingFace](https://huggingface.co/collections/ai4bharat/indicconformer-66d9e933a243cba4b679cb7f)


<!-- 
### For Production (Docker)
- **Prerequisites**: Docker and Docker Compose
- **Steps**:
  1. **Start the server**:
  For GPU
  ```bash
  docker compose -f compose.yaml up -d
  ```

-->

<!-- 
docker build -t dwani/asr-indic-server:latest -f Dockerfile .

export HF_TOKEN="HF_TKEN"

docker compose -f compose.yml up -d

Test for 
malayalam
odia
gujurati

-->
<!-- 
python -m pip install --upgrade pip setuptools wheel
python -m pip install "nemo_toolkit[asr]"

nohup python src/server/asr_api.py --port 7863 --host 0.0.0.0 --device cuda > asr.log 2>&1 &

sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 7863


sudo iptables -t nat -A OUTPUT -p tcp --dport 80 -j REDIRECT --to-port 7863


nohup python src/server/asr_api.py --port 7863 --host 0.0.0.0 --device cuda > asr.log 2>&1 &
-->
