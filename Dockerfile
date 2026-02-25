FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    sudo \
    wget \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY docker-requirements.txt .

RUN pip install --no-cache-dir --no-deps -r docker-requirements.txt
RUN pip install typing_extensions typing_inspection annotated_types
RUN pip install anyio torch==2.7.1 torchaudio torchcodec
RUN pip install packaging regex numpy
RUN pip install huggingface_hub safetensors transformers
RUN pip install python-multipart
WORKDIR /app

# Copy pre-downloaded HuggingFace model (run: huggingface-cli download ai4bharat/indic-conformer-600m-multilingual --local-dir ./hf_models)
COPY hf_models /app/hf_models
RUN test -f /app/hf_models/config.json || (echo "Build context must include hf_models. Run: huggingface-cli download ai4bharat/indic-conformer-600m-multilingual --local-dir ./hf_models" && exit 1)

COPY . .
RUN useradd -ms /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 10803

CMD ["python", "/app/src/server/asr_api.py", "--host", "0.0.0.0", "--port", "10803"]