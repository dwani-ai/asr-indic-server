FROM dwani/core-image-2:latest AS builder

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --no-deps -r requirements.txt


COPY . .
RUN pip install --upgrade pip
RUN useradd -ms /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

# Use absolute path for clarity
#CMD ["python", "/app/src/server/asr_api.py", "--host", "0.0.0.0", "--port", "7860", "--device", "cuda"]
CMD ["python", "/app/src/multi-lingual/asr_api.py", "--host", "0.0.0.0", "--port", "7860", "--device", "cuda"]