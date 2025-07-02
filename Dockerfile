FROM dwani/core-image-2:latest AS builder

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


COPY . .

RUN useradd -ms /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

# Use absolute path for clarity
CMD ["python", "/app/src/asr_api.py", "--host", "0.0.0.0", "--port", "7860", "--device", "cuda"]