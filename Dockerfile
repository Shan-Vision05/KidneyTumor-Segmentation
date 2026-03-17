# Base image
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install CPU-only PyTorch first so MONAI doesn't pull the full CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the package source — PYTHONPATH makes SpleenSeg.* importable without
# a packaging step (avoids setuptools version constraints in the slim image)
COPY pyproject.toml .
COPY SpleenSeg/ ./SpleenSeg/
ENV PYTHONPATH=/app

# Bake the ONNX model and sample cases into the image
# (dvc pull restores these paths before `docker build` is called in CI)
COPY models/unet25d.onnx /models/unet25d.onnx
COPY samples/ /samples/

# Runtime configuration — all can be overridden in docker-compose / docker run
ENV ONNX_MODEL_PATH=/models/unet25d.onnx \
    SAMPLES_DIR=/samples \
    RESULTS_DIR=/results \
    ROOT_PATH=""

EXPOSE 8000

CMD ["uvicorn", "SpleenSeg.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]