# Use an official PyTorch image with CUDA (for NVIDIA GPUs)
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system packages if any (soundfile needs libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app code and config
COPY main.py config_vocals_mel_band_roformer.yaml ./
COPY models/ ./models/


# Expose port 80 for the FastAPI service
EXPOSE 80

# Run the FastAPI server with Uvicorn (listening on port 80)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
