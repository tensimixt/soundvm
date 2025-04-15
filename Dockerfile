FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install any system dependencies (libsndfile1 for audio read/write)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy in Python requirements, install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire models folder + other code files
COPY models/ ./models/
COPY main.py config_vocals_mel_band_roformer.yaml ./

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--workers", "4"]

