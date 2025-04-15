import os
import io
import tempfile
import requests
import zipfile
import logging

import numpy as np
import torch
import soundfile as sf

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

# Configure logging: adjust level to DEBUG for even more detail
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# URL to model weights (Hugging Face)
MODEL_URL = "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt"
MODEL_PATH = "MelBandRoformer.ckpt"

# Load config (YAML) for model architecture and inference parameters
import yaml
from ml_collections import ConfigDict

with open("config_vocals_mel_band_roformer.yaml", "r") as f:
    config = ConfigDict(yaml.safe_load(f))

# Convert config.model (ConfigDict) to a plain dictionary
model_config = dict(config.model)

# Convert multi_stft_resolutions_window_sizes to a tuple if present
if "multi_stft_resolutions_window_sizes" in model_config:
    logger.info("Before conversion: %s (%s)", model_config["multi_stft_resolutions_window_sizes"],
                type(model_config["multi_stft_resolutions_window_sizes"]))
    model_config["multi_stft_resolutions_window_sizes"] = tuple(model_config["multi_stft_resolutions_window_sizes"])
    logger.info("After conversion: %s (%s)", model_config["multi_stft_resolutions_window_sizes"],
                type(model_config["multi_stft_resolutions_window_sizes"]))

# Import model class definition (ensure the models folder structure is included in the repo)
from models.mel_band_roformer import MelBandRoformer

# Initialize FastAPI
app = FastAPI()

# Global model variable and device configuration
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
def load_model():
    """Download and load the Mel-Band Roformer model at startup."""
    global model
    try:
        # Download weights if not already present
        if not os.path.exists(MODEL_PATH):
            logger.info("Model weights not found. Downloading model weights from: %s", MODEL_URL)
            resp = requests.get(MODEL_URL, stream=True)
            resp.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info("Model weights downloaded successfully.")

        # Initialize model with the plain dictionary configuration
        logger.info("Initializing the model with configuration: %s", model_config)
        model = MelBandRoformer(**model_config)
        logger.info("Model instance created successfully.")

        # Load model state
        state = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        logger.info("Model loaded and ready on device: %s", device)
    except Exception as e:
        logger.exception("Error during model initialization or loading.")
        raise e

@app.post("/separate")
async def separate_audio(file: UploadFile = File(...)):
    """
    Separate the uploaded WAV file into vocals and instrumental.
    Returns a ZIP containing 'vocals.wav' and 'instrumental.wav'.
    """
    try:
        # Read the uploaded file into a NumPy array
        data, sr = sf.read(file.file)
        logger.info("Audio file read successfully. Sample rate: %d, data shape: %s", sr, data.shape)
    except Exception as e:
        logger.exception("Failed to read input audio file.")
        raise HTTPException(status_code=400, detail="Invalid audio file format.")

    if data.ndim == 1:
        # Duplicate mono into stereo channels
        logger.info("Input audio is mono; duplicating channel to create stereo.")
        data = np.stack([data, data], axis=-1)

    # Convert to torch tensor and move to appropriate device
    mixture = torch.tensor(data.T, dtype=torch.float32).to(device)
    logger.info("Converted audio to tensor with shape: %s", mixture.shape)

    # Inference parameters from config
    C = config.inference.chunk_size       # chunk length (samples)
    N = config.inference.num_overlap       # overlap factor
    step = C // N
    fade = C // 10
    border = C - step
    if mixture.shape[1] > 2 * border and border > 0:
        mixture = torch.nn.functional.pad(mixture, (border, border), mode='reflect')
        logger.info("Padded the mixture for boundary processing.")

    # Determine output tensor shape
    if config.training.get("target_instrument", None):
        out_shape = (1,) + tuple(mixture.shape)
    else:
        out_shape = (len(config.training.instruments),) + tuple(mixture.shape)
    result = torch.zeros(out_shape, dtype=torch.float32, device=device)
    counter = torch.zeros(out_shape, dtype=torch.float32, device=device)
    logger.info("Output shape determined: %s", out_shape)

    # Process the audio in overlapping chunks
    logger.info("Beginning inference over %d total samples...", mixture.shape[1])
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            i = 0
            total_len = mixture.shape[1]
            while i < total_len:
                part = mixture[:, i:i+C]
                length = part.shape[-1]
                if length < C:
                    if length > C // 2:
                        part = torch.nn.functional.pad(part, (0, C - length), mode='reflect')
                    else:
                        part = torch.nn.functional.pad(part, (0, C - length, 0, 0), mode='constant', value=0)
                    logger.debug("Padded a short segment: original length %d, padded to %d", length, C)
                # Run inference on the chunk
                output_chunk = model(part.unsqueeze(0))[0]
                window = torch.ones(C, device=device)
                if fade > 0:
                    fadein = torch.linspace(0, 1, fade, device=device)
                    fadeout = torch.linspace(1, 0, fade, device=device)
                    window[:fade] = fadein
                    window[-fade:] = fadeout
                if i == 0:
                    window[:fade] = 1
                if i + C >= total_len:
                    window[-fade:] = 1
                # Add the chunk's contribution to the result
                result[..., i:i+length] += output_chunk[..., :length] * window[:length]
                counter[..., i:i+length] += window[:length]
                logger.debug("Processed chunk from %d to %d", i, i+length)
                i += step

    # Average overlapping regions
    estimated_sources = (result / counter).cpu().numpy()
    if mixture.shape[1] > 2 * border and border > 0:
        estimated_sources = estimated_sources[..., border:-border]
    logger.info("Inference completed. Estimated sources shape: %s", estimated_sources.shape)

    # Choose the vocals (assume vocals is the first stem)
    vocals_est = estimated_sources[0]
    if vocals_est.shape[0] == 1:
        vocals_est = vocals_est[0]

    # Compute instrumental as original mix minus vocals
    orig = data.T
    instrumental_est = orig - vocals_est
    logger.info("Separated vocals and instrumental.")

    # Write output files to temporary WAV files
    tmp_v = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_i = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp_v.name, vocals_est.T, sr, subtype="FLOAT")
    sf.write(tmp_i.name, instrumental_est.T, sr, subtype="FLOAT")
    tmp_v.close(); tmp_i.close()
    logger.info("Temporary WAV files written: %s, %s", tmp_v.name, tmp_i.name)

    # Package into a ZIP archive
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as z:
        z.write(tmp_v.name, arcname="vocals.wav")
        z.write(tmp_i.name, arcname="instrumental.wav")
    os.remove(tmp_v.name)
    os.remove(tmp_i.name)
    zip_buffer.seek(0)
    logger.info("Packaged output into ZIP archive.")

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=separated.zip"}
    )
