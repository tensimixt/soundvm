import os
import io
import tempfile
import requests
import zipfile

import numpy as np
import torch
import soundfile as sf

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse

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
    print("Before conversion:", model_config["multi_stft_resolutions_window_sizes"], type(model_config["multi_stft_resolutions_window_sizes"]))
    model_config["multi_stft_resolutions_window_sizes"] = tuple(model_config["multi_stft_resolutions_window_sizes"])
    print("After conversion:", model_config["multi_stft_resolutions_window_sizes"], type(model_config["multi_stft_resolutions_window_sizes"]))

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
    # Download weights if not already present
    if not os.path.exists(MODEL_PATH):
        print("Downloading model weights...")
        resp = requests.get(MODEL_URL, stream=True)
        resp.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    # Initialize model with the plain dictionary configuration
    try:
        model = MelBandRoformer(**model_config)
    except Exception as e:
        print("Error during model initialization:", e)
        raise e
    # Load model state
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print("Model loaded and ready")

@app.post("/separate")
async def separate_audio(file: UploadFile = File(...)):
    """
    Separate the uploaded WAV file into vocals and instrumental.
    Returns a ZIP containing 'vocals.wav' and 'instrumental.wav'.
    """
    # Read the uploaded file into a NumPy array
    data, sr = sf.read(file.file)
    if data.ndim == 1:
        # Duplicate mono into stereo channels
        data = np.stack([data, data], axis=-1)
    # Convert to torch tensor and move to appropriate device
    mixture = torch.tensor(data.T, dtype=torch.float32).to(device)
    
    # Inference parameters from config
    C = config.inference.chunk_size       # chunk length (samples)
    N = config.inference.num_overlap      # overlap factor
    step = C // N
    fade = C // 10
    border = C - step
    if mixture.shape[1] > 2 * border and border > 0:
        mixture = torch.nn.functional.pad(mixture, (border, border), mode='reflect')
    
    # Determine output tensor shape
    if config.training.get("target_instrument", None):
        out_shape = (1,) + tuple(mixture.shape)
    else:
        out_shape = (len(config.training.instruments),) + tuple(mixture.shape)
    result = torch.zeros(out_shape, dtype=torch.float32, device=device)
    counter = torch.zeros(out_shape, dtype=torch.float32, device=device)
    
    # Process the audio in overlapping chunks
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
                result[..., i:i+length] += output_chunk[..., :length] * window[:length]
                counter[..., i:i+length] += window[:length]
                i += step
    
    # Average overlapping regions
    estimated_sources = (result / counter).cpu().numpy()
    if mixture.shape[1] > 2 * border and border > 0:
        estimated_sources = estimated_sources[..., border:-border]
    
    # Choose the vocals (assume vocals is the first stem)
    vocals_est = estimated_sources[0]
    if vocals_est.shape[0] == 1:
        vocals_est = vocals_est[0]
    
    # Compute instrumental as original mix minus vocals
    orig = data.T
    instrumental_est = orig - vocals_est
    
    # Write output files to temporary WAV files
    tmp_v = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_i = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp_v.name, vocals_est.T, sr, subtype="FLOAT")
    sf.write(tmp_i.name, instrumental_est.T, sr, subtype="FLOAT")
    tmp_v.close(); tmp_i.close()
    
    # Package into a ZIP archive
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as z:
        z.write(tmp_v.name, arcname="vocals.wav")
        z.write(tmp_i.name, arcname="instrumental.wav")
    os.remove(tmp_v.name)
    os.remove(tmp_i.name)
    zip_buffer.seek(0)
    
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=separated.zip"}
    )
