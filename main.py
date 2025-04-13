import os, io, tempfile, requests, zipfile
import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse

# URL to model weights (Hugging Face)
MODEL_URL = "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt"
MODEL_PATH = "MelBandRoformer.ckpt"

# Load config (YAML) for model architecture and inference params
import yaml
from ml_collections import ConfigDict
with open("config_vocals_mel_band_roformer.yaml") as f:
    config = ConfigDict(yaml.safe_load(f))
    
# Import model class definition (from Kimberley Jensen's implementation)
from models.mel_band_roformer import MelBandRoformer  # assume we've included the model code in our project

# Initialize FastAPI
app = FastAPI()

# Global model variable
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
def load_model():
    """Download and load the Mel-Band Roformer model at app startup."""
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
    # Load model architecture and weights
    model = MelBandRoformer(**dict(config.model))
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    # Enable CUDA optimization if available
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print("Model loaded and ready")

@app.post("/separate")
async def separate_audio(file: UploadFile = File(...)):
    """
    Separate the uploaded WAV file into vocals and instrumental.
    Returns a ZIP with 'vocals.wav' and 'instrumental.wav'.
    """
    # Read the uploaded file into numpy array
    data, sr = sf.read(file.file)
    if data.ndim == 1:
        # if mono, duplicate channel to make stereo
        data = np.stack([data, data], axis=-1)
    # Convert to torch tensor (shape [channels, samples])
    mixture = torch.tensor(data.T, dtype=torch.float32)
    mixture = mixture.to(device)
    # Perform separation in chunks using model (as per original code)
    C = config.inference.chunk_size        # chunk length (samples)
    N = config.inference.num_overlap       # overlap factor
    step = C // N
    fade = C // 10
    # Pad the mixture for seamless chunking at edges
    border = C - step
    if mixture.shape[1] > 2 * border and border > 0:
        mixture = torch.nn.functional.pad(mixture, (border, border), mode='reflect')
    # Prepare output tensors
    if config.training.get("target_instrument", None):
        # only one target (vocals)
        out_shape = (1,) + tuple(mixture.shape)
    else:
        out_shape = (len(config.training.instruments),) + tuple(mixture.shape)
    result = torch.zeros(out_shape, dtype=torch.float32, device=device)
    counter = torch.zeros(out_shape, dtype=torch.float32, device=device)
    # Process in chunks with overlap
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            i = 0
            total_len = mixture.shape[1]
            while i < total_len:
                part = mixture[:, i:i+C]
                length = part.shape[-1]
                if length < C:
                    # Pad last chunk
                    if length > C // 2:
                        part = torch.nn.functional.pad(part, (0, C - length), mode='reflect')
                    else:
                        part = torch.nn.functional.pad(part, (0, C - length, 0, 0), mode='constant', value=0)
                # Run model on this chunk
                output_chunk = model(part.unsqueeze(0))[0]  # shape: [num_stems, channels, C]
                # Create fade window for overlap-add
                window = torch.ones(C, device=device)
                if fade > 0:
                    fadein = torch.linspace(0, 1, fade, device=device)
                    fadeout = torch.linspace(1, 0, fade, device=device)
                    window[:fade] = fadein
                    window[-fade:] = fadeout
                # For first and last chunks, adjust window to avoid double-fading
                if i == 0:
                    window[:fade] = 1
                if i + C >= total_len:
                    window[-fade:] = 1
                # Add weighted output to result
                result[..., i:i+length] += output_chunk[..., :length] * window[:length]
                counter[..., i:i+length] += window[:length]
                i += step
    # Normalize by overlap count
    estimated_sources = (result / counter).cpu().numpy()
    # Remove padding
    if mixture.shape[1] > 2 * border and border > 0:
        estimated_sources = estimated_sources[..., border:-border]
    # If only one target (vocals), get that; else assume first is vocals
    if config.training.get("target_instrument", None):
        vocals_est = estimated_sources[0]  # shape [channels, samples]
    else:
        vocals_est = estimated_sources[0]
    # Ensure shape back to original (if original was mono, take one channel)
    if vocals_est.shape[0] == 1:
        vocals_est = vocals_est[0]
    # Instrumental = original - vocals
    orig = data.T  # original audio as [channels, samples]
    instrumental_est = orig - vocals_est
    # Write output WAVs to temporary files
    tmp_v = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_i = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp_v.name, vocals_est.T, sr, subtype='FLOAT')
    sf.write(tmp_i.name, instrumental_est.T, sr, subtype='FLOAT')
    tmp_v.close(); tmp_i.close()
    # Package into a zip
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as z:
        z.write(tmp_v.name, arcname="vocals.wav")
        z.write(tmp_i.name, arcname="instrumental.wav")
    # Clean up temp files
    os.remove(tmp_v.name); os.remove(tmp_i.name)
    zip_buffer.seek(0)
    # Return the zip file as streaming response
    return StreamingResponse(zip_buffer, media_type="application/zip", 
                              headers={"Content-Disposition": "attachment; filename=separated.zip"})
