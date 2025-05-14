from cog import BasePredictor, Input, Path
import torch
import soundfile as sf
import numpy as np
import yaml
from ml_collections import ConfigDict
from models.mel_band_roformer import MelBandRoformer

MODEL_URL = "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt"
MODEL_PATH = "MelBandRoformer.ckpt"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        # Load config
        with open("config_vocals_mel_band_roformer.yaml", "r") as f:
            config = ConfigDict(yaml.safe_load(f))
        
        # Convert config to dict
        model_config = dict(config.model)
        if "multi_stft_resolutions_window_sizes" in model_config:
            model_config["multi_stft_resolutions_window_sizes"] = tuple(
                model_config["multi_stft_resolutions_window_sizes"]
            )

        # Initialize model
        self.model = MelBandRoformer(**model_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Download and load weights
        if not Path(MODEL_PATH).exists():
            torch.hub.download_url_to_file(MODEL_URL, MODEL_PATH)
        
        state = torch.load(MODEL_PATH, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        
        self.config = config

    def predict(
        self,
        audio: Path = Input(description="Input audio file to separate"),
    ) -> Path:
        """Run inference on the input audio file"""
        # Read audio
        data, sr = sf.read(str(audio))
        
        if data.ndim == 1:
            data = np.stack([data, data], axis=-1)

        # Convert to tensor
        mixture = torch.tensor(data.T, dtype=torch.float32).to(self.device)

        # Get inference params
        C = self.config.inference.chunk_size
        N = self.config.inference.num_overlap
        step = C // N
        fade = C // 10
        border = C - step

        if mixture.shape[1] > 2 * border and border > 0:
            mixture = torch.nn.functional.pad(mixture, (border, border), mode='reflect')

        # Setup output tensors
        out_shape = (1,) + tuple(mixture.shape)
        result = torch.zeros(out_shape, dtype=torch.float32, device=self.device)
        counter = torch.zeros(out_shape, dtype=torch.float32, device=self.device)

        # Process audio
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                i = 0
                total_len = mixture.shape[1]
                while i < total_len:
                    # Process chunks similar to main.py
                    # [... rest of processing logic ...]
                    pass

        # Generate output files
        output_path = Path("/tmp/output.wav")
        vocals_est = estimated_sources[0]
        if vocals_est.shape[0] == 1:
            vocals_est = vocals_est[0]
        
        sf.write(str(output_path), vocals_est.T, sr, subtype="FLOAT")
        
        return output_path
