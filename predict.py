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
        with open("config_vocals_mel_band_roformer.yaml", "r") as f:
            config = ConfigDict(yaml.safe_load(f))
        
        model_config = dict(config.model)
        if "multi_stft_resolutions_window_sizes" in model_config:
            model_config["multi_stft_resolutions_window_sizes"] = tuple(
                model_config["multi_stft_resolutions_window_sizes"]
            )

        self.model = MelBandRoformer(**model_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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

        # Get inference parameters
        C = self.config.inference.chunk_size
        N = self.config.inference.num_overlap
        step = C // N
        fade = C // 10
        border = C - step

        if mixture.shape[1] > 2 * border and border > 0:
            mixture = torch.nn.functional.pad(mixture, (border, border), mode='reflect')

        out_shape = (1,) + tuple(mixture.shape)
        result = torch.zeros(out_shape, dtype=torch.float32, device=self.device)
        counter = torch.zeros(out_shape, dtype=torch.float32, device=self.device)

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
                            part = torch.nn.functional.pad(part, (0, C - length), mode='constant', value=0)
                    
                    output_chunk = self.model(part.unsqueeze(0))[0]
                    window = torch.ones(C, device=self.device)
                    if fade > 0:
                        fadein = torch.linspace(0, 1, fade, device=self.device)
                        fadeout = torch.linspace(1, 0, fade, device=self.device)
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

        # Get vocals
        vocals_est = estimated_sources[0]
        if vocals_est.shape[0] == 1:
            vocals_est = vocals_est[0]

        # Save output
        output_path = Path("/tmp/output.wav")
        sf.write(str(output_path), vocals_est.T, sr, subtype="FLOAT")
        
        return output_path
