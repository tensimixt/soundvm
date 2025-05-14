import torch
from cog import BasePredictor, Input, Path
from models.mel_band_roformer import MelBandRoformer
import yaml
from ml_collections import ConfigDict
import soundfile as sf
import numpy as np

class Predictor(BasePredictor):
    def setup(self):
        # Load config
        with open("config_vocals_mel_band_roformer.yaml", "r") as f:
            config = yaml.safe_load(f)
        self.config = ConfigDict(config)
        
        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MelBandRoformer(**self.config["model"]).to(self.device)
        
        # Load weights
        state_dict = torch.load("model.pth", map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, audio: Path = Input(description="Input audio file")) -> Path:
        # Load audio
        audio_data, sr = sf.read(str(audio))
        if len(audio_data.shape) == 1:
            audio_data = np.stack([audio_data, audio_data])
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio_data).to(self.device)
        
        # Process
        with torch.no_grad():
            output = self.model(audio_tensor)
        
        # Save and return
        output_path = Path("output.wav")
        sf.write(str(output_path), output.cpu().numpy().T, sr)
        return output_path
