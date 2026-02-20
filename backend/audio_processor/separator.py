import os
import tempfile
import numpy as np
import torch
import librosa
import soundfile as sf

# Set backend BEFORE importing torchaudio
os.environ['TORCHAUDIO_USE_BACKEND'] = 'soundfile'

# Now import demucs (which will import torchaudio)
from demucs.pretrained import get_model
from demucs.apply import apply_model


class AudioSeparator:
    def __init__(self):
        """Initialize Demucs model"""
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def _load_model(self):
        """Lazy load the model (only when needed)"""
        if self.model is None:
            self.model = get_model('htdemucs')
            self.model.to(self.device)
            self.model.eval()
    
    def process(self, input_path, progress_callback=None):
        """
        Remove background music and return isolated vocals using Demucs.

        Args:
            input_path (str): Path to input audio file.
            progress_callback (callable): Called with float 0.0-1.0 for progress.

        Returns:
            str: Path to the output vocals WAV file.
        """
        if progress_callback is None:
            progress_callback = lambda x: None

        progress_callback(0.05)
        
        # Load audio with librosa (handles all formats, avoids torchcodec)
        y, sr = librosa.load(input_path, sr=None, mono=False)
        progress_callback(0.15)
        
        # Ensure stereo: librosa returns (samples,) for mono or (channels, samples) for stereo
        if y.ndim == 1:
            y = np.vstack([y, y])  # Mono to stereo: (samples,) -> (2, samples)
        elif y.ndim == 2:
            if y.shape[0] > 2:
                y = y[:2]  # Take first 2 channels
            elif y.shape[0] == 1:
                y = np.vstack([y[0], y[0]])  # Mono to stereo
        
        # Convert to torch tensor: shape should be (channels, samples)
        # Then add batch dimension: (batch, channels, samples)
        wav = torch.from_numpy(y).float()  # (channels, samples)
        wav = wav.unsqueeze(0)  # Add batch: (1, channels, samples)
        
        progress_callback(0.25)
        
        # Load model
        self._load_model()
        progress_callback(0.35)
        
        # Move to device
        wav = wav.to(self.device)
        
        # Apply model
        # Returns tensor of shape (batch, sources, channels, samples)
        # Sources: [drums, bass, other, vocals] = 4 sources
        progress_callback(0.4)
        with torch.no_grad():
            sources = apply_model(
                self.model, 
                wav, 
                shifts=1, 
                split=True, 
                overlap=0.25, 
                progress=False  # Disable progress to avoid issues
            )
        
        progress_callback(0.9)
        
        # sources shape: (batch=1, sources=4, channels=2, samples)
        # Extract vocals (last source, index 3)
        vocals = sources[0, 3].cpu().numpy()  # (channels, samples)
        
        # Convert to format soundfile expects: (samples, channels)
        if vocals.ndim == 2 and vocals.shape[0] == 2:
            # Stereo: (2, samples) -> (samples, 2)
            vocals_out = vocals.T
        elif vocals.ndim == 1:
            # Mono: (samples,)
            vocals_out = vocals
        else:
            # Fallback: take first channel
            vocals_out = vocals[0] if vocals.ndim > 1 else vocals
        
        progress_callback(0.95)
        
        # Save to temporary file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        tmp.close()
        sf.write(tmp.name, vocals_out, sr, format='WAV')
        
        progress_callback(1.0)
        return tmp.name


# Global instance (lazy loaded)
_separator_instance = None

def get_separator():
    """Get or create separator instance"""
    global _separator_instance
    if _separator_instance is None:
        _separator_instance = AudioSeparator()
    return _separator_instance
