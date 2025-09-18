import librosa
import numpy as np
import soundfile as sf
from scipy import signal
import os
import tempfile
from .filters import VocalRemovalFilter, HarmonicPercussiveFilter
from .utils import normalize_audio


class AudioSeparator:
    def __init__(self):
        self.vocal_removal_filter = VocalRemovalFilter()
        self.hp_filter = HarmonicPercussiveFilter()

    def process(self, input_path, method='vocal_removal', quality='medium', progress_callback=None):
        """
        Process the input audio file and separate components based on the selected method.

        Args:
            input_path (str): Path to the input audio file.
            method (str): Separation method to use ('vocal_removal', 'instrumental_isolation', 'harmonic_percussive').
            quality (str): Quality setting ('low', 'medium', 'high').
            progress_callback (callable): Function to call with progress updates.

        Returns:
            str: Path to the output file.
        """
        # default callback does nothing
        if progress_callback is None:
            progress_callback = lambda x: None

        progress_callback(0.1)

        # Load audio
        y, sr = librosa.load(input_path, sr=None, mono=False)
        progress_callback(0.2)

        # Ensure stereo shape
        if y.ndim == 1:
            y = np.array([y, y])  # Convert mono to stereo
        elif y.shape[0] > 2:
            y = y[:2]  # Take first two channels if more than stereo
        progress_callback(0.3)

        params = self._get_quality_params(quality)

        if method == 'vocal_removal':
            processed_audio = self._vocal_removal(y, sr, params, progress_callback)
        elif method == 'instrumental_isolation':
            processed_audio = self._instrumental_isolation(y, sr, params, progress_callback)
        elif method == 'harmonic_percussive':
            processed_audio = self._harmonic_percussive_separation(y, sr, params, progress_callback)
        else:
            raise ValueError(f"Unknown method: {method}")

        progress_callback(0.9)

        # Save output
        output_file = self._save_output(processed_audio, sr, input_path)

        progress_callback(1.0)

        return output_file

    def _get_quality_params(self, quality):
        """Get processing parameters based on quality setting"""
        if quality == 'low':
            return {'n_fft': 1024, 'hop_length': 512, 'win_length': 1024}
        elif quality == 'medium':
            return {'n_fft': 2048, 'hop_length': 512, 'win_length': 2048}
        else:  # high
            return {'n_fft': 4096, 'hop_length': 1024, 'win_length': 4096}

    def _vocal_removal(self, y, sr, params, progress_callback):
        """Remove vocals using center channel extraction and frequency filtering"""
        left, right = y[0], y[1]

        progress_callback(0.4)

        # Center channel extraction
        vocal_removed = left - right

        progress_callback(0.6)

        # Enhanced vocal removal with spectral processing
        stft_left = librosa.stft(left, n_fft=params['n_fft'], hop_length=params['hop_length'])
        stft_right = librosa.stft(right, n_fft=params['n_fft'], hop_length=params['hop_length'])

        mag_left = np.abs(stft_left)
        mag_right = np.abs(stft_right)
        phase_left = np.angle(stft_left)

        progress_callback(0.7)

        similarity = np.minimum(mag_left, mag_right) / (np.maximum(mag_left, mag_right) + 1e-10)
        vocal_mask = similarity > 0.8

        mag_processed = mag_left.copy()
        mag_processed[vocal_mask] *= 0.1

        stft_processed = mag_processed * np.exp(1j * phase_left)
        enhanced_removal = librosa.istft(stft_processed, hop_length=params['hop_length'])

        progress_callback(0.8)

        vocal_removed_normalized = normalize_audio(vocal_removed)
        enhanced_normalized = normalize_audio(enhanced_removal)

        result = 0.3 * vocal_removed_normalized + 0.7 * enhanced_normalized

        # Apply additional filtering to suppress vocal frequencies
        result = self.vocal_removal_filter.apply_vocal_suppression_filter(result, sr)

        return normalize_audio(result)

    def _instrumental_isolation(self, y, sr, params, progress_callback):
        """Isolate instrumental parts by enhancing them and reducing vocals"""
        left, right = y[0], y[1]
        progress_callback(0.4)

        stereo_enhancement = (left + right) * 0.7 + (left - right) * 0.3

        progress_callback(0.6)

        harmonic, percussive = librosa.effects.hpss(stereo_enhancement, margin=(1.0, 5.0))
        progress_callback(0.8)

        result = 0.8 * harmonic + 0.2 * percussive
        return normalize_audio(result)

    def _harmonic_percussive_separation(self, y, sr, params, progress_callback):
        """Separate harmonic and percussive components"""
        mono = np.mean(y, axis=0)
        progress_callback(0.4)

        harmonic, percussive = librosa.effects.hpss(mono, margin=(1.0, 5.0), kernel_size=(17, 17))
        progress_callback(0.7)

        result = harmonic
        progress_callback(0.8)

        return normalize_audio(result)

    def _save_output(self, audio, sr, original_file):
        """Save processed audio to temporary file"""
        _, ext = os.path.splitext(original_file)
        if ext.lower() not in ['.wav', '.flac']:
            ext = '.wav'

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            output_path = tmp_file.name

        sf.write(output_path, audio, sr, format='WAV' if ext == '.wav' else 'FLAC')
        return output_path
