import numpy as np
from scipy import signal
import librosa

class VocalRemovalFilter:
    """Filters for vocal removal and enhancement"""
    
    def apply_vocal_suppression_filter(self, audio, sr):
        """Apply frequency filtering to suppress vocal ranges"""
        # Typical vocal frequency ranges:
        # Male vocals: 85-255 Hz (fundamental), harmonics up to 4kHz
        # Female vocals: 165-265 Hz (fundamental), harmonics up to 5kHz
        
        # Create a notch filter for vocal fundamental frequencies
        nyquist = sr / 2
        
        # Define vocal frequency ranges to attenuate
        vocal_ranges = [
            (80, 300),    # Fundamental vocal frequencies
            (2000, 4000), # Vocal formants
        ]
        
        filtered_audio = audio.copy()
        
        for low_freq, high_freq in vocal_ranges:
            # Create bandstop filter
            low_norm = low_freq / nyquist
            high_norm = high_freq / nyquist
            
            # Ensure frequencies are within valid range
            low_norm = max(0.01, min(0.99, low_norm))
            high_norm = max(0.01, min(0.99, high_norm))
            
            if low_norm < high_norm:
                b, a = signal.butter(4, [low_norm, high_norm], btype='bandstop')
                filtered_audio = signal.filtfilt(b, a, filtered_audio)
        
        return filtered_audio
    
    def apply_center_channel_isolation(self, left_channel, right_channel, method='subtract'):
        """Isolate or remove center channel content"""
        if method == 'subtract':
            # Remove center channel (vocals)
            return left_channel - right_channel
        elif method == 'add':
            # Isolate center channel (vocals)
            return (left_channel + right_channel) / 2
        elif method == 'karaoke':
            # Advanced karaoke effect
            return (left_channel - right_channel) * 0.5 + (left_channel + right_channel) * 0.1
        else:
            raise ValueError(f"Unknown method: {method}")

class HarmonicPercussiveFilter:
    """Filters for harmonic-percussive separation"""
    
    def separate_harmonic_percussive(self, audio, sr, margin=(1.0, 5.0)):
        """Separate harmonic and percussive components"""
        harmonic, percussive = librosa.effects.hpss(audio, margin=margin)
        return harmonic, percussive
    
    def enhance_harmonic(self, audio, sr):
        """Enhance harmonic content (melodic instruments)"""
        harmonic, _ = self.separate_harmonic_percussive(audio, sr)
        return harmonic
    
    def enhance_percussive(self, audio, sr):
        """Enhance percussive content (drums, beats)"""
        _, percussive = self.separate_harmonic_percussive(audio, sr)
        return percussive

class FrequencyFilter:
    """General frequency filtering utilities"""
    
    @staticmethod
    def highpass_filter(audio, sr, cutoff_freq, order=5):
        """Apply high-pass filter"""
        nyquist = sr / 2
        cutoff_norm = cutoff_freq / nyquist
        cutoff_norm = max(0.01, min(0.99, cutoff_norm))
        
        b, a = signal.butter(order, cutoff_norm, btype='high')
        return signal.filtfilt(b, a, audio)
    
    @staticmethod
    def lowpass_filter(audio, sr, cutoff_freq, order=5):
        """Apply low-pass filter"""
        nyquist = sr / 2
        cutoff_norm = cutoff_freq / nyquist
        cutoff_norm = max(0.01, min(0.99, cutoff_norm))
        
        b, a = signal.butter(order, cutoff_norm, btype='low')
        return signal.filtfilt(b, a, audio)
    
    @staticmethod
    def bandpass_filter(audio, sr, low_freq, high_freq, order=5):
        """Apply band-pass filter"""
        nyquist = sr / 2
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        low_norm = max(0.01, min(0.99, low_norm))
        high_norm = max(0.01, min(0.99, high_norm))
        
        if low_norm >= high_norm:
            return audio  # Invalid range, return original
        
        b, a = signal.butter(order, [low_norm, high_norm], btype='band')
        return signal.filtfilt(b, a, audio)
    
    @staticmethod
    def bandstop_filter(audio, sr, low_freq, high_freq, order=5):
        """Apply band-stop (notch) filter"""
        nyquist = sr / 2
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        low_norm = max(0.01, min(0.99, low_norm))
        high_norm = max(0.01, min(0.99, high_norm))
        
        if low_norm >= high_norm:
            return audio  # Invalid range, return original
        
        b, a = signal.butter(order, [low_norm, high_norm], btype='bandstop')
        return signal.filtfilt(b, a, audio)

class SpectralFilter:
    """Spectral domain filtering"""
    
    @staticmethod
    def spectral_gate(audio, sr, threshold_db=-20, n_fft=2048):
        """Apply spectral gating to remove low-level noise"""
        # Compute STFT
        stft = librosa.stft(audio, n_fft=n_fft)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Convert threshold to linear scale
        threshold_linear = librosa.db_to_amplitude(threshold_db)
        
        # Apply gate
        gate_mask = magnitude > (threshold_linear * np.max(magnitude))
        gated_magnitude = magnitude * gate_mask
        
        # Reconstruct
        gated_stft = gated_magnitude * np.exp(1j * phase)
        return librosa.istft(gated_stft)
    
    @staticmethod
    def spectral_subtraction(audio, sr, alpha=2.0, beta=0.01):
        """Apply spectral subtraction for noise reduction"""
        # Compute STFT
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise from first few frames
        noise_frames = min(10, magnitude.shape[1] // 4)
        noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # Apply spectral subtraction
        enhanced_magnitude = magnitude - alpha * noise_spectrum
        
        # Ensure minimum level
        enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
        
        # Reconstruct
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        return librosa.istft(enhanced_stft)