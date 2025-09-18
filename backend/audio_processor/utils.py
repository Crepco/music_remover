import librosa
import numpy as np
import soundfile as sf
from mutagen import File
import os

def validate_audio_file(filepath):
    """Validate if file is a valid audio file"""
    try:
        # Try to load a small portion of the file
        y, sr = librosa.load(filepath, duration=1.0, sr=None)
        return len(y) > 0 and sr > 0
    except Exception:
        return False

def get_file_info(filepath):
    """Extract metadata and basic info from audio file"""
    try:
        # Get basic audio info using librosa
        y, sr = librosa.load(filepath, sr=None)
        duration = len(y) / sr

        info = {
            'sample_rate': sr,
            'duration': duration,
            'channels': 'stereo' if y.ndim > 1 else 'mono',
            'samples': len(y),
            'file_size': os.path.getsize(filepath)
        }

        # Try to get metadata using mutagen
        try:
            audio_file = File(filepath)
            if audio_file is not None:
                info['title'] = audio_file.get('TIT2', ['Unknown'])[0] if audio_file.get('TIT2') else 'Unknown'
                info['artist'] = audio_file.get('TPE1', ['Unknown'])[0] if audio_file.get('TPE1') else 'Unknown'
                info['album'] = audio_file.get('TALB', ['Unknown'])[0] if audio_file.get('TALB') else 'Unknown'

                # Handle different tag formats
                if not info['title'] or info['title'] == 'Unknown':
                    info['title'] = audio_file.get('title', ['Unknown'])[0] if audio_file.get('title') else 'Unknown'
                if not info['artist'] or info['artist'] == 'Unknown':
                    info['artist'] = audio_file.get('artist', ['Unknown'])[0] if audio_file.get('artist') else 'Unknown'
                if not info['album'] or info['album'] == 'Unknown':
                    info['album'] = audio_file.get('album', ['Unknown'])[0] if audio_file.get('album') else 'Unknown'
        except Exception:
            info['title'] = 'Unknown'
            info['artist'] = 'Unknown'
            info['album'] = 'Unknown'

        return info

    except Exception as e:
        return {'error': str(e)}

def normalize_audio(audio, target_level=-3.0):
    """Normalize audio to target dB level"""
    if len(audio) == 0:
        return audio

    # Calculate current RMS level
    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        return audio

    # Calculate target RMS from dB
    target_rms = librosa.db_to_amplitude(target_level)

    # Apply normalization
    gain = target_rms / rms
    normalized = audio * gain

    # Prevent clipping
    peak = np.max(np.abs(normalized))
    if peak > 1.0:
        normalized = normalized / peak * 0.95

    return normalized

def fade_in_out(audio, sr, fade_duration=0.1):
    """Apply fade in and fade out to audio"""
    fade_samples = int(fade_duration * sr)

    if len(audio) <= 2 * fade_samples:
        return audio

    audio_faded = audio.copy()

    # Fade in
    fade_in_curve = np.linspace(0, 1, fade_samples)
    audio_faded[:fade_samples] *= fade_in_curve

    # Fade out
    fade_out_curve = np.linspace(1, 0, fade_samples)
    audio_faded[-fade_samples:] *= fade_out_curve

    return audio_faded

def detect_audio_properties(audio, sr):
    """Detect various properties of audio signal"""
    properties = {}

    # Basic statistics
    properties['rms'] = librosa.feature.rms(y=audio)[0]
    properties['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio)[0]

    # Spectral features
    properties['spectral_centroid'] = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    properties['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    properties['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]

    # Tempo and beat
    try:
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        properties['tempo'] = tempo
        properties['beat_frames'] = len(beats)
    except Exception:
        properties['tempo'] = None
        properties['beat_frames'] = 0

    # Harmonic-percussive ratio
    try:
        harmonic, percussive = librosa.effects.hpss(audio)
        harmonic_energy = np.sum(harmonic**2)
        percussive_energy = np.sum(percussive**2)
        total_energy = harmonic_energy + percussive_energy

        if total_energy > 0:
            properties['harmonic_ratio'] = harmonic_energy / total_energy
            properties['percussive_ratio'] = percussive_energy / total_energy
        else:
            properties['harmonic_ratio'] = 0
            properties['percussive_ratio'] = 0
    except Exception:
        properties['harmonic_ratio'] = 0
        properties['percussive_ratio'] = 0

    return properties

def convert_to_mono(audio):
    """Convert stereo audio to mono"""
    if audio.ndim == 1:
        return audio
    elif audio.ndim == 2:
        return np.mean(audio, axis=0)
    else:
        return np.mean(audio[:2], axis=0)  # Take first two channels

def convert_to_stereo(audio):
    """Convert mono audio to stereo"""
    if audio.ndim == 1:
        return np.array([audio, audio])
    elif audio.ndim == 2 and audio.shape[0] == 1:
        return np.array([audio[0], audio[0]])
    else:
        return audio

def apply_gain(audio, gain_db):
    """Apply gain in dB to audio signal"""
    if gain_db == 0:
        return audio

    gain_linear = librosa.db_to_amplitude(gain_db)
    return audio * gain_linear

def remove_silence(audio, sr, threshold_db=-40, min_silence_duration=0.5):
    """Remove silence from beginning and end of audio"""
    # Convert threshold to amplitude
    threshold = librosa.db_to_amplitude(threshold_db)

    # Find non-silent regions
    frame_length = 2048
    hop_length = 512

    # Calculate RMS energy
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

    # Find frames above threshold
    active_frames = rms > threshold

    if not np.any(active_frames):
        return audio  # All silent, return original

    # Find first and last active frames
    first_active = np.argmax(active_frames)
    last_active = len(active_frames) - 1 - np.argmax(active_frames[::-1])

    # Convert frame indices to sample indices
    start_sample = first_active * hop_length
    end_sample = min(len(audio), (last_active + 1) * hop_length + frame_length)

    return audio[start_sample:end_sample]

def calculate_audio_fingerprint(audio, sr):
    """Calculate a simple audio fingerprint for similarity detection"""
    # Use chromagram as a simple fingerprint
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    fingerprint = np.mean(chroma, axis=1)
    return fingerprint

def estimate_vocal_presence(audio, sr):
    """Estimate the presence of vocals in audio"""
    # This is a simple heuristic - can be improved with ML models

    # Convert to mono if stereo
    if audio.ndim > 1:
        mono = np.mean(audio, axis=0)
    else:
        mono = audio

    # Calculate spectral features that might indicate vocals
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=mono, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(mono))

    # Vocal presence heuristic (very basic)
    # Vocals typically have moderate spectral centroid and ZCR
    vocal_score = 0.0

    if 1000 < spectral_centroid < 4000:  # Typical vocal range
        vocal_score += 0.3

    if 0.05 < zero_crossing_rate < 0.2:  # Typical vocal ZCR
        vocal_score += 0.3

    # Check for harmonic content in vocal frequency range
    try:
        harmonic, percussive = librosa.effects.hpss(mono)
        vocal_band = librosa.feature.spectral_centroid(y=harmonic, sr=sr)
        if np.mean(vocal_band) > 800:  # Harmonic content in vocal range
            vocal_score += 0.4
    except Exception:
        pass

    return min(1.0, vocal_score)
