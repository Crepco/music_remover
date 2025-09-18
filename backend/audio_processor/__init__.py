# Audio Processor Package
from .separator import AudioSeparator
from .filters import VocalRemovalFilter, HarmonicPercussiveFilter, FrequencyFilter, SpectralFilter
from .utils import validate_audio_file, get_file_info, normalize_audio

__version__ = "1.0.0"
__author__ = "Music Separator"

__all__ = [
    'AudioSeparator',
    'VocalRemovalFilter', 
    'HarmonicPercussiveFilter',
    'FrequencyFilter',
    'SpectralFilter',
    'validate_audio_file',
    'get_file_info',
    'normalize_audio'
]