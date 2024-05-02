import sys
import os
import importlib
import audio_mae

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'submodules', 'audio-mae'))
audiomae = importlib.import_module(os.path.join('models_mae'))

from ..submodules.audio_mae.models_mae import MaskedAutoEncoderViT

MaskedAutoEncoderViT = getattr(audiomae, 'MaskedAutoEncoderViT')

