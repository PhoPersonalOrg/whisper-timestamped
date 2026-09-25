"""Vendored CrisperWhisper inference (transformers backend).

Source: https://github.com/nyrahealth/CrisperWhisper (MIT).
This copy omits the CTranslate2 fork path so it coexists with faster-whisper
on Windows. Use ``backend="transformers"`` (the default for ``auto`` here).
"""

from whisper_timestamped.crisperwhisper import _nvidia_libs

# No-op off Linux; harmless when CT2 is unused.
_nvidia_libs.preload()

from whisper_timestamped.crisperwhisper.model import (  # noqa: E402
    DEFAULT_MODEL,
    OFFICIAL_MODELS,
    CrisperWhisperModel,
    resolve_model_id,
)
from whisper_timestamped.crisperwhisper.result import (  # noqa: E402
    ChunkResult,
    TranscriptionResult,
    WordTimestamp,
)

__all__ = [
    "CrisperWhisperModel",
    "TranscriptionResult",
    "ChunkResult",
    "WordTimestamp",
    "DEFAULT_MODEL",
    "OFFICIAL_MODELS",
    "resolve_model_id",
]
__version__ = "2.0.3"
