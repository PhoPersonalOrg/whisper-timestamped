"""Vendored CrisperWhisper inference (transformers + optional CT2 on Linux).

Source: https://github.com/nyrahealth/CrisperWhisper (MIT).
On Windows use ``backend="transformers"``. On Linux/WSL2 install the
``crisper_ct2`` extra for CTranslate2 + speculative decoding.
"""

from whisper_timestamped.crisperwhisper import _nvidia_libs

# Must run before anything imports ctranslate2 (see _nvidia_libs docstring).
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
    "check_speculative_support",
    "ct2_fork_available",
]
__version__ = "2.0.3"

# Fork APIs required for the CT2 engine (see engine._REQUIRED_FORK_APIS).
_CT2_FORK_APIS = (
    "prefill",
    "forward_step",
    "set_alignment_heads",
    "generate_greedy_with_attention",
)


def ct2_fork_available() -> bool:
    """True when the CrisperWhisper CT2 fork (not upstream ctranslate2) is usable."""
    import importlib.util

    if importlib.util.find_spec("whisper_timestamped.crisperwhisper.engine") is None:
        return False
    try:
        import ctranslate2
    except ImportError:
        return False
    whisper_cls = getattr(ctranslate2.models, "Whisper", None)
    if whisper_cls is None:
        return False
    return all(hasattr(whisper_cls, m) for m in _CT2_FORK_APIS)


def check_speculative_support() -> None:
    """Verify the CrisperWhisper CT2 fork is installed for speculative decoding.

    Raises ``ImportError`` if the installed ctranslate2 lacks the custom
    speculative-decoding APIs. Called lazily when speculative decoding is
    first requested.
    """
    import ctranslate2

    whisper_cls = getattr(ctranslate2.models, "Whisper", None)
    if whisper_cls is None:
        raise ImportError(
            "ctranslate2 Whisper model not available — "
            "install ctranslate2-crisperwhisper for speculative decoding."
        )

    required = ("prefill", "forward_step_greedy", "forward_batch_greedy")
    missing = [m for m in required if not hasattr(whisper_cls, m)]
    if missing:
        raise ImportError(
            f"The installed ctranslate2 ({ctranslate2.__version__}) is missing "
            f"speculative-decoding APIs: {', '.join(missing)}.\n"
            "You likely have the upstream package installed on top of the fork "
            "(both own the ctranslate2 import directory, so the last install "
            "wins). Fix with:\n"
            "  pip uninstall -y ctranslate2\n"
            "  pip install --force-reinstall ctranslate2-crisperwhisper"
        )
