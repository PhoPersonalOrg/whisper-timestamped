"""Adapter: vendored CrisperWhisper → whisper-timestamped result dict.

Uses the transformers inference path only (no CTranslate2 fork). Maps
``TranscriptionResult`` onto the existing ``{text, language, segments}``
schema so callers of ``transcribe_timestamped`` keep working.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from whisper_timestamped.crisperwhisper import (
    OFFICIAL_MODELS,
    CrisperWhisperModel,
    TranscriptionResult,
    resolve_model_id,
)

# OpenAI Whisper sizes that are not CrisperWhisper 2.0 shorthands.
# (large / turbo / medium / small ARE Crisper shorthands and resolve to Nyra weights.)
_STOCK_WHISPER_ONLY = {
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small.en",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
}

_CRISPER_SHORTHANDS = set(OFFICIAL_MODELS.keys())


class CrisperWhisperAsTimestamped:
    """Thin wrapper so ``isinstance`` / identity checks can detect the backend."""

    def __init__(self, model: CrisperWhisperModel, name: str, device: str):
        self.model = model
        self.name = name
        self.device = device

    def __repr__(self) -> str:
        return f"CrisperWhisperAsTimestamped(name={self.name!r}, device={self.device!r})"


def is_crisper_model(model: Any) -> bool:
    return isinstance(model, CrisperWhisperAsTimestamped)


def _resolve_device(device: Optional[Union[str, Any]]) -> str:
    if device is None:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    device_str = str(device)
    if device_str.startswith("cuda"):
        return "cuda"
    return device_str


def _compute_type_for_device(device: str) -> str:
    return "float16" if device == "cuda" else "float32"


def validate_crisper_model_name(name: str) -> str:
    """Resolve a Crisper model id; raise a clear error for stock-only Whisper names."""
    # Paths and HF ids pass through.
    if "/" in name or "\\" in name or name.endswith((".pt", ".bin", ".ckpt")):
        return resolve_model_id(name)

    if name in _STOCK_WHISPER_ONLY:
        shorthands = ", ".join(sorted(_CRISPER_SHORTHANDS))
        raise ValueError(
            f"Model name {name!r} is an OpenAI Whisper size, not a CrisperWhisper "
            f"checkpoint. With backend='crisperwhisper', use one of: {shorthands}, "
            f"or a HuggingFace id / local path (e.g. nyralabs/CrisperWhisper2.0_small)."
        )

    return resolve_model_id(name)


def load_crisper_model(
    name: str,
    device: Optional[Union[str, Any]] = None,
) -> CrisperWhisperAsTimestamped:
    """Load CrisperWhisper 2.0 weights on the transformers backend."""
    resolved = validate_crisper_model_name(name)
    device_str = _resolve_device(device)
    compute_type = _compute_type_for_device(device_str)
    model = CrisperWhisperModel(
        resolved,
        backend="transformers",
        device=device_str,
        compute_type=compute_type,
    )
    return CrisperWhisperAsTimestamped(model, name=resolved, device=device_str)


def map_crisper_result(
    result: TranscriptionResult,
    *,
    remove_punctuation_from_words: bool = False,
) -> dict:
    """Map a CrisperWhisper ``TranscriptionResult`` to the timestamped schema."""
    words_src = result.words or []
    flat_words = []
    for w in words_src:
        if w.start is None or w.end is None:
            continue
        text = w.word
        if remove_punctuation_from_words:
            text = _strip_punctuation(text)
        flat_words.append(
            {
                "text": text,
                "start": float(w.start),
                "end": float(w.end),
            }
        )

    segments = _segments_from_chunks_or_words(result, flat_words)
    return {
        "text": result.text,
        "language": result.language,
        "segments": segments,
    }


def _strip_punctuation(text: str) -> str:
    import string

    return text.strip(string.punctuation + " ")


def _segments_from_chunks_or_words(
    result: TranscriptionResult,
    flat_words: list[dict],
) -> list[dict]:
    chunks = result.chunks
    if not chunks:
        if flat_words:
            start = flat_words[0]["start"]
            end = flat_words[-1]["end"]
        else:
            start = 0.0
            end = float(result.duration) if result.duration else 0.0
        return [
            {
                "id": 0,
                "start": start,
                "end": end,
                "text": result.text,
                "words": flat_words,
            }
        ]

    segments = []
    remaining = list(flat_words)
    for i, chunk in enumerate(chunks):
        c_start = float(chunk.start_sec)
        c_end = float(chunk.end_sec)
        if i < len(chunks) - 1:
            seg_words = [w for w in remaining if w["start"] < c_end - 1e-6]
            remaining = [w for w in remaining if w["start"] >= c_end - 1e-6]
        else:
            seg_words = remaining
            remaining = []

        if seg_words:
            start = seg_words[0]["start"]
            end = seg_words[-1]["end"]
        else:
            start, end = c_start, c_end
        segments.append(
            {
                "id": i,
                "start": start,
                "end": end,
                "text": chunk.text,
                "words": seg_words,
            }
        )
    return segments


def _audio_to_numpy(audio: Any, sample_rate: int = 16000) -> tuple[np.ndarray, Optional[int]]:
    """Return mono float32 samples at 16 kHz, plus optional source sample rate for arrays."""
    if isinstance(audio, str):
        return audio, None  # path; CrisperWhisper loads it

    if hasattr(audio, "detach"):
        # torch.Tensor
        audio = audio.detach().cpu().numpy()

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=0) if audio.shape[0] <= 8 else audio.mean(axis=-1)
    return audio, sample_rate


def transcribe_crisper(
    model: CrisperWhisperAsTimestamped,
    audio: Any,
    *,
    language: Optional[str] = None,
    crisper_mode: str = "verbatim",
    remove_punctuation_from_words: bool = False,
    verbose: bool = False,
    **_ignored,
) -> dict:
    """Run CrisperWhisper and return a whisper-timestamped-compatible dict."""
    if crisper_mode not in ("verbatim", "intended"):
        raise ValueError(
            f"crisper_mode must be 'verbatim' or 'intended', got {crisper_mode!r}"
        )

    audio_in, sr = _audio_to_numpy(audio)
    kw = dict(
        language=language or "en",
        mode=crisper_mode,
        word_timestamps=True,
        hallucination_mitigation=True,
        longform_strategy="continuation",
    )
    if isinstance(audio_in, np.ndarray) and sr is not None:
        kw["sr"] = sr

    result = model.model.transcribe(audio_in, **kw)
    out = map_crisper_result(
        result, remove_punctuation_from_words=remove_punctuation_from_words
    )
    if verbose:
        print(out["text"])
    return out
