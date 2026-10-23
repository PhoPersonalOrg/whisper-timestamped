"""openai-whisper-style frame progress bars for longform chunk loops."""

from __future__ import annotations

import importlib
import sys
import threading
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

import numpy as np
from tqdm import tqdm

from whisper_timestamped.crisperwhisper.longform.base import SAMPLE_RATE

# Match Whisper / CT2Engine mel hop (see engine.HOP_LENGTH).
HOP_LENGTH = 160

_progress_tls = threading.local()


def set_audio_progress_callback(
    callback: Optional[Callable[[float], None]],
) -> None:
    """Install a thread-local ``(fraction: float) -> None`` progress hook."""
    _progress_tls.callback = callback


def get_audio_progress_callback() -> Optional[Callable[[float], None]]:
    return getattr(_progress_tls, "callback", None)


class _CallbackTqdm(tqdm):
    """tqdm that reports ``n / total`` to the thread-local progress hook."""

    def update(self, n: float = 1):
        previous = self.n
        result = super().update(n)
        callback = get_audio_progress_callback()
        total = self.total
        if callback is not None and total and self.n > previous:
            try:
                fraction = min(1.0, max(0.0, float(self.n) / float(total)))
                callback(fraction)
            except Exception:
                pass
        ## END if callback is not None and total....

        return result


def _install_openai_whisper_tqdm_hook() -> Callable[[], None]:
    """Point openai-whisper's ``tqdm.tqdm`` at ``_CallbackTqdm``.

    openai-whisper does ``import tqdm`` then ``tqdm.tqdm(...)`` in
    ``whisper/transcribe.py``.  CrisperWhisper already constructs
    ``_CallbackTqdm`` directly; this patch is what makes OpenAI Whisper
    frame bars report to ``set_audio_progress_callback``.
    """
    transcribe_mod = sys.modules.get("whisper.transcribe")
    if transcribe_mod is None:
        try:
            transcribe_mod = importlib.import_module("whisper.transcribe")
        except ImportError:
            return lambda: None
        ## END try import whisper.transcribe....

    ## END if whisper.transcribe not loaded....

    tqdm_ns = getattr(transcribe_mod, "tqdm", None)
    if tqdm_ns is None or not hasattr(tqdm_ns, "tqdm"):
        return lambda: None
    previous = tqdm_ns.tqdm
    if previous is _CallbackTqdm:
        return lambda: None
    tqdm_ns.tqdm = _CallbackTqdm

    def restore() -> None:
        if getattr(tqdm_ns, "tqdm", None) is _CallbackTqdm:
            tqdm_ns.tqdm = previous
        ## END if still patched....

    return restore


@contextmanager
def audio_progress_hook(
    callback: Optional[Callable[[float], None]],
) -> Iterator[None]:
    """Install the progress callback for CrisperWhisper and OpenAI Whisper.

    CrisperWhisper reports through ``_CallbackTqdm`` in longform loops.
    OpenAI Whisper reports through the same class after its tqdm is patched.
    """
    set_audio_progress_callback(callback)
    restore = (
        _install_openai_whisper_tqdm_hook()
        if callback is not None
        else (lambda: None)
    )
    try:
        yield
    finally:
        restore()
        set_audio_progress_callback(None)
    ## END try audio_progress_hook....


def frame_pos(sample_index: int) -> int:
    """Convert an audio sample index to a mel-frame index."""
    return max(0, int(sample_index)) // HOP_LENGTH


def covered_frame_pos(
    audio: np.ndarray,
    chunk_idx: int,
    chunk: np.ndarray,
    stride_sec: float,
) -> int:
    """Mel-frame index covered after finishing *chunk_idx* (end of window)."""
    covered = min(
        len(audio),
        int(chunk_idx * stride_sec * SAMPLE_RATE) + len(chunk),
    )
    return frame_pos(covered)


@contextmanager
def audio_progress_bar(
    audio: np.ndarray,
    verbose: bool = False,
) -> Iterator[tqdm]:
    """Yield a tqdm bar over mel frames (openai-whisper semantics).

    The bar is shown when ``verbose is False`` (default), disabled when
    ``verbose is True`` (transcript text is printed instead) or when
    ``verbose`` is any other non-False value that openai-whisper treats
    as ``disable=True`` via ``disable=verbose is not False``.
    """
    total = frame_pos(len(audio))
    with _CallbackTqdm(
        total=total,
        unit="frames",
        disable=verbose is not False,
    ) as pbar:
        yield pbar
