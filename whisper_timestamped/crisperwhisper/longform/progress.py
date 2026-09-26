"""openai-whisper-style frame progress bars for longform chunk loops."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import numpy as np
from tqdm import tqdm

from whisper_timestamped.crisperwhisper.longform.base import SAMPLE_RATE

# Match Whisper / CT2Engine mel hop (see engine.HOP_LENGTH).
HOP_LENGTH = 160


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
    with tqdm(
        total=total,
        unit="frames",
        disable=verbose is not False,
    ) as pbar:
        yield pbar
