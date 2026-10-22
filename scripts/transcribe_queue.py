"""Torch-free helpers for transcription job identity and queue control."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Set

import pandas as pd

ARM_DF_INDEX_COL = "_arm_df_index"


def coerce_row_index(value: Any) -> Optional[int]:
    """Normalize a filelist / DataFrame index to ``int``, or ``None``."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def callback_row_index(idx: Any, row: Optional[Mapping[str, Any]] = None) -> Optional[int]:
    """GUI DataFrame index for a filelist row, preferring ``_arm_df_index``."""
    if row is not None:
        try:
            raw = row.get(ARM_DF_INDEX_COL)
        except (AttributeError, TypeError):
            raw = None
        coerced = coerce_row_index(raw)
        if coerced is not None:
            return coerced
        ## END if coerced is not None....

    return coerce_row_index(idx)


def index_in_skip_set(skip_indices: Optional[Set[Any]], row_index: Any) -> bool:
    """True when *row_index* is in the skip set (int-coerced)."""
    coerced = coerce_row_index(row_index)
    if coerced is None:
        return False
    skip = skip_indices or set()
    if coerced in skip:
        return True
    for item in skip:
        if coerce_row_index(item) == coerced:
            return True
        ## END if coerce_row_index(item) == coerced....

    ## END for item in skip....

    return False


def job_loop_action(
    stop_after_current: bool,
    skip_indices: Optional[Set[Any]],
    row_index: Any,
) -> str:
    """Return ``stop``, ``skip``, or ``start`` for the next transcription job."""
    if stop_after_current:
        return "stop"
    if index_in_skip_set(skip_indices, row_index):
        return "skip"
    return "start"
