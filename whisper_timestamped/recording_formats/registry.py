"""Registry of recordings formats with lookup and folder auto-detect."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from whisper_timestamped.recording_formats.base import RecordingsFormat
from whisper_timestamped.recording_formats.debut import DebutFormat
from whisper_timestamped.recording_formats.ios_whisper_app import (
    IOSWhisperAppFormat,
)
from whisper_timestamped.recording_formats.just_press_record import (
    JustPressRecordFormat,
)
from whisper_timestamped.recording_formats.rec_continuous import (
    RecContinuousFormat,
    RecContinuousVideoRecorderFormat,
)
from whisper_timestamped.recording_formats.voice_memos import VoiceMemosFormat

# Most-specific-first for detect_format.
# Canonical CAM_* id is rec_continuous_video_recorder; rec_continuous is alias only.
_DETECT_ORDER: List[RecordingsFormat] = [
    VoiceMemosFormat(),
    JustPressRecordFormat(),
    DebutFormat(),
    RecContinuousVideoRecorderFormat(),
    IOSWhisperAppFormat(),
]

_ALIAS_FORMATS: List[RecordingsFormat] = [
    RecContinuousFormat(),
]

FORMATS: Dict[str, RecordingsFormat] = {
    f.id: f for f in (*_DETECT_ORDER, *_ALIAS_FORMATS)
}


def get_format(format_id: str) -> RecordingsFormat:
    """Return a registered format by id (e.g. ``voice_memos``)."""
    key = format_id.strip().lower()
    if key not in FORMATS:
        known = ", ".join(sorted(FORMATS))
        raise KeyError(f"Unknown recordings format {format_id!r}; known: {known}")
    return FORMATS[key]


def detect_format(path: Path) -> Optional[RecordingsFormat]:
    """
    Detect which registered format best matches *path* (file or folder).

    Tries formats in most-specific-first order. Returns None when nothing matches.
    """
    path = Path(path)
    if path.is_file():
        for fmt in _DETECT_ORDER:
            if fmt.matches_file(path):
                return fmt
            ## END if fmt.matches_file(path)....
        ## END for fmt in _DETECT_ORDER....

        return None
    ## END if path.is_file()....

    if path.is_dir():
        for fmt in _DETECT_ORDER:
            if fmt.matches_folder(path):
                return fmt
            ## END if fmt.matches_folder(path)....
        ## END for fmt in _DETECT_ORDER....
    ## END if path.is_dir()....

    return None


def list_canonical_formats() -> List[RecordingsFormat]:
    """Formats used for detect/watch (aliases like ``rec_continuous`` omitted)."""
    return list(_DETECT_ORDER)


def list_format_ids() -> List[str]:
    return sorted(FORMATS.keys())


def list_canonical_format_ids() -> List[str]:
    """Sorted ids for user-facing pickers (aliases omitted)."""
    return sorted(f.id for f in _DETECT_ORDER)


_ALIAS_TO_CANONICAL: Dict[str, str] = {
    "rec_continuous": "rec_continuous_video_recorder",
}


def canonical_format_id(format_id: str) -> str:
    """Map an alias id (e.g. ``rec_continuous``) to its canonical id."""
    key = (format_id or "").strip().lower()
    return _ALIAS_TO_CANONICAL.get(key, key)
