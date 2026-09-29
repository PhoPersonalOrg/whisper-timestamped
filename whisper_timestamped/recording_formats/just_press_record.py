"""Just Press Record nested date/time audio export format."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from whisper_timestamped.recording_formats.base import RecordingsFormat

_IPHONE_BACKUP_ROOT = Path(r"H:/backups/2026-09-21_iPhone15Pro")


def parse_just_press_record_path(path: Path) -> Optional[Tuple[str, str]]:
    """
    Parse Just Press Record export path: ``.../YYYY-MM-DD/HH-MM-SS.ext``.

    Returns ``(transcript_name, extracted_creation_time)`` or None when the
    parent folder / stem are not a valid date / time pair.

    transcript_name: ``YYYY-MM-DD_HH-MM-SS.ext``
    extracted_creation_time: ``YYYY-MM-DD HH:MM:SS`` (path wall clock; no TZ)
    """
    parent = path.parent.name
    stem = path.stem
    try:
        datetime.strptime(parent, "%Y-%m-%d")
        time_part = datetime.strptime(stem, "%H-%M-%S")
    except ValueError:
        return None

    transcript_name = f"{parent}_{stem}{path.suffix}"
    creation = f"{parent} {time_part.strftime('%H:%M:%S')}"
    return transcript_name, creation


class JustPressRecordFormat(RecordingsFormat):
    def __init__(self) -> None:
        root = _IPHONE_BACKUP_ROOT / "Just Press Record"
        super().__init__(
            id="just_press_record",
            label="Just Press Record",
            media_extensions=(".m4a", ".caf"),
            input_mode="filelist",
            default_recordings_dir=root,
            default_output_dir=root / "transcriptions",
            default_filelists_dir=root / "filelists",
            filelist_name_prefix="jpr_audio",
            nest_one_level=True,
        )

    def matches_file(self, path: Path) -> bool:
        if path.suffix.lower() not in self.media_extensions:
            return False
        return parse_just_press_record_path(path) is not None

    def transcript_name(self, path: Path) -> str:
        parsed = parse_just_press_record_path(path)
        if parsed is not None:
            return parsed[0]
        return path.name

    def extract_creation_time(
        self,
        path: Path,
        tz_name: str = "America/Los_Angeles",
    ) -> Optional[str]:
        parsed = parse_just_press_record_path(path)
        if parsed is None:
            return None
        return parsed[1]
