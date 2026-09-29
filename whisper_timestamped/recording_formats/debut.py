"""Debut screen-recorder video format."""

from __future__ import annotations

import re
from pathlib import Path

from whisper_timestamped.recording_formats.base import RecordingsFormat

_DEBUT_RE = re.compile(
    r"^Debut_\d{4}-\d{2}-\d{2}T\d{6}",
    re.IGNORECASE,
)


class DebutFormat(RecordingsFormat):
    def __init__(self) -> None:
        super().__init__(
            id="debut",
            label="Debut",
            media_extensions=(".mp4", ".mkv"),
            input_mode="directory",
            default_recordings_dir=Path(
                r"M:\ScreenRecordings\EyeTrackerVR_Recordings"
            ),
            default_output_dir=None,
            default_filelists_dir=None,
            filelist_name_prefix="debut",
            nest_one_level=False,
        )

    def matches_file(self, path: Path) -> bool:
        if path.suffix.lower() not in self.media_extensions:
            return False
        return _DEBUT_RE.match(path.stem) is not None
