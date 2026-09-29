"""REC continuous video recorder (CAM_*) format."""

from __future__ import annotations

import re
from pathlib import Path

from whisper_timestamped.recording_formats.base import RecordingsFormat

_CAM_RE = re.compile(
    r"^CAM_\d{4}-\d{2}-\d{2}T\d{6}",
    re.IGNORECASE,
)


class RecContinuousFormat(RecordingsFormat):
    def __init__(self) -> None:
        super().__init__(
            id="rec_continuous",
            label="REC_continuous_video_recorder",
            media_extensions=(".mp4", ".mkv"),
            input_mode="directory",
            default_recordings_dir=Path(
                r"I:/ScreenRecordings/REC_continuous_video_recorder"
            ),
            default_output_dir=None,
            default_filelists_dir=None,
            filelist_name_prefix="rec_continuous",
            nest_one_level=False,
        )

    def matches_file(self, path: Path) -> bool:
        if path.suffix.lower() not in self.media_extensions:
            return False
        return _CAM_RE.match(path.stem) is not None
