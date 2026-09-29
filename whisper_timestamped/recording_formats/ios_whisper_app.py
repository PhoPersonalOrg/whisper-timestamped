"""iOS WhisperApp flat audio export format."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from whisper_timestamped.recording_formats.base import RecordingsFormat

_IPHONE_BACKUP_ROOT = Path(r"H:/backups/2026-09-21_iPhone15Pro")


class IOSWhisperAppFormat(RecordingsFormat):
    def __init__(self) -> None:
        root = _IPHONE_BACKUP_ROOT / "WhisperApp"
        super().__init__(
            id="ios_whisper_app",
            label="iOSWhisperApp",
            media_extensions=(".m4a", ".caf"),
            input_mode="filelist",
            default_recordings_dir=root / "Audio" / "ACTIVE",
            default_output_dir=root / "transcriptions",
            default_filelists_dir=root / "filelists",
            filelist_name_prefix="audio",
            nest_one_level=False,
        )

    def matches_file(self, path: Path) -> bool:
        # Catch-all for flat WhisperApp audio that is not JPR / VoiceMemos.
        if path.suffix.lower() not in self.media_extensions:
            return False
        from whisper_timestamped.recording_formats.just_press_record import (
            parse_just_press_record_path,
        )
        from whisper_timestamped.recording_formats.voice_memos import (
            parse_voice_memos_filename,
        )

        if parse_just_press_record_path(path) is not None:
            return False
        if parse_voice_memos_filename(path) is not None:
            return False
        return True

    def extract_creation_time(
        self,
        path: Path,
        tz_name: str = "America/Los_Angeles",
    ) -> Optional[str]:
        # Prefer ffprobe embedded tags in the extract script.
        return None
