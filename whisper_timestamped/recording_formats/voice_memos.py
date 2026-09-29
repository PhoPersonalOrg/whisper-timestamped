"""Apple Voice Memos macOS / container export format."""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from whisper_timestamped.recording_formats.base import RecordingsFormat

_IPHONE_BACKUP_ROOT = Path(r"H:/backups/2026-09-21_iPhone15Pro")

# YYYYMMDD HHMMSS[-HEXID].m4a  (ID optional)
_VOICE_MEMOS_RE = re.compile(
    r"^(\d{8}) (\d{6})(?:-[0-9A-Fa-f]+)?$",
)


def parse_voice_memos_filename(path: Path) -> Optional[datetime]:
    """Parse Voice Memos stem ``YYYYMMDD HHMMSS[-ID]`` into a naive datetime."""
    match = _VOICE_MEMOS_RE.match(path.stem)
    if match is None:
        return None
    date_part, time_part = match.group(1), match.group(2)
    try:
        return datetime.strptime(f"{date_part}{time_part}", "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _default_cloud_recordings_db(voice_memos_root: Path) -> Path:
    return (
        voice_memos_root
        / "group.com.apple.VoiceMemos.shared"
        / "Recordings"
        / "CloudRecordings.db"
    )


def load_voice_memos_titles(db_path: Path) -> Dict[str, str]:
    """
    Map ``ZPATH`` basename -> display title from CloudRecordings.db.

    Returns empty dict if the DB is missing or unreadable.
    """
    if not db_path.is_file():
        return {}
    try:
        conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    except sqlite3.Error:
        return {}

    titles: Dict[str, str] = {}
    try:
        cur = conn.execute(
            "SELECT ZPATH, ZCUSTOMLABELFORSORTING FROM ZCLOUDRECORDING"
        )
        for zpath, label in cur.fetchall():
            if not zpath:
                continue
            name = Path(str(zpath)).name
            if label is None:
                continue
            title = str(label).strip()
            if title:
                titles[name] = title
            ## END if title....
        ## END for zpath, label in cur.fetchall()....
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    ## END try/except/finally sqlite....

    return titles


class VoiceMemosFormat(RecordingsFormat):
    def __init__(self) -> None:
        root = _IPHONE_BACKUP_ROOT / "VoiceMemos"
        super().__init__(
            id="voice_memos",
            label="Apple VoiceMemos",
            media_extensions=(".m4a",),
            input_mode="filelist",
            default_recordings_dir=root / "audio",
            default_output_dir=root / "transcriptions",
            default_filelists_dir=root / "filelists",
            filelist_name_prefix="voicememos_audio",
            nest_one_level=False,
            extra_filelist_columns=("title",),
        )
        self._voice_memos_root = root
        self._title_cache: Optional[Dict[str, str]] = None

    def _titles(self) -> Dict[str, str]:
        if self._title_cache is None:
            db = _default_cloud_recordings_db(self._voice_memos_root)
            self._title_cache = load_voice_memos_titles(db)
        ## END if self._title_cache is None....

        return self._title_cache

    def matches_file(self, path: Path) -> bool:
        if path.suffix.lower() not in self.media_extensions:
            return False
        return parse_voice_memos_filename(path) is not None

    def matches_folder(self, path: Path, sample_limit: int = 40) -> bool:
        # Prefer the flattened audio/ working set when given the export root.
        audio_child = path / "audio"
        if audio_child.is_dir() and super().matches_folder(
            audio_child, sample_limit=sample_limit
        ):
            return True
        return super().matches_folder(path, sample_limit=sample_limit)

    def extract_creation_time(
        self,
        path: Path,
        tz_name: str = "America/Los_Angeles",
    ) -> Optional[str]:
        dt = parse_voice_memos_filename(path)
        if dt is None:
            return None
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def extra_row_fields(self, path: Path) -> Dict[str, Any]:
        title = self._titles().get(path.name, "")
        return {"title": title}
