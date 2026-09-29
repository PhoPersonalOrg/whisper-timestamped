"""Apple Voice Memos macOS / container export format."""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from whisper_timestamped.recording_formats.base import RecordingsFormat

_IPHONE_BACKUP_ROOT = Path(r"H:/backups/2026-09-21_iPhone15Pro")

# Core Data / Apple epoch: 2001-01-01 UTC → Unix seconds.
_APPLE_EPOCH_UNIX = 978307200

# YYYYMMDD HHMMSS[-HEXID].m4a  (ID optional)
_VOICE_MEMOS_RE = re.compile(
    r"^(\d{8}) (\d{6})(?:-[0-9A-Fa-f]+)?$",
)

_EXTRA_COLUMNS = (
    "title",
    "duration_seconds",
    "recorded_at_utc",
    "unique_id",
    "folder",
    "apple_transcript",
    "encoder",
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


def _parse_recorded_at_utc(
    custom_label: Any,
    zdate: Any,
) -> str:
    """Prefer ZCUSTOMLABEL ISO UTC; else convert ZDATE (Core Data epoch)."""
    if custom_label is not None:
        raw = str(custom_label).strip()
        if raw:
            try:
                # e.g. 2019-04-16T00:01:01Z
                iso = raw.replace("Z", "+00:00")
                dt = datetime.fromisoformat(iso)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                ## END if dt.tzinfo is None....

                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
            ## END try/except ISO parse....
        ## END if raw....
    ## END if custom_label....

    if zdate is None:
        return ""
    try:
        unix_sec = float(zdate) + _APPLE_EPOCH_UNIX
        dt = datetime.fromtimestamp(unix_sec, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError, OverflowError):
        return ""


def load_voice_memos_metadata(db_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Map ``ZPATH`` basename -> recording metadata from CloudRecordings.db.

    Fields: title, duration_seconds, recorded_at_utc, unique_id, folder.

    Title comes from ``ZENCRYPTEDTITLE`` (Explorer / transcript name), with
    ``ZCUSTOMLABELFORSORTING`` as fallback. There is no GPS/location column in
    this schema; place names live only in the title when Apple geocoded them.

    Returns empty dict if the DB is missing or unreadable.
    """
    if not db_path.is_file():
        return {}
    try:
        conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    except sqlite3.Error:
        return {}

    meta: Dict[str, Dict[str, Any]] = {}
    try:
        folders: Dict[int, str] = {}
        try:
            for folder_pk, folder_name in conn.execute(
                "SELECT Z_PK, ZENCRYPTEDNAME FROM ZFOLDER"
            ):
                if folder_pk is None:
                    continue
                name = (
                    str(folder_name).strip() if folder_name is not None else ""
                )
                folders[int(folder_pk)] = name
            ## END for folder_pk, folder_name in conn.execute....

        except sqlite3.Error:
            folders = {}
        ## END try/except ZFOLDER....

        cur = conn.execute(
            """
            SELECT ZPATH, ZENCRYPTEDTITLE, ZCUSTOMLABELFORSORTING,
                   ZCUSTOMLABEL, ZDATE, ZDURATION, ZUNIQUEID, ZFOLDER
            FROM ZCLOUDRECORDING
            """
        )
        for (
            zpath,
            encrypted_title,
            label_for_sorting,
            custom_label,
            zdate,
            duration,
            unique_id,
            folder_pk,
        ) in cur.fetchall():
            if not zpath:
                continue
            name = Path(str(zpath)).name
            title = ""
            if encrypted_title is not None:
                title = str(encrypted_title).strip()
            ## END if encrypted_title....

            if not title and label_for_sorting is not None:
                title = str(label_for_sorting).strip()
            ## END if not title and label_for_sorting....

            folder = ""
            if folder_pk is not None:
                folder = folders.get(int(folder_pk), "")
            ## END if folder_pk....

            duration_seconds: Any = ""
            if duration is not None:
                try:
                    duration_seconds = float(duration)
                except (TypeError, ValueError):
                    duration_seconds = ""
                ## END try/except duration....
            ## END if duration....

            meta[name] = {
                "title": title,
                "duration_seconds": duration_seconds,
                "recorded_at_utc": _parse_recorded_at_utc(custom_label, zdate),
                "unique_id": (
                    str(unique_id).strip() if unique_id is not None else ""
                ),
                "folder": folder,
            }
        ## END for zpath, ... in cur.fetchall()....

    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    ## END try/except/finally sqlite....

    return meta


def load_voice_memos_titles(db_path: Path) -> Dict[str, str]:
    """
    Map ``ZPATH`` basename -> display title from CloudRecordings.db.

    Thin wrapper over :func:`load_voice_memos_metadata`.
    Returns empty dict if the DB is missing or unreadable.
    """
    return {
        name: str(row.get("title") or "")
        for name, row in load_voice_memos_metadata(db_path).items()
        if row.get("title")
    }


def _ffprobe_encoder(path: Path) -> str:
    """Return ``format.tags.encoder`` from ffprobe, or empty string."""
    if not path.is_file():
        return ""
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    ## END try/except ffprobe....

    if completed.returncode != 0 or not completed.stdout:
        return ""
    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return ""
    tags = (data.get("format") or {}).get("tags") or {}
    encoder = tags.get("encoder")
    if encoder is None:
        return ""
    return str(encoder).strip()


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
            extra_filelist_columns=_EXTRA_COLUMNS,
        )
        self._voice_memos_root = root
        self._meta_cache: Optional[Dict[str, Dict[str, Any]]] = None
        self._encoder_cache: Dict[str, str] = {}

    def _metadata(self) -> Dict[str, Dict[str, Any]]:
        if self._meta_cache is None:
            db = _default_cloud_recordings_db(self._voice_memos_root)
            self._meta_cache = load_voice_memos_metadata(db)
        ## END if self._meta_cache is None....

        return self._meta_cache

    def _encoder_for(self, path: Path) -> str:
        key = path.name
        if key not in self._encoder_cache:
            self._encoder_cache[key] = _ffprobe_encoder(path)
        ## END if key not in self._encoder_cache....

        return self._encoder_cache[key]

    def _apple_transcript_path(self, title: str) -> str:
        if not title:
            return ""
        candidate = self._voice_memos_root / "transcripts" / f"{title}.txt"
        if candidate.is_file():
            return str(candidate)
        return ""

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
        # Filename wall-clock is device-local; do not replace with ZDATE/LA.
        dt = parse_voice_memos_filename(path)
        if dt is None:
            return None
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def extra_row_fields(self, path: Path) -> Dict[str, Any]:
        row = self._metadata().get(path.name, {})
        title = str(row.get("title") or "")
        return {
            "title": title,
            "duration_seconds": row.get("duration_seconds", ""),
            "recorded_at_utc": row.get("recorded_at_utc", ""),
            "unique_id": row.get("unique_id", ""),
            "folder": row.get("folder", ""),
            "apple_transcript": self._apple_transcript_path(title),
            "encoder": self._encoder_for(path),
        }
