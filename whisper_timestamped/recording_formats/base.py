"""Base class for known recordings export formats."""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

DUP_DIR_NAME = "_DUP"

FILELIST_COLUMNS = [
    "name",
    "size_bytes",
    "size_mb",
    "creation_time",
    "modification_time",
    "full_path",
]


def utc_to_local_str(utc_dt: datetime, tz_name: str) -> str:
    """Convert aware UTC datetime to naive local YYYY-MM-DD HH:MM:SS."""
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=ZoneInfo("UTC"))
    local = utc_dt.astimezone(ZoneInfo(tz_name)).replace(tzinfo=None)
    return local.strftime("%Y-%m-%d %H:%M:%S")


def _fs_timestamp_to_local_str(epoch_sec: float, tz_name: str) -> str:
    utc_dt = datetime.fromtimestamp(epoch_sec, tz=ZoneInfo("UTC"))
    return utc_to_local_str(utc_dt, tz_name)


def _birth_time_epoch(stat_result: os.stat_result) -> float:
    if sys.platform == "win32":
        return float(stat_result.st_ctime)
    birth = getattr(stat_result, "st_birthtime", None)
    if birth is not None:
        return float(birth)
    return float(stat_result.st_ctime)


@dataclass
class RecordingsFormat(ABC):
    """Settings and helpers for one recordings export layout."""

    id: str
    label: str
    media_extensions: Tuple[str, ...]
    input_mode: str  # "directory" | "filelist"
    default_recordings_dir: Path
    default_output_dir: Optional[Path] = None
    default_filelists_dir: Optional[Path] = None
    filelist_name_prefix: str = "audio"
    nest_one_level: bool = False
    extra_filelist_columns: Tuple[str, ...] = field(default_factory=tuple)

    # --- detection ------------------------------------------------------------

    @abstractmethod
    def matches_file(self, path: Path) -> bool:
        """True when *path* looks like a media file of this format."""

    def matches_folder(self, path: Path, sample_limit: int = 40) -> bool:
        """True when a majority of sampled media under *path* match this format."""
        if not path.is_dir():
            return False
        media = self.discover_media(path)
        if not media:
            # VoiceMemos root may contain an `audio/` child.
            audio_child = path / "audio"
            if audio_child.is_dir() and audio_child != path:
                media = self.discover_media(audio_child)
            ## END if audio_child....
        ## END if not media....

        if not media:
            return False
        sample = media[:sample_limit]
        hits = sum(1 for p in sample if self.matches_file(p))
        return hits >= max(1, (len(sample) + 1) // 2)

    # --- naming / creation time ----------------------------------------------

    def transcript_name(self, path: Path) -> str:
        """CSV `name` / transcript basename (with extension)."""
        return path.name

    def extract_creation_time(
        self,
        path: Path,
        tz_name: str = "America/Los_Angeles",
    ) -> Optional[str]:
        """
        Format-specific wall-clock creation time as ``YYYY-MM-DD HH:MM:SS``.

        Return None so callers fall back to ffprobe (or other probes).
        """
        return None

    def extra_row_fields(self, path: Path) -> Dict[str, Any]:
        """Optional extra CSV columns for one media path (e.g. title)."""
        return {}

    # --- discovery / filelist ------------------------------------------------

    def discover_media(self, recordings_dir: Path) -> List[Path]:
        """Return sorted media paths under *recordings_dir* for this format."""
        if not recordings_dir.is_dir():
            return []

        seen: set[Path] = set()
        paths: List[Path] = []
        patterns: Sequence[str]
        if self.nest_one_level:
            patterns = ("*{ext}", "*/*{ext}")
        else:
            patterns = ("*{ext}",)
        ## END if nest_one_level....

        for ext in self.media_extensions:
            for pattern_tmpl in patterns:
                pattern = pattern_tmpl.format(ext=ext)
                for p in recordings_dir.glob(pattern):
                    if not p.is_file():
                        continue
                    if p.parent.name == DUP_DIR_NAME:
                        continue
                    if p in seen:
                        continue
                    seen.add(p)
                    paths.append(p)
                ## END for p in recordings_dir.glob(pattern)....
            ## END for pattern_tmpl in patterns....
        ## END for ext in self.media_extensions....

        return sorted(paths)

    def build_filelist_rows(
        self,
        recordings_dir: Path,
        tz_name: str = "America/Los_Angeles",
    ) -> List[Dict[str, Any]]:
        """Build filelist row dicts for media under *recordings_dir*."""
        rows: List[Dict[str, Any]] = []
        for path in self.discover_media(recordings_dir):
            st = path.stat()
            size_bytes = int(st.st_size)
            row: Dict[str, Any] = {
                "name": self.transcript_name(path),
                "size_bytes": size_bytes,
                "size_mb": f"{size_bytes / 1_048_576:.3f}",
                "creation_time": _fs_timestamp_to_local_str(
                    _birth_time_epoch(st), tz_name
                ),
                "modification_time": _fs_timestamp_to_local_str(
                    st.st_mtime, tz_name
                ),
                "full_path": str(path),
            }
            row.update(self.extra_row_fields(path))
            rows.append(row)
        ## END for path in self.discover_media(recordings_dir)....

        return rows

    def default_filelist_csv_path(
        self,
        when: Optional[datetime] = None,
    ) -> Path:
        """Dated default CSV path under ``default_filelists_dir``."""
        if self.default_filelists_dir is None:
            raise ValueError(
                f"Format {self.id!r} has no default_filelists_dir"
            )
        day = (when or datetime.now()).strftime("%Y-%m-%d")
        return (
            self.default_filelists_dir
            / f"{day}_{self.filelist_name_prefix}_file_list.csv"
        )

    def resolve_filelist_csv(self) -> Path:
        """Newest matching filelist CSV, else today's default path."""
        if self.default_filelists_dir is None:
            raise ValueError(
                f"Format {self.id!r} has no default_filelists_dir"
            )
        directory = self.default_filelists_dir
        if directory.is_dir():
            candidates = sorted(
                directory.glob("*file_list*.csv"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                return candidates[0]
            ## END if candidates....
        ## END if directory.is_dir()....

        return self.default_filelist_csv_path()

    def ensure_filelist_csv(
        self,
        csv_path: Path,
        recordings_dir: Optional[Path] = None,
        tz_name: str = "America/Los_Angeles",
    ) -> int:
        """
        Write a bootstrap filelist CSV. Returns row count.

        Raises ``FileNotFoundError`` / ``ValueError`` when the dir is missing
        or contains no media.
        """
        audio_dir = recordings_dir or self.default_recordings_dir
        if not audio_dir.is_dir():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

        rows = self.build_filelist_rows(audio_dir, tz_name=tz_name)
        if not rows:
            exts = ", ".join(self.media_extensions)
            raise ValueError(f"No media files ({exts}) found in: {audio_dir}")

        columns = list(FILELIST_COLUMNS)
        for col in self.extra_filelist_columns:
            if col not in columns:
                columns.append(col)
            ## END if col not in columns....
        ## END for col in self.extra_filelist_columns....

        df = pd.DataFrame(rows, columns=columns)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        return len(rows)

    def process_recordings_kwargs(
        self,
        recordings_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        filelist_csv: Optional[Path] = None,
        video_extensions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Kwargs suitable for ``process_recordings(**kwargs)``."""
        if self.input_mode == "directory":
            return {
                "recordings_dir": recordings_dir or self.default_recordings_dir,
                "output_dir": output_dir
                if output_dir is not None
                else self.default_output_dir,
                "video_extensions": video_extensions
                if video_extensions is not None
                else list(self.media_extensions),
            }
        ## END if directory mode....

        if self.input_mode != "filelist":
            raise ValueError(
                f"Unknown input_mode {self.input_mode!r} for format {self.id!r}"
            )
        ## END if unexpected input_mode....

        return {
            "filelist_csv": filelist_csv or self.resolve_filelist_csv(),
            "output_dir": output_dir
            if output_dir is not None
            else self.default_output_dir,
        }
