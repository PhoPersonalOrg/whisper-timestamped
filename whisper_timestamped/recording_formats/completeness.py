"""Non-blocking checks for whether a recording file is finished writing."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from whisper_timestamped.recording_formats.base import (
    DUP_DIR_NAME,
    FILELIST_COLUMNS,
    RecordingsFormat,
)
from whisper_timestamped.recording_formats.rec_continuous import (
    read_timing_sidecar,
    sidecar_path_for,
)
from whisper_timestamped.recording_formats.registry import list_canonical_formats

TIMING_SIDECAR_SUFFIX = ".timing.json"
DEFAULT_MIN_AGE_S = 10.0
CAM_FORMAT_IDS = frozenset({"rec_continuous_video_recorder", "rec_continuous"})


@dataclass(frozen=True)
class FileSnapshot:
    """Size and mtime sample for a path; used to detect a still-growing file."""

    path: Path
    size: int
    mtime: float


def file_snapshot(path: Path) -> Optional[FileSnapshot]:
    """Return a size/mtime snapshot, or None if the file is missing."""
    try:
        stat = path.stat()
    except OSError:
        return None
    if not path.is_file():
        return None
    return FileSnapshot(path=path, size=int(stat.st_size), mtime=float(stat.st_mtime))


def is_timing_sidecar_path(path: Path) -> bool:
    """True when *path* is a ``*.ext.timing.json`` sidecar, not media."""
    return path.name.lower().endswith(TIMING_SIDECAR_SUFFIX)


def recording_path_for_event(path: Path) -> Path:
    """Map a watch event path to the media file (sidecar events → recording)."""
    path = Path(path)
    if is_timing_sidecar_path(path):
        stripped = path.name[: -len(TIMING_SIDECAR_SUFFIX)]
        return path.with_name(stripped)
    ## END if is_timing_sidecar_path(path)....

    return path


def path_in_watch_scope(path: Path, watch_root: Path, recursive: bool) -> bool:
    """True when *path* is inside *watch_root* at the format's discover depth."""
    try:
        rel = path.resolve().relative_to(Path(watch_root).resolve())
    except ValueError:
        return False
    depth = len(rel.parts)
    if recursive:
        return 1 <= depth <= 2
    return depth == 1


def skip_watch_reason(path: Path, fmt: RecordingsFormat) -> Optional[str]:
    """Return a skip reason, or None if the path is a candidate recording."""
    if is_timing_sidecar_path(path):
        return "timing_sidecar"
    if DUP_DIR_NAME in path.parts:
        return "dup_dir"
    if not fmt.matches_file(path):
        return "format_mismatch"
    return None


def sidecar_is_finalized(recording_file: Path) -> bool:
    """True when the CAM timing sidecar includes a stop timestamp."""
    sidecar = read_timing_sidecar(recording_file)
    if sidecar is None:
        return False
    return sidecar.get("wall_clock_stop_unix") is not None


def exclusive_open_succeeds(path: Path) -> bool:
    """True when this process can open *path* with no sharing (writer has closed)."""
    if not path.is_file():
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            generic_read = 0x80000000
            open_existing = 3
            file_attribute_normal = 0x80
            file_share_none = 0
            invalid_handle = ctypes.c_void_p(-1).value
            create_file = ctypes.windll.kernel32.CreateFileW
            create_file.argtypes = [
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.LPVOID,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.HANDLE,
            ]
            create_file.restype = wintypes.HANDLE
            handle = create_file(
                str(path),
                generic_read,
                file_share_none,
                None,
                open_existing,
                file_attribute_normal,
                None,
            )
            if handle == invalid_handle or handle is None:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        except (OSError, AttributeError, ValueError, OverflowError):
            return False
        ## END try CreateFileW....

    try:
        fd = os.open(path, os.O_RDONLY)
        os.close(fd)
        return True
    except OSError:
        return False


def is_size_mtime_stable(
    current: FileSnapshot,
    previous: Optional[FileSnapshot],
    *,
    min_age_s: float = DEFAULT_MIN_AGE_S,
    now: Optional[float] = None,
) -> bool:
    """True when *current* is old enough and unchanged vs *previous*. Never sleeps."""
    if current.size <= 0:
        return False
    clock = time.time() if now is None else now
    if (clock - current.mtime) < float(min_age_s):
        return False
    if previous is None:
        return False
    return current.size == previous.size


def is_recording_complete(
    path: Path,
    fmt: RecordingsFormat,
    previous: Optional[FileSnapshot] = None,
    *,
    min_age_s: float = DEFAULT_MIN_AGE_S,
    now: Optional[float] = None,
    require_exclusive_open: bool = True,
) -> Tuple[bool, Optional[FileSnapshot]]:
    """
    Return ``(complete, current_snapshot)`` without sleeping.

    Callers should keep *previous* from the last tick and re-check until True.
    """
    media = recording_path_for_event(path)
    if skip_watch_reason(media, fmt) is not None:
        return False, None
    snap = file_snapshot(media)
    if snap is None:
        return False, None
    if not is_size_mtime_stable(snap, previous, min_age_s=min_age_s, now=now):
        return False, snap
    ## END if not is_size_mtime_stable....

    if fmt.id in CAM_FORMAT_IDS:
        sidecar = sidecar_path_for(media)
        if sidecar.is_file():
            return sidecar_is_finalized(media), snap
        if require_exclusive_open and not exclusive_open_succeeds(media):
            return False, snap
        ## END if sidecar missing and exclusive open failed....

    return True, snap


def default_watch_folders(
    formats: Optional[Sequence[RecordingsFormat]] = None,
    *,
    require_existing: bool = True,
) -> List[Dict[str, Any]]:
    """
    Watch specs for canonical formats whose default recordings dir exists.

    Each item: ``{format_id, path, enabled, recursive}``.
    """
    chosen: Iterable[RecordingsFormat]
    if formats is None:
        chosen = list_canonical_formats()
    else:
        chosen = formats
    ## END if formats is None....

    folders: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for fmt in chosen:
        root = Path(fmt.default_recordings_dir)
        if require_existing and not root.is_dir():
            continue
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        folders.append(
            {
                "format_id": fmt.id,
                "path": key,
                "enabled": True,
                "recursive": bool(fmt.nest_one_level),
            }
        )
    ## END for fmt in chosen....

    return folders


def normalize_watch_folders(raw: Any) -> List[Dict[str, Any]]:
    """Coerce a prefs JSON list into watch-folder dicts; drop invalid entries."""
    if not isinstance(raw, list):
        return []
    folders: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        format_id = str(item.get("format_id") or "").strip()
        path = str(item.get("path") or "").strip()
        if not format_id or not path:
            continue
        recursive = item.get("recursive")
        if recursive is None:
            try:
                from whisper_timestamped.recording_formats.registry import get_format

                recursive = bool(get_format(format_id).nest_one_level)
            except (KeyError, Exception):
                recursive = False
            ## END try get_format....

        folders.append(
            {
                "format_id": format_id,
                "path": path,
                "enabled": bool(item.get("enabled", True)),
                "recursive": bool(recursive),
            }
        )
    ## END for item in raw....

    return folders


def output_dir_for_format(
    fmt: RecordingsFormat,
    recordings_dir: Optional[Path] = None,
) -> Path:
    """Format default output dir, else ``<recordings_dir>/transcriptions``."""
    if fmt.default_output_dir is not None:
        return Path(fmt.default_output_dir)
    root = Path(recordings_dir) if recordings_dir is not None else Path(
        fmt.default_recordings_dir
    )
    return root / "transcriptions"


def upsert_filelist_csv_row(
    csv_path: Path,
    row: Dict[str, Any],
    extra_columns: Sequence[str] = (),
) -> None:
    """Insert or update a filelist CSV row keyed by ``full_path``."""
    columns = list(FILELIST_COLUMNS)
    for col in extra_columns:
        if col not in columns:
            columns.append(col)
        ## END if col not in columns....

    ## END for col in extra_columns....

    for key in row:
        if key not in columns:
            columns.append(key)
        ## END if key not in columns....

    ## END for key in row....

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.is_file():
        try:
            df = pd.read_csv(csv_path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            df = pd.DataFrame(columns=columns)
        ## END try read csv....

    else:
        df = pd.DataFrame(columns=columns)
    ## END if csv_path.is_file()....

    for col in columns:
        if col not in df.columns:
            df[col] = ""
        ## END if col not in df.columns....

    ## END for col in columns....

    full_path = str(row.get("full_path") or "")
    updated = False
    if full_path and "full_path" in df.columns and not df.empty:
        mask = df["full_path"].astype(str) == full_path
        if bool(mask.any()):
            idx = df.index[mask][0]
            for key, value in row.items():
                if key not in df.columns:
                    df[key] = ""
                if df[key].dtype != "object":
                    df[key] = df[key].astype("object")
                df.at[idx, key] = value
            ## END for key, value in row.items()....

            updated = True
        ## END if mask.any()....

    ## END if matching full_path....

    if not updated:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    ## END if not updated....

    df.to_csv(csv_path, index=False, encoding="utf-8")

