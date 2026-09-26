"""
Extract embedded m4a/caf creation_time (UTC -> local) into a file-list CSV.

Adds/overwrites columns:
  - extracted_creation_time
  - duration (seconds)
  - duration_hms
  - is_duplicate (True for non-keepers that share the same
    extracted_creation_time + duration + size_mb; keeper is False)

Also sets each media file's Windows "Date created" / Creation Time (the
column sortable in Explorer) to the probed recording creation_time when present.

If the input CSV is missing, builds it from --audio-dir (*.m4a, *.caf) and saves it
before probing.

After probing, flags content duplicates that match on creation time, duration,
and size_mb. Within each group, keeps the row with extracted_creation_time
present and the longest duration; marks the rest as duplicates. Prints each
duplicate group. Pass --move-duplicates to move non-keepers into
--audio-dir/_DUP/ (created as needed).

Duration fields are optional: missing values leave empty cells (no failure).

Example:
```bash
./.venv/Scripts/python.exe scripts/iOSWhisperAppHelpers/extract_m4a_creation_times.py
./.venv/Scripts/python.exe scripts/iOSWhisperAppHelpers/extract_m4a_creation_times.py --move-duplicates
```
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
from ctypes import wintypes
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

DEFAULT_AUDIO_DIR = Path(r"H:/backups/2026-09-21_iPhone15Pro/WhisperApp/Audio/ACTIVE")
DEFAULT_TZ = "America/Los_Angeles"
DEFAULT_CSV_PATH = (
    Path(r"H:/backups/2026-09-21_iPhone15Pro/WhisperApp/filelists")
    / f"{datetime.now(ZoneInfo(DEFAULT_TZ)).strftime('%Y-%m-%d')}_audio_file_list.csv"
)
COL_CREATION = "extracted_creation_time"
COL_DURATION_SEC = "duration (seconds)"
COL_DURATION_HMS = "duration_hms"
COL_IS_DUPLICATE = "is_duplicate"
COL_SIZE_MB = "size_mb"
AUDIO_EXTENSIONS = (".m4a", ".caf")
DUP_DIR_NAME = "_DUP"

# Windows FILETIME: 100-ns intervals since 1601-01-01; Unix epoch offset.
_EPOCH_AS_FILETIME = 116444736000000000
_FILE_WRITE_ATTRIBUTES = 0x100
_FILE_SHARE_READ = 0x1
_FILE_SHARE_WRITE = 0x2
_FILE_SHARE_DELETE = 0x4
_OPEN_EXISTING = 3
_FILE_ATTRIBUTE_NORMAL = 0x80
_INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value


def resolve_audio_path(
    row: pd.Series,
    audio_dir: Path,
) -> Optional[Path]:
    """Prefer existing full_path; else audio_dir / name."""
    full_path_raw = row.get("full_path")
    if pd.notna(full_path_raw) and str(full_path_raw).strip():
        candidate = Path(str(full_path_raw).strip())
        if candidate.is_file():
            return candidate
    ## END if full_path present....

    name_raw = row.get("name")
    if pd.notna(name_raw) and str(name_raw).strip():
        candidate = audio_dir / str(name_raw).strip()
        if candidate.is_file():
            return candidate
    ## END if name present....

    return None


def probe_format_metadata(
    path: Path,
) -> Tuple[Optional[datetime], Optional[float]]:
    """
    Return (creation_time_utc, duration_seconds) from ffprobe format JSON.

    Either field may be None if missing or unparseable; never raises for that.
    """
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                str(path),
            ],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  ! ffprobe failed for {path.name}: {exc}")
        return None, None

    fmt = json.loads(out).get("format") or {}
    tags = fmt.get("tags") or {}

    utc_dt: Optional[datetime] = None
    raw_ct = tags.get("creation_time")
    if raw_ct:
        cleaned = str(raw_ct).strip().replace("Z", "+00:00")
        try:
            utc_dt = datetime.fromisoformat(cleaned)
        except ValueError:
            print(f"  ! Unparseable creation_time for {path.name}: {raw_ct!r}")
    ## END if raw_ct....

    duration_sec: Optional[float] = None
    raw_dur = fmt.get("duration")
    if raw_dur is not None and str(raw_dur).strip() != "":
        try:
            duration_sec = float(raw_dur)
        except (TypeError, ValueError):
            print(f"  ! Unparseable duration for {path.name}: {raw_dur!r}")
    ## END if raw_dur....

    return utc_dt, duration_sec


def utc_to_local_str(utc_dt: datetime, tz_name: str) -> str:
    """Convert aware UTC datetime to naive local YYYY-MM-DD HH:MM:SS."""
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=ZoneInfo("UTC"))
    local = utc_dt.astimezone(ZoneInfo(tz_name)).replace(tzinfo=None)
    return local.strftime("%Y-%m-%d %H:%M:%S")


def format_duration_seconds(duration_sec: float) -> str:
    return f"{duration_sec:.3f}"


def format_duration_hms(duration_sec: float) -> str:
    total = int(duration_sec)  # floor
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _datetime_to_filetime(utc_dt: datetime) -> wintypes.FILETIME:
    """Convert aware (or naive-as-UTC) datetime to Windows FILETIME."""
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=ZoneInfo("UTC"))
    timestamp = int(utc_dt.timestamp() * 10_000_000) + _EPOCH_AS_FILETIME
    return wintypes.FILETIME(timestamp & 0xFFFFFFFF, timestamp >> 32)


def set_windows_creation_time(path: Path, utc_dt: datetime) -> None:
    """
    Set Windows Explorer "Date created" (Creation Time); leave atime/mtime alone.

    Raises OSError on failure. Only supported on win32.
    """
    if sys.platform != "win32":
        raise OSError("set_windows_creation_time is only supported on Windows")

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    creation = _datetime_to_filetime(utc_dt)
    handle = kernel32.CreateFileW(
        str(path),
        _FILE_WRITE_ATTRIBUTES,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE,
        None,
        _OPEN_EXISTING,
        _FILE_ATTRIBUTE_NORMAL,
        None,
    )
    if handle == _INVALID_HANDLE_VALUE:
        raise ctypes.WinError(ctypes.get_last_error())

    try:
        ok = kernel32.SetFileTime(handle, ctypes.byref(creation), None, None)
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel32.CloseHandle(handle)


def _fs_timestamp_to_local_str(epoch_sec: float, tz_name: str) -> str:
    """Format a Unix epoch seconds value as local YYYY-MM-DD HH:MM:SS."""
    utc_dt = datetime.fromtimestamp(epoch_sec, tz=ZoneInfo("UTC"))
    return utc_to_local_str(utc_dt, tz_name)


def _birth_time_epoch(stat_result: os.stat_result) -> float:
    """Windows birth time is st_ctime; prefer st_birthtime elsewhere when present."""
    if sys.platform == "win32":
        return float(stat_result.st_ctime)
    birth = getattr(stat_result, "st_birthtime", None)
    if birth is not None:
        return float(birth)
    return float(stat_result.st_ctime)


def build_file_list_from_audio_dir(
    audio_dir: Path,
    csv_path: Path,
    tz_name: str,
) -> int:
    """
    Scan audio_dir for *.m4a / *.caf and write a file-list CSV.

    Columns: name, size_bytes, size_mb, creation_time, modification_time, full_path.
    Returns the number of rows written.
    """
    if not audio_dir.is_dir():
        raise SystemExit(f"Audio directory not found: {audio_dir}")

    paths = sorted(
        p
        for ext in AUDIO_EXTENSIONS
        for p in audio_dir.glob(f"*{ext}")
        if p.is_file()
    )
    if not paths:
        exts = ", ".join(AUDIO_EXTENSIONS)
        raise SystemExit(f"No audio files ({exts}) found in: {audio_dir}")

    rows: list[dict[str, object]] = []
    for path in paths:
        st = path.stat()
        size_bytes = int(st.st_size)
        rows.append(
            {
                "name": path.name,
                "size_bytes": size_bytes,
                "size_mb": f"{size_bytes / 1_048_576:.3f}",
                "creation_time": _fs_timestamp_to_local_str(
                    _birth_time_epoch(st), tz_name
                ),
                "modification_time": _fs_timestamp_to_local_str(st.st_mtime, tz_name),
                "full_path": str(path),
            }
        )
    ## END for path in paths....

    df = pd.DataFrame(
        rows,
        columns=[
            "name",
            "size_bytes",
            "size_mb",
            "creation_time",
            "modification_time",
            "full_path",
        ],
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Built file-list CSV ({len(rows)} rows) -> {csv_path}")
    return len(rows)


def _cell_str(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _parse_float_or_neg_inf(value: object) -> float:
    raw = _cell_str(value)
    if not raw:
        return float("-inf")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float("-inf")


def _duplicate_key_series(df: pd.DataFrame) -> pd.Series:
    """
    Key for duplicate grouping: creation_time|duration|size_mb.

    Empty string when any field is missing (those rows never join a group).
    """
    ct = (
        df[COL_CREATION].map(_cell_str)
        if COL_CREATION in df.columns
        else pd.Series([""] * len(df), index=df.index)
    )
    dur = (
        df[COL_DURATION_SEC].map(_cell_str)
        if COL_DURATION_SEC in df.columns
        else pd.Series([""] * len(df), index=df.index)
    )
    size_mb = (
        df[COL_SIZE_MB].map(_cell_str)
        if COL_SIZE_MB in df.columns
        else pd.Series([""] * len(df), index=df.index)
    )
    valid = ct.ne("") & dur.ne("") & size_mb.ne("")
    key = ct + "|" + dur + "|" + size_mb
    return key.where(valid, "")


def _keeper_sort_key(row: pd.Series) -> Tuple[int, float, float, int, int, str]:
    """
    Prefer has extracted_creation_time, then longest duration, then size_mb,
    then fewer 'recovered_' prefixes / shorter name (stable).
    """
    has_ct = 1 if _cell_str(row.get(COL_CREATION)) else 0
    duration = _parse_float_or_neg_inf(row.get(COL_DURATION_SEC))
    size_mb = _parse_float_or_neg_inf(row.get(COL_SIZE_MB))
    name = _cell_str(row.get("name"))
    recovered_depth = name.lower().count("recovered_")
    return (has_ct, duration, size_mb, -recovered_depth, -len(name), name)


def choose_keeper_index(group: pd.DataFrame) -> object:
    """Return the index label of the preferred keeper row in group."""
    best_idx = group.index[0]
    best_key = _keeper_sort_key(group.loc[best_idx])
    for idx in group.index[1:]:
        key = _keeper_sort_key(group.loc[idx])
        if key > best_key:
            best_key = key
            best_idx = idx
        ## END if key > best_key....
    ## END for idx in group.index[1:]....

    return best_idx


def mark_creation_time_duplicates(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Set is_duplicate from matching (creation_time, duration, size_mb).

    Rows are duplicates IFF they share the same non-empty triple. Within each
    group of size >= 2, the keeper (has creation_time, longest duration) is
    False; all other members are True. Returns (dup_groups, dup_rows).
    """
    key = _duplicate_key_series(df)
    is_dup = pd.Series(False, index=df.index)
    dup_groups = 0

    for group_key, group in df.groupby(key, sort=False):
        if not group_key or len(group) < 2:
            continue
        ## END if singleton or empty key....

        dup_groups += 1
        keeper_idx = choose_keeper_index(group)
        for idx in group.index:
            if idx != keeper_idx:
                is_dup.loc[idx] = True
            ## END if idx != keeper_idx....
        ## END for idx in group.index....
    ## END for group_key, group in df.groupby(key, sort=False)....

    df[COL_IS_DUPLICATE] = is_dup
    dup_rows = int(is_dup.sum())
    return dup_groups, dup_rows


def print_creation_time_duplicates(df: pd.DataFrame) -> None:
    """Print every group that shares creation_time + duration + size_mb."""
    key = _duplicate_key_series(df)
    counts = key[key != ""].value_counts()
    dup_keys = sorted(k for k, n in counts.items() if n >= 2)
    if not dup_keys:
        print("No creation-time duplicate groups.")
        return
    ## END if not dup_keys....

    print(f"Creation-time duplicate groups: {len(dup_keys)}")
    print()
    for group_key in dup_keys:
        group = df[key == group_key]
        creation_time, duration, size_mb = group_key.split("|", 2)
        print(
            f"* {creation_time}  dur={duration}  size_mb={size_mb}  "
            f"(x{len(group)})"
        )
        keeper_idx = choose_keeper_index(group)
        # Keeper first, then others by name.
        ordered = pd.concat(
            [
                group.loc[[keeper_idx]],
                group.drop(index=keeper_idx).sort_values(
                    by="name", key=lambda s: s.map(_cell_str)
                ),
            ]
        )
        for idx, row in ordered.iterrows():
            label = "KEEP  " if idx == keeper_idx else "DUP   "
            name = _cell_str(row.get("name")) or "?"
            parts = [f"    {label}{name}"]
            ct_raw = row.get(COL_CREATION)
            if _cell_str(ct_raw):
                parts.append(f"extracted_creation_time={_cell_str(ct_raw)}")
            ## END if creation_time present....

            size_raw = row.get("size_bytes")
            if _cell_str(size_raw):
                parts.append(f"size_bytes={_cell_str(size_raw)}")
            ## END if size_bytes present....

            dur_raw = row.get(COL_DURATION_SEC)
            if _cell_str(dur_raw):
                parts.append(f"dur={_cell_str(dur_raw)}")
            ## END if duration present....

            print("  ".join(parts))
        ## END for idx, row in ordered.iterrows()....

        print()
    ## END for group_key in dup_keys....


def _unique_dest_path(dest_dir: Path, filename: str) -> Path:
    """Return dest_dir/filename, or dest_dir/stem_N.suffix if that name exists."""
    candidate = dest_dir / filename
    if not candidate.exists():
        return candidate
    ## END if not candidate.exists....

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    n = 2
    while True:
        candidate = dest_dir / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        ## END if not candidate.exists....

        n += 1
    ## END while True....


def move_duplicates_to_dup_dir(
    df: pd.DataFrame,
    audio_dir: Path,
) -> Tuple[int, int, int]:
    """
    Move is_duplicate rows into audio_dir/_DUP/.

    Updates full_path (and name if renamed for collision) on moved rows.
    Returns (moved, skipped, fail).
    """
    if COL_IS_DUPLICATE not in df.columns:
        return 0, 0, 0
    ## END if COL_IS_DUPLICATE not in df.columns....

    dup_dir = audio_dir / DUP_DIR_NAME
    dup_dir_resolved = dup_dir.resolve()
    moved = 0
    skipped = 0
    fail = 0

    dup_mask = df[COL_IS_DUPLICATE].fillna(False).astype(bool)
    if not dup_mask.any():
        print(f"No duplicates to move -> {dup_dir}")
        return 0, 0, 0
    ## END if not dup_mask.any()....

    dup_dir.mkdir(parents=True, exist_ok=True)
    print(f"Moving duplicates -> {dup_dir}")

    for idx in df.index[dup_mask]:
        row = df.loc[idx]
        source = resolve_audio_path(row, audio_dir)
        if source is None:
            name = row.get("name", "?")
            print(f"  ! Skip missing duplicate name={name!r}")
            skipped += 1
            continue
        ## END if source is None....

        try:
            source_resolved = source.resolve()
        except OSError as exc:
            print(f"  ! Resolve failed for {source}: {exc}")
            fail += 1
            continue
        ## END try resolve....

        if source_resolved.parent == dup_dir_resolved or dup_dir_resolved in source_resolved.parents:
            print(f"  ~ Already in {DUP_DIR_NAME}/: {source.name}")
            skipped += 1
            continue
        ## END if already under _DUP....

        dest = _unique_dest_path(dup_dir, source.name)
        try:
            source.rename(dest)
        except OSError as exc:
            print(f"  ! Move failed {source.name} -> {dest.name}: {exc}")
            fail += 1
            continue
        ## END try rename....

        df.at[idx, "full_path"] = str(dest)
        if "name" in df.columns:
            df.at[idx, "name"] = dest.name
        ## END if name column....

        moved += 1
        if dest.name != source.name:
            print(f"  moved {source.name} -> {DUP_DIR_NAME}/{dest.name}")
        else:
            print(f"  moved {source.name} -> {DUP_DIR_NAME}/")
        ## END if dest.name != source.name....
    ## END for idx in df.index[dup_mask]....

    print(f"moved={moved} skipped={skipped} fail={fail} -> {dup_dir}")
    return moved, skipped, fail


def extract_for_csv(
    csv_path: Path,
    output_path: Path,
    audio_dir: Path,
    tz_name: str,
    move_duplicates: bool = False,
) -> Tuple[int, int, int, int, int, int, int, int, int, int, int, int]:
    """
    Probe each row, set Windows creation time when present, write CSV columns.

    Returns
    (probed, creation_ok, duration_ok, missing_file, ffprobe_fail,
     fs_set_ok, fs_set_fail, dup_groups, dup_rows, moved, skipped, fail).
    """
    df = pd.read_csv(csv_path)
    creations: list[str] = []
    durations_sec: list[str] = []
    durations_hms: list[str] = []
    probed = 0
    creation_ok = 0
    duration_ok = 0
    missing_file = 0
    ffprobe_fail = 0
    fs_set_ok = 0
    fs_set_fail = 0

    for _, row in df.iterrows():
        path = resolve_audio_path(row, audio_dir)
        if path is None:
            name = row.get("name", "?")
            print(f"  ! Missing file for row name={name!r}")
            creations.append("")
            durations_sec.append("")
            durations_hms.append("")
            missing_file += 1
            continue
        ## END if path is None....

        probed += 1
        utc_dt, duration_sec = probe_format_metadata(path)

        if utc_dt is None and duration_sec is None:
            # Likely ffprobe failure or empty format; count once when both absent
            # after a successful path resolve (probe already logged on failure).
            ffprobe_fail += 1

        if utc_dt is not None:
            creations.append(utc_to_local_str(utc_dt, tz_name))
            creation_ok += 1
            if sys.platform == "win32":
                try:
                    set_windows_creation_time(path, utc_dt)
                    fs_set_ok += 1
                except OSError as exc:
                    print(f"  ! SetFileTime failed for {path.name}: {exc}")
                    fs_set_fail += 1
            ## END if win32....
        else:
            creations.append("")
        ## END if utc_dt....

        if duration_sec is not None:
            durations_sec.append(format_duration_seconds(duration_sec))
            durations_hms.append(format_duration_hms(duration_sec))
            duration_ok += 1
        else:
            durations_sec.append("")
            durations_hms.append("")
        ## END if duration_sec....
    ## END for _, row in df.iterrows()....

    df[COL_CREATION] = creations
    df[COL_DURATION_SEC] = durations_sec
    df[COL_DURATION_HMS] = durations_hms
    dup_groups, dup_rows = mark_creation_time_duplicates(df)
    print_creation_time_duplicates(df)

    moved = 0
    skipped = 0
    fail = 0
    if move_duplicates:
        moved, skipped, fail = move_duplicates_to_dup_dir(df, audio_dir)
    ## END if move_duplicates....

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return (
        probed,
        creation_ok,
        duration_ok,
        missing_file,
        ffprobe_fail,
        fs_set_ok,
        fs_set_fail,
        dup_groups,
        dup_rows,
        moved,
        skipped,
        fail,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract embedded m4a/caf creation_time (UTC -> local) and optional "
            f"duration columns into a file-list CSV "
            f"({COL_CREATION}, {COL_DURATION_SEC!r}, {COL_DURATION_HMS}, "
            f"{COL_IS_DUPLICATE}). "
            "If the input CSV is missing, builds it from --audio-dir "
            f"({', '.join('*' + e for e in AUDIO_EXTENSIONS)}). "
            "Flags non-keeper rows that share the same non-empty "
            f"{COL_CREATION} + {COL_DURATION_SEC!r} + {COL_SIZE_MB} as "
            "duplicates (keeper prefers creation_time present and longest "
            "duration). Pass --move-duplicates to move non-keepers into "
            f"--audio-dir/{DUP_DIR_NAME}/. "
            "On Windows, also sets each file's Explorer Date created "
            "(Creation Time) column."
        )
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        default=DEFAULT_CSV_PATH,
        help=(
            f"Input CSV (default: {DEFAULT_CSV_PATH}); "
            "created from --audio-dir if missing"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: overwrite input in place)",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help=(
            f"Audio directory for missing-CSV bootstrap and when full_path "
            f"is missing (default: {DEFAULT_AUDIO_DIR})"
        ),
    )
    parser.add_argument(
        "--tz",
        default=DEFAULT_TZ,
        help=f"Target timezone for extracted_creation_time (default: {DEFAULT_TZ})",
    )
    parser.add_argument(
        "--move-duplicates",
        action="store_true",
        help=(
            f"Move rows marked is_duplicate into --audio-dir/{DUP_DIR_NAME}/ "
            "(created as needed); keepers stay in place"
        ),
    )
    args = parser.parse_args()

    csv_path: Path = args.csv_path
    output_path: Path = args.output if args.output is not None else csv_path

    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}")
        print(f"Building from audio dir: {args.audio_dir}")
        build_file_list_from_audio_dir(
            audio_dir=args.audio_dir,
            csv_path=csv_path,
            tz_name=args.tz,
        )
    ## END if not csv_path.is_file()....

    if sys.platform != "win32":
        print(
            "Warning: not Windows; filesystem creation times will not be updated."
        )

    print(f"Input:  {csv_path}")
    print(f"Output: {output_path}")
    print(f"TZ:     {args.tz}")
    print(f"Move duplicates: {args.move_duplicates}")

    (
        probed,
        creation_ok,
        duration_ok,
        missing_file,
        ffprobe_fail,
        fs_set_ok,
        fs_set_fail,
        dup_groups,
        dup_rows,
        moved,
        skipped,
        fail,
    ) = extract_for_csv(
        csv_path=csv_path,
        output_path=output_path,
        audio_dir=args.audio_dir,
        tz_name=args.tz,
        move_duplicates=args.move_duplicates,
    )

    print(
        f"Done. probed={probed} creation_ok={creation_ok} duration_ok={duration_ok} "
        f"missing_file={missing_file} ffprobe_fail={ffprobe_fail} "
        f"fs_set_ok={fs_set_ok} fs_set_fail={fs_set_fail} "
        f"dup_groups={dup_groups} dup_rows={dup_rows} "
        f"moved={moved} skipped={skipped} fail={fail} -> {output_path}"
    )


if __name__ == "__main__":
    main()
