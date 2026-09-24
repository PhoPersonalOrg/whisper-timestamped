"""
Extract embedded m4a creation_time (UTC -> local) into a file-list CSV.

Adds/overwrites columns:
  - extracted_creation_time
  - duration (seconds)
  - duration_hms

Duration fields are optional: missing values leave empty cells (no failure).

Example:
```bash
./.venv/Scripts/python.exe scripts/iOSWhisperAppHelpers/extract_m4a_creation_times.py
```
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

DEFAULT_CSV_PATH = Path(
    r"H:\backups\2026-09-21_iPhone15Pro\WhisperApp\filelists"
    r"\2026-09-24_1pm_DEDUPE_m4a_file_list.csv"
)
DEFAULT_AUDIO_DIR = Path(r"H:\backups\2026-09-21_iPhone15Pro\WhisperApp\Audio")
DEFAULT_TZ = "America/Los_Angeles"
COL_CREATION = "extracted_creation_time"
COL_DURATION_SEC = "duration (seconds)"
COL_DURATION_HMS = "duration_hms"


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


def extract_for_csv(
    csv_path: Path,
    output_path: Path,
    audio_dir: Path,
    tz_name: str,
) -> Tuple[int, int, int, int, int]:
    """
    Probe each row and write CSV with creation_time + optional duration columns.

    Returns (probed, creation_ok, duration_ok, missing_file, ffprobe_fail).
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return probed, creation_ok, duration_ok, missing_file, ffprobe_fail


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract embedded m4a creation_time (UTC -> local) and optional "
            f"duration columns into a file-list CSV "
            f"({COL_CREATION}, {COL_DURATION_SEC!r}, {COL_DURATION_HMS})."
        )
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        default=DEFAULT_CSV_PATH,
        help=f"Input CSV (default: {DEFAULT_CSV_PATH})",
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
        help=f"Fallback audio directory when full_path missing (default: {DEFAULT_AUDIO_DIR})",
    )
    parser.add_argument(
        "--tz",
        default=DEFAULT_TZ,
        help=f"Target timezone for extracted_creation_time (default: {DEFAULT_TZ})",
    )
    args = parser.parse_args()

    csv_path: Path = args.csv_path
    output_path: Path = args.output if args.output is not None else csv_path

    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    print(f"Input:  {csv_path}")
    print(f"Output: {output_path}")
    print(f"TZ:     {args.tz}")

    probed, creation_ok, duration_ok, missing_file, ffprobe_fail = extract_for_csv(
        csv_path=csv_path,
        output_path=output_path,
        audio_dir=args.audio_dir,
        tz_name=args.tz,
    )

    print(
        f"Done. probed={probed} creation_ok={creation_ok} duration_ok={duration_ok} "
        f"missing_file={missing_file} ffprobe_fail={ffprobe_fail} -> {output_path}"
    )


if __name__ == "__main__":
    main()
