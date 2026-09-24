"""
Extract embedded m4a creation_time (UTC -> local) into a file-list CSV.

Adds/overwrites column `extracted_creation_time` using ffprobe format tags.

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
COLUMN_NAME = "extracted_creation_time"


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


def probe_creation_time_utc(path: Path) -> Optional[datetime]:
    """Return embedded creation_time as timezone-aware UTC datetime, or None."""
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
        return None

    fmt = json.loads(out).get("format") or {}
    tags = fmt.get("tags") or {}
    raw = tags.get("creation_time")
    if not raw:
        return None

    # ffprobe typically returns e.g. 2026-07-10T02:57:31.000000Z
    cleaned = str(raw).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        print(f"  ! Unparseable creation_time for {path.name}: {raw!r}")
        return None


def utc_to_local_str(utc_dt: datetime, tz_name: str) -> str:
    """Convert aware UTC datetime to naive local YYYY-MM-DD HH:MM:SS."""
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=ZoneInfo("UTC"))
    local = utc_dt.astimezone(ZoneInfo(tz_name)).replace(tzinfo=None)
    return local.strftime("%Y-%m-%d %H:%M:%S")


def extract_for_csv(
    csv_path: Path,
    output_path: Path,
    audio_dir: Path,
    tz_name: str,
) -> Tuple[int, int, int, int]:
    """
    Probe each row and write CSV with extracted_creation_time.

    Returns (probed, succeeded, missing_tag, missing_file).
    """
    df = pd.read_csv(csv_path)
    extracted: list[str] = []
    probed = 0
    succeeded = 0
    missing_tag = 0
    missing_file = 0

    for _, row in df.iterrows():
        path = resolve_audio_path(row, audio_dir)
        if path is None:
            name = row.get("name", "?")
            print(f"  ! Missing file for row name={name!r}")
            extracted.append("")
            missing_file += 1
            continue
        ## END if path is None....

        probed += 1
        utc_dt = probe_creation_time_utc(path)
        if utc_dt is None:
            extracted.append("")
            missing_tag += 1
            continue
        ## END if utc_dt is None....

        extracted.append(utc_to_local_str(utc_dt, tz_name))
        succeeded += 1
    ## END for _, row in df.iterrows()....

    df[COLUMN_NAME] = extracted
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return probed, succeeded, missing_tag, missing_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract embedded m4a creation_time (UTC -> local) into "
            f"column '{COLUMN_NAME}' on a file-list CSV."
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

    probed, succeeded, missing_tag, missing_file = extract_for_csv(
        csv_path=csv_path,
        output_path=output_path,
        audio_dir=args.audio_dir,
        tz_name=args.tz,
    )

    print(
        f"Done. probed={probed} succeeded={succeeded} "
        f"missing_tag={missing_tag} missing_file={missing_file} -> {output_path}"
    )


if __name__ == "__main__":
    main()
