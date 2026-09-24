"""
Parse an iOS WhisperApp bulk transcript export log into a CSV.

This is for text copied from WhisperApp on iOS via the "Bulk > Export with Timestamps" option, which is then pasted into Notes.app or something and eventually to a .txt file

Entries are delimited by a datetime-only header line, then a body until the
next header. Bodies may be a status stub (recovered / interrupted / saved) or
a multi-paragraph transcription.

Example: 
```bash
./.venv/Scripts/python.exe scripts/iOSWhisperAppHelpers/parse_iOSWhisperApp_bulk_transcript_export.py "H:/backups/2026-09-21_iPhone15Pro/WhisperApp/Bulk/2026-09-24_ExportLog.txt"
```
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

_ISO_HEADER_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
_US_HEADER_RE = re.compile(
    r"^(\d{1,2})/(\d{1,2})/(\d{2}),\s+(\d{1,2}):(\d{2})\s*([AP]M)$",
    re.IGNORECASE,
)

_STATUS_BY_BODY = {
    "Recovered recording (transcription lost)": "recovered_transcription_lost",
    "Transcription interrupted. Tap retry to continue.": "transcription_interrupted",
    "Saved recording": "saved_recording",
}


def normalize_spaces(text: str) -> str:
    """Replace narrow/no-break spaces with regular spaces."""
    return text.replace("\u202f", " ").replace("\xa0", " ")


def parse_header_datetime(raw: str) -> Optional[Tuple[str, datetime]]:
    """
    If `raw` is a datetime header, return (normalized_raw, parsed_datetime).
    Otherwise return None.
    """
    cleaned = normalize_spaces(raw.strip())
    if not cleaned:
        return None

    if _ISO_HEADER_RE.match(cleaned):
        return cleaned, datetime.strptime(cleaned, "%Y-%m-%d %H:%M:%S")

    match = _US_HEADER_RE.match(cleaned)
    if match:
        # Rebuild with a single space before AM/PM for a stable datetime_raw.
        month, day, year, hour, minute, ampm = match.groups()
        normalized = f"{int(month)}/{int(day)}/{year}, {int(hour)}:{minute} {ampm.upper()}"
        parsed = datetime.strptime(normalized, "%m/%d/%y, %I:%M %p")
        return normalized, parsed

    return None


def classify_body(body: str) -> Tuple[str, str, bool]:
    """
    Return (status, transcription, has_transcription).

    Status stubs leave transcription empty; other non-empty bodies are transcripts.
    """
    if not body:
        return "empty", "", False

    status = _STATUS_BY_BODY.get(body)
    if status is not None:
        return status, "", False

    return "transcribed", body, True


def parse_export_log(text: str) -> pd.DataFrame:
    """Split export log text into one DataFrame row per datetime-headed entry."""
    lines = text.splitlines()
    headers: List[Tuple[int, str, datetime]] = []

    for i, line in enumerate(lines):
        parsed = parse_header_datetime(line)
        if parsed is None:
            continue
        datetime_raw, dt = parsed
        headers.append((i, datetime_raw, dt))
    ## END for i, line in enumerate(lines)....

    rows = []
    for entry_index, (line_idx, datetime_raw, dt) in enumerate(headers):
        body_start = line_idx + 1
        body_end = headers[entry_index + 1][0] if entry_index + 1 < len(headers) else len(lines)
        body = "\n".join(lines[body_start:body_end]).strip()
        status, transcription, has_transcription = classify_body(body)
        rows.append(
            {
                "entry_index": entry_index,
                "datetime": dt,
                "datetime_raw": datetime_raw,
                "status": status,
                "transcription": transcription,
                "has_transcription": has_transcription,
            }
        )
    ## END for entry_index, (line_idx, datetime_raw, dt) in enumerate(headers)....

    df = pd.DataFrame(
        rows,
        columns=[
            "entry_index",
            "datetime",
            "datetime_raw",
            "status",
            "transcription",
            "has_transcription",
        ],
    )
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def parse_export_file(input_path: Path) -> pd.DataFrame:
    text = input_path.read_text(encoding="utf-8")
    return parse_export_log(text)


def run_parse(input_path: Path, output_path: Path) -> pd.DataFrame:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")

    df = parse_export_file(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    status_counts = df["status"].value_counts().to_dict() if not df.empty else {}
    counts_str = ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items()))
    print(f"Parsed {len(df)} entr{'y' if len(df) == 1 else 'ies'} ({counts_str}) -> {output_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse an iOS WhisperApp bulk transcript export log into a CSV DataFrame."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the export .txt log",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: same stem as input with _ParsedTable.csv)",
    )
    args = parser.parse_args()
    output_path = (
        args.output
        if args.output is not None
        else args.input_path.with_name(f"{args.input_path.stem}_ParsedTable.csv")
    )
    run_parse(input_path=args.input_path, output_path=output_path)


if __name__ == "__main__":
    main()
