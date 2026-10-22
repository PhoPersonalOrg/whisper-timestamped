"""REC continuous video recorder (CAM_*) format with timing sidecars."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from whisper_timestamped.recording_formats.base import (
    RecordingsFormat,
    utc_to_local_str,
)

_CAM_RE = re.compile(
    r"^CAM_(\d{4}-\d{2}-\d{2}T\d{6})$",
    re.IGNORECASE,
)

_DEFAULT_RECORDINGS_DIR = Path(r"H:/ScreenRecordings/REC_continuous_video_recorder")

_EXTRA_COLUMNS = (
    "timing_sidecar",
    "segment_index",
    "duration_perf_seconds",
    "lsl_local_clock_start",
)


def sidecar_path_for(recording_file: Path) -> Path:
    """Mirror continuous_video_recorder: ``file.ext`` → ``file.ext.timing.json``."""
    return recording_file.with_suffix(recording_file.suffix + ".timing.json")


def parse_cam_filename(path: Path) -> Optional[datetime]:
    """Parse ``CAM_YYYY-MM-DDTHHMMSS`` stem into a naive local datetime."""
    match = _CAM_RE.match(path.stem)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%dT%H%M%S")
    except ValueError:
        return None


def read_timing_sidecar(recording_file: Path) -> Optional[Dict[str, Any]]:
    """Load the timing sidecar JSON next to *recording_file*, or None."""
    sidecar = sidecar_path_for(recording_file)
    if not sidecar.is_file():
        return None
    try:
        with open(sidecar, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


class RecContinuousVideoRecorderFormat(RecordingsFormat):
    """CAM_* videos from continuous_video_recorder (debut-example and peers)."""

    def __init__(self) -> None:
        root = _DEFAULT_RECORDINGS_DIR
        super().__init__(
            id="rec_continuous_video_recorder",
            label="REC_continuous_video_recorder",
            media_extensions=(".mkv", ".mp4"),
            input_mode="directory",
            default_recordings_dir=root,
            default_output_dir=root / "transcriptions",
            default_filelists_dir=root / "filelists",
            filelist_name_prefix="rec_continuous_video_recorder",
            nest_one_level=False,
            extra_filelist_columns=_EXTRA_COLUMNS,
        )

    def matches_file(self, path: Path) -> bool:
        if path.suffix.lower() not in self.media_extensions:
            return False
        return parse_cam_filename(path) is not None

    def extract_creation_time(
        self,
        path: Path,
        tz_name: str = "America/Los_Angeles",
    ) -> Optional[str]:
        sidecar = read_timing_sidecar(path)
        if sidecar is not None:
            wall_unix = sidecar.get("wall_clock_start_unix")
            if wall_unix is not None:
                try:
                    utc_dt = datetime.fromtimestamp(
                        float(wall_unix), tz=timezone.utc
                    )
                    return utc_to_local_str(utc_dt, tz_name)
                except (TypeError, ValueError, OSError, OverflowError):
                    pass
            ## END if wall_unix is not None....
        ## END if sidecar is not None....

        parsed = parse_cam_filename(path)
        if parsed is None:
            return None
        return parsed.strftime("%Y-%m-%d %H:%M:%S")

    def extra_row_fields(self, path: Path) -> Dict[str, Any]:
        fields: Dict[str, Any] = {
            "timing_sidecar": "",
            "segment_index": "",
            "duration_perf_seconds": "",
            "lsl_local_clock_start": "",
        }
        sidecar_path = sidecar_path_for(path)
        sidecar = read_timing_sidecar(path)
        if sidecar is None:
            if sidecar_path.is_file():
                fields["timing_sidecar"] = str(sidecar_path)
            return fields
        ## END if sidecar is None....

        fields["timing_sidecar"] = str(sidecar_path)
        if "segment_index" in sidecar and sidecar["segment_index"] is not None:
            fields["segment_index"] = sidecar["segment_index"]
        ## END if segment_index....
        if (
            "duration_perf_seconds" in sidecar
            and sidecar["duration_perf_seconds"] is not None
        ):
            fields["duration_perf_seconds"] = sidecar["duration_perf_seconds"]
        ## END if duration_perf_seconds....
        if (
            "lsl_local_clock_start" in sidecar
            and sidecar["lsl_local_clock_start"] is not None
        ):
            fields["lsl_local_clock_start"] = sidecar["lsl_local_clock_start"]
        ## END if lsl_local_clock_start....
        return fields



class RecContinuousFormat(RecContinuousVideoRecorderFormat):
    """Backward-compatible alias id ``rec_continuous``."""

    def __init__(self) -> None:
        super().__init__()
        self.id = "rec_continuous"
        self.label = "REC_continuous_video_recorder"
        self.filelist_name_prefix = "rec_continuous"
