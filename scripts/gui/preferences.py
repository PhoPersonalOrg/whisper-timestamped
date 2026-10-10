"""Persist GUI path/filelist preferences via QSettings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from PyQt6.QtCore import QSettings


def _settings() -> QSettings:
    return QSettings()


def _load_loaded_csvs(s: QSettings) -> Dict[str, Path]:
    """Deserialize format_id → CSV path map; skip missing files."""
    loaded_csvs: Dict[str, Path] = {}
    raw = s.value("loaded_csvs", "")
    mapping: Dict[str, Any] = {}
    if isinstance(raw, dict):
        mapping = raw
    elif isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                mapping = parsed
        except json.JSONDecodeError:
            mapping = {}

    for format_id, path_str in mapping.items():
        if not format_id or not path_str:
            continue
        path = Path(str(path_str))
        if path.is_file():
            loaded_csvs[str(format_id)] = path
    ## END for format_id, path_str in mapping.items()....

    return loaded_csvs


def load_prefs() -> Dict[str, Any]:
    """Load path-related prefs. Drop loaded CSV paths that no longer exist."""
    s = _settings()
    return {
        "scan_audio_dir": str(s.value("scan/audio_dir", "") or ""),
        "scan_format_id": str(s.value("scan/format_id", "") or ""),
        "scan_timezone": str(s.value("scan/timezone", "") or ""),
        "scan_csv_path": str(s.value("scan/csv_path", "") or ""),
        "loaded_csvs": _load_loaded_csvs(s),
        "browse_load_csv_dir": str(s.value("browse/load_csv_dir", "") or ""),
        "transcribe_output_dir": str(s.value("transcribe/output_dir", "") or ""),
    }


def save_prefs(prefs: Dict[str, Any]) -> None:
    """Write path-related prefs to QSettings."""
    s = _settings()
    s.setValue("scan/audio_dir", str(prefs.get("scan_audio_dir", "") or ""))
    s.setValue("scan/format_id", str(prefs.get("scan_format_id", "") or ""))
    s.setValue("scan/timezone", str(prefs.get("scan_timezone", "") or ""))
    s.setValue("scan/csv_path", str(prefs.get("scan_csv_path", "") or ""))
    s.setValue("browse/load_csv_dir", str(prefs.get("browse_load_csv_dir", "") or ""))
    s.setValue(
        "transcribe/output_dir",
        str(prefs.get("transcribe_output_dir", "") or ""),
    )

    loaded = prefs.get("loaded_csvs") or {}
    serializable = {
        str(fid): str(Path(path))
        for fid, path in loaded.items()
        if fid and path
    }
    s.setValue("loaded_csvs", json.dumps(serializable))
    s.sync()
