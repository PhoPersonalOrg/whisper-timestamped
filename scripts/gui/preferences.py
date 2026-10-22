"""Persist GUI path/filelist preferences via QSettings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from PyQt6.QtCore import QSettings

from whisper_timestamped.recording_formats.completeness import (
    default_watch_folders,
    normalize_watch_folders,
)


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
        ## END try json.loads....

    ## END if raw type....

    for format_id, path_str in mapping.items():
        if not format_id or not path_str:
            continue
        path = Path(str(path_str))
        if path.is_file():
            loaded_csvs[str(format_id)] = path
        ## END if path.is_file()....

    ## END for format_id, path_str in mapping.items()....

    return loaded_csvs


def _load_watch_folders(s: QSettings) -> List[Dict[str, Any]]:
    raw = s.value("watch/folders", "")
    parsed: Any = raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = []
        ## END try json.loads....

    folders = normalize_watch_folders(parsed)
    if folders:
        return folders
    return default_watch_folders()


def load_prefs() -> Dict[str, Any]:
    """Load path-related prefs. Drop loaded CSV paths that no longer exist."""
    s = _settings()
    watch_enabled_raw = s.value("watch/enabled", True)
    if isinstance(watch_enabled_raw, str):
        watch_enabled = watch_enabled_raw.strip().lower() not in ("0", "false", "no")
    else:
        watch_enabled = bool(watch_enabled_raw)
    ## END if watch_enabled_raw is str....

    hidden_raw = s.value("table/hidden_columns", "")
    hidden_columns: List[str] = []
    if isinstance(hidden_raw, list):
        hidden_columns = [str(name) for name in hidden_raw if name]
    elif isinstance(hidden_raw, str) and hidden_raw.strip():
        try:
            parsed_hidden = json.loads(hidden_raw)
            if isinstance(parsed_hidden, list):
                hidden_columns = [str(name) for name in parsed_hidden if name]
            ## END if parsed_hidden is list....

        except json.JSONDecodeError:
            hidden_columns = []
        ## END try json.loads....

    ## END if hidden_raw type....

    return {
        "scan_audio_dir": str(s.value("scan/audio_dir", "") or ""),
        "scan_format_id": str(s.value("scan/format_id", "") or ""),
        "scan_timezone": str(s.value("scan/timezone", "") or ""),
        "scan_csv_path": str(s.value("scan/csv_path", "") or ""),
        "loaded_csvs": _load_loaded_csvs(s),
        "browse_load_csv_dir": str(s.value("browse/load_csv_dir", "") or ""),
        "transcribe_output_dir": str(s.value("transcribe/output_dir", "") or ""),
        "transcribe_backend": str(
            s.value("transcribe/backend", "crisperwhisper") or "crisperwhisper"
        ),
        "transcribe_model": str(s.value("transcribe/model", "turbo") or "turbo"),
        "transcribe_crisper_mode": str(
            s.value("transcribe/crisper_mode", "verbatim") or "verbatim"
        ),
        "watch_enabled": watch_enabled,
        "watch_folders": _load_watch_folders(s),
        "hidden_columns": hidden_columns,
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
    s.setValue(
        "transcribe/backend",
        str(prefs.get("transcribe_backend", "crisperwhisper") or "crisperwhisper"),
    )
    s.setValue(
        "transcribe/model",
        str(prefs.get("transcribe_model", "turbo") or "turbo"),
    )
    s.setValue(
        "transcribe/crisper_mode",
        str(prefs.get("transcribe_crisper_mode", "verbatim") or "verbatim"),
    )
    s.setValue("watch/enabled", bool(prefs.get("watch_enabled", True)))
    hidden_columns = [
        str(name) for name in (prefs.get("hidden_columns") or []) if name
    ]
    s.setValue("table/hidden_columns", json.dumps(hidden_columns))
    s.setValue(
        "watch/folders",
        json.dumps(normalize_watch_folders(prefs.get("watch_folders") or [])),
    )

    loaded = prefs.get("loaded_csvs") or {}
    serializable = {
        str(fid): str(Path(path))
        for fid, path in loaded.items()
        if fid and path
    }
    s.setValue("loaded_csvs", json.dumps(serializable))
    s.sync()
