"""Torch-free helpers for locating, merging, and describing transcript outputs."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd


PARAMS_SUFFIX = ".params.json"
TRANSCRIPTION_PARAMS_KEY = "transcription_params"

PARAM_FILELIST_COLUMNS: Tuple[str, ...] = (
    "transcript_params",
    "transcript_backend",
    "transcript_model",
    "transcript_crisper_mode",
)

# Segment-level suffix per output format; whisper-timestamped uses .words.json for JSON.
_OUTPUT_SUFFIX_BY_FORMAT = {
    "json": ".words.json",
    "csv": ".csv",
    "txt": ".txt",
    "vtt": ".vtt",
    "srt": ".srt",
    "tsv": ".tsv",
}

# All write_results keys → on-disk suffix (for filling transcript_* on skip/re-run).
_TRANSCRIPT_OUTPUT_SPECS: List[Tuple[str, str]] = [
    ("json", ".words.json"),
    ("csv", ".csv"),
    ("words.csv", ".words.csv"),
    ("txt", ".txt"),
    ("vtt", ".vtt"),
    ("words.vtt", ".words.vtt"),
    ("srt", ".srt"),
    ("words.srt", ".words.srt"),
    ("tsv", ".tsv"),
    ("words.tsv", ".words.tsv"),
]


def _transcript_column_for_key(key: str) -> str:
    return f"transcript_{key.replace('.', '_')}"


TRANSCRIPT_PATH_COLUMNS: Tuple[str, ...] = tuple(
    _transcript_column_for_key(key) for key, _ in _TRANSCRIPT_OUTPUT_SPECS
)


def transcript_status_columns() -> List[str]:
    """Filelist columns that describe transcription outputs and run parameters."""
    return list(TRANSCRIPT_PATH_COLUMNS) + list(PARAM_FILELIST_COLUMNS)


def _running_in_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        with open("/proc/version", encoding="utf-8") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def host_path(path: Union[str, Path]) -> Path:
    """Map Windows drive paths ↔ WSL ``/mnt/<drive>/...`` for the current host."""
    s = str(path).replace("\\", "/")
    drive_m = re.match(r"^([A-Za-z]):/(.*)$", s)
    if drive_m:
        letter, rest = drive_m.group(1), drive_m.group(2)
        if _running_in_wsl():
            return Path(f"/mnt/{letter.lower()}/{rest}")
        return Path(f"{letter}:/{rest}")

    mnt_m = re.match(r"^/mnt/([a-zA-Z])/(.*)$", s)
    if mnt_m and sys.platform == "win32" and not _running_in_wsl():
        letter, rest = mnt_m.group(1).upper(), mnt_m.group(2)
        return Path(f"{letter}:/{rest}")

    return Path(path)


def _cell_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def resolve_existing_file(path_val: Any) -> Optional[Path]:
    """Resolve a stored path string to an existing local file (WSL-aware)."""
    raw = _cell_str(path_val)
    if not raw:
        return None
    p = host_path(raw)
    if p.is_file():
        return p
    orig = Path(raw)
    return orig if orig.is_file() else None


def resolve_existing_dir(path_val: Any) -> Optional[Path]:
    """Resolve a path to an existing directory (WSL-aware)."""
    if path_val is None:
        return None
    if isinstance(path_val, Path):
        raw = str(path_val)
    else:
        raw = _cell_str(path_val)
        if not raw:
            return None
    p = host_path(raw)
    if p.is_dir():
        return p
    orig = Path(raw)
    return orig if orig.is_dir() else None


def base_name_from_row(row: Mapping[str, Any]) -> str:
    """Transcript basename: ``name`` stem, else ``full_path`` stem."""
    name_raw = _cell_str(row.get("name", "") if hasattr(row, "get") else "")
    if name_raw:
        return Path(name_raw).stem
    full_path_raw = _cell_str(row.get("full_path", "") if hasattr(row, "get") else "")
    if full_path_raw:
        return Path(full_path_raw).stem
    return ""


def csv_derived_output_dir(csv_path: Path) -> Path:
    """Default transcriptions dir for a filelist CSV (filelists/ sibling, else beside CSV)."""
    if csv_path.parent.name.lower() == "filelists":
        return csv_path.parent.parent / "transcriptions"
    return csv_path.parent / "transcriptions"


def params_sidecar_path(output_dir: Path, base_name: str) -> Path:
    return output_dir.joinpath(base_name).with_suffix(PARAMS_SUFFIX)


def write_params_sidecar(output_dir: Path, base_name: str, params: Mapping[str, Any]) -> Path:
    """Write ``{base_name}.params.json`` next to transcript outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = params_sidecar_path(output_dir, base_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(params), f, indent=2, ensure_ascii=False)
    return path


def _load_json_object(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def load_params_sidecar(output_dir: Path, base_name: str) -> Optional[Dict[str, Any]]:
    return _load_json_object(params_sidecar_path(output_dir, base_name))


def load_params_from_words_json(output_dir: Path, base_name: str) -> Optional[Dict[str, Any]]:
    data = _load_json_object(output_dir.joinpath(base_name).with_suffix(".words.json"))
    if not data:
        return None
    nested = data.get(TRANSCRIPTION_PARAMS_KEY)
    return nested if isinstance(nested, dict) else None


def load_transcription_params(output_dir: Path, base_name: str) -> Optional[Dict[str, Any]]:
    """Prefer the params sidecar; fall back to ``transcription_params`` in ``.words.json``."""
    return load_params_sidecar(output_dir, base_name) or load_params_from_words_json(
        output_dir, base_name
    )


def params_to_filelist_cols(
    params: Optional[Mapping[str, Any]],
    sidecar_path: Optional[Path] = None,
) -> Dict[str, str]:
    """Map run parameters to filelist columns."""
    sidecar_str = ""
    if sidecar_path is not None:
        existing = resolve_existing_file(sidecar_path)
        if existing is not None:
            sidecar_str = str(existing.resolve())
        elif sidecar_path.exists():
            sidecar_str = str(sidecar_path.resolve())
    cols = {
        "transcript_params": sidecar_str,
        "transcript_backend": "",
        "transcript_model": "",
        "transcript_crisper_mode": "",
    }
    if params:
        cols["transcript_backend"] = str(params.get("backend") or "")
        cols["transcript_model"] = str(params.get("model_name") or "")
        cols["transcript_crisper_mode"] = str(params.get("crisper_mode") or "")
    return cols


def collect_extant_transcript_paths(output_dir: Path, base_name: str) -> Dict[str, str]:
    """Map transcript_* column -> absolute path for outputs that already exist."""
    output_file_path = output_dir.joinpath(base_name)
    cols: Dict[str, str] = {}
    for key, suffix in _TRANSCRIPT_OUTPUT_SPECS:
        a_file = output_file_path.with_suffix(suffix)
        col = _transcript_column_for_key(key)
        if a_file.exists() and a_file.is_file():
            cols[col] = str(a_file.resolve())
        else:
            cols[col] = ""
        ## END if a_file exists....
    ## END for key, suffix in _TRANSCRIPT_OUTPUT_SPECS....

    return cols


def collect_extant_status_cols(output_dir: Path, base_name: str) -> Dict[str, str]:
    """Transcript path columns plus param sidecar/filelist fields, if present."""
    cols = collect_extant_transcript_paths(output_dir, base_name)
    params = load_transcription_params(output_dir, base_name)
    sidecar = params_sidecar_path(output_dir, base_name)
    cols.update(params_to_filelist_cols(params, sidecar if sidecar.is_file() else None))
    return cols


def flatten_write_results_to_transcript_cols(
    curr_output_files_dict: dict, base_name: str
) -> Dict[str, str]:
    """Map write_results nested dict to transcript_* absolute path strings."""
    cols: Dict[str, str] = {
        _transcript_column_for_key(key): "" for key, _ in _TRANSCRIPT_OUTPUT_SPECS
    }
    for key, by_base in curr_output_files_dict.items():
        path = by_base.get(base_name)
        col = _transcript_column_for_key(key)
        if path is not None:
            cols[col] = str(Path(path).resolve())
        ## END if path is not None....
    ## END for key, by_base in curr_output_files_dict.items()....

    return cols


def find_extant_output_files(
    output_dir: Path,
    base_name: str,
    output_formats: Sequence[str] = ("json", "csv", "srt", "vtt", "txt"),
) -> List[Path]:
    """Return any of the output files that would be created upon transcribe completion."""
    output_file_path: Path = output_dir.joinpath(base_name)
    found_output_files: List[Path] = []
    for fmt in output_formats:
        suffix = _OUTPUT_SUFFIX_BY_FORMAT.get(fmt, f".{fmt}")
        a_file: Path = output_file_path.with_suffix(suffix)
        if a_file.exists() and a_file.is_file():
            found_output_files.append(a_file)
        ## END if a_file exists....
    ## END for fmt in output_formats....

    return found_output_files


def _discard_transcript_outputs(output_dir: Path, base_name: str) -> List[Path]:
    """Remove any transcript outputs for base_name so the file can be reprocessed."""
    removed: List[Path] = []
    output_file_path = output_dir.joinpath(base_name)
    seen = set()
    suffixes = [suffix for _, suffix in _TRANSCRIPT_OUTPUT_SPECS] + [PARAMS_SUFFIX]
    for suffix in suffixes:
        a_file = output_file_path.with_suffix(suffix)
        key = a_file.as_posix()
        if key in seen:
            continue
        seen.add(key)
        if a_file.exists() and a_file.is_file():
            try:
                a_file.unlink()
                removed.append(a_file)
            except OSError as e:
                print(f"  ~ Could not remove partial output {a_file.name}: {e}")
            ## END try unlink....
        ## END if a_file exists....
    ## END for suffix in suffixes....

    return removed


def has_transcript_path_values(row: Mapping[str, Any]) -> bool:
    """True if any transcript_* path column is non-empty (file need not exist)."""
    for col in TRANSCRIPT_PATH_COLUMNS:
        if _cell_str(row.get(col, "") if hasattr(row, "get") else ""):
            return True
        ## END if non-empty transcript path....
    ## END for col in TRANSCRIPT_PATH_COLUMNS....

    return False


def ensure_transcript_status_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure transcript path and param columns exist as object dtype."""
    for col in transcript_status_columns():
        if col not in df.columns:
            df[col] = ""
        elif df[col].dtype != "object":
            df[col] = df[col].astype("object")
        ## END if col missing or non-object....
    ## END for col in transcript_status_columns()....

    return df


def merge_transcript_status_columns(
    new_df: pd.DataFrame, previous_df: pd.DataFrame
) -> pd.DataFrame:
    """Copy transcript_* and param columns from previous_df onto matching new_df rows.

    Match by ``full_path`` first, then ``name``.
    """
    cols = [c for c in transcript_status_columns() if c in previous_df.columns]
    if not cols or previous_df.empty or new_df.empty:
        return new_df

    ensure_transcript_status_columns(new_df)

    def _norm_path(val: Any) -> str:
        raw = _cell_str(val)
        if not raw:
            return ""
        return str(host_path(raw)).replace("\\", "/").lower()

    prev_by_path: Dict[str, pd.Series] = {}
    prev_by_name: Dict[str, pd.Series] = {}
    for _, prow in previous_df.iterrows():
        fp = _norm_path(prow.get("full_path", "") if "full_path" in previous_df.columns else "")
        if fp:
            prev_by_path[fp] = prow
        name = _cell_str(prow.get("name", "") if "name" in previous_df.columns else "").lower()
        if name:
            prev_by_name[name] = prow
        ## END if name....
    ## END for _, prow in previous_df.iterrows()....

    for idx, nrow in new_df.iterrows():
        match: Optional[pd.Series] = None
        fp = _norm_path(nrow.get("full_path", "") if "full_path" in new_df.columns else "")
        if fp and fp in prev_by_path:
            match = prev_by_path[fp]
        else:
            name = _cell_str(nrow.get("name", "") if "name" in new_df.columns else "").lower()
            if name and name in prev_by_name:
                match = prev_by_name[name]
            ## END if name match....
        ## END if full_path match....

        if match is None:
            continue
        for col in cols:
            val = match.get(col, "")
            if _cell_str(val):
                new_df.at[idx, col] = val
            ## END if previous col non-empty....
        ## END for col in cols....
    ## END for idx, nrow in new_df.iterrows()....

    return new_df


def candidate_output_dirs(
    row: Mapping[str, Any],
    *,
    format_output_dir: Optional[Path] = None,
    prefs_output_dir: Optional[Path] = None,
) -> List[Path]:
    """Deduped existing directories to search for a row's transcript outputs."""
    dirs: List[Path] = []
    seen: set[str] = set()

    def _add(path_val: Any) -> None:
        resolved = resolve_existing_dir(path_val)
        if resolved is None:
            return
        key = str(resolved.resolve())
        if key in seen:
            return
        seen.add(key)
        dirs.append(resolved.resolve())

    for col in TRANSCRIPT_PATH_COLUMNS:
        raw = _cell_str(row.get(col, "") if hasattr(row, "get") else "")
        if not raw:
            continue
        _add(Path(raw).parent)
        ## END if stored transcript path....
    ## END for col in TRANSCRIPT_PATH_COLUMNS....

    _add(format_output_dir)

    source_csv = _cell_str(row.get("source_csv", "") if hasattr(row, "get") else "")
    if source_csv:
        _add(csv_derived_output_dir(Path(source_csv)))
    ## END if source_csv....

    _add(prefs_output_dir)
    return dirs


def discover_row_transcript_updates(
    row: Mapping[str, Any],
    extra_dirs: Optional[Iterable[Path]] = None,
    *,
    format_output_dir: Optional[Path] = None,
    prefs_output_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """Return transcript/param column updates if outputs exist for this row.

    Prefers directories of already-stored ``transcript_*`` paths, then extra /
    format / CSV / prefs candidate dirs.
    """
    base_name = base_name_from_row(row)
    if not base_name:
        return {}

    search_dirs = candidate_output_dirs(
        row,
        format_output_dir=format_output_dir,
        prefs_output_dir=prefs_output_dir,
    )
    if extra_dirs is not None:
        seen = {str(p) for p in search_dirs}
        for extra in extra_dirs:
            resolved = resolve_existing_dir(extra)
            if resolved is None:
                continue
            key = str(resolved.resolve())
            if key in seen:
                continue
            seen.add(key)
            search_dirs.append(resolved.resolve())
        ## END for extra in extra_dirs....
    ## END if extra_dirs is not None....

    for output_dir in search_dirs:
        cols = collect_extant_status_cols(output_dir, base_name)
        if any(cols.get(col) for col in TRANSCRIPT_PATH_COLUMNS):
            return cols
        ## END if any transcript path found....
    ## END for output_dir in search_dirs....

    return {}
