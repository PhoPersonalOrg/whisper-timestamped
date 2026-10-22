"""QAbstractTableModel backed by a merged pandas DataFrame of filelist CSVs.

Provides:
- RecordingTableModel  — the core data model
- RecordingFilterProxy — QSortFilterProxyModel with text / format / status filters
- load_format_csv / merge_format_csvs — CSV loading helpers
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from PyQt6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    pyqtSignal,
)
from PyQt6.QtGui import QBrush, QColor, QFont

from scripts.gui.style import ROW_TINTS
from scripts.transcript_outputs import (
    ensure_transcript_status_columns,
    has_transcript_path_values,
)


def parse_created_datetime(value: Any) -> Optional[datetime]:
    """Parse a filelist creation timestamp into naive local wall-clock time."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (ValueError, TypeError):
        pass
    text = str(value).strip()
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    dt = parsed.to_pydatetime()
    if getattr(dt, "tzinfo", None) is not None:
        dt = dt.astimezone().replace(tzinfo=None)
    ## END if tzinfo....

    return dt


def format_created_display(value: Any) -> str:
    """Local datetime, e.g. ``2026-10-07 (Wed) 4:56pm``."""
    dt = parse_created_datetime(value)
    if dt is None:
        return ""
    hour = dt.strftime("%I").lstrip("0") or "12"
    minute = dt.strftime("%M")
    ampm = dt.strftime("%p").lower()
    weekday = dt.strftime("%a")
    return f"{dt.strftime('%Y-%m-%d')} ({weekday}) {hour}:{minute}{ampm}"


def format_created_ago(value: Any, now: Optional[datetime] = None) -> str:
    """Now-relative label for a creation timestamp.

    Ladder (elapsed time from *now*):
      - under 1 hour        -> ``<1 hr ago`` (also used for small future skew)
      - under 24 hours      -> ``~N hr ago``
      - 1-6 calendar days   -> ``1 day ago (Tue)`` / ``N days ago (Mon)``
      - 7-13 calendar days  -> ``Last Wed``
      - 2-7 weeks           -> ``N weeks ago``
      - under 2 years       -> ``N months ago``
      - otherwise           -> ``N years ago``
    """
    dt = parse_created_datetime(value)
    if dt is None:
        return ""
    if now is None:
        now = datetime.now()
    ## END if now is None....

    elapsed = now - dt
    seconds = elapsed.total_seconds()
    if seconds < 3600:
        return "<1 hr ago"
    if seconds < 86400:
        hours = max(1, int(round(seconds / 3600)))
        return f"~{hours} hr ago"
    ## END if under 24 hours....

    days = (now.date() - dt.date()).days
    weekday = dt.strftime("%a")
    if days <= 0:
        # Over 24h elapsed but same calendar date cannot happen; guard anyway.
        days = 1
    if days <= 6:
        unit = "day" if days == 1 else "days"
        return f"{days} {unit} ago ({weekday})"
    if days <= 13:
        return f"Last {weekday}"
    if days < 56:
        weeks = max(2, int(round(days / 7)))
        return f"{weeks} weeks ago"
    if days < 730:
        months = max(2, int(round(days / 30)))
        return f"{months} months ago"
    ## END if days ladder....

    years = max(2, int(round(days / 365)))
    return f"{years} years ago"


# -- Column configuration -----------------------------------------------------

# Columns shown in the table (in order). The model stores the full DataFrame
# but only exposes these to the view.
DISPLAY_COLUMNS: List[str] = [
    "status",
    "name",
    "format_id",
    "title",
    "size_mb",
    "duration_hms",
    "extracted_creation_time",
    "ago",
    "is_duplicate",
    "transcript_txt",
]

# Display-only columns derived at render time; never materialized in the
# backing DataFrame (so they never leak into saved filelist CSVs).
VIRTUAL_COLUMNS: frozenset = frozenset({"ago"})

# Nice header labels
COLUMN_HEADERS: Dict[str, str] = {
    "status": "",
    "name": "Name",
    "format_id": "Format",
    "title": "Title",
    "size_mb": "Size (MB)",
    "duration_hms": "Duration",
    "extracted_creation_time": "Created",
    "ago": "Ago",
    "is_duplicate": "Dup",
    "transcript_txt": "Transcript",
}

# Status emoji/symbol for the first column
_STATUS_ICONS = {
    "transcribed": "✓",
    "pending": "⏳",
    "duplicate": "✕",
    "queued": "⋯",
    "transcribing": "▶",
    "cancelling": "⏹",
    "failed": "!",
}

PENDING_LIKE_STATUSES = {
    "pending",
    "queued",
    "transcribing",
    "cancelling",
    "failed",
}

PROGRESS_ROLE = Qt.ItemDataRole.UserRole + 1
JOB_PHASE_ROLE = Qt.ItemDataRole.UserRole + 2


# -- CSV loading helpers -------------------------------------------------------

def load_format_csv(csv_path: Path, format_id: str) -> pd.DataFrame:
    """Load a single format's filelist CSV, adding metadata columns."""
    df = pd.read_csv(csv_path)
    df["format_id"] = format_id
    df["source_csv"] = str(csv_path)
    return df


def merge_format_csvs(csv_paths: Dict[str, Path]) -> pd.DataFrame:
    """Merge multiple format CSVs into one DataFrame for the table."""
    frames: List[pd.DataFrame] = []
    for format_id, csv_path in csv_paths.items():
        if csv_path.is_file():
            frames.append(load_format_csv(csv_path, format_id))
    ## END for format_id, csv_path in csv_paths.items()...

    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    return merged


def compute_status(row: pd.Series) -> str:
    """Derive display status from a filelist row."""
    # Check duplicate first
    dup = row.get("is_duplicate")
    if pd.notna(dup) and (dup is True or str(dup).strip().lower() in ("true", "1", "yes")):
        return "duplicate"
    if has_transcript_path_values(row):
        return "transcribed"
    return "pending"


# -- Model ---------------------------------------------------------------------

class RecordingTableModel(QAbstractTableModel):
    """Table model backed by a merged filelist DataFrame."""

    data_changed_signal = pyqtSignal()  # emitted after bulk data changes

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._df: pd.DataFrame = pd.DataFrame()
        self._columns: List[str] = list(DISPLAY_COLUMNS)
        self._dup_font = QFont()
        self._dup_font.setStrikeOut(True)
        self._job_state: Dict[int, Dict[str, Any]] = {}


    def set_dataframe(self, df: pd.DataFrame) -> None:
        """Replace the model's backing DataFrame."""
        self.beginResetModel()
        self._job_state = {}
        self._df = df.copy()
        # Ensure all display columns exist
        for col in self._columns:
            if col in VIRTUAL_COLUMNS:
                continue
            if col not in self._df.columns:
                self._df[col] = ""
        ## END for col in self._columns...

        ensure_transcript_status_columns(self._df)

        # Compute status column
        self._df["status"] = self._df.apply(compute_status, axis=1)
        self.endResetModel()
        self.data_changed_signal.emit()


    def _emit_row_changed(self, df_index: int) -> None:
        try:
            visual_row = list(self._df.index).index(df_index)
            left = self.index(visual_row, 0)
            right = self.index(visual_row, len(self._columns) - 1)
            self.dataChanged.emit(left, right)
        except ValueError:
            pass


    def job_state(self, df_index: int) -> Optional[Dict[str, Any]]:
        """Return runtime job overlay for *df_index*, if any."""
        return self._job_state.get(int(df_index))


    def set_job_state(
        self,
        df_index: int,
        phase: str,
        *,
        progress: Optional[float] = None,
        position: Optional[int] = None,
        total: Optional[int] = None,
        message: Optional[str] = None,
    ) -> None:
        """Set ephemeral queued/transcribing/cancelling/failed overlay."""
        key = int(df_index)
        prev = self._job_state.get(key, {})
        state: Dict[str, Any] = {
            "phase": phase,
            "progress": progress if progress is not None else prev.get("progress"),
            "position": position if position is not None else prev.get("position"),
            "total": total if total is not None else prev.get("total"),
            "message": message if message is not None else prev.get("message"),
        }
        if progress is None and phase in ("queued", "failed"):
            state["progress"] = None
        self._job_state[key] = state
        self._emit_row_changed(key)


    def set_job_states(
        self,
        df_indices: List[int],
        phase: str,
    ) -> None:
        """Mark many rows with the same phase (e.g. queued at batch start)."""
        total = len(df_indices)
        for i, df_index in enumerate(df_indices, start=1):
            key = int(df_index)
            self._job_state[key] = {
                "phase": phase,
                "progress": None,
                "position": i,
                "total": total,
                "message": None,
            }
        ## END for i, df_index in enumerate....

        if self._df.empty:
            return
        top_left = self.index(0, 0)
        bottom_right = self.index(len(self._df) - 1, len(self._columns) - 1)
        self.dataChanged.emit(top_left, bottom_right)


    def clear_job_state(self, df_index: int) -> None:
        """Drop runtime overlay so persisted status (pending/transcribed) shows."""
        key = int(df_index)
        if self._job_state.pop(key, None) is not None:
            self._emit_row_changed(key)


    def clear_all_job_state(self) -> None:
        if not self._job_state:
            return
        self._job_state = {}
        if self._df.empty:
            return
        top_left = self.index(0, 0)
        bottom_right = self.index(len(self._df) - 1, len(self._columns) - 1)
        self.dataChanged.emit(top_left, bottom_right)


    def refresh_ago_column(self) -> None:
        """Re-render the now-relative Ago column without touching the DataFrame."""
        if self._df.empty or "ago" not in self._columns:
            return
        col = self._columns.index("ago")
        top = self.index(0, col)
        bottom = self.index(len(self._df) - 1, col)
        self.dataChanged.emit(top, bottom, [Qt.ItemDataRole.DisplayRole])


    def queued_df_indices(self) -> List[int]:
        return [
            idx for idx, state in self._job_state.items()
            if state.get("phase") == "queued"
        ]


    def transcribing_df_index(self) -> Optional[int]:
        for idx, state in self._job_state.items():
            if state.get("phase") in ("transcribing", "cancelling"):
                return idx
            ## END if transcribing or cancelling....

        ## END for idx, state in self._job_state.items()....

        return None


    def _status_for_visual_row(self, row: int) -> str:
        df_index = int(self._df.index[row])
        job = self._job_state.get(df_index)
        if job:
            return str(job.get("phase") or "pending")
        return str(self._df.iloc[row].get("status", "pending"))


    def _job_for_visual_row(self, row: int) -> Optional[Dict[str, Any]]:
        df_index = int(self._df.index[row])
        return self._job_state.get(df_index)


    def upsert_row(self, row: Dict[str, Any]) -> int:
        """Insert or update a row keyed by ``full_path``. Returns the DataFrame index."""
        full_path = str(row.get("full_path") or "")
        if full_path and "full_path" in self._df.columns and not self._df.empty:
            mask = self._df["full_path"].astype(str) == full_path
            if bool(mask.any()):
                idx = int(self._df.index[mask][0])
                self.update_row(idx, row)
                return idx
            ## END if mask.any()....

        ## END if matching full_path....

        for col in list(row.keys()) + list(self._columns):
            if col in VIRTUAL_COLUMNS:
                continue
            if col not in self._df.columns:
                self._df[col] = ""
            ## END if col not in self._df.columns....

        ## END for col in list(row.keys())....

        ensure_transcript_status_columns(self._df)
        new_index = 0 if self._df.empty else int(self._df.index.max()) + 1
        values: Dict[str, Any] = {col: "" for col in self._df.columns}
        values.update(row)
        visual = len(self._df)
        self.beginInsertRows(QModelIndex(), visual, visual)
        self._df.loc[new_index] = values
        self._df.at[new_index, "status"] = compute_status(self._df.loc[new_index])
        self.endInsertRows()
        self.data_changed_signal.emit()
        return new_index


    def update_row(self, df_index: int, updates: Dict[str, Any]) -> None:
        """Update specific cells in a row by DataFrame index."""
        if df_index not in self._df.index:
            return
        for col, value in updates.items():
            if col not in self._df.columns:
                self._df[col] = ""
            if self._df[col].dtype != "object":
                self._df[col] = self._df[col].astype("object")
            self._df.at[df_index, col] = value
        ## END for col, value in updates.items()...

        # Recompute persisted status; keep in-flight overlay unless transcripts landed
        self._df.at[df_index, "status"] = compute_status(self._df.loc[df_index])
        if has_transcript_path_values(self._df.loc[df_index]):
            self._job_state.pop(int(df_index), None)
        self._emit_row_changed(int(df_index))


    def get_dataframe(self) -> pd.DataFrame:
        """Return a copy of the backing DataFrame."""
        return self._df.copy()


    def get_row_data(self, row: int) -> Optional[pd.Series]:
        """Return the full Series for a visual row index."""
        if 0 <= row < len(self._df):
            series = self._df.iloc[row].copy()
            series["status"] = self._status_for_visual_row(row)
            return series
        return None


    def df_index_for_row(self, visual_row: int) -> Optional[int]:
        """Map visual row to DataFrame index."""
        if 0 <= visual_row < len(self._df):
            return self._df.index[visual_row]
        return None


    # -- QAbstractTableModel overrides -----------------------------------------

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._df)


    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._columns)


    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None

        row = index.row()
        col_name = self._columns[index.column()]

        if row < 0 or row >= len(self._df):
            return None

        raw_value = self._df.iloc[row].get(col_name, "")
        status = self._status_for_visual_row(row)
        job = self._job_for_visual_row(row)

        if role == PROGRESS_ROLE:
            if job is None:
                return None
            return job.get("progress")

        if role == JOB_PHASE_ROLE:
            return job.get("phase") if job else None

        if role == Qt.ItemDataRole.DisplayRole:
            if col_name == "status":
                return _STATUS_ICONS.get(status, "")
            if col_name == "is_duplicate":
                val = self._df.iloc[row].get("is_duplicate")
                if pd.notna(val) and (val is True or str(val).strip().lower() in ("true", "1", "yes")):
                    return "Yes"
                return ""
            if col_name == "transcript_txt":
                val = str(raw_value).strip() if pd.notna(raw_value) else ""
                if val:
                    return Path(val).name
                return ""
            if col_name == "extracted_creation_time":
                return format_created_display(raw_value)
            if col_name == "ago":
                return format_created_ago(self._df.iloc[row].get("extracted_creation_time", ""))
            if pd.isna(raw_value):
                return ""
            return str(raw_value)

        if role == Qt.ItemDataRole.UserRole:
            if col_name in ("extracted_creation_time", "ago"):
                dt = parse_created_datetime(self._df.iloc[row].get("extracted_creation_time", ""))
                return dt.strftime("%Y-%m-%d %H:%M:%S") if dt is not None else ""
            if pd.isna(raw_value):
                return ""
            return str(raw_value)

        if role == Qt.ItemDataRole.BackgroundRole:
            tint = ROW_TINTS.get(status)
            if tint is not None:
                return QBrush(tint)
            return None

        if role == Qt.ItemDataRole.FontRole:
            if status == "duplicate" and col_name == "name":
                return self._dup_font
            return None

        if role == Qt.ItemDataRole.ForegroundRole:
            if status == "duplicate":
                return QBrush(QColor("#9898b0"))
            return None

        if role == Qt.ItemDataRole.ToolTipRole:
            if col_name == "status":
                if job:
                    phase = str(job.get("phase") or "")
                    pos = job.get("position")
                    total = job.get("total")
                    progress = job.get("progress")
                    if phase == "queued":
                        if pos and total:
                            return f"Queued {pos}/{total}"
                        return "Queued"
                    if phase == "transcribing":
                        pct = f" {int(round(progress * 100))}%" if isinstance(progress, (int, float)) else ""
                        if pos and total:
                            return f"Transcribing {pos}/{total}{pct}"
                        return f"Transcribing{pct}"
                    if phase == "cancelling":
                        return "Cancelling — finishing this file, then stopping"
                    if phase == "failed":
                        return str(job.get("message") or "Transcription failed")
                return status
            if col_name == "transcript_txt":
                val = str(raw_value).strip() if pd.notna(raw_value) else ""
                if not val:
                    return "Not transcribed"
                backend = str(self._df.iloc[row].get("transcript_backend", "") or "").strip()
                model = str(self._df.iloc[row].get("transcript_model", "") or "").strip()
                if backend or model:
                    bits = [p for p in (backend, model) if p]
                    return f"{val}\n{' / '.join(bits)}"
                return val
            if col_name == "name":
                full_path = self._df.iloc[row].get("full_path", "")
                return str(full_path) if pd.notna(full_path) else ""
            if col_name in ("extracted_creation_time", "ago"):
                dt = parse_created_datetime(self._df.iloc[row].get("extracted_creation_time", ""))
                return dt.strftime("%Y-%m-%d %H:%M:%S") if dt is not None else ""
            return None

        if role == Qt.ItemDataRole.TextAlignmentRole:
            if col_name in ("status", "is_duplicate"):
                return Qt.AlignmentFlag.AlignCenter
            if col_name == "size_mb":
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            return None

        return None


    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal and 0 <= section < len(self._columns):
                col = self._columns[section]
                return COLUMN_HEADERS.get(col, col)
            if orientation == Qt.Orientation.Vertical:
                return str(section + 1)
        return None


    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable


    # -- Convenience -----------------------------------------------------------

    def status_counts(self) -> Dict[str, int]:
        """Return counts by display status for the status bar."""
        counts = {
            "transcribed": 0,
            "pending": 0,
            "duplicate": 0,
            "queued": 0,
            "transcribing": 0,
            "cancelling": 0,
            "failed": 0,
            "total": 0,
        }
        if "status" not in self._df.columns or self._df.empty:
            return counts
        counts["total"] = len(self._df)
        for visual_row in range(len(self._df)):
            status = self._status_for_visual_row(visual_row)
            counts[status] = counts.get(status, 0) + 1
        ## END for visual_row in range....

        return counts


    def format_ids(self) -> List[str]:
        """Return unique format_id values for filter dropdown."""
        if "format_id" not in self._df.columns:
            return []
        return sorted(self._df["format_id"].dropna().unique().tolist())


    def column_name(self, column: int) -> str:
        """Return the backing column name for a visual column index."""
        if 0 <= column < len(self._columns):
            return self._columns[column]
        return ""


# -- Filter Proxy Model -------------------------------------------------------

class RecordingFilterProxy(QSortFilterProxyModel):
    """Filter proxy supporting text search, format filter, and status filter."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._text_filter: str = ""
        self._format_filter: str = ""      # "" = all
        self._status_filter: str = ""      # "" = all
        self._hide_duplicates: bool = False
        self.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)


    def set_text_filter(self, text: str) -> None:
        self._text_filter = text.strip().lower()
        self.invalidateFilter()


    def set_format_filter(self, format_id: str) -> None:
        self._format_filter = format_id
        self.invalidateFilter()


    def set_status_filter(self, status: str) -> None:
        self._status_filter = status
        self.invalidateFilter()


    def set_hide_duplicates(self, hide: bool) -> None:
        self._hide_duplicates = hide
        self.invalidateFilter()


    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        model = self.sourceModel()
        if not isinstance(model, RecordingTableModel):
            return True
        row_data = model.get_row_data(source_row)
        if row_data is None:
            return False

        # Status filter (Pending includes in-flight / failed-retryable rows)
        status = str(row_data.get("status", ""))
        if self._status_filter == "pending":
            if status not in PENDING_LIKE_STATUSES:
                return False
        elif self._status_filter and status != self._status_filter:
            return False

        # Hide duplicates toggle
        if self._hide_duplicates and status == "duplicate":
            return False

        # Format filter
        if self._format_filter:
            fmt = str(row_data.get("format_id", ""))
            if fmt != self._format_filter:
                return False

        # Text search (name, title, full_path)
        if self._text_filter:
            searchable = " ".join([
                str(row_data.get("name", "")),
                str(row_data.get("title", "")),
                str(row_data.get("full_path", "")),
            ]).lower()
            if self._text_filter not in searchable:
                return False

        return True


    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        """Sort Created by the original timestamp, not the 12-hour display string.

        Empty / unparseable dates always sort after real timestamps, in both
        ascending (oldest first) and descending (newest first) order.
        """
        model = self.sourceModel()
        if isinstance(model, RecordingTableModel) and model.column_name(left.column()) in ("extracted_creation_time", "ago"):
            left_key = model.data(left, Qt.ItemDataRole.UserRole) or ""
            right_key = model.data(right, Qt.ItemDataRole.UserRole) or ""
            left_empty = not left_key
            right_empty = not right_key
            if left_empty or right_empty:
                if left_empty and right_empty:
                    return False
                # Qt swaps the operands when sorting descending, so invert
                # the empty comparison to keep blanks at the bottom either way.
                if self.sortOrder() == Qt.SortOrder.AscendingOrder:
                    return not left_empty
                return left_empty
            ## END if left_empty or right_empty....

            return left_key < right_key
        ## END if created column....

        return super().lessThan(left, right)
