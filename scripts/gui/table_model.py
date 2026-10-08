"""QAbstractTableModel backed by a merged pandas DataFrame of filelist CSVs.

Provides:
- RecordingTableModel  — the core data model
- RecordingFilterProxy — QSortFilterProxyModel with text / format / status filters
- load_format_csv / merge_format_csvs — CSV loading helpers
"""

from __future__ import annotations

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
    "is_duplicate",
    "transcript_txt",
]

# Nice header labels
COLUMN_HEADERS: Dict[str, str] = {
    "status": "",
    "name": "Name",
    "format_id": "Format",
    "title": "Title",
    "size_mb": "Size (MB)",
    "duration_hms": "Duration",
    "extracted_creation_time": "Created",
    "is_duplicate": "Dup",
    "transcript_txt": "Transcript",
}

# Status emoji/symbol for the first column
_STATUS_ICONS = {
    "transcribed": "✓",
    "pending": "⏳",
    "duplicate": "✕",
}


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
    # Check for any transcript output
    txt = row.get("transcript_txt", "")
    json_col = row.get("transcript_json", "")
    if (pd.notna(txt) and str(txt).strip()) or (pd.notna(json_col) and str(json_col).strip()):
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


    def set_dataframe(self, df: pd.DataFrame) -> None:
        """Replace the model's backing DataFrame."""
        self.beginResetModel()
        self._df = df.copy()
        # Ensure all display columns exist
        for col in self._columns:
            if col not in self._df.columns:
                self._df[col] = ""
        ## END for col in self._columns...

        # Compute status column
        self._df["status"] = self._df.apply(compute_status, axis=1)
        self.endResetModel()
        self.data_changed_signal.emit()


    def update_row(self, df_index: int, updates: Dict[str, Any]) -> None:
        """Update specific cells in a row by DataFrame index."""
        if df_index not in self._df.index:
            return
        for col, value in updates.items():
            if col in self._df.columns:
                self._df.at[df_index, col] = value
        ## END for col, value in updates.items()...

        # Recompute status
        self._df.at[df_index, "status"] = compute_status(self._df.loc[df_index])
        # Find the visual row for this df_index
        try:
            visual_row = list(self._df.index).index(df_index)
            left = self.index(visual_row, 0)
            right = self.index(visual_row, len(self._columns) - 1)
            self.dataChanged.emit(left, right)
        except ValueError:
            pass


    def get_dataframe(self) -> pd.DataFrame:
        """Return a copy of the backing DataFrame."""
        return self._df.copy()


    def get_row_data(self, row: int) -> Optional[pd.Series]:
        """Return the full Series for a visual row index."""
        if 0 <= row < len(self._df):
            return self._df.iloc[row].copy()
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
        status = str(self._df.iloc[row].get("status", "pending"))

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
            if col_name == "transcript_txt":
                val = str(raw_value).strip() if pd.notna(raw_value) else ""
                return val if val else "Not transcribed"
            if col_name == "name":
                full_path = self._df.iloc[row].get("full_path", "")
                return str(full_path) if pd.notna(full_path) else ""
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
        """Return counts by status for the status bar."""
        if "status" not in self._df.columns or self._df.empty:
            return {"transcribed": 0, "pending": 0, "duplicate": 0, "total": 0}
        counts = self._df["status"].value_counts().to_dict()
        counts["total"] = len(self._df)
        return counts


    def format_ids(self) -> List[str]:
        """Return unique format_id values for filter dropdown."""
        if "format_id" not in self._df.columns:
            return []
        return sorted(self._df["format_id"].dropna().unique().tolist())


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

        # Status filter
        status = str(row_data.get("status", ""))
        if self._status_filter and status != self._status_filter:
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
