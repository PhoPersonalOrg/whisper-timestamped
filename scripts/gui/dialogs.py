"""Configuration dialogs for scanning and transcription."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from whisper_timestamped.recording_formats import (
    get_format,
    list_format_ids,
    detect_format,
    RecordingsFormat,
)


# -- Scan Config Dialog --------------------------------------------------------

class ScanConfigDialog(QDialog):
    """Pre-scan configuration: audio dir, format, timezone, CSV output path."""

    def __init__(
        self,
        parent=None,
        *,
        initial_audio_dir: str = "",
        initial_format_id: str = "",
        initial_timezone: Optional[str] = None,
        initial_csv_path: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Scan Audio Directory")
        self.setMinimumWidth(550)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setSpacing(10)

        # Audio directory
        dir_row = QHBoxLayout()
        self._dir_edit = QLineEdit(initial_audio_dir or "")
        self._dir_edit.setPlaceholderText("Select audio directory…")
        dir_row.addWidget(self._dir_edit, stretch=1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_dir)
        dir_row.addWidget(browse_btn)
        form.addRow("Audio Directory:", dir_row)

        # Format
        self._format_combo = QComboBox()
        self._format_combo.addItem("Auto-detect", "")
        for fid in list_format_ids():
            fmt = get_format(fid)
            self._format_combo.addItem(f"{fid} — {fmt.label}", fid)
        ## END for fid in list_format_ids()...
        self._format_combo.currentIndexChanged.connect(self._on_format_changed)
        if initial_format_id:
            idx = self._format_combo.findData(initial_format_id)
            if idx >= 0:
                self._format_combo.setCurrentIndex(idx)
        form.addRow("Format:", self._format_combo)

        # Timezone
        self._tz_edit = QLineEdit(initial_timezone or "America/Los_Angeles")
        form.addRow("Timezone:", self._tz_edit)

        # Output CSV
        csv_row = QHBoxLayout()
        self._csv_edit = QLineEdit(initial_csv_path or "")
        self._csv_edit.setPlaceholderText("Auto-generated from format defaults")
        csv_row.addWidget(self._csv_edit, stretch=1)
        csv_browse = QPushButton("Browse…")
        csv_browse.clicked.connect(self._browse_csv)
        csv_row.addWidget(csv_browse)
        form.addRow("Output CSV:", csv_row)

        layout.addLayout(form)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        scan_btn = QPushButton("Scan")
        scan_btn.setObjectName("accentButton")
        scan_btn.clicked.connect(self.accept)
        btn_layout.addWidget(scan_btn)
        layout.addLayout(btn_layout)


    def _browse_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select Audio Directory",
            self._dir_edit.text() or str(Path.home()),
        )
        if path:
            self._dir_edit.setText(path)
            self._auto_fill_csv()


    def _browse_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Output CSV Path",
            self._csv_edit.text() or "",
            "CSV files (*.csv)",
        )
        if path:
            self._csv_edit.setText(path)


    def _on_format_changed(self, index: int) -> None:
        self._auto_fill_csv()


    def _auto_fill_csv(self) -> None:
        """Auto-fill the CSV path from the selected format's defaults."""
        fmt = self.selected_format()
        if fmt is not None and fmt.default_filelists_dir is not None:
            try:
                csv_path = fmt.default_filelist_csv_path(
                    datetime.now(ZoneInfo(self.timezone())).replace(tzinfo=None)
                )
                self._csv_edit.setText(str(csv_path))
            except Exception:
                pass


    def audio_dir(self) -> Path:
        return Path(self._dir_edit.text().strip())


    def format_id(self) -> str:
        """Raw combo value; empty string means Auto-detect."""
        return self._format_combo.currentData() or ""


    def selected_format(self) -> Optional[RecordingsFormat]:
        fid = self.format_id()
        if not fid:
            # Auto-detect from the selected directory
            audio_dir = self.audio_dir()
            if audio_dir.is_dir():
                detected = detect_format(audio_dir)
                if detected is not None:
                    return detected
            return get_format("ios_whisper_app")
        return get_format(fid)


    def timezone(self) -> str:
        tz = self._tz_edit.text().strip()
        return tz if tz else "America/Los_Angeles"


    def csv_path(self) -> Path:
        text = self._csv_edit.text().strip()
        if text:
            return Path(text)
        # Fall back to format default
        fmt = self.selected_format()
        if fmt is not None:
            try:
                return fmt.default_filelist_csv_path(
                    datetime.now(ZoneInfo(self.timezone())).replace(tzinfo=None)
                )
            except Exception:
                pass
        return Path("audio_file_list.csv")


# -- Transcribe Config Dialog --------------------------------------------------

class TranscribeConfigDialog(QDialog):
    """Pre-transcription configuration: backend, model, output dir."""

    def __init__(
        self,
        default_output_dir: str = "",
        parent=None,
        *,
        initial_backend: str = "crisperwhisper",
        initial_model: str = "turbo",
        initial_crisper_mode: str = "verbatim",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Transcription Settings")
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setSpacing(10)

        # Backend
        self._backend_combo = QComboBox()
        self._backend_combo.addItem("CrisperWhisper", "crisperwhisper")
        self._backend_combo.addItem("OpenAI Whisper", "openai-whisper")
        self._backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        idx = self._backend_combo.findData(initial_backend)
        if idx >= 0:
            self._backend_combo.setCurrentIndex(idx)
        form.addRow("Backend:", self._backend_combo)

        # Model
        self._model_edit = QLineEdit("turbo")
        form.addRow("Model:", self._model_edit)

        # CrisperWhisper mode
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Verbatim", "verbatim")
        self._mode_combo.addItem("Intended", "intended")
        mode_idx = self._mode_combo.findData(initial_crisper_mode)
        if mode_idx >= 0:
            self._mode_combo.setCurrentIndex(mode_idx)
        self._mode_label = QLabel("CrisperWhisper Mode:")
        form.addRow(self._mode_label, self._mode_combo)
        self._on_backend_changed(self._backend_combo.currentIndex())
        self._model_edit.setText(initial_model or "turbo")

        # Output directory
        dir_row = QHBoxLayout()
        self._dir_edit = QLineEdit(default_output_dir)
        self._dir_edit.setPlaceholderText("Transcriptions output directory")
        dir_row.addWidget(self._dir_edit, stretch=1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_dir)
        dir_row.addWidget(browse_btn)
        form.addRow("Output Dir:", dir_row)

        layout.addLayout(form)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        start_btn = QPushButton("Start Transcription")
        start_btn.setObjectName("accentButton")
        start_btn.clicked.connect(self.accept)
        btn_layout.addWidget(start_btn)
        layout.addLayout(btn_layout)


    def _on_backend_changed(self, index: int) -> None:
        is_crisper = self._backend_combo.currentData() == "crisperwhisper"
        self._mode_combo.setVisible(is_crisper)
        self._mode_label.setVisible(is_crisper)
        # Default model name based on backend
        if is_crisper:
            if self._model_edit.text() in ("medium.en", "medium"):
                self._model_edit.setText("turbo")
        else:
            if self._model_edit.text() in ("turbo", "medium"):
                self._model_edit.setText("medium.en")


    def _browse_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory",
            self._dir_edit.text() or str(Path.home()),
        )
        if path:
            self._dir_edit.setText(path)


    def backend(self) -> str:
        return self._backend_combo.currentData() or "crisperwhisper"


    def model_name(self) -> str:
        return self._model_edit.text().strip() or "turbo"


    def crisper_mode(self) -> str:
        return self._mode_combo.currentData() or "verbatim"


    def output_dir(self) -> Path:
        text = self._dir_edit.text().strip()
        return Path(text) if text else Path("transcriptions")


# -- Watch Folders Dialog ------------------------------------------------------

class WatchFoldersDialog(QDialog):
    """Enable/disable watched recordings folders and auto-transcribe."""

    def __init__(
        self,
        folders: Optional[List[Dict[str, Any]]] = None,
        *,
        auto_transcribe: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Watch Folders")
        self.setMinimumWidth(720)
        self.setMinimumHeight(360)

        layout = QVBoxLayout(self)
        self._auto_cb = QCheckBox("Auto-transcribe completed recordings")
        self._auto_cb.setChecked(auto_transcribe)
        layout.addWidget(self._auto_cb)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(
            ["On", "Format", "Folder", "Recursive"]
        )
        self._table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        layout.addWidget(self._table)

        for spec in folders or []:
            self._add_row(spec)
        ## END for spec in folders....

        row_btns = QHBoxLayout()
        add_btn = QPushButton("Add folder…")
        add_btn.clicked.connect(self._add_folder)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_selected)
        row_btns.addWidget(add_btn)
        row_btns.addWidget(remove_btn)
        row_btns.addStretch()
        layout.addLayout(row_btns)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton("Save")
        save_btn.setObjectName("accentButton")
        save_btn.clicked.connect(self.accept)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

    def _add_row(self, spec: Dict[str, Any]) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        enabled_item = QTableWidgetItem()
        enabled_item.setFlags(
            Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
        )
        enabled_item.setCheckState(
            Qt.CheckState.Checked
            if spec.get("enabled", True)
            else Qt.CheckState.Unchecked
        )
        self._table.setItem(row, 0, enabled_item)

        format_combo = QComboBox()
        for fid in list_format_ids():
            format_combo.addItem(fid, fid)
        ## END for fid in list_format_ids()....

        format_id = str(spec.get("format_id") or "")
        idx = format_combo.findData(format_id)
        if idx >= 0:
            format_combo.setCurrentIndex(idx)
        self._table.setCellWidget(row, 1, format_combo)

        path_item = QTableWidgetItem(str(spec.get("path") or ""))
        self._table.setItem(row, 2, path_item)

        recursive_item = QTableWidgetItem()
        recursive_item.setFlags(
            Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
        )
        recursive_item.setCheckState(
            Qt.CheckState.Checked
            if spec.get("recursive")
            else Qt.CheckState.Unchecked
        )
        self._table.setItem(row, 3, recursive_item)

    def _add_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select recordings folder", str(Path.home())
        )
        if not path:
            return
        detected = detect_format(Path(path))
        format_id = detected.id if detected is not None else "ios_whisper_app"
        recursive = bool(detected.nest_one_level) if detected is not None else False
        self._add_row(
            {
                "format_id": format_id,
                "path": path,
                "enabled": True,
                "recursive": recursive,
            }
        )

    def _remove_selected(self) -> None:
        row = self._table.currentRow()
        if row >= 0:
            self._table.removeRow(row)
        ## END if row >= 0....

    def auto_transcribe(self) -> bool:
        return self._auto_cb.isChecked()

    def folders(self) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for row in range(self._table.rowCount()):
            enabled_item = self._table.item(row, 0)
            path_item = self._table.item(row, 2)
            recursive_item = self._table.item(row, 3)
            combo = self._table.cellWidget(row, 1)
            format_id = ""
            if isinstance(combo, QComboBox):
                format_id = str(combo.currentData() or combo.currentText() or "")
            path = path_item.text().strip() if path_item is not None else ""
            if not format_id or not path:
                continue
            result.append(
                {
                    "format_id": format_id,
                    "path": path,
                    "enabled": (
                        enabled_item is not None
                        and enabled_item.checkState() == Qt.CheckState.Checked
                    ),
                    "recursive": (
                        recursive_item is not None
                        and recursive_item.checkState() == Qt.CheckState.Checked
                    ),
                }
            )
        ## END for row in range(self._table.rowCount())....

        return result

