"""Configuration dialogs for scanning and transcription."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
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

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Scan Audio Directory")
        self.setMinimumWidth(550)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setSpacing(10)

        # Audio directory
        dir_row = QHBoxLayout()
        self._dir_edit = QLineEdit()
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
        form.addRow("Format:", self._format_combo)

        # Timezone
        self._tz_edit = QLineEdit("America/Los_Angeles")
        form.addRow("Timezone:", self._tz_edit)

        # Output CSV
        csv_row = QHBoxLayout()
        self._csv_edit = QLineEdit()
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


    def selected_format(self) -> Optional[RecordingsFormat]:
        fid = self._format_combo.currentData()
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

    def __init__(self, default_output_dir: str = "", parent=None) -> None:
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
        form.addRow("Backend:", self._backend_combo)

        # Model
        self._model_edit = QLineEdit("medium")
        form.addRow("Model:", self._model_edit)

        # CrisperWhisper mode
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Verbatim", "verbatim")
        self._mode_combo.addItem("Clean", "clean")
        self._mode_label = QLabel("CrisperWhisper Mode:")
        form.addRow(self._mode_label, self._mode_combo)

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
            if self._model_edit.text() == "medium.en":
                self._model_edit.setText("medium")
        else:
            if self._model_edit.text() == "medium":
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
        return self._model_edit.text().strip() or "medium"


    def crisper_mode(self) -> str:
        return self._mode_combo.currentData() or "verbatim"


    def output_dir(self) -> Path:
        text = self._dir_edit.text().strip()
        return Path(text) if text else Path("transcriptions")
