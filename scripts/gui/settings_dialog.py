"""Persisted Settings dialog for the Audio Recording Manager."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from scripts.gui.idle import (
    DEFAULT_IDLE_SECONDS,
    DEFAULT_LOW_PRIORITY,
    DEFAULT_RUN_WHEN,
    IDLE_SECONDS_CHOICES,
    normalize_idle_seconds,
    normalize_run_when,
)
from whisper_timestamped.recording_formats import (
    canonical_format_id,
    detect_format,
    list_canonical_format_ids,
)

IDLE_LABELS = {
    60: "1 minute",
    120: "2 minutes",
    300: "5 minutes",
    600: "10 minutes",
    900: "15 minutes",
}


def alt_held(modifiers: Optional[Qt.KeyboardModifier] = None) -> bool:
    """True when Alt is down. Pass *modifiers* in tests; otherwise read the app."""
    if modifiers is None:
        modifiers = QApplication.keyboardModifiers()
    return bool(modifiers & Qt.KeyboardModifier.AltModifier)


class SettingsDialog(QDialog):
    """Edit persisted watch, transcription, and background preferences."""

    def __init__(
        self,
        folders: Optional[List[Dict[str, Any]]] = None,
        *,
        auto_transcribe: bool = True,
        start_with_windows: bool = False,
        show_autostart: bool = True,
        scan_timezone: str = "America/Los_Angeles",
        transcribe_backend: str = "crisperwhisper",
        transcribe_model: str = "turbo",
        transcribe_crisper_mode: str = "verbatim",
        transcribe_output_dir: str = "",
        run_when: str = DEFAULT_RUN_WHEN,
        idle_seconds: int = DEFAULT_IDLE_SECONDS,
        low_priority: bool = DEFAULT_LOW_PRIORITY,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(720)
        self.setMinimumHeight(520)

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(
            self._build_general_tab(
                auto_transcribe=auto_transcribe,
                start_with_windows=start_with_windows,
                show_autostart=show_autostart,
                scan_timezone=scan_timezone,
            ),
            "General",
        )
        tabs.addTab(
            self._build_transcription_tab(
                transcribe_backend=transcribe_backend,
                transcribe_model=transcribe_model,
                transcribe_crisper_mode=transcribe_crisper_mode,
                transcribe_output_dir=transcribe_output_dir,
            ),
            "Transcription",
        )
        tabs.addTab(
            self._build_background_tab(
                run_when=run_when,
                idle_seconds=idle_seconds,
                low_priority=low_priority,
            ),
            "Background",
        )
        tabs.addTab(self._build_watch_tab(folders or []), "Watch folders")
        layout.addWidget(tabs)

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

    def _build_general_tab(
        self,
        *,
        auto_transcribe: bool,
        start_with_windows: bool,
        show_autostart: bool,
        scan_timezone: str,
    ) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        form.setSpacing(10)

        self._auto_cb = QCheckBox("Auto-transcribe completed recordings")
        self._auto_cb.setChecked(bool(auto_transcribe))
        form.addRow(self._auto_cb)

        self._autostart_cb: Optional[QCheckBox] = None
        if show_autostart:
            self._autostart_cb = QCheckBox("Start with Windows")
            self._autostart_cb.setChecked(bool(start_with_windows))
            form.addRow(self._autostart_cb)
        ## END if show_autostart....

        self._tz_edit = QLineEdit(scan_timezone or "America/Los_Angeles")
        form.addRow("Default timezone:", self._tz_edit)
        return tab

    def _build_transcription_tab(
        self,
        *,
        transcribe_backend: str,
        transcribe_model: str,
        transcribe_crisper_mode: str,
        transcribe_output_dir: str,
    ) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        form.setSpacing(10)

        self._backend_combo = QComboBox()
        self._backend_combo.addItem("CrisperWhisper", "crisperwhisper")
        self._backend_combo.addItem("OpenAI Whisper", "openai-whisper")
        form.addRow("Backend:", self._backend_combo)

        self._model_edit = QLineEdit(transcribe_model or "turbo")
        form.addRow("Model:", self._model_edit)

        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Verbatim", "verbatim")
        self._mode_combo.addItem("Intended", "intended")
        mode_idx = self._mode_combo.findData(transcribe_crisper_mode)
        if mode_idx >= 0:
            self._mode_combo.setCurrentIndex(mode_idx)
        ## END if mode_idx >= 0....

        self._mode_label = QLabel("CrisperWhisper Mode:")
        form.addRow(self._mode_label, self._mode_combo)

        idx = self._backend_combo.findData(transcribe_backend)
        self._backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        if idx >= 0:
            self._backend_combo.setCurrentIndex(idx)
        ## END if idx >= 0....

        dir_row = QHBoxLayout()
        self._dir_edit = QLineEdit(transcribe_output_dir or "")
        self._dir_edit.setPlaceholderText("Transcriptions output directory")
        dir_row.addWidget(self._dir_edit, stretch=1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_output_dir)
        dir_row.addWidget(browse_btn)
        form.addRow("Default output dir:", dir_row)

        hint = QLabel(
            "Used for auto-transcribe and for Transcribe without Alt-click."
        )
        hint.setWordWrap(True)
        form.addRow(hint)
        self._on_backend_changed(self._backend_combo.currentIndex())
        return tab

    def _build_background_tab(
        self,
        *,
        run_when: str,
        idle_seconds: int,
        low_priority: bool,
    ) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        group = QGroupBox("When to auto-transcribe")
        bg_layout = QVBoxLayout(group)
        self._run_when_group = QButtonGroup(self)
        self._always_radio = QRadioButton("Whenever files are ready")
        self._idle_radio = QRadioButton("When this computer is idle")
        self._run_when_group.addButton(self._always_radio)
        self._run_when_group.addButton(self._idle_radio)
        bg_layout.addWidget(self._always_radio)
        bg_layout.addWidget(self._idle_radio)
        if normalize_run_when(run_when) == "always":
            self._always_radio.setChecked(True)
        else:
            self._idle_radio.setChecked(True)
        ## END if run_when is always....

        idle_row = QHBoxLayout()
        idle_row.addWidget(QLabel("Idle for:"))
        self._idle_combo = QComboBox()
        chosen_idle = normalize_idle_seconds(idle_seconds)
        for seconds in IDLE_SECONDS_CHOICES:
            self._idle_combo.addItem(IDLE_LABELS[seconds], seconds)
        ## END for seconds in IDLE_SECONDS_CHOICES....

        idle_idx = self._idle_combo.findData(chosen_idle)
        if idle_idx >= 0:
            self._idle_combo.setCurrentIndex(idle_idx)
        ## END if idle_idx >= 0....

        idle_row.addWidget(self._idle_combo)
        idle_row.addStretch()
        bg_layout.addLayout(idle_row)

        self._low_priority_cb = QCheckBox("Run auto-transcribe at low CPU priority")
        self._low_priority_cb.setChecked(bool(low_priority))
        bg_layout.addWidget(self._low_priority_cb)
        self._idle_radio.toggled.connect(self._sync_idle_combo_enabled)
        self._sync_idle_combo_enabled(self._idle_radio.isChecked())
        layout.addWidget(group)
        layout.addStretch()
        return tab

    def _build_watch_tab(self, folders: List[Dict[str, Any]]) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
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

        for spec in folders:
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
        return tab

    def _on_backend_changed(self, index: int) -> None:
        is_crisper = self._backend_combo.currentData() == "crisperwhisper"
        self._mode_combo.setVisible(is_crisper)
        self._mode_label.setVisible(is_crisper)
        if is_crisper:
            if self._model_edit.text() in ("medium.en", "medium"):
                self._model_edit.setText("turbo")
            ## END if model is openai default....

        else:
            if self._model_edit.text() in ("turbo", "medium"):
                self._model_edit.setText("medium.en")
            ## END if model is crisper default....

        ## END if is_crisper....

    def _browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self._dir_edit.text() or str(Path.home()),
        )
        if path:
            self._dir_edit.setText(path)
        ## END if path....

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
        for fid in list_canonical_format_ids():
            format_combo.addItem(fid, fid)
        ## END for fid in list_canonical_format_ids()....

        format_id = canonical_format_id(str(spec.get("format_id") or ""))
        idx = format_combo.findData(format_id)
        if idx >= 0:
            format_combo.setCurrentIndex(idx)
        ## END if idx >= 0....

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

    def _sync_idle_combo_enabled(self, idle_selected: bool) -> None:
        self._idle_combo.setEnabled(bool(idle_selected))

    def auto_transcribe(self) -> bool:
        return self._auto_cb.isChecked()

    def start_with_windows(self) -> bool:
        if self._autostart_cb is None:
            return False
        return self._autostart_cb.isChecked()

    def scan_timezone(self) -> str:
        tz = self._tz_edit.text().strip()
        return tz if tz else "America/Los_Angeles"

    def transcribe_backend(self) -> str:
        return self._backend_combo.currentData() or "crisperwhisper"

    def transcribe_model(self) -> str:
        return self._model_edit.text().strip() or "turbo"

    def transcribe_crisper_mode(self) -> str:
        return self._mode_combo.currentData() or "verbatim"

    def transcribe_output_dir(self) -> str:
        return self._dir_edit.text().strip()

    def run_when(self) -> str:
        if self._always_radio.isChecked():
            return "always"
        return DEFAULT_RUN_WHEN

    def idle_seconds(self) -> int:
        data = self._idle_combo.currentData()
        return normalize_idle_seconds(
            data if data is not None else DEFAULT_IDLE_SECONDS
        )

    def low_priority(self) -> bool:
        return self._low_priority_cb.isChecked()

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
