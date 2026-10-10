"""Main application window for the Audio Recording Manager GUI."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QFont, QKeySequence
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QStatusBar,
    QTableView,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QCheckBox,
)

# Ensure repository root is on sys.path
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gui.audio_player import AudioPlayer
from scripts.gui.dialogs import ScanConfigDialog, TranscribeConfigDialog
from scripts.gui.preferences import load_prefs, save_prefs
from scripts.gui.scan_worker import ScanWorker
from scripts.gui.table_model import (
    RecordingFilterProxy,
    RecordingTableModel,
    load_format_csv,
    merge_format_csvs,
)
from scripts.gui.transcript_format import (
    TranscriptDoc,
    has_any_transcript,
    load_transcript,
    render_html,
    render_plain,
)
from scripts.gui.transcript_viewer import TranscriptViewerDialog
from scripts.gui.transcribe_worker import TranscribeWorker

from scripts.iOSWhisperAppHelpers.extract_m4a_creation_times import (
    move_duplicates_to_dup_dir,
)


def _format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS for the playback scrubber."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class MainWindow(QMainWindow):
    """Central window for the Audio Recording Manager."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Audio Recording Manager")
        self.setMinimumSize(1000, 600)
        self.resize(1400, 850)

        # -- State ---------------------------------------------------------------
        self._model = RecordingTableModel(self)
        self._proxy = RecordingFilterProxy(self)
        self._proxy.setSourceModel(self._model)
        self._player = AudioPlayer(self)
        self._scan_worker: Optional[ScanWorker] = None
        self._transcribe_worker: Optional[TranscribeWorker] = None
        # Track loaded CSVs: format_id -> csv_path
        self._loaded_csvs: Dict[str, Path] = {}
        self._prefs = load_prefs()
        self._playing_row: int = -1
        self._current_preview_row_data: Optional[pd.Series] = None
        self._current_preview_doc: Optional[TranscriptDoc] = None

        # -- Build UI ------------------------------------------------------------
        self._build_toolbar()
        self._build_table()
        self._build_preview_dock()
        self._build_playback_dock()
        self._build_log_dock()
        self._build_status_bar()
        self._connect_signals()
        self._restore_prefs()


    # ── Toolbar ──────────────────────────────────────────────────────────────

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.addToolBar(toolbar)

        # Scan button
        scan_btn = QPushButton("⟳ Scan Audio Dir")
        scan_btn.setObjectName("accentButton")
        scan_btn.clicked.connect(self._on_scan_clicked)
        toolbar.addWidget(scan_btn)

        # Load CSV button
        load_btn = QPushButton("📂 Load CSV")
        load_btn.clicked.connect(self._on_load_csv_clicked)
        toolbar.addWidget(load_btn)

        toolbar.addSeparator()

        # Transcribe button
        self._transcribe_btn = QPushButton("🎙 Transcribe Selected")
        self._transcribe_btn.clicked.connect(self._on_transcribe_selected)
        toolbar.addWidget(self._transcribe_btn)

        self._transcribe_all_btn = QPushButton("⏩ Transcribe All Pending")
        self._transcribe_all_btn.clicked.connect(self._on_transcribe_all_pending)
        toolbar.addWidget(self._transcribe_all_btn)

        toolbar.addSeparator()

        # Format filter
        toolbar.addWidget(QLabel("  Format: "))
        self._format_combo = QComboBox()
        self._format_combo.addItem("All", "")
        self._format_combo.setMinimumWidth(140)
        self._format_combo.currentIndexChanged.connect(self._on_format_filter_changed)
        toolbar.addWidget(self._format_combo)

        # Status filter
        toolbar.addWidget(QLabel("  Status: "))
        self._status_combo = QComboBox()
        self._status_combo.addItem("All", "")
        self._status_combo.addItem("✓ Transcribed", "transcribed")
        self._status_combo.addItem("⏳ Pending", "pending")
        self._status_combo.addItem("✕ Duplicate", "duplicate")
        self._status_combo.currentIndexChanged.connect(self._on_status_filter_changed)
        toolbar.addWidget(self._status_combo)

        # Hide duplicates checkbox
        toolbar.addWidget(QLabel("  "))
        self._hide_dups_cb = QCheckBox("Hide Duplicates")
        self._hide_dups_cb.toggled.connect(self._on_hide_dups_toggled)
        toolbar.addWidget(self._hide_dups_cb)

        toolbar.addSeparator()

        # Search filter
        toolbar.addWidget(QLabel("  🔍 "))
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Filter by name, title, path…")
        self._search_edit.setMinimumWidth(200)
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.textChanged.connect(self._on_text_filter_changed)
        toolbar.addWidget(self._search_edit)


    # ── Table ────────────────────────────────────────────────────────────────

    def _build_table(self) -> None:
        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setSortingEnabled(True)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_table_context_menu)
        self._table.doubleClicked.connect(self._on_table_double_click)

        # Column widths
        header = self._table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
            header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            header.setDefaultSectionSize(120)
            # Status column narrow
            header.resizeSection(0, 36)

        self.setCentralWidget(self._table)

        # Selection change listener for transcript preview
        sel_model = self._table.selectionModel()
        if sel_model is not None:
            sel_model.selectionChanged.connect(self._on_selection_changed)
            sel_model.currentChanged.connect(self._on_selection_changed)


    # ── Transcript Preview Dock (Right Side) ─────────────────────────────────

    def _build_preview_dock(self) -> None:
        dock = QDockWidget("Transcript Preview", self)
        dock.setObjectName("transcriptPreviewDock")
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
        )

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        # Header with filename and actions
        header_layout = QHBoxLayout()
        self._preview_title = QLabel("No recording selected")
        self._preview_title.setStyleSheet("font-weight: 600; font-size: 13px;")
        self._preview_title.setWordWrap(True)
        header_layout.addWidget(self._preview_title, stretch=1)

        self._copy_transcript_btn = QPushButton("📋 Copy")
        self._copy_transcript_btn.setToolTip("Copy transcript text to clipboard")
        self._copy_transcript_btn.setEnabled(False)
        self._copy_transcript_btn.clicked.connect(self._on_copy_preview_clicked)
        header_layout.addWidget(self._copy_transcript_btn)

        self._view_full_btn = QPushButton("🔍 Full View")
        self._view_full_btn.setToolTip("Open in full tabbed transcript viewer")
        self._view_full_btn.setEnabled(False)
        self._view_full_btn.clicked.connect(self._on_open_full_viewer_clicked)
        header_layout.addWidget(self._view_full_btn)

        layout.addLayout(header_layout)

        # Transcript viewer editor (rich formatted with timestamps)
        self._preview_text = QTextEdit()
        self._preview_text.setReadOnly(True)
        self._preview_text.setPlaceholderText("Select a recording from the table to view its transcript…")
        layout.addWidget(self._preview_text, stretch=1)

        dock.setWidget(container)
        dock.setMinimumWidth(320)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._preview_dock = dock


    # ── Playback Dock ────────────────────────────────────────────────────────

    def _build_playback_dock(self) -> None:
        dock = QDockWidget("Playback", self)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
        )

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        # Transport controls row
        transport = QHBoxLayout()

        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedWidth(40)
        self._play_btn.clicked.connect(self._on_play_pause)
        transport.addWidget(self._play_btn)

        self._stop_btn = QPushButton("■")
        self._stop_btn.setFixedWidth(40)
        self._stop_btn.clicked.connect(self._player.stop)
        transport.addWidget(self._stop_btn)

        self._position_label = QLabel("00:00")
        self._position_label.setFixedWidth(50)
        transport.addWidget(self._position_label)

        self._seek_slider = QSlider(Qt.Orientation.Horizontal)
        self._seek_slider.setRange(0, 0)
        self._seek_slider.sliderPressed.connect(self._on_seek_pressed)
        self._seek_slider.sliderReleased.connect(self._on_seek_released)
        transport.addWidget(self._seek_slider, stretch=1)

        self._duration_label = QLabel("00:00")
        self._duration_label.setFixedWidth(50)
        transport.addWidget(self._duration_label)

        transport.addWidget(QLabel("🔊"))
        self._volume_slider = QSlider(Qt.Orientation.Horizontal)
        self._volume_slider.setRange(0, 100)
        self._volume_slider.setValue(100)
        self._volume_slider.setFixedWidth(80)
        self._volume_slider.valueChanged.connect(
            lambda v: self._player.set_volume(v / 100.0)
        )
        transport.addWidget(self._volume_slider)

        layout.addLayout(transport)

        # Now-playing label
        self._now_playing = QLabel("")
        self._now_playing.setObjectName("dimLabel")
        layout.addWidget(self._now_playing)

        dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

        # Seek state
        self._is_seeking = False


    # ── Log Dock ─────────────────────────────────────────────────────────────

    def _build_log_dock(self) -> None:
        dock = QDockWidget("Transcription Log", self)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        self._log_text = QPlainTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(180)
        dock.setWidget(self._log_text)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        dock.hide()  # Hidden by default, shown during transcription
        self._log_dock = dock


    # ── Status Bar ───────────────────────────────────────────────────────────

    def _build_status_bar(self) -> None:
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        self._status_label = QLabel()
        status_bar.addPermanentWidget(self._status_label)


    # ── Signal Connections ───────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self._model.data_changed_signal.connect(self._update_status_bar)
        self._player.position_changed.connect(self._on_position_changed)
        self._player.duration_changed.connect(self._on_duration_changed)
        self._player.state_changed.connect(self._on_player_state_changed)
        self._player.error.connect(self._on_player_error)


    # ── Preferences ──────────────────────────────────────────────────────────

    def _restore_prefs(self) -> None:
        """Apply persisted paths: reload last filelists into the table."""
        loaded = self._prefs.get("loaded_csvs") or {}
        self._loaded_csvs = dict(loaded)
        if self._loaded_csvs:
            self._reload_merged_data()
            names = ", ".join(p.name for p in self._loaded_csvs.values())
            self._status_label.setText(f"Restored {names}")
        else:
            self._status_label.setText(
                "Ready — use Scan Audio Dir or Load CSV to get started"
            )


    def _save_prefs(self) -> None:
        """Persist current path/filelist preferences."""
        self._prefs["loaded_csvs"] = dict(self._loaded_csvs)
        save_prefs(self._prefs)


    # ── Toolbar Actions ──────────────────────────────────────────────────────

    def _on_scan_clicked(self) -> None:
        dlg = ScanConfigDialog(
            self,
            initial_audio_dir=self._prefs.get("scan_audio_dir", ""),
            initial_format_id=self._prefs.get("scan_format_id", ""),
            initial_timezone=self._prefs.get("scan_timezone") or None,
            initial_csv_path=self._prefs.get("scan_csv_path", ""),
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        audio_dir = dlg.audio_dir()
        fmt = dlg.selected_format()
        tz = dlg.timezone()
        csv_path = dlg.csv_path()

        if not audio_dir.is_dir():
            QMessageBox.warning(
                self, "Invalid Directory",
                f"The selected directory does not exist:\n{audio_dir}",
            )
            return

        if fmt is None:
            QMessageBox.warning(self, "No Format", "Could not determine format.")
            return

        self._prefs["scan_audio_dir"] = str(audio_dir)
        self._prefs["scan_format_id"] = dlg.format_id()
        self._prefs["scan_timezone"] = tz
        self._prefs["scan_csv_path"] = str(csv_path)
        self._save_prefs()

        self._status_label.setText(f"Scanning {audio_dir.name}…")
        self._scan_worker = ScanWorker(audio_dir, fmt, tz, csv_path, parent=self)
        self._scan_worker.progress.connect(
            lambda msg: self._status_label.setText(msg)
        )
        self._scan_worker.finished.connect(self._on_scan_finished)
        self._scan_worker.error.connect(self._on_scan_error)
        self._scan_worker.start()


    def _on_scan_finished(self, format_id: str, csv_path: object) -> None:
        csv_path = Path(str(csv_path))
        self._loaded_csvs[format_id] = csv_path
        self._prefs["scan_csv_path"] = str(csv_path)
        self._save_prefs()
        self._reload_merged_data()
        self._status_label.setText(f"Scan complete — loaded {csv_path.name}")


    def _on_scan_error(self, msg: str) -> None:
        self._status_label.setText(f"Scan error: {msg}")
        QMessageBox.critical(self, "Scan Error", msg)


    def _on_load_csv_clicked(self) -> None:
        start_dir = self._prefs.get("browse_load_csv_dir", "") or ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Filelist CSV",
            start_dir,
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return

        csv_path = Path(path)
        # Try to infer format_id from the CSV filename or content
        format_id = self._infer_format_id(csv_path)

        self._loaded_csvs[format_id] = csv_path
        self._prefs["browse_load_csv_dir"] = str(csv_path.parent)
        self._save_prefs()
        self._reload_merged_data()
        self._status_label.setText(f"Loaded {csv_path.name} as {format_id}")


    def _infer_format_id(self, csv_path: Path) -> str:
        """Best-effort format_id inference from CSV path/content."""
        name = csv_path.name.lower()
        if "voicememos" in name or "voice_memos" in name:
            return "voice_memos"
        if "jpr" in name or "just_press" in name:
            return "just_press_record"
        if "debut" in name:
            return "debut"
        if "rec_continuous" in name:
            return "rec_continuous"
        # Default to ios_whisper_app for WhisperApp CSVs
        return "ios_whisper_app"


    def _reload_merged_data(self) -> None:
        """Re-merge all loaded CSVs and update the table model."""
        df = merge_format_csvs(self._loaded_csvs)
        self._model.set_dataframe(df)

        # Update format filter combo
        current_format = self._format_combo.currentData()
        self._format_combo.blockSignals(True)
        self._format_combo.clear()
        self._format_combo.addItem("All", "")
        for fid in self._model.format_ids():
            self._format_combo.addItem(fid, fid)
        ## END for fid in self._model.format_ids()...
        # Restore selection if still valid
        idx = self._format_combo.findData(current_format)
        if idx >= 0:
            self._format_combo.setCurrentIndex(idx)
        self._format_combo.blockSignals(False)

        self._auto_size_columns()
        QTimer.singleShot(0, self._auto_size_columns)


    def _auto_size_columns(self) -> None:
        """Auto-size table columns to fit both header names and cell contents."""
        self._table.resizeColumnsToContents()
        header = self._table.horizontalHeader()
        if header is not None:
            if header.sectionSize(0) < 36:
                header.resizeSection(0, 36)
            for col in range(1, self._model.columnCount()):
                new_width = max(header.sectionSize(col) + 20, 60)
                header.resizeSection(col, new_width)
            ## END for col in range(1, self._model.columnCount())...



    # -- Transcription ---------------------------------------------------------

    def _get_selected_source_rows(self) -> List[int]:
        """Get source model row indices for the current table selection."""
        rows = set()
        for idx in self._table.selectionModel().selectedRows():
            source_idx = self._proxy.mapToSource(idx)
            rows.add(source_idx.row())
        ## END for idx in selectedRows...
        return sorted(rows)


    def _on_transcribe_selected(self) -> None:
        source_rows = self._get_selected_source_rows()
        if not source_rows:
            QMessageBox.information(
                self, "No Selection",
                "Select one or more recordings to transcribe.",
            )
            return
        self._start_transcription(source_rows)


    def _on_transcribe_all_pending(self) -> None:
        df = self._model.get_dataframe()
        if df.empty:
            QMessageBox.information(self, "No Data", "Load recordings first.")
            return

        # Find pending rows (not transcribed, not duplicate)
        pending_mask = df["status"] == "pending"
        pending_rows = list(df.index[pending_mask])
        if not pending_rows:
            QMessageBox.information(
                self, "Nothing Pending",
                "All recordings are already transcribed or marked as duplicates.",
            )
            return

        reply = QMessageBox.question(
            self, "Transcribe All Pending",
            f"Transcribe {len(pending_rows)} pending recording(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._start_transcription(pending_rows)


    def _start_transcription(self, source_rows: List[int]) -> None:
        """Launch the TranscribeWorker for the given source model rows."""
        if self._transcribe_worker is not None and self._transcribe_worker.isRunning():
            QMessageBox.warning(
                self, "Already Running",
                "A transcription is already in progress.",
            )
            return

        df = self._model.get_dataframe()
        selected_df = df.loc[source_rows].copy()

        # Determine output dir from the first row's source CSV
        first_csv = selected_df.iloc[0].get("source_csv", "")
        if pd.notna(first_csv) and str(first_csv).strip():
            csv_path = Path(str(first_csv).strip())
            if csv_path.parent.name.lower() == "filelists":
                default_out = csv_path.parent.parent / "transcriptions"
            else:
                default_out = csv_path.parent / "transcriptions"
        else:
            saved_out = self._prefs.get("transcribe_output_dir", "") or ""
            default_out = Path(saved_out) if saved_out else Path("transcriptions")

        # Show config dialog
        dlg = TranscribeConfigDialog(str(default_out), parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        output_dir = dlg.output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._prefs["transcribe_output_dir"] = str(output_dir)
        self._save_prefs()

        # Show log dock
        self._log_dock.show()
        self._log_text.clear()
        self._log_text.appendPlainText(
            f"Starting transcription of {len(selected_df)} file(s)…\n"
        )

        self._transcribe_worker = TranscribeWorker(
            rows_df=selected_df,
            output_dir=output_dir,
            backend=dlg.backend(),
            model_name=dlg.model_name(),
            crisper_mode=dlg.crisper_mode(),
            parent=self,
        )
        self._transcribe_worker.output_line.connect(self._on_transcribe_output)
        self._transcribe_worker.file_completed.connect(self._on_file_transcribed)
        self._transcribe_worker.all_done.connect(self._on_transcription_done)
        self._transcribe_worker.error.connect(self._on_transcription_error)
        self._transcribe_worker.start()

        self._transcribe_btn.setEnabled(False)
        self._transcribe_all_btn.setEnabled(False)
        self._status_label.setText("Transcribing…")


    def _on_transcribe_output(self, line: str) -> None:
        self._log_text.appendPlainText(line)
        # Auto-scroll
        sb = self._log_text.verticalScrollBar()
        if sb is not None:
            sb.setValue(sb.maximum())


    def _on_file_transcribed(self, df_index: int, transcript_cols: dict) -> None:
        self._model.update_row(df_index, transcript_cols)
        # Also update the backing CSV
        self._save_affected_csv(df_index)
        # Refresh preview if this row is currently selected
        selection = self._table.selectionModel()
        if selection is not None and selection.hasSelection():
            proxy_index = selection.currentIndex()
            if proxy_index.isValid():
                source_index = self._proxy.mapToSource(proxy_index)
                if source_index.isValid() and source_index.row() == df_index:
                    row_data = self._model.get_row_data(source_index.row())
                    if row_data is not None:
                        self._update_transcript_preview(row_data)
                    ## END if row_data is not None....
                ## END if source_index matches transcribed row....
            ## END if proxy_index.isValid()....
        ## END if selection hasSelection....



    def _on_transcription_done(self, completed: int, failed: int) -> None:
        self._transcribe_btn.setEnabled(True)
        self._transcribe_all_btn.setEnabled(True)
        self._status_label.setText(
            f"Transcription complete: {completed} succeeded, {failed} failed"
        )
        self._log_text.appendPlainText(
            f"\n✓ Done — {completed} completed, {failed} failed"
        )
        self._update_status_bar()
        self._auto_size_columns()


    def _on_transcription_error(self, msg: str) -> None:
        self._transcribe_btn.setEnabled(True)
        self._transcribe_all_btn.setEnabled(True)
        self._status_label.setText(f"Transcription error: {msg}")
        self._log_text.appendPlainText(f"\n✗ Error: {msg}")
        QMessageBox.critical(self, "Transcription Error", msg)


    def _save_affected_csv(self, df_index: int) -> None:
        """Persist changes to the source CSV for a given row."""
        df = self._model.get_dataframe()
        if df_index not in df.index:
            return
        source_csv = df.at[df_index, "source_csv"] if "source_csv" in df.columns else None
        if pd.isna(source_csv) or not str(source_csv).strip():
            return

        csv_path = Path(str(source_csv).strip())
        # Save only the rows belonging to this CSV
        mask = df["source_csv"] == str(csv_path)
        subset = df[mask].drop(columns=["status", "format_id", "source_csv"], errors="ignore")
        try:
            subset.to_csv(csv_path, index=False, encoding="utf-8")
        except OSError:
            pass


    # -- Filter handlers -------------------------------------------------------

    def _on_format_filter_changed(self) -> None:
        fid = self._format_combo.currentData() or ""
        self._proxy.set_format_filter(fid)


    def _on_status_filter_changed(self) -> None:
        status = self._status_combo.currentData() or ""
        self._proxy.set_status_filter(status)


    def _on_hide_dups_toggled(self, checked: bool) -> None:
        self._proxy.set_hide_duplicates(checked)


    def _on_text_filter_changed(self, text: str) -> None:
        self._proxy.set_text_filter(text)


    # -- Table interactions ----------------------------------------------------

    def _on_table_double_click(self, proxy_index) -> None:
        """Double-click → play audio."""
        source_index = self._proxy.mapToSource(proxy_index)
        row_data = self._model.get_row_data(source_index.row())
        if row_data is None:
            return
        self._play_row(row_data, source_index.row())


    def _on_table_context_menu(self, pos) -> None:
        proxy_index = self._table.indexAt(pos)
        if not proxy_index.isValid():
            return

        source_index = self._proxy.mapToSource(proxy_index)
        row_data = self._model.get_row_data(source_index.row())
        if row_data is None:
            return

        menu = QMenu(self)

        play_action = menu.addAction("▶ Play Audio")
        play_action.triggered.connect(
            lambda: self._play_row(row_data, source_index.row())
        )

        # View transcript (only if any transcript_* path is set)
        view_action = menu.addAction("📄 View Transcript")
        view_action.setEnabled(has_any_transcript(row_data))
        view_action.triggered.connect(lambda: self._view_transcript(row_data))

        menu.addSeparator()

        # Transcribe this file
        transcribe_action = menu.addAction("🎙 Transcribe This File")
        transcribe_action.triggered.connect(
            lambda: self._start_transcription([source_index.row()])
        )

        menu.addSeparator()

        # Open folder in Explorer
        reveal_action = menu.addAction("📁 Open in Explorer")
        reveal_action.triggered.connect(lambda: self._reveal_in_explorer(row_data))

        menu.addSeparator()

        # Duplicate management
        is_dup = row_data.get("is_duplicate")
        is_dup_bool = pd.notna(is_dup) and (
            is_dup is True or str(is_dup).strip().lower() in ("true", "1", "yes")
        )
        if is_dup_bool:
            move_dup_action = menu.addAction("🗑 Move Duplicates to _DUP/")
            move_dup_action.triggered.connect(self._on_move_duplicates)

        menu.exec(self._table.viewport().mapToGlobal(pos))


    def _play_row(self, row_data: pd.Series, visual_row: int) -> None:
        """Load and play audio for a row."""
        full_path = row_data.get("full_path", "")
        if pd.isna(full_path) or not str(full_path).strip():
            return

        audio_path = Path(str(full_path).strip())
        # Handle WSL paths
        if not audio_path.is_file() and str(audio_path).startswith("/mnt/"):
            import re
            m = re.match(r"^/mnt/([a-zA-Z])/(.*)$", str(audio_path))
            if m:
                audio_path = Path(f"{m.group(1).upper()}:/{m.group(2)}")
        ## END if WSL path...

        if not audio_path.is_file():
            QMessageBox.warning(
                self, "File Not Found",
                f"Audio file not found:\n{audio_path}",
            )
            return

        name = row_data.get("name", audio_path.name)
        self._now_playing.setText(f"Now playing: {name}")
        self._playing_row = visual_row

        # Update transcript preview for the playing row
        self._update_transcript_preview(row_data)

        self._player.load(audio_path)
        self._player.play()


    def _on_selection_changed(self, *args) -> None:
        """Update transcript preview when table row selection changes."""
        sel_model = self._table.selectionModel()
        if sel_model is None:
            self._update_transcript_preview(None)
            return

        selected_rows = sel_model.selectedRows()
        if not selected_rows:
            self._update_transcript_preview(None)
            return

        proxy_idx = selected_rows[0]
        source_idx = self._proxy.mapToSource(proxy_idx)
        if not source_idx.isValid():
            self._update_transcript_preview(None)
            return

        row_data = self._model.get_row_data(source_idx.row())
        self._update_transcript_preview(row_data)


    def _update_transcript_preview(self, row_data: Optional[pd.Series]) -> None:
        """Display the transcript for the currently selected row with timestamps in clean readable HTML."""
        if row_data is None:
            self._preview_title.setText("No recording selected")
            self._preview_text.clear()
            self._preview_text.setPlaceholderText(
                "Select a recording from the table to view its transcript…"
            )
            self._current_preview_row_data = None
            self._current_preview_doc = None
            self._copy_transcript_btn.setEnabled(False)
            self._view_full_btn.setEnabled(False)
            return

        self._current_preview_row_data = row_data
        name = row_data.get("name", "Unknown")
        self._preview_title.setText(str(name))

        doc = load_transcript(row_data)
        self._current_preview_doc = doc
        if doc is not None:
            self._preview_text.setHtml(render_html(doc, include_words=False))
            self._copy_transcript_btn.setEnabled(True)
            self._view_full_btn.setEnabled(True)
            return
        ## END if doc is not None....

        self._preview_text.clear()
        self._preview_text.setPlaceholderText("No transcript available for this recording.")
        self._copy_transcript_btn.setEnabled(False)
        self._view_full_btn.setEnabled(has_any_transcript(row_data))


    def _on_copy_preview_clicked(self) -> None:
        """Copy the current transcript preview text to the clipboard."""
        if self._current_preview_doc is not None:
            text = render_plain(self._current_preview_doc, include_words=False)
        else:
            text = self._preview_text.toPlainText()
        ## END if self._current_preview_doc is not None....

        if text:
            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(text)
                self._status_label.setText("Transcript copied to clipboard")
            ## END if clipboard is not None....
        ## END if text....



    def _on_open_full_viewer_clicked(self) -> None:
        """Open the full tabbed transcript viewer dialog for the previewed recording."""
        if self._current_preview_row_data is not None:
            self._view_transcript(self._current_preview_row_data)


    def _view_transcript(self, row_data: pd.Series) -> None:
        dlg = TranscriptViewerDialog(row_data, parent=self)
        dlg.exec()


    def _reveal_in_explorer(self, row_data: pd.Series) -> None:
        full_path = row_data.get("full_path", "")
        if pd.isna(full_path) or not str(full_path).strip():
            return

        path = Path(str(full_path).strip())
        # Handle WSL paths
        if str(path).startswith("/mnt/"):
            import re
            m = re.match(r"^/mnt/([a-zA-Z])/(.*)$", str(path))
            if m:
                path = Path(f"{m.group(1).upper()}:/{m.group(2)}")

        if path.is_file():
            # Open Explorer with the file selected
            subprocess.Popen(["explorer", "/select,", str(path)])
        elif path.parent.is_dir():
            subprocess.Popen(["explorer", str(path.parent)])


    def _on_move_duplicates(self) -> None:
        """Move duplicate rows to _DUP/ for all loaded CSVs."""
        reply = QMessageBox.question(
            self, "Move Duplicates",
            "Move all files marked as duplicates to _DUP/ directories?\n\n"
            "This will physically move files on disk.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        df = self._model.get_dataframe()

        # Group by source_csv and process each
        for csv_str, group_df in df.groupby("source_csv"):
            csv_path = Path(str(csv_str))
            if not csv_path.is_file():
                continue

            # Determine audio_dir from the format or the CSV location
            source_df = pd.read_csv(csv_path)

            # Infer audio_dir from full_path of first row
            first_path = group_df.iloc[0].get("full_path", "")
            if pd.notna(first_path) and str(first_path).strip():
                audio_dir = Path(str(first_path).strip()).parent
                # Handle WSL path
                if str(audio_dir).startswith("/mnt/"):
                    import re
                    m = re.match(r"^/mnt/([a-zA-Z])/(.*)$", str(audio_dir))
                    if m:
                        audio_dir = Path(f"{m.group(1).upper()}:/{m.group(2)}")
            else:
                continue

            moved, skipped, fail = move_duplicates_to_dup_dir(source_df, audio_dir)
            # Save updated CSV
            source_df.to_csv(csv_path, index=False, encoding="utf-8")

            self._status_label.setText(
                f"Duplicates: {moved} moved, {skipped} skipped, {fail} failed"
            )
        ## END for csv_str, group_df in df.groupby("source_csv")...

        # Reload data
        self._reload_merged_data()


    # -- Playback controls -----------------------------------------------------

    def _on_play_pause(self) -> None:
        if self._player.state == "playing":
            self._player.pause()
        elif self._player.state == "paused":
            self._player.play()
        elif self._player.state == "stopped" and self._player.current_path:
            self._player.play()


    def _on_position_changed(self, seconds: float) -> None:
        self._position_label.setText(_format_time(seconds))
        if not self._is_seeking:
            self._seek_slider.blockSignals(True)
            self._seek_slider.setValue(int(seconds * 10))
            self._seek_slider.blockSignals(False)


    def _on_duration_changed(self, seconds: float) -> None:
        self._duration_label.setText(_format_time(seconds))
        self._seek_slider.setRange(0, int(seconds * 10))


    def _on_player_state_changed(self, state: str) -> None:
        if state == "playing":
            self._play_btn.setText("⏸")
        else:
            self._play_btn.setText("▶")

        if state == "stopped":
            self._now_playing.setText("")
            self._playing_row = -1


    def _on_player_error(self, msg: str) -> None:
        self._status_label.setText(f"Playback error: {msg}")


    def _on_seek_pressed(self) -> None:
        self._is_seeking = True


    def _on_seek_released(self) -> None:
        self._is_seeking = False
        seconds = self._seek_slider.value() / 10.0
        self._player.seek(seconds)


    # -- Status bar ------------------------------------------------------------

    def _update_status_bar(self) -> None:
        counts = self._model.status_counts()
        total = counts.get("total", 0)
        transcribed = counts.get("transcribed", 0)
        pending = counts.get("pending", 0)
        duplicate = counts.get("duplicate", 0)
        self._status_label.setText(
            f"{total} recordings  │  ✓ {transcribed} transcribed  │  "
            f"⏳ {pending} pending  │  ✕ {duplicate} duplicates"
        )


    # -- Cleanup ---------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._save_prefs()
        self._player.cleanup()
        if self._scan_worker is not None and self._scan_worker.isRunning():
            self._scan_worker.quit()
            self._scan_worker.wait(3000)
        if self._transcribe_worker is not None and self._transcribe_worker.isRunning():
            self._transcribe_worker.request_cancel()
            self._transcribe_worker.wait(5000)
        super().closeEvent(event)


def main() -> None:
    """Launch the Audio Recording Manager GUI application."""
    from PyQt6.QtWidgets import QApplication
    from scripts.gui.style import apply_dark_theme

    app = QApplication(sys.argv)
    app.setOrganizationName("whisper-timestamped")
    app.setApplicationName("AudioRecordingManager")
    apply_dark_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
