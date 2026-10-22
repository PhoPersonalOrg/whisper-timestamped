"""Main application window for the Audio Recording Manager GUI."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

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
    QSystemTrayIcon,
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
from scripts.gui.autostart import is_start_with_windows, set_start_with_windows
from scripts.gui.dialogs import (
    ScanConfigDialog,
    TranscribeConfigDialog,
    WatchFoldersDialog,
)
from scripts.gui.folder_watch import FolderWatchWorker
from scripts.gui.discover_worker import DiscoverTranscriptsWorker
from scripts.gui.preferences import load_prefs, save_prefs
from scripts.gui.scan_worker import ScanWorker
from scripts.gui.status_delegate import NameProgressDelegate
from scripts.gui.table_model import (
    DISPLAY_COLUMNS,
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
from scripts.gui.tray import TrayController
from scripts.transcript_outputs import (
    csv_derived_output_dir,
    find_extant_output_files,
    has_transcript_path_values,
    transcript_status_columns,
)
from whisper_timestamped.recording_formats.completeness import (
    output_dir_for_format,
    upsert_filelist_csv_row,
)
from whisper_timestamped.recording_formats.registry import get_format

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
        self._discover_worker: Optional[DiscoverTranscriptsWorker] = None
        # Track loaded CSVs: format_id -> csv_path
        self._loaded_csvs: Dict[str, Path] = {}
        self._prefs = load_prefs()
        self._playing_row: int = -1
        self._current_preview_row_data: Optional[pd.Series] = None
        self._current_preview_doc: Optional[TranscriptDoc] = None
        self._transcribe_current_name: str = ""
        self._transcribe_current_progress: Optional[float] = None
        self._transcribe_batch_n: int = 0
        self._transcribe_current_i: int = 0
        self._force_quit: bool = False
        self._tray: Optional[TrayController] = None
        self._watch_worker: Optional[FolderWatchWorker] = None
        self._auto_queue: List[int] = []
        self._auto_queue_seen: Set[int] = set()
        self._transcribe_origin: str = ""

        # -- Build UI ------------------------------------------------------------
        self._build_toolbar()
        self._build_table()
        self._build_preview_dock()
        self._build_playback_dock()
        self._build_log_dock()
        self._build_status_bar()
        self._connect_signals()
        self._restore_prefs()
        self._setup_tray()
        self._start_folder_watch()

        # Keep the now-relative "Ago" column fresh while the window stays open.
        self._ago_refresh_timer = QTimer(self)
        self._ago_refresh_timer.setInterval(60_000)
        self._ago_refresh_timer.timeout.connect(self._model.refresh_ago_column)
        self._ago_refresh_timer.start()


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

        watch_btn = QPushButton("👁 Watch Folders")
        watch_btn.clicked.connect(self._on_watch_folders_clicked)
        toolbar.addWidget(watch_btn)

        toolbar.addSeparator()

        # Transcribe button
        self._transcribe_btn = QPushButton("🎙 Transcribe Selected")
        self._transcribe_btn.clicked.connect(self._on_transcribe_selected)
        toolbar.addWidget(self._transcribe_btn)

        self._transcribe_all_btn = QPushButton("⏩ Transcribe All Pending")
        self._transcribe_all_btn.clicked.connect(self._on_transcribe_all_pending)
        toolbar.addWidget(self._transcribe_all_btn)

        self._cancel_transcribe_btn = QPushButton("Cancel")
        self._cancel_transcribe_btn.setEnabled(False)
        self._cancel_transcribe_btn.clicked.connect(self._on_cancel_transcription)
        toolbar.addWidget(self._cancel_transcribe_btn)

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
        self._status_combo.addItem("⋯ Queued", "queued")
        self._status_combo.addItem("▶ Transcribing", "transcribing")
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
        self._sort_table_by_created_newest_first()
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

        try:
            name_col = DISPLAY_COLUMNS.index("name")
        except ValueError:
            name_col = 1
        self._table.setItemDelegateForColumn(name_col, NameProgressDelegate(self._table))
        self._table.clicked.connect(self._on_table_clicked)

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


    # ── Tray / folder watch ──────────────────────────────────────────────────

    def _setup_tray(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self._tray = None
            return
        from scripts.gui.style import app_icon

        self._tray = TrayController(
            app_icon(),
            self,
            on_open=self.show_from_tray,
            on_quit=self._quit_from_tray,
            on_watch_folders=self._on_watch_folders_clicked,
            on_auto_transcribe_toggled=self._on_auto_transcribe_toggled,
            on_start_with_windows_toggled=self._on_start_with_windows_toggled,
            auto_transcribe=bool(self._prefs.get("watch_enabled", True)),
            start_with_windows=is_start_with_windows(),
            show_autostart=sys.platform == "win32",
        )
        self._update_tray_tooltip()

    def show_from_tray(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _quit_from_tray(self) -> None:
        self._force_quit = True
        self.close()
        app = QApplication.instance()
        if app is not None:
            app.quit()
        ## END if app is not None....

    def _on_auto_transcribe_toggled(self, enabled: bool) -> None:
        self._prefs["watch_enabled"] = bool(enabled)
        self._save_prefs()
        if enabled:
            self._drain_auto_queue()
        ## END if enabled....

        self._update_tray_tooltip()

    def _on_start_with_windows_toggled(self, enabled: bool) -> None:
        ok = set_start_with_windows(bool(enabled))
        if not ok and enabled:
            QMessageBox.warning(
                self,
                "Start with Windows",
                "Could not create a Startup shortcut.",
            )
            if self._tray is not None:
                self._tray.set_start_with_windows(False)
            ## END if self._tray....

        ## END if not ok....

    def _on_watch_folders_clicked(self) -> None:
        dlg = WatchFoldersDialog(
            self._prefs.get("watch_folders") or [],
            auto_transcribe=bool(self._prefs.get("watch_enabled", True)),
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._prefs["watch_folders"] = dlg.folders()
        self._prefs["watch_enabled"] = dlg.auto_transcribe()
        self._save_prefs()
        if self._tray is not None:
            self._tray.set_auto_transcribe(bool(self._prefs["watch_enabled"]))
        ## END if self._tray....

        self._start_folder_watch()
        self._update_tray_tooltip()

    def _start_folder_watch(self) -> None:
        folders = self._prefs.get("watch_folders") or []
        if self._watch_worker is not None and self._watch_worker.isRunning():
            self._watch_worker.set_folders(folders)
            return
        self._watch_worker = FolderWatchWorker(folders, parent=self)
        self._watch_worker.file_ready.connect(self._on_watch_file_ready)
        self._watch_worker.status_message.connect(self._on_watch_status)
        self._watch_worker.error.connect(self._on_watch_error)
        self._watch_worker.start()

    def _on_watch_status(self, message: str) -> None:
        if self._transcribe_worker is None or not self._transcribe_worker.isRunning():
            self._status_label.setText(message)
        ## END if not transcribing....

        self._update_tray_tooltip(watch_status=message)

    def _on_watch_error(self, message: str) -> None:
        self._status_label.setText(f"Watch error: {message}")
        self._update_tray_tooltip()

    def _watch_root_for(self, format_id: str, path: Path) -> Path:
        for spec in self._prefs.get("watch_folders") or []:
            if str(spec.get("format_id") or "") != format_id:
                continue
            root = Path(str(spec.get("path") or ""))
            try:
                path.resolve().relative_to(root.resolve())
                return root
            except ValueError:
                continue
            ## END try relative_to....

        ## END for spec in watch_folders....

        return path.parent

    def _upsert_watched_recording(self, format_id: str, path: Path) -> Optional[int]:
        try:
            fmt = get_format(format_id)
        except KeyError:
            return None
        tz = self._prefs.get("scan_timezone") or "America/Los_Angeles"
        try:
            row = fmt.build_filelist_row(path, tz_name=tz)
        except OSError:
            return None
        watch_root = self._watch_root_for(format_id, path)
        if format_id in self._loaded_csvs:
            csv_path = self._loaded_csvs[format_id]
        else:
            csv_path = fmt.resolve_watch_filelist_csv(watch_root)
            self._loaded_csvs[format_id] = csv_path
            self._save_prefs()
        ## END if format_id in self._loaded_csvs....

        upsert_filelist_csv_row(csv_path, row, fmt.extra_filelist_columns)
        row["format_id"] = format_id
        row["source_csv"] = str(csv_path)
        return self._model.upsert_row(row)

    def _on_watch_file_ready(self, format_id: str, path_str: str) -> None:
        path = Path(path_str)
        df_index = self._upsert_watched_recording(format_id, path)
        if df_index is None:
            return
        self._enqueue_auto_transcribe(df_index)

    def _enqueue_auto_transcribe(self, df_index: int) -> None:
        if not self._prefs.get("watch_enabled", True):
            return
        df = self._model.get_dataframe()
        if df_index not in df.index:
            return
        row = df.loc[df_index]
        if has_transcript_path_values(row):
            return
        status = str(row.get("status") or "")
        if status == "duplicate":
            return
        job = self._model.job_state(df_index)
        if job and job.get("phase") in ("queued", "transcribing", "cancelling"):
            return
        format_id = str(row.get("format_id") or "")
        try:
            fmt = get_format(format_id)
        except KeyError:
            return
        full_path = Path(str(row.get("full_path") or ""))
        output_dir = output_dir_for_format(fmt, self._watch_root_for(format_id, full_path))
        name_raw = row.get("name")
        if name_raw is not None and pd.notna(name_raw) and str(name_raw).strip():
            base_name = Path(str(name_raw).strip()).stem
        else:
            base_name = full_path.stem
        ## END if name_raw....

        if find_extant_output_files(output_dir, base_name):
            return
        if df_index in self._auto_queue_seen:
            return
        self._auto_queue.append(int(df_index))
        self._auto_queue_seen.add(int(df_index))
        self._drain_auto_queue()

    def _output_dir_for_row(self, row: pd.Series) -> Path:
        format_id = str(row.get("format_id") or "")
        try:
            fmt = get_format(format_id)
        except KeyError:
            saved = self._prefs.get("transcribe_output_dir", "") or ""
            return Path(saved) if saved else Path("transcriptions")
        full_path = Path(str(row.get("full_path") or ""))
        return output_dir_for_format(fmt, self._watch_root_for(format_id, full_path))

    def _drain_auto_queue(self) -> None:
        if not self._prefs.get("watch_enabled", True):
            return
        if self._transcribe_worker is not None and self._transcribe_worker.isRunning():
            return
        if not self._auto_queue:
            return
        df = self._model.get_dataframe()
        pending: List[int] = []
        for df_index in self._auto_queue:
            if df_index in df.index:
                pending.append(df_index)
            ## END if df_index in df.index....

        ## END for df_index in self._auto_queue....

        self._auto_queue = []
        if not pending:
            return
        first_row = df.loc[pending[0]]
        first_out = self._output_dir_for_row(first_row)
        same: List[int] = []
        rest: List[int] = []
        for df_index in pending:
            if self._output_dir_for_row(df.loc[df_index]) == first_out:
                same.append(df_index)
            else:
                rest.append(df_index)
            ## END if same output dir....

        ## END for df_index in pending....

        self._auto_queue = rest
        self._start_transcription(same, prompt=False, output_dir=first_out)

    def _update_tray_tooltip(self, watch_status: Optional[str] = None) -> None:
        if self._tray is None:
            return
        folders = self._prefs.get("watch_folders") or []
        enabled_n = sum(1 for spec in folders if spec.get("enabled", True))
        auto = "on" if self._prefs.get("watch_enabled", True) else "paused"
        if self._transcribe_worker is not None and self._transcribe_worker.isRunning():
            n = self._transcribe_batch_n
            i = self._transcribe_current_i
            name = self._transcribe_current_name
            progress = self._transcribe_current_progress
            if name:
                if isinstance(progress, float):
                    text = (
                        f"Transcribing {i}/{n} — {name} "
                        f"{int(round(progress * 100))}%"
                    )
                else:
                    text = f"Transcribing {i}/{n} — {name}"
            else:
                text = f"Transcribing {i}/{n}"
            ## END if name....

        elif watch_status:
            text = f"{watch_status} · auto-transcribe {auto}"
        else:
            text = f"Watching {enabled_n} folder(s) · auto-transcribe {auto}"
        ## END if transcribing....

        self._tray.set_tooltip(f"Audio Recording Manager — {text}")


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
        if "rec_continuous_video_recorder" in name:
            return "rec_continuous_video_recorder"
        if "rec_continuous" in name:
            return "rec_continuous"
        if "debut" in name:
            return "debut"
        # Default to ios_whisper_app for WhisperApp CSVs
        return "ios_whisper_app"


    def _reload_merged_data(self) -> None:
        """Re-merge all loaded CSVs and update the table model."""
        df = merge_format_csvs(self._loaded_csvs)
        self._model.set_dataframe(df)
        self._sort_table_by_created_newest_first()

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
        self._start_transcript_discovery()


    def _discovery_jobs_from_model(self) -> List[dict]:
        """Snapshot table rows for background transcript discovery."""
        df = self._model.get_dataframe()
        if df.empty:
            return []
        cols = ["name", "full_path", "format_id", "source_csv"] + transcript_status_columns()
        jobs: List[dict] = []
        for idx, row in df.iterrows():
            job: dict = {"df_index": int(idx)}
            for col in cols:
                val = row.get(col, "") if col in df.columns else ""
                if pd.isna(val):
                    job[col] = ""
                else:
                    job[col] = str(val)
                ## END if pd.isna(val)....
            ## END for col in cols....

            jobs.append(job)
        ## END for idx, row in df.iterrows()....

        return jobs


    def _stop_transcript_discovery(self) -> None:
        worker = self._discover_worker
        if worker is None:
            return
        for signal, slot in (
            (worker.row_updated, self._on_transcript_discovered),
            (worker.progress, self._on_discovery_progress),
            (worker.finished, self._on_discovery_finished),
            (worker.error, self._on_discovery_error),
        ):
            try:
                signal.disconnect(slot)
            except TypeError:
                pass
            ## END try disconnect....
        ## END for signal, slot....

        worker.request_cancel()
        if worker.isRunning():
            worker.wait(1500)
        ## END if worker.isRunning()....

        self._discover_worker = None


    def _start_transcript_discovery(self) -> None:
        """Probe output folders for existing transcripts after a table load."""
        self._stop_transcript_discovery()
        if self._transcribe_worker is not None and self._transcribe_worker.isRunning():
            return

        jobs = self._discovery_jobs_from_model()
        if not jobs:
            return

        saved_out = self._prefs.get("transcribe_output_dir", "") or ""
        prefs_dir = Path(saved_out) if saved_out else None
        self._discover_worker = DiscoverTranscriptsWorker(
            jobs=jobs,
            prefs_output_dir=prefs_dir,
            parent=self,
        )
        self._discover_worker.row_updated.connect(self._on_transcript_discovered)
        self._discover_worker.progress.connect(self._on_discovery_progress)
        self._discover_worker.finished.connect(self._on_discovery_finished)
        self._discover_worker.error.connect(self._on_discovery_error)
        self._discover_worker.start()


    def _on_discovery_progress(self, msg: str) -> None:
        if self._transcribe_worker is not None and self._transcribe_worker.isRunning():
            return
        self._status_label.setText(msg)


    def _on_transcript_discovered(self, df_index: int, transcript_cols: dict) -> None:
        self._model.update_row(df_index, transcript_cols)
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
                ## END if source_index matches discovered row....
            ## END if proxy_index.isValid()....
        ## END if selection hasSelection....


    def _on_discovery_finished(self, found: int) -> None:
        if found:
            self._save_all_loaded_csvs()
        ## END if found....

        self._update_status_bar()
        self._auto_size_columns()
        if found:
            counts = self._model.status_counts()
            self._status_label.setText(
                f"Found {found} existing transcript(s)  │  "
                f"✓ {counts.get('transcribed', 0)} transcribed"
            )
        ## END if found....


    def _on_discovery_error(self, msg: str) -> None:
        self._status_label.setText(msg)


    def _sort_table_by_created_newest_first(self) -> None:
        """Default table order: newest recording dates first, empty dates last."""
        try:
            created_col = DISPLAY_COLUMNS.index("extracted_creation_time")
        except ValueError:
            return
        self._table.sortByColumn(created_col, Qt.SortOrder.DescendingOrder)


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


    def _visual_rows_to_df_indices(self, visual_rows: List[int]) -> List[int]:
        indices: List[int] = []
        for visual_row in visual_rows:
            df_index = self._model.df_index_for_row(visual_row)
            if df_index is not None:
                indices.append(int(df_index))
            ## END if df_index is not None....

        ## END for visual_row in visual_rows....

        return indices


    def _on_transcribe_selected(self) -> None:
        source_rows = self._get_selected_source_rows()
        if not source_rows:
            QMessageBox.information(
                self, "No Selection",
                "Select one or more recordings to transcribe.",
            )
            return
        self._start_transcription(self._visual_rows_to_df_indices(source_rows))


    def _on_transcribe_all_pending(self) -> None:
        df = self._model.get_dataframe()
        if df.empty:
            QMessageBox.information(self, "No Data", "Load recordings first.")
            return

        # Find pending rows (not transcribed, not duplicate)
        pending_mask = df["status"] == "pending"
        pending_rows = [int(i) for i in df.index[pending_mask]]
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


    def _start_transcription(
        self,
        source_rows: List[int],
        *,
        prompt: bool = True,
        output_dir: Optional[Path] = None,
    ) -> None:
        """Launch the TranscribeWorker for the given DataFrame indices."""
        if self._transcribe_worker is not None and self._transcribe_worker.isRunning():
            if prompt:
                QMessageBox.warning(
                    self, "Already Running",
                    "A transcription is already in progress.",
                )
            ## END if prompt....

            return

        if not source_rows:
            return

        self._stop_transcript_discovery()

        df = self._model.get_dataframe()
        selected_df = df.loc[source_rows].copy()

        # Determine output dir from the first row's source CSV
        first_csv = selected_df.iloc[0].get("source_csv", "")
        if pd.notna(first_csv) and str(first_csv).strip():
            default_out = csv_derived_output_dir(Path(str(first_csv).strip()))
        else:
            saved_out = self._prefs.get("transcribe_output_dir", "") or ""
            default_out = Path(saved_out) if saved_out else Path("transcriptions")

        if prompt:
            dlg = TranscribeConfigDialog(
                str(output_dir or default_out),
                parent=self,
                initial_backend=str(
                    self._prefs.get("transcribe_backend") or "crisperwhisper"
                ),
                initial_model=str(self._prefs.get("transcribe_model") or "turbo"),
                initial_crisper_mode=str(
                    self._prefs.get("transcribe_crisper_mode") or "verbatim"
                ),
            )
            if dlg.exec() != QDialog.DialogCode.Accepted:
                self._start_transcript_discovery()
                return
            output_dir = dlg.output_dir()
            backend = dlg.backend()
            model_name = dlg.model_name()
            crisper_mode = dlg.crisper_mode()
            self._prefs["transcribe_output_dir"] = str(output_dir)
            self._prefs["transcribe_backend"] = backend
            self._prefs["transcribe_model"] = model_name
            self._prefs["transcribe_crisper_mode"] = crisper_mode
            self._save_prefs()
            self._transcribe_origin = "manual"
        else:
            output_dir = Path(output_dir) if output_dir is not None else default_out
            backend = str(self._prefs.get("transcribe_backend") or "crisperwhisper")
            model_name = str(self._prefs.get("transcribe_model") or "turbo")
            crisper_mode = str(self._prefs.get("transcribe_crisper_mode") or "verbatim")
            self._transcribe_origin = "auto"
        ## END if prompt....

        output_dir.mkdir(parents=True, exist_ok=True)

        # Show log dock
        self._log_dock.show()
        self._log_text.clear()
        origin = "auto" if not prompt else "manual"
        self._log_text.appendPlainText(
            f"Starting {origin} transcription of {len(selected_df)} file(s)…\n"
        )

        self._transcribe_worker = TranscribeWorker(
            rows_df=selected_df,
            output_dir=output_dir,
            backend=backend,
            model_name=model_name,
            crisper_mode=crisper_mode,
            parent=self,
        )
        self._transcribe_worker.output_line.connect(self._on_transcribe_output)
        self._transcribe_worker.file_started.connect(self._on_file_transcribe_started)
        self._transcribe_worker.file_progress.connect(self._on_file_transcribe_progress)
        self._transcribe_worker.file_completed.connect(self._on_file_transcribed)
        self._transcribe_worker.file_skipped.connect(self._on_file_transcribe_skipped)
        self._transcribe_worker.file_failed.connect(self._on_file_transcribe_failed)
        self._transcribe_worker.all_done.connect(self._on_transcription_done)
        self._transcribe_worker.error.connect(self._on_transcription_error)

        self._transcribe_batch_n = len(source_rows)
        self._transcribe_current_i = 0
        self._transcribe_current_name = ""
        self._transcribe_current_progress = None
        self._model.set_job_states(source_rows, "queued")
        self._transcribe_worker.start()

        self._transcribe_btn.setEnabled(False)
        self._transcribe_all_btn.setEnabled(False)
        self._cancel_transcribe_btn.setEnabled(True)
        self._status_label.setText(f"Transcribing 0/{self._transcribe_batch_n}…")
        self._update_tray_tooltip()


    def _refresh_transcribe_status_bar(self) -> None:
        n = self._transcribe_batch_n
        i = self._transcribe_current_i
        name = self._transcribe_current_name
        progress = self._transcribe_current_progress
        if name:
            if isinstance(progress, float):
                self._status_label.setText(
                    f"Transcribing {i}/{n} — {name} {int(round(progress * 100))}%"
                )
            else:
                self._status_label.setText(f"Transcribing {i}/{n} — {name}")
        else:
            self._status_label.setText(f"Transcribing {i}/{n}…")
        self._update_tray_tooltip()


    def _on_file_transcribe_started(self, df_index: int, name: str, i: int, n: int) -> None:
        self._transcribe_current_i = i
        self._transcribe_batch_n = n
        self._transcribe_current_name = name
        self._transcribe_current_progress = None
        job = self._model.job_state(df_index)
        phase = "cancelling" if job and job.get("phase") == "cancelling" else "transcribing"
        self._model.set_job_state(
            df_index, phase, progress=None, position=i, total=n
        )
        self._refresh_transcribe_status_bar()


    def _on_file_transcribe_progress(self, df_index: int, fraction: float) -> None:
        self._transcribe_current_progress = fraction
        job = self._model.job_state(df_index)
        phase = "cancelling" if job and job.get("phase") == "cancelling" else "transcribing"
        self._model.set_job_state(df_index, phase, progress=fraction)
        self._refresh_transcribe_status_bar()


    def _on_file_transcribe_skipped(self, df_index: int, reason: str) -> None:
        had_job = self._model.job_state(df_index) is not None
        self._model.clear_job_state(df_index)
        if had_job:
            self._log_text.appendPlainText(f"Skipped ({reason})")


    def _on_file_transcribe_failed(self, df_index: int, message: str) -> None:
        self._model.set_job_state(df_index, "failed", message=message)
        self._log_text.appendPlainText(f"✗ Failed row {df_index}: {message}")


    def _cancel_remaining_transcription(self) -> None:
        """Finish the current file, skip the rest of the queue."""
        if self._transcribe_worker is None or not self._transcribe_worker.isRunning():
            return
        current = self._model.transcribing_df_index()
        for df_index in self._model.queued_df_indices():
            self._model.clear_job_state(df_index)
        ## END for df_index in queued....

        if current is not None:
            self._model.set_job_state(current, "cancelling")
        self._transcribe_worker.request_cancel()
        self._cancel_transcribe_btn.setEnabled(False)
        self._status_label.setText("Cancelling… finishing current file")


    def _remove_queued_rows(self, df_indices: List[int]) -> None:
        if self._transcribe_worker is None or not self._transcribe_worker.isRunning():
            return
        removed = 0
        for df_index in df_indices:
            job = self._model.job_state(df_index)
            if job is None:
                continue
            if job.get("phase") == "queued":
                self._transcribe_worker.skip_row(df_index)
                self._model.clear_job_state(df_index)
                removed += 1
            elif job.get("phase") in ("transcribing", "cancelling"):
                self._cancel_remaining_transcription()
                return
            ## END if job phase....

        ## END for df_index in df_indices....

        if removed:
            self._log_text.appendPlainText(
                f"Removed {removed} recording(s) from the transcription queue"
            )


    def _on_cancel_transcription(self) -> None:
        """Soft-stop the transcription queue after the current file finishes."""
        self._cancel_remaining_transcription()


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
                selected_df_index = self._model.df_index_for_row(source_index.row())
                if source_index.isValid() and selected_df_index == df_index:
                    row_data = self._model.get_row_data(source_index.row())
                    if row_data is not None:
                        self._update_transcript_preview(row_data)
                    ## END if row_data is not None....
                ## END if source_index matches transcribed row....
            ## END if proxy_index.isValid()....
        ## END if selection hasSelection....



    def _on_transcription_done(self, completed: int, failed: int, cancelled: int = 0) -> None:
        self._transcribe_btn.setEnabled(True)
        self._transcribe_all_btn.setEnabled(True)
        self._cancel_transcribe_btn.setEnabled(False)
        self._model.clear_all_job_state()
        self._transcribe_current_name = ""
        self._transcribe_current_progress = None
        parts = [f"{completed} succeeded"]
        if failed:
            parts.append(f"{failed} failed")
        if cancelled:
            parts.append(f"{cancelled} cancelled")
        summary = ", ".join(parts)
        self._status_label.setText(f"Transcription complete: {summary}")
        self._log_text.appendPlainText(f"\n✓ Done — {summary}")
        self._auto_size_columns()
        self._transcribe_origin = ""
        self._update_tray_tooltip()
        QTimer.singleShot(0, self._drain_auto_queue)


    def _on_transcription_error(self, msg: str) -> None:
        self._transcribe_btn.setEnabled(True)
        self._transcribe_all_btn.setEnabled(True)
        self._cancel_transcribe_btn.setEnabled(False)
        self._model.clear_all_job_state()
        self._status_label.setText(f"Transcription error: {msg}")
        self._log_text.appendPlainText(f"\n✗ Error: {msg}")
        origin = self._transcribe_origin
        self._transcribe_origin = ""
        self._update_tray_tooltip()
        if origin != "auto":
            QMessageBox.critical(self, "Transcription Error", msg)
        ## END if origin != auto....

        QTimer.singleShot(0, self._drain_auto_queue)


    def _save_affected_csv(self, df_index: int) -> None:
        """Persist changes to the source CSV for a given row."""
        df = self._model.get_dataframe()
        if df_index not in df.index:
            return
        source_csv = df.at[df_index, "source_csv"] if "source_csv" in df.columns else None
        if pd.isna(source_csv) or not str(source_csv).strip():
            return

        csv_path = Path(str(source_csv).strip())
        self._write_source_csv(df, csv_path)


    def _write_source_csv(self, df: pd.DataFrame, csv_path: Path) -> None:
        """Write rows belonging to *csv_path* back to that filelist."""
        if "source_csv" not in df.columns:
            return
        mask = df["source_csv"] == str(csv_path)
        subset = df[mask].drop(columns=["status", "format_id", "source_csv"], errors="ignore")
        try:
            subset.to_csv(csv_path, index=False, encoding="utf-8")
        except OSError:
            pass


    def _save_all_loaded_csvs(self) -> None:
        """Persist transcript/status columns for every loaded filelist CSV."""
        df = self._model.get_dataframe()
        if df.empty or "source_csv" not in df.columns:
            return
        for csv_str in df["source_csv"].dropna().unique():
            if not str(csv_str).strip():
                continue
            self._write_source_csv(df, Path(str(csv_str).strip()))
        ## END for csv_str in unique source_csv....


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


    def _on_table_clicked(self, proxy_index) -> None:
        """Click the status cell to dequeue or cancel remaining."""
        if not proxy_index.isValid():
            return
        source_index = self._proxy.mapToSource(proxy_index)
        if not source_index.isValid():
            return
        if self._model.column_name(source_index.column()) != "status":
            return
        df_index = self._model.df_index_for_row(source_index.row())
        if df_index is None:
            return
        job = self._model.job_state(df_index)
        if job is None:
            return
        phase = job.get("phase")
        if phase == "queued":
            self._remove_queued_rows([int(df_index)])
        elif phase in ("transcribing", "cancelling"):
            self._cancel_remaining_transcription()
        ## END if phase....



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

        df_index = self._model.df_index_for_row(source_index.row())
        job = self._model.job_state(df_index) if df_index is not None else None
        job_phase = job.get("phase") if job else None

        if job_phase == "queued" and df_index is not None:
            remove_action = menu.addAction("✕ Remove from queue")
            selected_queued = [
                idx for idx in self._visual_rows_to_df_indices(self._get_selected_source_rows())
                if (self._model.job_state(idx) or {}).get("phase") == "queued"
            ]
            targets = selected_queued if df_index in selected_queued and len(selected_queued) > 1 else [df_index]
            remove_action.triggered.connect(
                lambda _checked=False, rows=targets: self._remove_queued_rows(rows)
            )
        elif job_phase in ("transcribing", "cancelling"):
            cancel_action = menu.addAction("⏹ Cancel remaining (finish this file)")
            cancel_action.triggered.connect(self._cancel_remaining_transcription)
        else:
            # Transcribe this file
            transcribe_action = menu.addAction("🎙 Transcribe This File")
            transcribe_action.triggered.connect(
                lambda: self._start_transcription(
                    [df_index] if df_index is not None else []
                )
            )
        ## END if job_phase....

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
        if (
            not self._force_quit
            and self._tray is not None
            and self._tray.is_visible()
        ):
            self._save_prefs()
            event.ignore()
            self.hide()
            return
        ## END if hide to tray....

        self._save_prefs()
        self._player.cleanup()
        self._stop_transcript_discovery()
        if self._watch_worker is not None and self._watch_worker.isRunning():
            self._watch_worker.stop_watch()
            self._watch_worker.wait(3000)
        ## END if watch worker running....

        if self._scan_worker is not None and self._scan_worker.isRunning():
            self._scan_worker.quit()
            self._scan_worker.wait(3000)
        if self._transcribe_worker is not None and self._transcribe_worker.isRunning():
            self._transcribe_worker.request_cancel()
            self._transcribe_worker.wait(5000)
        if self._tray is not None:
            self._tray.hide()
        ## END if self._tray....

        super().closeEvent(event)


def main(argv: Optional[List[str]] = None) -> None:
    """Launch the Audio Recording Manager GUI application."""
    import argparse

    from PyQt6.QtWidgets import QApplication
    from scripts.gui.single_instance import SingleInstance, raise_window
    from scripts.gui.style import apply_dark_theme, app_icon

    parser = argparse.ArgumentParser(description="Audio Recording Manager")
    parser.add_argument(
        "--hidden",
        action="store_true",
        help="Start in the system tray without showing the window",
    )
    args, qt_args = parser.parse_known_args(argv if argv is not None else sys.argv[1:])

    app = QApplication([sys.argv[0], *qt_args])
    app.setOrganizationName("whisper-timestamped")
    app.setApplicationName("AudioRecordingManager")
    app.setQuitOnLastWindowClosed(False)

    instance = SingleInstance()
    if not instance.try_acquire():
        instance.notify_existing()
        return
    ## END if not instance.try_acquire()....

    apply_dark_theme(app)
    icon = app_icon()
    app.setWindowIcon(icon)
    window = MainWindow()
    window.setWindowIcon(icon)
    instance.listen(lambda: raise_window(window))
    if args.hidden and QSystemTrayIcon.isSystemTrayAvailable():
        window.hide()
    else:
        window.show()
    ## END if args.hidden....

    try:
        sys.exit(app.exec())
    finally:
        instance.release()



if __name__ == "__main__":
    main()
