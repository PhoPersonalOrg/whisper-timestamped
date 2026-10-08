# PyQt6 Audio Recording Manager — Implementation Plan

## Goal

Build a PyQt6 desktop GUI that unifies the existing CLI workflows — audio file discovery (`extract_m4a_creation_times.py`), centralized CSV database management, batch transcription (`process_recordings.py`), transcript viewing, and audio playback — into a single application. All existing CLI functionality remains untouched; the GUI is a new consumer of the same library functions.

## Architecture Overview

```mermaid
graph TB
    subgraph "New: scripts/gui/"
        main["main.py<br/>(entry point)"]
        mw["main_window.py<br/>(QMainWindow)"]
        tm["table_model.py<br/>(QAbstractTableModel)"]
        scan["scan_worker.py<br/>(QThread: discover + probe)"]
        txn["transcribe_worker.py<br/>(QThread: process_recordings)"]
        player["audio_player.py<br/>(sounddevice playback)"]
        tv["transcript_viewer.py<br/>(QDialog: show results)"]
        dlg["dialogs.py<br/>(ScanConfigDialog, etc.)"]
        style["style.py<br/>(QSS dark theme)"]
    end

    subgraph "Existing (unchanged)"
        extract["extract_m4a_creation_times.py"]
        process["process_recordings.py"]
        rf["recording_formats/*"]
    end

    mw --> tm
    mw --> player
    mw --> tv
    mw --> dlg
    scan --> extract
    scan --> rf
    txn --> process
    main --> mw
    main --> style
```

---

## User Review Required

> [!IMPORTANT]
> **PyQt6 dependency**: PyQt6 is not currently in the project. It will be added as a new optional extra `gui` in `pyproject.toml`. You'll install with `uv sync --extra gui`.

> [!IMPORTANT]
> **Transcription runs in a background QThread**: The GUI will remain responsive during transcription. Since `process_recordings()` loads a Whisper model (~2–4 GB VRAM), the user should be aware that launching transcription from the GUI has the same resource requirements as the CLI. The GUI will show a progress panel with live output capture.

> [!WARNING]
> **`process_recordings()` path handling**: The existing function uses WSL path mapping via `host_path()`. The GUI will run natively on Windows, so all paths passed to it will be native Windows paths. This should work correctly — `host_path()` is a no-op for native Windows paths.

---

## Proposed Changes

### 1. Dependency — `pyproject.toml`

#### [MODIFY] `pyproject.toml`

Add a new `gui` optional extra:

```diff
 [project.optional-dependencies]
 dev = [
     "matplotlib>=3.10.3",
     "transformers>=4.53.2",
 ]
+gui = [
+    "PyQt6>=6.6.0",
+]
 live = [
```

`sounddevice` and `soundfile` are already available in the `live` extra and are currently installed.

---

### 2. GUI Module — `scripts/gui/`

All new files. The GUI lives under `scripts/gui/` to keep it alongside other scripts and outside the installable `whisper_timestamped` package.

#### [NEW] `scripts/gui/__init__.py`

Empty package marker.

#### [NEW] `scripts/gui/main.py`

Entry point. Instantiates `QApplication`, applies the dark theme from `style.py`, creates `MainWindow`, runs the event loop.

```python
"""Launch the Audio Recording Manager GUI.

Usage:
    .venv/Scripts/python.exe -m scripts.gui.main
    # or:
    .venv/Scripts/python.exe scripts/gui/main.py
"""
import sys
from PyQt6.QtWidgets import QApplication
from scripts.gui.style import apply_dark_theme
from scripts.gui.main_window import MainWindow

def main() -> None:
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
```

#### [NEW] `scripts/gui/style.py`

Centralised QSS dark theme. Rich colour palette with accent colours for status badges (transcribed, pending, duplicate). Font: Segoe UI (Windows native) with fallbacks.

Key status colours:
| Status | Colour | Usage |
|---|---|---|
| Transcribed | `#4ade80` (green) | Row background tint, status badge |
| Pending | `#facc15` (amber) | Row background tint, status badge |
| Duplicate | `#f87171` (red) | Row background tint, strikethrough text |
| Playing | `#60a5fa` (blue) | Highlight on the currently-playing row |

#### [NEW] `scripts/gui/main_window.py`

`MainWindow(QMainWindow)` — the central widget. Layout:

```
┌──────────────────────────────────────────────────────┐
│  Toolbar:  [Scan Audio Dir ▾]  [Transcribe ▾]        │
│            [Format: ▾ combo]   [Filter: _____]        │
├──────────────────────────────────────────────────────┤
│                                                        │
│  QTableView  (sortable, filterable, multi-select)      │
│  Columns: status_icon │ name │ format │ title │        │
│           size_mb │ duration │ creation_time │          │
│           is_duplicate │ transcript_txt (path) │ ...   │
│                                                        │
│  (right-click context menu: Play, View Transcript,     │
│   Transcribe Selected, Reveal in Explorer, Mark Dup)   │
│                                                        │
├──────────────────────────────────────────────────────┤
│  Detail / Playback panel (collapsible bottom dock)     │
│  ┌─────────────────────────────────────────────┐      │
│  │ ▶ ■  00:00 / 01:23:45   ████████░░░  🔊    │      │
│  │ Transcript preview (first 500 chars of .txt)│      │
│  └─────────────────────────────────────────────┘      │
├──────────────────────────────────────────────────────┤
│  Status bar: "243 recordings │ 180 transcribed │ ..."  │
└──────────────────────────────────────────────────────┘
```

**Key behaviours:**

1. **Scan Audio Dir** — opens `ScanConfigDialog` (pick dir, format, timezone), runs `ScanWorker` in a QThread. On completion, merges results into the in-memory model and saves the per-format filelist CSV.

2. **Load Existing CSV** — toolbar button to directly load a previously-generated filelist CSV.

3. **Transcribe** — dropdown with:
   - **Transcribe Selected** — sends selected rows to `TranscribeWorker`
   - **Transcribe All Pending** — sends all rows where `transcript_json` is empty/missing

4. **Table** — built on `RecordingTableModel` (QAbstractTableModel) backed by a merged `pd.DataFrame`. Supports:
   - Sorting by any column
   - Filter text box (searches name, title, full_path)
   - Format dropdown filter (All / ios_whisper_app / voice_memos / just_press_record / ...)
   - Status filter (All / Transcribed / Pending / Duplicate)

5. **Row double-click** → plays audio in the detail panel
6. **Row right-click** → context menu: Play, View Transcript, Transcribe, Open Folder, Mark/Unmark Duplicate

#### [NEW] `scripts/gui/table_model.py`

`RecordingTableModel(QAbstractTableModel)` — wraps a `pd.DataFrame`.

**Unified column set** (superset of all format CSVs):
```python
DISPLAY_COLUMNS = [
    "status",           # computed: "transcribed" / "pending" / "duplicate"
    "name",
    "format_id",        # added by GUI on load: which RecordingsFormat this came from
    "title",            # VoiceMemos only; blank for others
    "size_mb",
    "duration_hms",
    "extracted_creation_time",
    "is_duplicate",
    "transcript_txt",   # path to .txt output
]
```

- Custom `data()` returns `Qt.BackgroundRole` colours based on status.
- Custom `QSortFilterProxyModel` subclass handles the toolbar filter controls.

**Loading logic:**
```python
def load_format_csv(csv_path: Path, format_id: str) -> pd.DataFrame:
    """Load a single format's filelist CSV, adding format_id column."""
    df = pd.read_csv(csv_path)
    df["format_id"] = format_id
    df["source_csv"] = str(csv_path)
    return df

def merge_format_csvs(csv_paths: Dict[str, Path]) -> pd.DataFrame:
    """Merge multiple format CSVs into one DataFrame for the table."""
    frames = []
    for format_id, csv_path in csv_paths.items():
        if csv_path.is_file():
            frames.append(load_format_csv(csv_path, format_id))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
```

#### [NEW] `scripts/gui/scan_worker.py`

`ScanWorker(QThread)` — performs audio discovery and metadata extraction.

Wraps the existing functions:
1. `RecordingsFormat.ensure_filelist_csv()` — discovers media files, writes CSV
2. `extract_m4a_creation_times.extract_for_csv()` — probes ffprobe metadata, sets creation times, flags duplicates

Emits signals:
- `progress(str)` — status messages for the status bar
- `finished(Path)` — path to the output CSV when done
- `error(str)` — on failure

```python
class ScanWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(Path)
    error = pyqtSignal(str)

    def __init__(self, audio_dir: Path, fmt: RecordingsFormat, tz_name: str, csv_path: Path):
        super().__init__()
        self.audio_dir = audio_dir
        self.fmt = fmt
        self.tz_name = tz_name
        self.csv_path = csv_path

    def run(self):
        try:
            # Step 1: Build filelist CSV
            self.progress.emit(f"Scanning {self.audio_dir.name}...")
            build_file_list_from_audio_dir(
                audio_dir=self.audio_dir,
                csv_path=self.csv_path,
                tz_name=self.tz_name,
                fmt=self.fmt,
            )
            # Step 2: Extract metadata (creation times, durations, duplicates)
            self.progress.emit("Probing metadata with ffprobe...")
            extract_for_csv(
                csv_path=self.csv_path,
                output_path=self.csv_path,  # overwrite in-place
                audio_dir=self.audio_dir,
                tz_name=self.tz_name,
                fmt=self.fmt,
            )
            self.finished.emit(self.csv_path)
        except Exception as e:
            self.error.emit(str(e))
```

#### [NEW] `scripts/gui/transcribe_worker.py`

`TranscribeWorker(QThread)` — runs transcription via `process_recordings()`.

Since `process_recordings()` prints progress to stdout, we redirect stdout to capture output for the GUI's progress panel. The worker writes a temporary filelist CSV containing only the selected rows, then calls `process_recordings(filelist_csv=temp_csv, ...)`.

Emits signals:
- `output_line(str)` — captured stdout line (for the log panel)
- `file_completed(int, dict)` — (row_index, transcript_paths) after each file finishes
- `all_done(int, int)` — (completed, failed) counts
- `error(str)` — on fatal error

```python
class TranscribeWorker(QThread):
    output_line = pyqtSignal(str)
    file_completed = pyqtSignal(int, dict)
    all_done = pyqtSignal(int, int)
    error = pyqtSignal(str)

    def __init__(self, rows_df: pd.DataFrame, output_dir: Path, backend: str = "crisperwhisper", model_name: str = "medium"):
        super().__init__()
        self.rows_df = rows_df
        self.output_dir = output_dir
        self.backend = backend
        self.model_name = model_name
```

**Stdout capture approach:** We use `contextlib.redirect_stdout` with a custom `io.TextIOBase` subclass that emits `output_line` on each `write()` call, so all of `process_recordings()`'s print output appears in the GUI log panel.

#### [NEW] `scripts/gui/audio_player.py`

`AudioPlayer(QObject)` — non-blocking audio playback using `sounddevice` + `soundfile`.

Key API:
```python
class AudioPlayer(QObject):
    position_changed = pyqtSignal(float)  # seconds
    duration_changed = pyqtSignal(float)  # total seconds
    state_changed = pyqtSignal(str)       # "playing" / "paused" / "stopped"

    def load(self, path: Path) -> None: ...
    def play(self) -> None: ...
    def pause(self) -> None: ...
    def stop(self) -> None: ...
    def seek(self, seconds: float) -> None: ...
    def set_volume(self, volume: float) -> None: ...  # 0.0–1.0
```

Implementation:
- Reads the audio file using `soundfile.SoundFile` in streaming mode (to handle large files without loading into memory)
- Uses `sounddevice.OutputStream` with a callback that feeds chunks
- A QTimer polls position for the progress slider
- Supports .m4a and .caf via the `soundfile` (libsndfile) backend; if libsndfile can't decode m4a, falls back to `subprocess` calling `ffmpeg` to convert to WAV into a temp file first

#### [NEW] `scripts/gui/transcript_viewer.py`

`TranscriptViewerDialog(QDialog)` — shows the transcription results for a selected recording.

Layout:
```
┌─────────────────────────────────────────────────┐
│  Transcript: 0EFCD68F-9DE0-47DF-804F-0BD...     │
│  Tabs: [Plain Text] [SRT] [VTT] [JSON]          │
├─────────────────────────────────────────────────┤
│                                                   │
│  QPlainTextEdit (read-only, monospace, with       │
│  search/Ctrl+F)                                   │
│                                                   │
├─────────────────────────────────────────────────┤
│  [Copy to Clipboard]  [Open in Editor]  [Close]  │
└─────────────────────────────────────────────────┘
```

Loads from the `transcript_*` paths in the filelist CSV. Tabs only appear for transcript files that actually exist on disk.

#### [NEW] `scripts/gui/dialogs.py`

`ScanConfigDialog(QDialog)` — pre-scan configuration:
- Audio directory picker (QFileDialog)
- Format dropdown (populated from `list_format_ids()`, with "Auto-detect" default)
- Timezone text input (default: "America/Los_Angeles")
- Output CSV path (auto-filled from format defaults, editable)
- [Scan] / [Cancel] buttons

`TranscribeConfigDialog(QDialog)` — pre-transcription configuration:
- Backend dropdown: openai-whisper / crisperwhisper
- Model name: text input (default: "medium")
- CrisperWhisper mode: verbatim / clean (only visible when backend=crisperwhisper)
- Output directory picker
- [Start] / [Cancel] buttons

---

### 3. Duplicate Management

The GUI exposes duplicate groups directly in the table:

1. **Visual highlighting**: Rows with `is_duplicate=True` get a red-tinted background and strikethrough text on the name column.
2. **Duplicate group badge**: A small grouped indicator showing how many duplicates share the same creation_time+duration+size_mb key.
3. **Context menu actions**:
   - **"Move Duplicates to _DUP/"** — calls `move_duplicates_to_dup_dir()` from `extract_m4a_creation_times.py`, updates the in-memory model and saves the CSV.
   - **"Show Duplicate Group"** — filters the table to show only the group members.
4. **Toolbar toggle**: "Show/Hide Duplicates" checkbox to filter duplicates from the main view.

---

### 4. File Structure Summary

```
scripts/gui/
├── __init__.py           # empty package marker
├── main.py               # entry point
├── main_window.py        # MainWindow (QMainWindow)
├── table_model.py        # RecordingTableModel + proxy filter model
├── scan_worker.py        # ScanWorker (QThread)
├── transcribe_worker.py  # TranscribeWorker (QThread)
├── audio_player.py       # AudioPlayer (sounddevice/soundfile)
├── transcript_viewer.py  # TranscriptViewerDialog
├── dialogs.py            # ScanConfigDialog, TranscribeConfigDialog
└── style.py              # QSS dark theme
```

All existing files remain completely unchanged — the GUI only imports from the existing modules:
- `scripts.iOSWhisperAppHelpers.extract_m4a_creation_times` — `build_file_list_from_audio_dir`, `extract_for_csv`, `move_duplicates_to_dup_dir`
- `scripts.process_recordings` — `process_recordings`, `find_extant_output_files`, `collect_extant_transcript_paths`
- `whisper_timestamped.recording_formats` — `get_format`, `list_format_ids`, `detect_format`, `RecordingsFormat`

---

## Verification Plan

### Automated Tests

No new automated tests in this phase (the GUI is a thin orchestrator over already-tested library functions). The existing test suite should continue to pass:

```bash
.venv\Scripts\python.exe -m pytest tests/ -x
```

### Manual Verification

1. **Launch**: `.venv\Scripts\python.exe scripts/gui/main.py` — verify window appears with dark theme
2. **Scan**: Click "Scan Audio Dir" → pick a known audio directory → verify table populates with all recordings, durations, creation times
3. **Load CSV**: Use toolbar to load an existing filelist CSV → verify columns display correctly
4. **Multi-format merge**: Load both a VoiceMemos and WhisperApp CSV → verify combined view with format column
5. **Filter/Sort**: Test text filter, format dropdown, status dropdown, column sorting
6. **Playback**: Double-click a row → verify audio plays in the detail panel with progress slider
7. **Transcript Viewer**: Right-click a transcribed row → "View Transcript" → verify tabs appear for each existing output format
8. **Transcribe Selected**: Select 1–2 untranscribed rows → right-click → "Transcribe Selected" → verify progress output appears, row status updates to "transcribed" on completion
9. **Transcribe All Pending**: Verify batch button processes only rows without transcripts
10. **Duplicate management**: Verify duplicate rows show red highlight; right-click → "Move Duplicates" works; table refreshes
11. **Status bar**: Verify counts update after each operation
12. **Existing CLI**: Run `process_recordings.py` and `extract_m4a_creation_times.py` CLI commands — verify they still work identically
