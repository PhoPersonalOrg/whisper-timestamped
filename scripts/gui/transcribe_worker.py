"""QThread worker for running transcription via process_recordings.

Writes a temporary filelist CSV containing only the selected rows, then
calls process_recordings() with stdout captured so the GUI can display
live progress.
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal


class _SignalWriter(io.TextIOBase):
    """io.TextIOBase wrapper that emits a signal for each line written."""

    def __init__(self, signal: pyqtSignal) -> None:
        super().__init__()
        self._signal = signal
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._signal.emit(line)
        ## END while newline in buffer...
        return len(text)

    def flush(self) -> None:
        if self._buf:
            self._signal.emit(self._buf)
            self._buf = ""


class TranscribeWorker(QThread):
    """Run process_recordings() in a background thread with live output capture."""

    output_line = pyqtSignal(str)
    file_completed = pyqtSignal(int, dict)   # (df_index, transcript_cols)
    all_done = pyqtSignal(int, int)           # (completed_count, failed_count)
    error = pyqtSignal(str)

    def __init__(self, rows_df: pd.DataFrame, output_dir: Path, backend: str = "crisperwhisper", model_name: str = "medium", crisper_mode: str = "verbatim", crisper_runtime: str = "auto", parent=None) -> None:
        super().__init__(parent)
        self.rows_df = rows_df.copy()
        self.output_dir = output_dir
        self.backend = backend
        self.model_name = model_name
        self.crisper_mode = crisper_mode
        self.crisper_runtime = crisper_runtime
        self._cancel_requested = False


    def request_cancel(self) -> None:
        """Request a soft-stop after the current file finishes."""
        self._cancel_requested = True


    def run(self) -> None:
        try:
            # Lazy import to avoid loading torch/whisper until needed
            from scripts.process_recordings import (
                process_recordings,
                collect_extant_transcript_paths,
            )

            # Write selected rows to a temporary filelist CSV
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False,
                prefix="arm_transcribe_", encoding="utf-8",
            )
            tmp_path = Path(tmp.name)
            self.rows_df.to_csv(tmp_path, index=False, encoding="utf-8")
            tmp.close()

            self.output_line.emit(f"Transcribing {len(self.rows_df)} file(s)…")
            self.output_line.emit(f"Backend: {self.backend}, Model: {self.model_name}")
            self.output_line.emit(f"Output dir: {self.output_dir}")
            self.output_line.emit("")

            # Redirect stdout to capture process_recordings output
            writer = _SignalWriter(self.output_line)
            old_stdout = sys.stdout
            sys.stdout = writer

            try:
                output_files = process_recordings(
                    filelist_csv=tmp_path,
                    output_dir=self.output_dir,
                    backend=self.backend,
                    model_name=self.model_name,
                    crisper_mode=self.crisper_mode,
                    crisper_runtime=self.crisper_runtime,
                )
            finally:
                sys.stdout = old_stdout
                writer.flush()
            ## END try redirect stdout...

            # Count results and emit per-row transcript paths
            completed = 0
            failed = 0
            for idx, row in self.rows_df.iterrows():
                name_raw = row.get("name", "")
                if pd.notna(name_raw) and str(name_raw).strip():
                    base_name = Path(str(name_raw).strip()).stem
                else:
                    full_path_raw = row.get("full_path", "")
                    base_name = Path(str(full_path_raw).strip()).stem if pd.notna(full_path_raw) else ""
                ## END if name present...

                if not base_name:
                    failed += 1
                    continue

                transcript_cols = collect_extant_transcript_paths(
                    self.output_dir, base_name
                )
                # Check if any transcript was produced
                has_output = any(
                    v for v in transcript_cols.values() if v
                )
                if has_output:
                    completed += 1
                    self.file_completed.emit(int(idx), transcript_cols)
                else:
                    failed += 1
            ## END for idx, row in self.rows_df.iterrows()...

            # Clean up temp CSV
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

            self.all_done.emit(completed, failed)

        except Exception as exc:
            self.error.emit(f"Transcription failed: {exc}")
