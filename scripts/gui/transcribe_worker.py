"""QThread worker for running transcription via process_recordings.

Writes a temporary filelist CSV containing only the selected rows, then
calls process_recordings() with stdout captured so the GUI can display
live progress.
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path
from typing import Optional, Set

import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal

from scripts.gui.idle import apply_current_thread_low_priority
from scripts.transcribe_queue import ARM_DF_INDEX_COL, coerce_row_index


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
    file_started = pyqtSignal(int, str, int, int)  # (df_index, name, i, n)
    file_progress = pyqtSignal(int, float)         # (df_index, 0..1)
    file_completed = pyqtSignal(int, dict)         # (df_index, transcript_cols)
    file_skipped = pyqtSignal(int, str)            # (df_index, reason)
    file_failed = pyqtSignal(int, str)             # (df_index, message)
    all_done = pyqtSignal(int, int, int)           # (completed, failed, cancelled)
    error = pyqtSignal(str)

    def __init__(
        self,
        rows_df: pd.DataFrame,
        output_dir: Path,
        backend: str = "crisperwhisper",
        model_name: str = "turbo",
        crisper_mode: str = "verbatim",
        crisper_runtime: str = "auto",
        low_priority: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.rows_df = rows_df.copy()
        self.rows_df[ARM_DF_INDEX_COL] = [int(i) for i in self.rows_df.index]
        self.output_dir = output_dir
        self.backend = backend
        self.model_name = model_name
        self.crisper_mode = crisper_mode
        self.crisper_runtime = crisper_runtime
        self.low_priority = bool(low_priority)
        self._cancel_requested = False
        self._pause_requested = False
        self._interrupt = None  # set in run() after lazy import
        self._skipped_indices: Set[int] = set()
        self._current_df_index: Optional[int] = None
        self._completed_count = 0
        self._failed_count = 0
        self._cancelled_count = 0


    def skip_row(self, df_index: int) -> None:
        """Remove a not-yet-started row from the transcription queue."""
        idx = int(df_index)
        if self._current_df_index is not None and idx == self._current_df_index:
            return
        self._skipped_indices.add(idx)
        if self._interrupt is not None:
            self._interrupt.skip_indices.add(idx)


    def request_pause(self) -> None:
        """Finish the current file, then stop without cancelling remaining rows."""
        if self._cancel_requested:
            return
        self._pause_requested = True
        if self._interrupt is not None:
            self._interrupt.stop_after_current = True
        ## END if self._interrupt is not None....

        self.output_line.emit(
            "User activity — finishing current file, then waiting for idle…"
        )


    def request_cancel(self) -> None:
        """Request a soft-stop after the current file finishes."""
        self._cancel_requested = True
        self._pause_requested = False
        remaining = [
            int(i) for i in self.rows_df.index
            if self._current_df_index is None or int(i) != self._current_df_index
        ]
        self._skipped_indices.update(remaining)
        if self._interrupt is not None:
            self._interrupt.stop_after_current = True
            self._interrupt.skip_indices.update(self._skipped_indices)
        ## END if self._interrupt is not None....

        self.output_line.emit(
            "Cancel requested — finishing current file, then stopping…"
        )


    def _on_file_start(self, row_index, base_name, i, n) -> None:
        df_index = coerce_row_index(row_index)
        if df_index is None:
            return
        self._current_df_index = df_index
        self.file_started.emit(df_index, str(base_name), int(i), int(n))


    def _on_file_done(self, row_index, transcript_cols) -> None:
        df_index = coerce_row_index(row_index)
        if df_index is None:
            return
        self._completed_count += 1
        if self._current_df_index == df_index:
            self._current_df_index = None
        self.file_completed.emit(df_index, dict(transcript_cols or {}))


    def _on_file_skip(self, row_index, reason) -> None:
        df_index = coerce_row_index(row_index)
        if df_index is None:
            return
        reason_s = str(reason or "cancelled")
        if reason_s == "cancelled":
            self._cancelled_count += 1
        if self._current_df_index == df_index:
            self._current_df_index = None
        self.file_skipped.emit(df_index, reason_s)


    def _on_file_error(self, row_index, message) -> None:
        df_index = coerce_row_index(row_index)
        if df_index is None:
            return
        self._failed_count += 1
        if self._current_df_index == df_index:
            self._current_df_index = None
        self.file_failed.emit(df_index, str(message or "Transcription failed"))


    def _on_progress(self, fraction: float) -> None:
        if self._current_df_index is None:
            return
        try:
            value = float(fraction)
        except (TypeError, ValueError):
            return
        self.file_progress.emit(self._current_df_index, max(0.0, min(1.0, value)))


    def run(self) -> None:
        try:
            # Lazy import to avoid loading torch/whisper until needed
            from scripts.process_recordings import (
                process_recordings,
                _GracefulInterruptState,
            )

            self._interrupt = _GracefulInterruptState()
            self._interrupt.skip_indices.update(self._skipped_indices)
            if self._cancel_requested or self._pause_requested:
                self._interrupt.stop_after_current = True
            ## END if self._cancel_requested or self._pause_requested....

            prev_threads = None
            if self.low_priority:
                apply_current_thread_low_priority()
                try:
                    import torch

                    prev_threads = int(torch.get_num_threads())
                    torch.set_num_threads(1)
                    try:
                        torch.set_num_interop_threads(1)
                    except RuntimeError:
                        pass
                    ## END try set_num_interop_threads....

                except Exception:
                    prev_threads = None
                ## END try torch thread cap....

            ## END if self.low_priority....

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
                process_recordings(
                    filelist_csv=tmp_path,
                    output_dir=self.output_dir,
                    backend=self.backend,
                    model_name=self.model_name,
                    crisper_mode=self.crisper_mode,
                    crisper_runtime=self.crisper_runtime,
                    interrupt=self._interrupt,
                    on_file_start=self._on_file_start,
                    on_file_done=self._on_file_done,
                    on_file_skip=self._on_file_skip,
                    on_file_error=self._on_file_error,
                    on_progress=self._on_progress,
                )
            finally:
                sys.stdout = old_stdout
                writer.flush()
                if prev_threads is not None:
                    try:
                        import torch

                        torch.set_num_threads(prev_threads)
                    except Exception:
                        pass
                    ## END try restore torch threads....

                ## END if prev_threads is not None....

            ## END try redirect stdout...

            # Clean up temp CSV
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

            self.all_done.emit(
                self._completed_count, self._failed_count, self._cancelled_count
            )

        except Exception as exc:
            self.error.emit(f"Transcription failed: {exc}")
