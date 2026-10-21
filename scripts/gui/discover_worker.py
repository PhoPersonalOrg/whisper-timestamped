"""QThread worker that probes disk for existing transcripts after table load."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QThread, pyqtSignal

from scripts.transcript_outputs import discover_row_transcript_updates
from whisper_timestamped.recording_formats import get_format


class DiscoverTranscriptsWorker(QThread):
    """Check output folders for existing transcripts without blocking the GUI."""

    row_updated = pyqtSignal(int, dict)  # (df_index, column updates)
    progress = pyqtSignal(str)
    finished = pyqtSignal(int)  # found_count
    error = pyqtSignal(str)

    def __init__(
        self,
        jobs: List[Dict[str, Any]],
        prefs_output_dir: Optional[Path] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.jobs = jobs
        self.prefs_output_dir = prefs_output_dir
        self._cancel_requested = False
        self._format_dirs: Dict[str, Optional[Path]] = {}


    def request_cancel(self) -> None:
        self._cancel_requested = True


    def _format_output_dir(self, format_id: str) -> Optional[Path]:
        if not format_id:
            return None
        if format_id not in self._format_dirs:
            try:
                fmt = get_format(format_id)
                self._format_dirs[format_id] = fmt.default_output_dir
            except (KeyError, ValueError):
                self._format_dirs[format_id] = None
            ## END try get_format....
        ## END if format_id not cached....

        return self._format_dirs[format_id]


    def run(self) -> None:
        try:
            found = 0
            total = len(self.jobs)
            if total:
                self.progress.emit(f"Checking {total} recording(s) for existing transcripts…")
            ## END if total....

            for i, job in enumerate(self.jobs):
                if self._cancel_requested:
                    break
                df_index = int(job.get("df_index", -1))
                if df_index < 0:
                    continue
                updates = discover_row_transcript_updates(
                    job,
                    format_output_dir=self._format_output_dir(str(job.get("format_id") or "")),
                    prefs_output_dir=self.prefs_output_dir,
                )
                if updates:
                    found += 1
                    self.row_updated.emit(df_index, updates)
                ## END if updates....

                if i == 0 or (i + 1) % 25 == 0 or (i + 1) == total:
                    self.progress.emit(
                        f"Checking transcripts… {i + 1}/{total}"
                    )
                ## END if progress tick....
            ## END for i, job in enumerate(self.jobs)....

            self.finished.emit(found)
        except Exception as exc:
            self.error.emit(f"Transcript discovery failed: {exc}")
