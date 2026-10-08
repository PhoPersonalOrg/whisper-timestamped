"""QThread worker for scanning audio directories and extracting metadata.

Wraps the existing functions from extract_m4a_creation_times and recording_formats
to run the potentially slow ffprobe + filesystem probing off the GUI thread.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

from whisper_timestamped.recording_formats import (
    RecordingsFormat,
    detect_format,
    get_format,
)
from scripts.iOSWhisperAppHelpers.extract_m4a_creation_times import (
    build_file_list_from_audio_dir,
    extract_for_csv,
)


class ScanWorker(QThread):
    """Discover audio files, probe metadata, flag duplicates — off the GUI thread."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(str, object)  # (format_id, csv_path)
    error = pyqtSignal(str)

    def __init__(self, audio_dir: Path, fmt: RecordingsFormat, tz_name: str, csv_path: Path, parent=None) -> None:
        super().__init__(parent)
        self.audio_dir = audio_dir
        self.fmt = fmt
        self.tz_name = tz_name
        self.csv_path = csv_path


    def run(self) -> None:
        try:
            # Step 1: Build filelist CSV from the audio directory
            self.progress.emit(f"Scanning {self.audio_dir.name} ({self.fmt.id})…")
            build_file_list_from_audio_dir(
                audio_dir=self.audio_dir,
                csv_path=self.csv_path,
                tz_name=self.tz_name,
                fmt=self.fmt,
            )

            # Step 2: Extract metadata — ffprobe creation times, durations, duplicate flags
            self.progress.emit("Probing metadata with ffprobe…")
            (
                probed, creation_ok, duration_ok, missing_file, ffprobe_fail,
                fs_set_ok, fs_set_fail, dup_groups, dup_rows,
                moved, skipped, fail,
            ) = extract_for_csv(
                csv_path=self.csv_path,
                output_path=self.csv_path,  # overwrite in-place
                audio_dir=self.audio_dir,
                tz_name=self.tz_name,
                fmt=self.fmt,
            )

            summary = (
                f"Scan complete: {probed} probed, {creation_ok} creation times, "
                f"{duration_ok} durations, {dup_groups} duplicate groups ({dup_rows} rows)"
            )
            self.progress.emit(summary)
            self.finished.emit(self.fmt.id, self.csv_path)

        except Exception as exc:
            self.error.emit(f"Scan failed: {exc}")
