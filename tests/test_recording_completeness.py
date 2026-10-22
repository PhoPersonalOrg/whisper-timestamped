"""Unit tests for recording completeness and default watch-folder seeding."""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from whisper_timestamped.recording_formats.completeness import (
    default_watch_folders,
    is_recording_complete,
    skip_watch_reason,
)
from whisper_timestamped.recording_formats.just_press_record import (
    JustPressRecordFormat,
)
from whisper_timestamped.recording_formats.rec_continuous import (
    RecContinuousVideoRecorderFormat,
    sidecar_path_for,
)


def _age_file(path: Path, age_s: float = 30.0) -> None:
    now = time.time()
    os.utime(path, (now - age_s, now - age_s))


class TestRecordingCompleteness(unittest.TestCase):
    def setUp(self) -> None:
        self.fmt = RecContinuousVideoRecorderFormat()
        self.min_age_s = 10.0

    def test_cam_sidecar_without_stop_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "CAM_2026-01-09T081552.mkv"
            path.write_bytes(b"x" * 100)
            sidecar_path_for(path).write_text(
                json.dumps({"wall_clock_start_unix": 1.0}),
                encoding="utf-8",
            )
            _age_file(path)
            now = time.time()
            first, snap1 = is_recording_complete(
                path, self.fmt, None, min_age_s=self.min_age_s, now=now
            )
            self.assertFalse(first)
            self.assertIsNotNone(snap1)
            second, _snap2 = is_recording_complete(
                path, self.fmt, snap1, min_age_s=self.min_age_s, now=now
            )
            self.assertFalse(second)
        ## END with tempfile....

    def test_cam_sidecar_with_stop_and_stable_size_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "CAM_2026-01-09T081552.mkv"
            path.write_bytes(b"x" * 100)
            sidecar_path_for(path).write_text(
                json.dumps(
                    {
                        "wall_clock_start_unix": 1.0,
                        "wall_clock_stop_unix": 2.0,
                    }
                ),
                encoding="utf-8",
            )
            _age_file(path)
            now = time.time()
            first, snap1 = is_recording_complete(
                path, self.fmt, None, min_age_s=self.min_age_s, now=now
            )
            self.assertFalse(first)
            second, _snap2 = is_recording_complete(
                path, self.fmt, snap1, min_age_s=self.min_age_s, now=now
            )
            self.assertTrue(second)
        ## END with tempfile....

    def test_young_mtime_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "CAM_2026-01-09T081552.mkv"
            path.write_bytes(b"x" * 100)
            sidecar_path_for(path).write_text(
                json.dumps({"wall_clock_stop_unix": 2.0}),
                encoding="utf-8",
            )
            now = time.time()
            complete, snap = is_recording_complete(
                path, self.fmt, None, min_age_s=self.min_age_s, now=now
            )
            self.assertFalse(complete)
            self.assertIsNotNone(snap)
        ## END with tempfile....

    def test_growing_size_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "CAM_2026-01-09T081552.mkv"
            path.write_bytes(b"a" * 10)
            sidecar_path_for(path).write_text(
                json.dumps({"wall_clock_stop_unix": 2.0}),
                encoding="utf-8",
            )
            _age_file(path)
            now = time.time()
            _complete, snap1 = is_recording_complete(
                path, self.fmt, None, min_age_s=self.min_age_s, now=now
            )
            path.write_bytes(b"a" * 50)
            _age_file(path)
            complete, _snap2 = is_recording_complete(
                path, self.fmt, snap1, min_age_s=self.min_age_s, now=now
            )
            self.assertFalse(complete)
        ## END with tempfile....

    def test_dup_dir_and_non_cam_names_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dup = Path(tmp) / "_DUP" / "CAM_2026-01-09T081552.mkv"
            dup.parent.mkdir()
            dup.write_bytes(b"x")
            self.assertEqual(skip_watch_reason(dup, self.fmt), "dup_dir")
            other = Path(tmp) / "notes.mkv"
            other.write_bytes(b"x")
            self.assertEqual(skip_watch_reason(other, self.fmt), "format_mismatch")
            sidecar = Path(tmp) / "CAM_2026-01-09T081552.mkv.timing.json"
            sidecar.write_text("{}", encoding="utf-8")
            self.assertEqual(skip_watch_reason(sidecar, self.fmt), "timing_sidecar")
            complete, snap = is_recording_complete(dup, self.fmt)
            self.assertFalse(complete)
            self.assertIsNone(snap)
        ## END with tempfile....


class TestDefaultWatchFolders(unittest.TestCase):
    def test_only_existing_default_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "cam"
            existing.mkdir()
            cam = RecContinuousVideoRecorderFormat()
            cam.default_recordings_dir = existing
            missing = RecContinuousVideoRecorderFormat()
            missing.id = "missing_format"
            missing.default_recordings_dir = Path(tmp) / "does_not_exist"
            folders = default_watch_folders(
                formats=[cam, missing], require_existing=True
            )
            self.assertEqual(len(folders), 1)
            self.assertEqual(
                folders[0]["format_id"], "rec_continuous_video_recorder"
            )
            self.assertEqual(folders[0]["path"], str(existing))
            self.assertTrue(folders[0]["enabled"])
            self.assertFalse(folders[0]["recursive"])
        ## END with tempfile....

    def test_just_press_record_is_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "jpr"
            root.mkdir()
            fmt = JustPressRecordFormat()
            fmt.default_recordings_dir = root
            folders = default_watch_folders(formats=[fmt], require_existing=True)
            self.assertEqual(len(folders), 1)
            self.assertTrue(folders[0]["recursive"])
            self.assertEqual(folders[0]["format_id"], "just_press_record")
        ## END with tempfile....


if __name__ == "__main__":
    unittest.main()
