"""Unit tests for idle-gated auto-transcribe helpers and prefs."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PyQt6.QtCore import QSettings

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.gui.idle import (
    DEFAULT_IDLE_SECONDS,
    DEFAULT_LOW_PRIORITY,
    DEFAULT_RUN_WHEN,
    IDLE_SECONDS_CHOICES,
    is_user_idle,
    last_input_idle_seconds,
    leftover_queued_for_requeue,
    normalize_idle_seconds,
    normalize_run_when,
    should_start_auto,
)
from scripts.gui.preferences import load_prefs, save_prefs
from scripts.transcribe_queue import job_loop_action


def _ini_settings(path: Path) -> QSettings:
    return QSettings(str(path), QSettings.Format.IniFormat)


class TestNormalize(unittest.TestCase):
    def test_run_when_defaults_to_idle(self):
        self.assertEqual(normalize_run_when(None), "idle")
        self.assertEqual(normalize_run_when(""), DEFAULT_RUN_WHEN)
        self.assertEqual(normalize_run_when("ALWAYS"), "always")
        self.assertEqual(normalize_run_when("idle"), "idle")
        self.assertEqual(normalize_run_when("bogus"), "idle")

    def test_idle_seconds_clamped_to_choices(self):
        self.assertEqual(normalize_idle_seconds(None), DEFAULT_IDLE_SECONDS)
        self.assertEqual(normalize_idle_seconds("300"), 300)
        self.assertEqual(normalize_idle_seconds(180), 120)
        for seconds in IDLE_SECONDS_CHOICES:
            self.assertEqual(normalize_idle_seconds(seconds), seconds)
        ## END for seconds in IDLE_SECONDS_CHOICES....



class TestShouldStartAuto(unittest.TestCase):
    def test_false_when_watch_off_or_worker_running(self):
        self.assertFalse(
            should_start_auto(
                watch_enabled=False,
                worker_running=False,
                run_when="always",
                idle_seconds=300,
                idle_for=999,
            )
        )
        self.assertFalse(
            should_start_auto(
                watch_enabled=True,
                worker_running=True,
                run_when="always",
                idle_seconds=300,
                idle_for=999,
            )
        )

    def test_idle_mode_requires_threshold(self):
        self.assertFalse(
            should_start_auto(
                watch_enabled=True,
                worker_running=False,
                run_when="idle",
                idle_seconds=300,
                idle_for=12,
            )
        )
        self.assertTrue(
            should_start_auto(
                watch_enabled=True,
                worker_running=False,
                run_when="idle",
                idle_seconds=300,
                idle_for=300,
            )
        )
        self.assertFalse(
            is_user_idle(run_when="idle", idle_seconds=60, idle_for=59.9)
        )
        self.assertTrue(
            is_user_idle(run_when="idle", idle_seconds=60, idle_for=60)
        )

    def test_always_ignores_idle_for(self):
        self.assertTrue(
            should_start_auto(
                watch_enabled=True,
                worker_running=False,
                run_when="always",
                idle_seconds=300,
                idle_for=0,
            )
        )
        self.assertTrue(
            is_user_idle(run_when="always", idle_seconds=300, idle_for=0)
        )

    def test_last_input_idle_seconds_non_negative(self):
        idle_for = last_input_idle_seconds()
        self.assertIsInstance(idle_for, float)
        self.assertGreaterEqual(idle_for, 0.0)


class TestLeftoverRequeue(unittest.TestCase):
    def test_pause_requeues_queued_indices(self):
        leftover = leftover_queued_for_requeue(
            origin="auto",
            paused=True,
            queued_indices=[10, 20, 10, 30],
        )
        self.assertEqual(leftover, [10, 20, 30])

    def test_cancel_or_manual_does_not_requeue(self):
        self.assertEqual(
            leftover_queued_for_requeue(
                origin="auto",
                paused=False,
                queued_indices=[10, 20],
            ),
            [],
        )
        self.assertEqual(
            leftover_queued_for_requeue(
                origin="manual",
                paused=True,
                queued_indices=[10, 20],
            ),
            [],
        )

    def test_pause_is_stop_without_skip(self):
        started = []
        leftover = []
        stop = False
        for row_index in (10, 20, 30):
            action = job_loop_action(stop, set(), row_index)
            if action == "stop":
                leftover.append(row_index)
                break
            if action == "skip":
                self.fail(f"pause should not skip {row_index}")
            started.append(row_index)
            if row_index == 10:
                stop = True
            ## END if row_index == 10....

        ## END for row_index in (10, 20, 30)....

        self.assertEqual(started, [10])
        self.assertEqual(leftover, [20])


class TestWatchIdlePrefs(unittest.TestCase):
    def test_defaults_when_ini_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            ini = Path(tmp) / "prefs.ini"
            with patch(
                "scripts.gui.preferences._settings",
                side_effect=lambda: _ini_settings(ini),
            ):
                prefs = load_prefs()
            ## END with patch _settings....

        ## END with tempfile.TemporaryDirectory....

        self.assertEqual(prefs["watch_run_when"], DEFAULT_RUN_WHEN)
        self.assertEqual(prefs["watch_idle_seconds"], DEFAULT_IDLE_SECONDS)
        self.assertEqual(prefs["watch_low_priority"], DEFAULT_LOW_PRIORITY)

    def test_round_trip_new_keys(self):
        payload = {
            "scan_audio_dir": "",
            "scan_format_id": "",
            "scan_timezone": "America/New_York",
            "scan_csv_path": "",
            "browse_load_csv_dir": "",
            "transcribe_output_dir": "C:/tmp/transcripts",
            "transcribe_backend": "openai-whisper",
            "transcribe_model": "medium.en",
            "transcribe_crisper_mode": "intended",
            "watch_enabled": True,
            "watch_folders": [],
            "watch_run_when": "always",
            "watch_idle_seconds": 120,
            "watch_low_priority": False,
            "hidden_columns": [],
            "loaded_csvs": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            ini = Path(tmp) / "prefs.ini"
            with patch(
                "scripts.gui.preferences._settings",
                side_effect=lambda: _ini_settings(ini),
            ):
                save_prefs(payload)
                loaded = load_prefs()
            ## END with patch _settings....

        ## END with tempfile.TemporaryDirectory....

        self.assertEqual(loaded["watch_run_when"], "always")
        self.assertEqual(loaded["watch_idle_seconds"], 120)
        self.assertFalse(loaded["watch_low_priority"])
        self.assertEqual(loaded["scan_timezone"], "America/New_York")
        self.assertEqual(loaded["transcribe_backend"], "openai-whisper")
        self.assertEqual(loaded["transcribe_model"], "medium.en")
        self.assertEqual(loaded["transcribe_crisper_mode"], "intended")
        self.assertEqual(loaded["transcribe_output_dir"], "C:/tmp/transcripts")


if __name__ == "__main__":
    unittest.main()
