"""Unit tests for Settings dialog accessors and Alt-click helper."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.gui.dialogs import TranscribeConfigDialog
from scripts.gui.settings_dialog import SettingsDialog, alt_held
from whisper_timestamped.recording_formats import list_format_ids


def _app() -> QApplication:
    existing = QApplication.instance()
    if existing is not None:
        return existing
    return QApplication([])


class TestAltHeld(unittest.TestCase):
    def test_alt_on_and_off(self):
        self.assertTrue(alt_held(Qt.KeyboardModifier.AltModifier))
        self.assertFalse(alt_held(Qt.KeyboardModifier.NoModifier))
        self.assertFalse(alt_held(Qt.KeyboardModifier.ControlModifier))
        self.assertTrue(
            alt_held(
                Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.ShiftModifier
            )
        )


class TestSettingsDialogAccessors(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def test_round_trip_accessors(self):
        format_ids = list_format_ids()
        self.assertTrue(format_ids)
        format_id = format_ids[0]
        folders = [
            {
                "format_id": format_id,
                "path": "C:/recordings/cam",
                "enabled": True,
                "recursive": False,
            },
            {
                "format_id": format_id,
                "path": "C:/recordings/extra",
                "enabled": False,
                "recursive": True,
            },
        ]
        dlg = SettingsDialog(
            folders,
            auto_transcribe=False,
            start_with_windows=False,
            show_autostart=True,
            scan_timezone="America/Chicago",
            transcribe_backend="openai-whisper",
            transcribe_model="medium.en",
            transcribe_crisper_mode="intended",
            transcribe_output_dir="D:/out",
            run_when="always",
            idle_seconds=120,
            low_priority=False,
        )
        self.assertFalse(dlg.auto_transcribe())
        self.assertFalse(dlg.start_with_windows())
        self.assertEqual(dlg.scan_timezone(), "America/Chicago")
        self.assertEqual(dlg.transcribe_backend(), "openai-whisper")
        self.assertEqual(dlg.transcribe_model(), "medium.en")
        self.assertEqual(dlg.transcribe_crisper_mode(), "intended")
        self.assertEqual(dlg.transcribe_output_dir(), "D:/out")
        self.assertEqual(dlg.run_when(), "always")
        self.assertEqual(dlg.idle_seconds(), 120)
        self.assertFalse(dlg.low_priority())
        self.assertEqual(dlg.folders(), folders)
        dlg.close()


class TestTranscribeConfigDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def test_openai_backend_initializes_without_mode_combo_error(self):
        dlg = TranscribeConfigDialog(
            default_output_dir="D:/out",
            initial_backend="openai-whisper",
            initial_model="large-v3",
        )
        self.assertEqual(dlg.backend(), "openai-whisper")
        self.assertEqual(dlg.model_name(), "large-v3")
        self.assertTrue(hasattr(dlg, "_mode_combo"))
        dlg.close()
