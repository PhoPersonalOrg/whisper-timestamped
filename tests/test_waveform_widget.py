"""Unit tests for playback-dock waveform peaks, seek mapping, and stale workers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QSignalSpy, QTest
from PyQt6.QtWidgets import QApplication

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.gui.audio_player import AudioPlayer
from scripts.gui.waveform_widget import (
    WAVEFORM_HEIGHT,
    WaveformPeaksWorker,
    WaveformWidget,
    downsample_peaks,
)

SR = 8000
N_BINS = 32


def _app() -> QApplication:
    existing = QApplication.instance()
    if existing is not None:
        return existing
    return QApplication([])


def _write_wav(path: Path, data: np.ndarray, samplerate: int = SR) -> None:
    sf.write(str(path), data, samplerate, subtype="FLOAT")


class TestDownsamplePeaks(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_silence_peaks_are_near_zero(self):
        path = self.dir / "silence.wav"
        _write_wav(path, np.zeros(SR, dtype=np.float32))
        mins, maxs, duration = downsample_peaks(path, n_bins=N_BINS)
        self.assertEqual(len(mins), N_BINS)
        self.assertEqual(len(maxs), N_BINS)
        self.assertAlmostEqual(duration, 1.0, places=5)
        self.assertTrue(np.all(np.abs(mins) < 1e-6))
        self.assertTrue(np.all(np.abs(maxs) < 1e-6))

    def test_sine_extrema_near_full_scale(self):
        path = self.dir / "sine.wav"
        t = np.linspace(0, 1, SR, endpoint=False, dtype=np.float32)
        sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        _write_wav(path, sine)
        mins, maxs, duration = downsample_peaks(path, n_bins=N_BINS)
        self.assertEqual(len(mins), N_BINS)
        self.assertAlmostEqual(duration, 1.0, places=5)
        self.assertGreater(float(np.max(maxs)), 0.9)
        self.assertLess(float(np.min(mins)), -0.9)

    def test_stereo_is_mixed_to_mono(self):
        path = self.dir / "stereo.wav"
        n = SR
        left = np.full(n, 0.8, dtype=np.float32)
        right = np.full(n, 0.4, dtype=np.float32)
        stereo = np.stack([left, right], axis=1)
        _write_wav(path, stereo)
        mins, maxs, duration = downsample_peaks(path, n_bins=N_BINS)
        self.assertAlmostEqual(duration, 1.0, places=5)
        self.assertTrue(np.allclose(mins, 0.6, atol=1e-5))
        self.assertTrue(np.allclose(maxs, 0.6, atol=1e-5))


class TestWaveformSeekMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def _widget(self, duration: float = 10.0) -> WaveformWidget:
        widget = WaveformWidget()
        widget.setFixedWidth(400)
        widget.resize(400, WAVEFORM_HEIGHT)
        zeros = np.zeros(N_BINS, dtype=np.float32)
        widget.set_peaks(zeros, zeros, duration)
        return widget

    def test_seconds_at_x_maps_fraction_of_duration(self):
        widget = self._widget(10.0)
        self.assertAlmostEqual(widget.seconds_at_x(0), 0.0, places=5)
        self.assertAlmostEqual(widget.seconds_at_x(100), 2.5, places=5)
        self.assertAlmostEqual(widget.seconds_at_x(400), 10.0, places=5)
        self.assertAlmostEqual(widget.seconds_at_x(-10), 0.0, places=5)
        self.assertAlmostEqual(widget.seconds_at_x(999), 10.0, places=5)

    def test_click_at_quarter_width_seeks_to_quarter_duration(self):
        widget = self._widget(10.0)
        widget.show()
        self.app.processEvents()
        spy = QSignalSpy(widget.seek_requested)
        QTest.mouseClick(
            widget,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(100, WAVEFORM_HEIGHT // 2),
        )
        self.app.processEvents()
        self.assertGreaterEqual(len(spy), 1)
        self.assertAlmostEqual(float(spy[-1][0]), 2.5, delta=0.2)
        widget.close()


class TestStaleWaveformWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "tone.wav"
        t = np.linspace(0, 0.25, SR // 4, endpoint=False, dtype=np.float32)
        _write_wav(self.path, (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run_worker(self, generation: int, current_gen: int) -> list[int]:
        applied: list[int] = []

        def on_ready(mins, maxs, duration, result_gen: int) -> None:
            if result_gen != current_gen:
                return
            applied.append(result_gen)

        worker = WaveformPeaksWorker(self.path, generation=generation, n_bins=N_BINS)
        worker.peaks_ready.connect(on_ready, Qt.ConnectionType.DirectConnection)
        worker.start()
        self.assertTrue(worker.wait(10000))
        return applied

    def test_stale_generation_is_ignored(self):
        self.assertEqual(self._run_worker(generation=1, current_gen=2), [])

    def test_current_generation_is_applied(self):
        self.assertEqual(self._run_worker(generation=2, current_gen=2), [2])


class TestAudioPlayerLoad(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.player = AudioPlayer()

    def tearDown(self) -> None:
        self.player.cleanup()
        self._tmp.cleanup()

    def test_load_wav_sets_decode_path_without_ffmpeg(self):
        path = self.dir / "tone.wav"
        t = np.linspace(0, 0.25, SR // 4, endpoint=False, dtype=np.float32)
        _write_wav(path, (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))
        converting: list[str] = []
        self.player.converting.connect(converting.append)
        self.player.load(path)
        self.assertEqual(converting, [])
        self.assertEqual(self.player.decode_path, path)
        self.assertGreater(self.player.duration, 0.2)

    def test_garbage_mkv_does_not_count_as_loaded(self):
        path = self.dir / "clip.mkv"
        path.write_bytes(b"not a matroska file")
        errors: list[str] = []
        self.player.error.connect(errors.append)
        self.player.load(path)
        self.assertIsNone(self.player.decode_path)
        self.assertEqual(self.player.duration, 0.0)
        self.assertTrue(errors)

    def test_placeholder_keeps_duration_for_playhead(self):
        widget = WaveformWidget()
        widget.set_placeholder(12.5)
        self.assertAlmostEqual(widget.seconds_at_x(0), 0.0, places=5)
        widget.resize(200, WAVEFORM_HEIGHT)
        widget.setFixedWidth(200)
        self.assertAlmostEqual(widget.seconds_at_x(100), 6.25, places=5)


if __name__ == "__main__":
    unittest.main()
