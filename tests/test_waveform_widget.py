"""Unit tests for playback-dock waveform peaks, cache, and prefetch."""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
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
from scripts.gui.waveform_cache import (
    WaveformCache,
    WaveformPrefetchWorker,
    extract_peaks,
)
from scripts.gui.waveform_widget import (
    WAVEFORM_HEIGHT,
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


def _tone_wav(path: Path, seconds: float = 0.25) -> Path:
    n = max(1, int(SR * seconds))
    t = np.linspace(0, seconds, n, endpoint=False, dtype=np.float32)
    _write_wav(path, (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))
    return path


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


class TestWaveformCache(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.calls: list[str] = []

        def extract(path: Path, cancel_check=None) -> tuple:
            self.calls.append(Path(path).name)
            mins = np.full(4, -0.4, dtype=np.float32)
            maxs = np.full(4, 0.4, dtype=np.float32)
            return mins, maxs, 1.25

        self.cache = WaveformCache(cache_dir=self.dir / "cache", extract_fn=extract)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_disk_cache_hit_skips_extract(self):
        wav = _tone_wav(self.dir / "clip.wav")
        first = self.cache.get_or_extract(wav)
        self.assertEqual(self.calls, ["clip.wav"])
        self.assertTrue(self.cache.cache_file(wav).is_file())
        second = self.cache.get(wav)
        self.assertIsNotNone(second)
        self.assertEqual(self.calls, ["clip.wav"])
        self.assertAlmostEqual(first[2], 1.25, places=5)
        self.assertAlmostEqual(second[2], 1.25, places=5)

    def test_stale_mtime_recomputes(self):
        wav = _tone_wav(self.dir / "clip.wav")
        self.cache.get_or_extract(wav)
        self.assertEqual(self.calls, ["clip.wav"])
        later = time.time() + 5
        os.utime(wav, (later, later))
        self.assertIsNone(self.cache.get(wav))
        self.cache.get_or_extract(wav)
        self.assertEqual(self.calls, ["clip.wav", "clip.wav"])

    def test_native_wav_extract_does_not_invoke_ffmpeg(self):
        wav = _tone_wav(self.dir / "tone.wav", seconds=0.5)

        def boom(*args, **kwargs):
            raise AssertionError("ffmpeg should not be called for a native WAV")

        mins, maxs, duration = extract_peaks(wav, n_bins=N_BINS, ffmpeg_popen=boom)
        self.assertGreater(len(mins), 0)
        self.assertGreater(duration, 0.4)

    def test_garbage_mkv_extract_fails_without_cache(self):
        path = self.dir / "clip.mkv"
        path.write_bytes(b"not a matroska file")
        real_cache = WaveformCache(cache_dir=self.dir / "mkv-cache")
        with self.assertRaises(Exception):
            real_cache.get_or_extract(path)
        self.assertIsNone(real_cache.get(path))
        self.assertFalse(real_cache.cache_file(path).is_file())


class TestWaveformPrefetchPrioritize(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_prioritize_runs_bumped_path_next(self):
        names = ("a.wav", "b.wav", "c.wav")
        paths = [_tone_wav(self.dir / name, seconds=0.05) for name in names]
        order: list[str] = []
        started = threading.Event()
        release = threading.Event()

        def extract(path: Path, cancel_check=None) -> tuple:
            order.append(Path(path).name)
            if len(order) == 1:
                started.set()
                release.wait(timeout=3)
            ## END if first extract....

            mins = np.full(4, -0.1, dtype=np.float32)
            maxs = np.full(4, 0.1, dtype=np.float32)
            return mins, maxs, 0.05

        cache = WaveformCache(cache_dir=self.dir / "cache", extract_fn=extract)
        worker = WaveformPrefetchWorker(cache)
        worker.set_paths(paths)
        worker.start()
        self.assertTrue(started.wait(timeout=3))
        worker.prioritize(paths[2])
        release.set()
        deadline = time.time() + 5
        while len(order) < 3 and time.time() < deadline:
            self.app.processEvents()
            time.sleep(0.03)
        ## END while waiting for queue....

        worker.request_cancel()
        self.assertTrue(worker.wait(3000))
        self.assertEqual(order[0], "a.wav")
        self.assertEqual(order[1], "c.wav")
        self.assertEqual(sorted(order), ["a.wav", "b.wav", "c.wav"])


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
        path = _tone_wav(self.dir / "tone.wav")
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
