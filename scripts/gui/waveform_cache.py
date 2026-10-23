"""Disk-backed peak envelopes and a background prefetch queue.

Peaks are extracted from the original media path (soundfile for PCM, ffmpeg
float32 pipe for video) and stored under Qt's cache directory. Playback temp
WAVs are never written for waveform preview.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Set, Tuple

import numpy as np
from PyQt6.QtCore import QStandardPaths, QThread, pyqtSignal

from scripts.gui.audio_player import _VIDEO_SUFFIXES
from scripts.gui.waveform_widget import DEFAULT_N_BINS, downsample_peaks

CACHE_VERSION = 1
FFMPEG_PEAK_RATE = 8000
PIPE_CHUNK_BYTES = 65536 * 4
PREBIN_SAMPLES = 2048

ExtractFn = Callable[..., Tuple[np.ndarray, np.ndarray, float]]
Peaks = Tuple[np.ndarray, np.ndarray, float]


def cache_dir_default() -> Path:
    root = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation)
    if not root:
        root = str(Path.home() / ".cache" / "whisper-timestamped")
    ## END if no Qt cache location....

    return Path(root) / "waveforms"


def source_fingerprint(path: Path) -> Tuple[int, int]:
    stat = path.stat()
    return int(stat.st_mtime_ns), int(stat.st_size)


def extract_peaks(
    path: Path,
    n_bins: int = DEFAULT_N_BINS,
    cancel_check: Optional[Callable[[], bool]] = None,
    ffmpeg_popen=subprocess.Popen,
) -> Peaks:
    """Return (mins, maxs, duration) for *path*, using ffmpeg only when needed."""
    source = Path(path)
    if source.suffix.lower() not in _VIDEO_SUFFIXES:
        try:
            mins, maxs, duration = downsample_peaks(
                source, n_bins=n_bins, cancel_check=cancel_check
            )
            if len(mins) > 0 and duration > 0:
                return mins, maxs, duration
            ## END if native peaks usable....

        except Exception:
            pass
        ## END try native soundfile....

    return downsample_peaks_ffmpeg(
        source,
        n_bins=n_bins,
        cancel_check=cancel_check,
        ffmpeg_popen=ffmpeg_popen,
    )


def downsample_peaks_ffmpeg(
    path: Path,
    n_bins: int = DEFAULT_N_BINS,
    cancel_check: Optional[Callable[[], bool]] = None,
    sample_rate: int = FFMPEG_PEAK_RATE,
    ffmpeg_popen=subprocess.Popen,
) -> Peaks:
    """Stream mono float32 from ffmpeg and bin min/max peaks."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-f",
        "f32le",
        "pipe:1",
    ]
    kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "bufsize": PIPE_CHUNK_BYTES,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    ## END if win32....

    proc = ffmpeg_popen(cmd, **kwargs)
    pre_mins: List[np.ndarray] = []
    pre_maxs: List[np.ndarray] = []
    leftover = np.zeros(0, dtype=np.float32)
    total_samples = 0
    try:
        stdout = proc.stdout
        if stdout is None:
            raise RuntimeError("ffmpeg stdout pipe missing")
        while True:
            if cancel_check is not None and cancel_check():
                raise InterruptedError("waveform peak extraction cancelled")
            raw = stdout.read(PIPE_CHUNK_BYTES)
            if not raw:
                break
            samples = np.frombuffer(raw, dtype="<f4")
            if samples.size == 0:
                continue
            leftover, total_samples = _append_prebins(
                samples, leftover, pre_mins, pre_maxs, total_samples
            )
        ## END while ffmpeg pipe....

        if leftover.size:
            pre_mins.append(np.array([float(leftover.min())], dtype=np.float32))
            pre_maxs.append(np.array([float(leftover.max())], dtype=np.float32))
        ## END if leftover....

    finally:
        _stop_ffmpeg(proc)
    ## END try ffmpeg pipe....

    returncode = proc.poll()
    if total_samples <= 0:
        stderr = b""
        if proc.stderr is not None:
            stderr = proc.stderr.read() or b""
        detail = stderr.decode("utf-8", errors="replace").strip().splitlines()
        tail = detail[-1] if detail else f"exit {returncode}"
        raise RuntimeError(f"ffmpeg produced no audio for {path.name}: {tail}")
    ## END if no samples....

    duration = total_samples / float(sample_rate)
    if not pre_mins:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), 0.0

    mins_pre = np.concatenate(pre_mins)
    maxs_pre = np.concatenate(pre_maxs)
    return _rebin_peaks(mins_pre, maxs_pre, n_bins, duration)


def _append_prebins(
    samples: np.ndarray,
    leftover: np.ndarray,
    pre_mins: List[np.ndarray],
    pre_maxs: List[np.ndarray],
    total_samples: int,
) -> Tuple[np.ndarray, int]:
    if leftover.size:
        samples = np.concatenate([leftover, np.asarray(samples, dtype=np.float32)])
    else:
        samples = np.asarray(samples, dtype=np.float32)
    ## END if leftover....

    total_samples += int(samples.size)
    n = (samples.size // PREBIN_SAMPLES) * PREBIN_SAMPLES
    if n:
        shaped = samples[:n].reshape(-1, PREBIN_SAMPLES)
        pre_mins.append(shaped.min(axis=1).astype(np.float32, copy=False))
        pre_maxs.append(shaped.max(axis=1).astype(np.float32, copy=False))
    ## END if full prebins....

    return samples[n:].copy(), total_samples


def _rebin_peaks(
    mins_pre: np.ndarray,
    maxs_pre: np.ndarray,
    n_bins: int,
    duration: float,
) -> Peaks:
    n_pre = int(mins_pre.size)
    n_bins = max(1, min(int(n_bins), n_pre))
    out_min = np.empty(n_bins, dtype=np.float32)
    out_max = np.empty(n_bins, dtype=np.float32)
    for i in range(n_bins):
        start = int(i * n_pre / n_bins)
        end = max(start + 1, int((i + 1) * n_pre / n_bins))
        out_min[i] = float(mins_pre[start:end].min())
        out_max[i] = float(maxs_pre[start:end].max())
    ## END for i in range(n_bins)....

    return out_min, out_max, float(duration)


def _stop_ffmpeg(proc) -> None:
    try:
        if proc.poll() is None:
            proc.kill()
        ## END if still running....

        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()
        proc.wait(timeout=2)
    except Exception:
        pass
    ## END try stop ffmpeg....


class WaveformCache:
    """Memory + npz peak cache keyed by resolved path, mtime, and size."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        extract_fn: Optional[ExtractFn] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else cache_dir_default()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._extract_fn: ExtractFn = extract_fn or extract_peaks
        self._memory: Dict[str, Tuple[np.ndarray, np.ndarray, float, int, int]] = {}

    def key_for(self, path: Path) -> str:
        try:
            return str(Path(path).resolve())
        except OSError:
            return str(Path(path))
        ## END try resolve....

    def cache_file(self, path: Path) -> Path:
        digest = hashlib.sha1(self.key_for(path).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.npz"

    def get(self, path: Path) -> Optional[Peaks]:
        source = Path(path)
        if not source.is_file():
            return None
        try:
            mtime_ns, size = source_fingerprint(source)
        except OSError:
            return None
        ## END try fingerprint....

        key = self.key_for(source)
        mem = self._memory.get(key)
        if mem is not None:
            mins, maxs, duration, mem_mtime, mem_size = mem
            if mem_mtime == mtime_ns and mem_size == size:
                return mins, maxs, duration
            self._memory.pop(key, None)
        ## END if memory entry....

        disk = self._load_disk(source, mtime_ns, size)
        if disk is None:
            return None
        mins, maxs, duration = disk
        self._memory[key] = (mins, maxs, duration, mtime_ns, size)
        return mins, maxs, duration

    def put(self, path: Path, mins: np.ndarray, maxs: np.ndarray, duration: float) -> None:
        source = Path(path)
        if not source.is_file():
            return
        try:
            mtime_ns, size = source_fingerprint(source)
        except OSError:
            return
        ## END try fingerprint....

        mins_arr = np.asarray(mins, dtype=np.float32)
        maxs_arr = np.asarray(maxs, dtype=np.float32)
        key = self.key_for(source)
        self._memory[key] = (mins_arr, maxs_arr, float(duration), mtime_ns, size)
        dest = self.cache_file(source)
        tmp = dest.with_name(dest.stem + "-tmp.npz")
        np.savez(
            tmp,
            mins=mins_arr,
            maxs=maxs_arr,
            duration=np.float64(duration),
            mtime_ns=np.int64(mtime_ns),
            size=np.int64(size),
            n_bins=np.int64(len(mins_arr)),
            version=np.int64(CACHE_VERSION),
        )
        tmp.replace(dest)

    def get_or_extract(
        self,
        path: Path,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Peaks:
        hit = self.get(path)
        if hit is not None:
            return hit
        mins, maxs, duration = self._extract_fn(path, cancel_check=cancel_check)
        if len(mins) == 0 or duration <= 0:
            raise RuntimeError(f"no waveform data for {Path(path).name}")
        ## END if empty extract....

        self.put(path, mins, maxs, duration)
        return mins, maxs, duration

    def _load_disk(
        self,
        source: Path,
        mtime_ns: int,
        size: int,
    ) -> Optional[Peaks]:
        dest = self.cache_file(source)
        if not dest.is_file():
            return None
        try:
            with np.load(dest, allow_pickle=False) as data:
                version = int(data["version"])
                if version != CACHE_VERSION:
                    return None
                if int(data["mtime_ns"]) != mtime_ns or int(data["size"]) != size:
                    return None
                mins = np.asarray(data["mins"], dtype=np.float32)
                maxs = np.asarray(data["maxs"], dtype=np.float32)
                duration = float(data["duration"])
            ## END with np.load....

        except (OSError, ValueError, KeyError, EOFError):
            return None
        ## END try load npz....

        if len(mins) == 0 or len(mins) != len(maxs):
            return None
        return mins, maxs, duration


class WaveformPrefetchWorker(QThread):
    """Lowest-priority queue: extract and cache peaks for many files."""

    peaks_ready = pyqtSignal(str, object, object, float)
    peaks_failed = pyqtSignal(str, str)
    progress = pyqtSignal(str)

    def __init__(self, cache: WaveformCache, parent=None) -> None:
        super().__init__(parent)
        self._cache = cache
        self._lock = threading.Lock()
        self._queue: Deque[Path] = deque()
        self._queued: Set[str] = set()
        self._cancel_requested = False
        self._wake = threading.Event()
        self._total = 0
        self._done = 0

    def set_paths(self, paths: List[Path]) -> None:
        unique: Deque[Path] = deque()
        seen: Set[str] = set()
        for path in paths:
            key = self._cache.key_for(path)
            if key in seen:
                continue
            seen.add(key)
            unique.append(Path(path))
        ## END for path in paths....

        with self._lock:
            self._queue = unique
            self._queued = seen
            self._total = len(unique)
            self._done = 0
        ## END with lock....

        self._wake.set()

    def enqueue(self, path: Path) -> None:
        key = self._cache.key_for(path)
        with self._lock:
            if key in self._queued:
                return
            self._queue.append(Path(path))
            self._queued.add(key)
            self._total += 1
        ## END with lock....

        self._wake.set()

    def prioritize(self, path: Path) -> None:
        target = Path(path)
        key = self._cache.key_for(target)
        with self._lock:
            self._queue = deque(
                item for item in self._queue if self._cache.key_for(item) != key
            )
            self._queue.appendleft(target)
            if key not in self._queued:
                self._queued.add(key)
                self._total += 1
            ## END if newly queued....

        ## END with lock....

        self._wake.set()

    def request_cancel(self) -> None:
        self._cancel_requested = True
        self._wake.set()

    def run(self) -> None:
        while not self._cancel_requested:
            path: Optional[Path] = None
            with self._lock:
                if self._queue:
                    path = self._queue.popleft()
                ## END if queue....

            ## END with lock....

            if path is None:
                self._wake.wait(0.4)
                self._wake.clear()
                continue
            ## END if idle....

            try:
                hit = self._cache.get(path)
                if hit is None:
                    mins, maxs, duration = self._cache.get_or_extract(
                        path,
                        cancel_check=lambda: self._cancel_requested,
                    )
                    if self._cancel_requested:
                        return
                    self.peaks_ready.emit(str(path), mins, maxs, duration)
                ## END if cache miss....

            except InterruptedError:
                return
            except Exception as exc:
                if not self._cancel_requested:
                    self.peaks_failed.emit(str(path), str(exc))
                ## END if not cancelled....

            ## END try extract....

            with self._lock:
                self._done += 1
                done = self._done
                total = self._total
            ## END with lock....

            if done == 1 or done == total or done % 10 == 0:
                self.progress.emit(f"Waveforms {done}/{total}")
            ## END if progress tick....

        ## END while not cancelled....
