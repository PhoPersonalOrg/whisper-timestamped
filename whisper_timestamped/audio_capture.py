"""Refcounted shared microphone capture for live transcription and raw WAV recording."""

from __future__ import annotations

import threading
from typing import Callable, List, Optional, Union

import numpy as np

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - optional at import time
    sd = None


AudioCallback = Callable[[np.ndarray], None]


class MicCaptureHub:
    """One PortAudio input stream shared by N listeners. First subscribe opens; last unsubscribe closes."""

    def __init__(self, stream_factory: Optional[Callable[..., object]] = None):
        self._stream_factory = stream_factory
        self._lock = threading.RLock()
        self._listeners: List[AudioCallback] = []
        self._stream = None
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = "float32"
        self.device: Optional[Union[int, str]] = None

    @property
    def is_running(self) -> bool:
        return self._stream is not None

    @property
    def listener_count(self) -> int:
        with self._lock:
            return len(self._listeners)

    def subscribe(
        self,
        callback: AudioCallback,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "float32",
        device: Optional[Union[int, str]] = None,
    ) -> None:
        with self._lock:
            if callback not in self._listeners:
                self._listeners.append(callback)
            if self._stream is not None:
                return
            self.sample_rate = int(sample_rate)
            self.channels = int(channels)
            self.dtype = dtype
            self.device = device
            self._open_stream()

    def unsubscribe(self, callback: AudioCallback) -> None:
        with self._lock:
            try:
                self._listeners.remove(callback)
            except ValueError:
                return
            if not self._listeners:
                self._close_stream()

    def _open_stream(self) -> None:
        blocksize = int(self.sample_rate * 0.05)
        kwargs = dict(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            device=self.device,
            blocksize=blocksize,
            callback=self._audio_callback,
        )
        if self._stream_factory is not None:
            self._stream = self._stream_factory(**kwargs)
        else:
            if sd is None:
                raise RuntimeError("sounddevice is not available. Please install sounddevice.")
            self._stream = sd.InputStream(**kwargs)
        self._stream.start()

    def _close_stream(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    def _audio_callback(self, indata, frames, time_info, status):  # called by PortAudio thread
        if getattr(indata, "ndim", 1) == 2:
            mono = indata[:, 0]
        else:
            mono = indata
        data = np.ascontiguousarray(mono, dtype=np.float32).copy()
        with self._lock:
            listeners = list(self._listeners)
        for cb in listeners:
            try:
                cb(data)
            except Exception:
                pass
        ## END for cb in listeners....


_hub_lock = threading.Lock()
_shared_hub: Optional[MicCaptureHub] = None


def get_shared_mic_hub() -> MicCaptureHub:
    global _shared_hub
    with _hub_lock:
        if _shared_hub is None:
            _shared_hub = MicCaptureHub()
        return _shared_hub


def reset_shared_mic_hub() -> None:
    """Close and drop the process-wide hub (tests)."""
    global _shared_hub
    with _hub_lock:
        hub = _shared_hub
        _shared_hub = None
    if hub is None:
        return
    with hub._lock:
        hub._listeners.clear()
        hub._close_stream()
