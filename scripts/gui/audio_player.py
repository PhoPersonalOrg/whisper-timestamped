"""Non-blocking audio playback using sounddevice + soundfile.

Supports .m4a and .caf via libsndfile.  When libsndfile cannot decode a
file (older AAC-LC m4a on some builds), automatically converts to WAV via
ffmpeg into a temp file.
"""

from __future__ import annotations

import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from PyQt6.QtCore import QObject, QTimer, pyqtSignal


class AudioPlayer(QObject):
    """Lightweight audio player with play/pause/stop/seek and volume."""

    position_changed = pyqtSignal(float)   # current position in seconds
    duration_changed = pyqtSignal(float)   # total duration in seconds
    state_changed = pyqtSignal(str)        # "playing" / "paused" / "stopped"
    error = pyqtSignal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._stream: Optional[sd.OutputStream] = None
        self._sf: Optional[sf.SoundFile] = None
        self._lock = threading.Lock()
        self._volume: float = 1.0
        self._position_frames: int = 0
        self._total_frames: int = 0
        self._samplerate: int = 44100
        self._channels: int = 1
        self._state: str = "stopped"
        self._loaded_path: Optional[Path] = None
        self._temp_wav: Optional[Path] = None

        # Timer to emit position updates while playing
        self._timer = QTimer(self)
        self._timer.setInterval(100)  # 10 Hz updates
        self._timer.timeout.connect(self._emit_position)


    def load(self, path: Path) -> None:
        """Load an audio file for playback."""
        self.stop()
        self._cleanup_temp()

        resolved = Path(path)
        # Try opening directly with soundfile
        try:
            sfile = sf.SoundFile(str(resolved))
        except (sf.LibsndfileError, RuntimeError):
            # libsndfile can't decode — try ffmpeg conversion to WAV
            resolved = self._convert_to_wav(resolved)
            if resolved is None:
                return
            try:
                sfile = sf.SoundFile(str(resolved))
            except Exception as exc:
                self.error.emit(f"Cannot open audio: {exc}")
                return
        ## END try open soundfile...

        self._sf = sfile
        self._samplerate = sfile.samplerate
        self._channels = sfile.channels
        self._total_frames = sfile.frames
        self._position_frames = 0
        self._loaded_path = Path(path)

        duration = self._total_frames / self._samplerate
        self.duration_changed.emit(duration)
        self._set_state("stopped")


    def play(self) -> None:
        """Start or resume playback."""
        if self._sf is None:
            return
        if self._state == "playing":
            return

        if self._stream is not None:
            # Resume from pause
            self._stream.start()
            self._set_state("playing")
            self._timer.start()
            return
        ## END if resuming...

        # Start fresh stream
        try:
            self._stream = sd.OutputStream(
                samplerate=self._samplerate,
                channels=self._channels,
                callback=self._audio_callback,
                finished_callback=self._on_stream_finished,
                blocksize=4096,
            )
            self._stream.start()
            self._set_state("playing")
            self._timer.start()
        except Exception as exc:
            self.error.emit(f"Playback error: {exc}")


    def pause(self) -> None:
        """Pause playback."""
        if self._stream is not None and self._state == "playing":
            self._stream.stop()
            self._set_state("paused")
            self._timer.stop()


    def stop(self) -> None:
        """Stop playback and reset position."""
        self._timer.stop()
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        ## END if stream...

        if self._sf is not None:
            with self._lock:
                self._sf.seek(0)
                self._position_frames = 0
        ## END if sf...

        self._set_state("stopped")
        self.position_changed.emit(0.0)


    def seek(self, seconds: float) -> None:
        """Seek to a position in seconds."""
        if self._sf is None:
            return
        target_frame = int(seconds * self._samplerate)
        target_frame = max(0, min(target_frame, self._total_frames))
        with self._lock:
            self._sf.seek(target_frame)
            self._position_frames = target_frame
        self.position_changed.emit(seconds)


    def set_volume(self, volume: float) -> None:
        """Set volume (0.0 to 1.0)."""
        self._volume = max(0.0, min(1.0, volume))


    @property
    def current_path(self) -> Optional[Path]:
        return self._loaded_path


    @property
    def duration(self) -> float:
        if self._total_frames > 0 and self._samplerate > 0:
            return self._total_frames / self._samplerate
        return 0.0


    @property
    def state(self) -> str:
        return self._state


    def cleanup(self) -> None:
        """Release all resources. Call on application exit."""
        self.stop()
        if self._sf is not None:
            self._sf.close()
            self._sf = None
        self._cleanup_temp()


    # -- Private ---------------------------------------------------------------

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        """Sounddevice output callback — runs on the audio thread."""
        with self._lock:
            if self._sf is None:
                outdata[:] = 0
                raise sd.CallbackStop
            data = self._sf.read(frames, dtype="float32")
        ## END with lock...

        if len(data) == 0:
            outdata[:] = 0
            raise sd.CallbackStop

        # Mono → multi-channel expansion if needed
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if len(data) < frames:
            outdata[:len(data)] = data * self._volume
            outdata[len(data):] = 0
            self._position_frames += len(data)
            raise sd.CallbackStop
        else:
            outdata[:] = data * self._volume
            self._position_frames += frames


    def _on_stream_finished(self) -> None:
        """Called when the stream ends (track finished)."""
        # Use QTimer.singleShot so we're back on the GUI thread
        QTimer.singleShot(0, self._handle_finished)


    def _handle_finished(self) -> None:
        """Handle end-of-track on the GUI thread."""
        if self._state == "playing":
            self.stop()


    def _emit_position(self) -> None:
        """Emit current position for the progress slider."""
        if self._samplerate > 0:
            pos = self._position_frames / self._samplerate
            self.position_changed.emit(pos)


    def _set_state(self, state: str) -> None:
        self._state = state
        self.state_changed.emit(state)


    def _convert_to_wav(self, source: Path) -> Optional[Path]:
        """Convert audio to WAV via ffmpeg. Returns temp WAV path or None."""
        try:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False, prefix="arm_playback_"
            )
            tmp.close()
            self._temp_wav = Path(tmp.name)
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(source),
                    "-acodec", "pcm_s16le", "-ar", "44100",
                    str(self._temp_wav),
                ],
                capture_output=True,
                check=True,
                timeout=120,
            )
            return self._temp_wav
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
            self.error.emit(f"ffmpeg conversion failed for {source.name}: {exc}")
            self._cleanup_temp()
            return None


    def _cleanup_temp(self) -> None:
        """Remove any temporary WAV file."""
        if self._temp_wav is not None:
            try:
                self._temp_wav.unlink(missing_ok=True)
            except OSError:
                pass
            self._temp_wav = None
