"""Compact DAW-style waveform overview for the playback dock.

Peak extraction opens its own SoundFile so it never moves the live playback
read pointer. The widget paints a downsampled min/max envelope and emits
seek requests on click/drag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import soundfile as sf
from PyQt6.QtCore import QPointF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QSizePolicy, QWidget

from scripts.gui.style import _ACCENT, _BG_WIDGET, _FG, _FG_DIM

DEFAULT_N_BINS = 4096
CHUNK_FRAMES = 65536
WAVEFORM_HEIGHT = 72


def downsample_peaks(
    path: Path,
    n_bins: int = DEFAULT_N_BINS,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Read *path* in chunks and return per-bin min/max envelopes.

    Mixes channels to mono. Never loads the whole file into memory.
    """
    with sf.SoundFile(str(path)) as sfile:
        samplerate = int(sfile.samplerate)
        total_frames = int(sfile.frames)
        if samplerate <= 0 or total_frames <= 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), 0.0

        duration = total_frames / float(samplerate)
        n_bins = max(1, min(int(n_bins), total_frames))
        mins = np.full(n_bins, np.inf, dtype=np.float32)
        maxs = np.full(n_bins, -np.inf, dtype=np.float32)
        frames_per_bin = total_frames / n_bins
        frame_index = 0

        while True:
            if cancel_check is not None and cancel_check():
                raise InterruptedError("waveform peak extraction cancelled")
            data = sfile.read(CHUNK_FRAMES, dtype="float32", always_2d=True)
            if len(data) == 0:
                break
            mono = data.mean(axis=1)
            n = len(mono)
            indices = np.floor((np.arange(n) + frame_index) / frames_per_bin).astype(
                np.int64
            )
            indices = np.clip(indices, 0, n_bins - 1)
            np.minimum.at(mins, indices, mono)
            np.maximum.at(maxs, indices, mono)
            frame_index += n
        ## END while reading chunks....

    empty = ~np.isfinite(mins)
    mins[empty] = 0.0
    maxs[empty] = 0.0
    return mins, maxs, duration


class WaveformWidget(QWidget):
    """Overview waveform with a playhead; click or drag to seek."""

    seek_requested = pyqtSignal(float)
    seek_pressed = pyqtSignal()
    seek_released = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._mins: Optional[np.ndarray] = None
        self._maxs: Optional[np.ndarray] = None
        self._duration: float = 0.0
        self._position: float = 0.0
        self._loading: bool = False
        self._dragging: bool = False

        self.setFixedHeight(WAVEFORM_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def sizeHint(self) -> QSize:
        return QSize(400, WAVEFORM_HEIGHT)

    def minimumSizeHint(self) -> QSize:
        return QSize(100, WAVEFORM_HEIGHT)

    def set_peaks(
        self,
        mins: np.ndarray,
        maxs: np.ndarray,
        duration: float,
    ) -> None:
        self._mins = np.asarray(mins, dtype=np.float32)
        self._maxs = np.asarray(maxs, dtype=np.float32)
        self._duration = max(0.0, float(duration))
        self._loading = False
        self.update()

    def set_position(self, seconds: float) -> None:
        if self._duration > 0:
            seconds = min(max(0.0, float(seconds)), self._duration)
        else:
            seconds = 0.0
        ## END if duration....

        if abs(seconds - self._position) < 1e-4:
            return
        self._position = seconds
        self.update()

    def set_loading(self) -> None:
        self._mins = None
        self._maxs = None
        self._duration = 0.0
        self._position = 0.0
        self._loading = True
        self.update()

    def clear(self) -> None:
        self._mins = None
        self._maxs = None
        self._duration = 0.0
        self._position = 0.0
        self._loading = False
        self.update()

    def set_placeholder(self, duration: float) -> None:
        """Show an empty envelope so the playhead can still scrub."""
        self._mins = np.zeros(2, dtype=np.float32)
        self._maxs = np.zeros(2, dtype=np.float32)
        self._duration = max(0.0, float(duration))
        self._loading = False
        self.update()

    def seconds_at_x(self, x: float) -> float:
        """Map a widget x coordinate to seconds along the current duration."""
        if self._duration <= 0:
            return 0.0
        width = max(1, self.width())
        frac = min(1.0, max(0.0, float(x) / width))
        return frac * self._duration

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._duration <= 0:
            return
        self._dragging = True
        self.seek_pressed.emit()
        self._emit_seek(event.position().x())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self._dragging:
            return
        self._emit_seek(event.position().x())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self._dragging:
            return
        self._dragging = False
        self._emit_seek(event.position().x())
        self.seek_released.emit()

    def paintEvent(self, event) -> None:  # noqa: ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(_BG_WIDGET))

        if self._loading:
            self._draw_message(painter, "Loading waveform…")
            painter.end()
            return
        ## END if loading....

        if self._mins is None or self._maxs is None or len(self._mins) == 0:
            self._draw_message(painter, "Select a recording")
            painter.end()
            return
        ## END if no peaks....

        width = max(1, self.width())
        height = max(1, self.height())
        center = height / 2.0
        mins = self._mins
        maxs = self._maxs
        n = len(mins)
        x_scale = (width - 1) / max(n - 1, 1)

        top: list[QPointF] = []
        bottom: list[QPointF] = []
        for i in range(n):
            x = i * x_scale
            peak = float(np.clip(maxs[i], -1.0, 1.0))
            trough = float(np.clip(mins[i], -1.0, 1.0))
            top.append(QPointF(x, center - peak * center))
            bottom.append(QPointF(x, center - trough * center))
        ## END for i in range(n)....

        polygon = QPolygonF(top + list(reversed(bottom)))
        accent = QColor(_ACCENT)
        accent.setAlpha(180)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(accent)
        painter.drawPolygon(polygon)

        if self._duration > 0 and self._position > 0:
            played_w = int(round((self._position / self._duration) * width))
            overlay = QColor(_ACCENT)
            overlay.setAlpha(40)
            painter.fillRect(0, 0, played_w, height, overlay)
        ## END if played region....

        if self._duration > 0:
            x = (self._position / self._duration) * (width - 1)
            painter.setPen(QPen(QColor(_FG), 1))
            painter.drawLine(QPointF(x, 0), QPointF(x, height))
        ## END if playhead....

        painter.end()

    def _emit_seek(self, x: float) -> None:
        seconds = self.seconds_at_x(x)
        self.set_position(seconds)
        self.seek_requested.emit(seconds)

    def _draw_message(self, painter: QPainter, text: str) -> None:
        painter.setPen(QColor(_FG_DIM))
        font = QFont(self.font())
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, text)
