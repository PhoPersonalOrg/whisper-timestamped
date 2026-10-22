"""Item delegates for per-row transcription progress in the recordings table."""

from __future__ import annotations

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import QStyledItemDelegate, QStyleOptionViewItem

from scripts.gui.style import PROGRESS_FILL, PROGRESS_TEXT, PROGRESS_TRACK
from scripts.gui.table_model import JOB_PHASE_ROLE, PROGRESS_ROLE


class NameProgressDelegate(QStyledItemDelegate):
    """Paint a progress overlay in the Name cell of the active transcription row."""

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        super().paint(painter, option, index)
        phase = index.data(JOB_PHASE_ROLE)
        if phase not in ("transcribing", "cancelling"):
            return
        progress = index.data(PROGRESS_ROLE)
        rect = option.rect.adjusted(3, 3, -3, -3)
        painter.save()
        painter.setClipRect(option.rect)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(PROGRESS_TRACK)
        painter.drawRoundedRect(rect, 4, 4)
        if isinstance(progress, (int, float)) and progress > 0:
            fill_w = max(4, int(rect.width() * min(1.0, float(progress))))
            fill = QRect(rect.x(), rect.y(), fill_w, rect.height())
            painter.setBrush(PROGRESS_FILL)
            painter.drawRoundedRect(fill, 4, 4)
        elif progress is None:
            fill_w = max(4, int(rect.width() * 0.12))
            fill = QRect(rect.x(), rect.y(), fill_w, rect.height())
            painter.setBrush(PROGRESS_FILL)
            painter.drawRoundedRect(fill, 4, 4)
        ## END if progress known....

        if phase == "cancelling":
            label = "Finishing…"
        elif isinstance(progress, (int, float)):
            label = f"{int(round(float(progress) * 100))}%"
        else:
            label = "Starting…"
        painter.setPen(PROGRESS_TEXT)
        painter.drawText(
            option.rect.adjusted(0, 0, -8, 0),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            label,
        )
        painter.restore()
