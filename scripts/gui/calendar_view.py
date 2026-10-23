"""Scrollable month calendar of recordings as tiny day-dots."""

from __future__ import annotations

from calendar import monthrange
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PyQt6.QtCore import (
    QLocale,
    QPoint,
    QRect,
    QSize,
    Qt,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QLayout,
    QLayoutItem,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from scripts.gui.style import SELECTION_BG, STATUS_COLORS
from scripts.gui.table_model import format_created_display, parse_created_datetime

MonthKey = Tuple[int, int]
DayMap = Dict[date, List["CalendarEntry"]]
DOT_SIZE = 5
DOT_SPACING = 1
CELL_HEIGHT = 44


@dataclass(frozen=True)
class CalendarEntry:
    """One dated recording to plot on the calendar."""

    df_index: int
    dt: datetime
    name: str
    status: str = "pending"
    day_number: int = 0


def make_calendar_entry(
    df_index: int,
    created_value: object,
    name: object,
    status: object = "pending",
) -> Optional[CalendarEntry]:
    """Build an entry, or None when *created_value* is not a real timestamp."""
    dt = parse_created_datetime(created_value)
    if dt is None:
        return None
    return CalendarEntry(
        df_index=int(df_index),
        dt=dt,
        name=str(name or ""),
        status=str(status or "pending") or "pending",
    )


def group_calendar_entries(
    entries: Sequence[CalendarEntry],
) -> "OrderedDict[MonthKey, DayMap]":
    """Group dated entries by month, then day; number recordings within each day.

    Empty months are omitted. Within a day, order is chronological then ``df_index``.
    """
    by_month: Dict[MonthKey, Dict[date, List[CalendarEntry]]] = {}
    ordered = sorted(entries, key=lambda entry: (entry.dt, entry.df_index))
    for entry in ordered:
        month_key = (entry.dt.year, entry.dt.month)
        day = entry.dt.date()
        day_map = by_month.setdefault(month_key, {})
        day_map.setdefault(day, []).append(entry)
    ## END for entry in ordered....

    grouped: "OrderedDict[MonthKey, DayMap]" = OrderedDict()
    for month_key in sorted(by_month):
        numbered_days: DayMap = {}
        for day in sorted(by_month[month_key]):
            numbered: List[CalendarEntry] = []
            for i, entry in enumerate(by_month[month_key][day], start=1):
                numbered.append(replace(entry, day_number=i))
            ## END for i, entry in enumerate....

            numbered_days[day] = numbered
        ## END for day in sorted....

        grouped[month_key] = numbered_days
    ## END for month_key in sorted....

    return grouped


class _FlowLayout(QLayout):
    """Left-to-right wrapping layout for recording dots inside a day cell."""

    def __init__(self, parent: Optional[QWidget] = None, spacing: int = DOT_SPACING) -> None:
        super().__init__(parent)
        self._items: List[QLayoutItem] = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)

    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> Optional[QLayoutItem]:  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int) -> Optional[QLayoutItem]:  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        # Do not report wrapped height — day cells stay a uniform calendar size.
        return False

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        margins = self.contentsMargins()
        return QSize(
            DOT_SIZE + margins.left() + margins.right(),
            DOT_SIZE + margins.top() + margins.bottom(),
        )

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        x = rect.x()
        y = rect.y()
        line_height = 0
        space_x = self.spacing()
        space_y = self.spacing()
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + space_x
            if next_x - space_x > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + space_y
                next_x = x + hint.width() + space_x
                line_height = 0
            ## END if wrap....

            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            ## END if not test_only....

            x = next_x
            line_height = max(line_height, hint.height())
        ## END for item in self._items....

        return y + line_height - rect.y()


class _RecordingDot(QWidget):
    """Tiny painted pin for one recording; avoids QPushButton min-size/padding."""

    clicked_index = pyqtSignal(int)

    def __init__(self, entry: CalendarEntry, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.df_index = entry.df_index
        self._status = entry.status
        self._selected = False
        self._hover = False
        self.setObjectName("calendarDot")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFixedSize(DOT_SIZE, DOT_SIZE)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setToolTip(
            f"#{entry.day_number} {entry.name}\n{format_created_display(entry.dt)}"
        )

    def set_selected(self, selected: bool) -> None:
        self._selected = bool(selected)
        self.setProperty("selected", self._selected)
        self.update()

    def enterEvent(self, event) -> None:  # noqa: N802
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked_index.emit(self.df_index)
            event.accept()
            return
        ## END if left click....

        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        fill = QColor(STATUS_COLORS.get(self._status, STATUS_COLORS["pending"]))
        if self._selected:
            border = QColor(SELECTION_BG)
        elif self._hover:
            border = QColor("#9b80fd")
        else:
            border = fill
        ## END if selected/hover....

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(fill)
        painter.setPen(QPen(border, 1))
        inset = 0.5
        painter.drawEllipse(
            QRect(
                int(inset),
                int(inset),
                DOT_SIZE - 1,
                DOT_SIZE - 1,
            )
        )
        painter.end()


class _DayCell(QWidget):
    """One in-month day: date number plus a compact cluster of pin-dots."""

    def __init__(
        self,
        day: date,
        entries: Sequence[CalendarEntry],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("calendarDayCell")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(24, CELL_HEIGHT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 2)
        layout.setSpacing(1)

        number = QLabel(str(day.day))
        number.setObjectName("calendarDayNumber")
        layout.addWidget(number)

        dots_host = QWidget()
        dots_host.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        dots_layout = _FlowLayout(dots_host, spacing=DOT_SPACING)
        dots_host.setLayout(dots_layout)
        self.dots: List[_RecordingDot] = []
        for entry in entries:
            dot = _RecordingDot(entry, dots_host)
            dots_layout.addWidget(dot)
            self.dots.append(dot)
        ## END for entry in entries....

        layout.addWidget(dots_host, stretch=1)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(32, CELL_HEIGHT)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(24, CELL_HEIGHT)


class _MonthGrid(QWidget):
    """One month of day cells."""

    def __init__(
        self,
        year: int,
        month: int,
        day_map: DayMap,
        first_day_of_week: Qt.DayOfWeek,
        weekday_labels: Sequence[str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("calendarMonth")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.setSpacing(4)

        header = QLabel(datetime(year, month, 1).strftime("%B %Y"))
        header.setObjectName("calendarMonthHeader")
        layout.addWidget(header)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(3)
        grid.setVerticalSpacing(3)
        for col, label_text in enumerate(weekday_labels):
            label = QLabel(label_text)
            label.setObjectName("calendarWeekday")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(label, 0, col)
        ## END for col, label_text in enumerate....

        first_column = _column_for_date(date(year, month, 1), first_day_of_week)
        days_in_month = monthrange(year, month)[1]
        for day_num in range(1, days_in_month + 1):
            day = date(year, month, day_num)
            row = 1 + (first_column + day_num - 1) // 7
            col = (first_column + day_num - 1) % 7
            cell = _DayCell(day, day_map.get(day, []), self)
            grid.addWidget(cell, row, col)
        ## END for day_num in range....

        for col in range(7):
            grid.setColumnStretch(col, 1)
        ## END for col in range....

        for row in range(1, 7):
            grid.setRowMinimumHeight(row, CELL_HEIGHT)
            grid.setRowStretch(row, 1)
        ## END for row in range....

        layout.addLayout(grid)


def _qt_weekday_number(day: Qt.DayOfWeek) -> int:
    """Qt weekday as 1=Monday … 7=Sunday."""
    value = getattr(day, "value", day)
    return int(value)


def _column_for_date(day: date, first_day_of_week: Qt.DayOfWeek) -> int:
    """0-6 column index for *day* given locale first day of week (Qt 1=Mon … 7=Sun)."""
    qt_weekday = day.weekday() + 1
    return (qt_weekday - _qt_weekday_number(first_day_of_week)) % 7


def _weekday_labels(locale: QLocale, first_day_of_week: Qt.DayOfWeek) -> List[str]:
    labels: List[str] = []
    first = _qt_weekday_number(first_day_of_week)
    for offset in range(7):
        qt_day = ((first - 1 + offset) % 7) + 1
        name = locale.standaloneDayName(qt_day, QLocale.FormatType.ShortFormat)
        labels.append(name[:2] if len(name) > 2 else name)
    ## END for offset in range....

    return labels


class RecordingCalendarWidget(QScrollArea):
    """Scrollable stack of month grids with tiny recording dots."""

    dot_clicked = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("recordingCalendar")
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setMinimumHeight(160)

        self._dots: Dict[int, _RecordingDot] = {}
        self._selected: List[int] = []
        self._did_scroll = False
        self._locale = QLocale.system()
        self._first_day = self._locale.firstDayOfWeek()
        self._weekday_labels = _weekday_labels(self._locale, self._first_day)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(6, 6, 6, 6)
        self._body_layout.setSpacing(4)
        self._placeholder = QLabel("No dated recordings in the current view.")
        self._placeholder.setObjectName("dimLabel")
        self._placeholder.setWordWrap(True)
        self._body_layout.addWidget(self._placeholder)
        self._body_layout.addStretch(1)
        self.setWidget(self._body)

    def set_entries(self, entries: Sequence[CalendarEntry]) -> None:
        """Rebuild month grids from *entries*. Does not scroll."""
        previous_selected = list(self._selected)
        self._clear_months()
        grouped = group_calendar_entries(entries)
        if not grouped:
            self._placeholder.show()
            self._selected = []
            return
        ## END if not grouped....

        self._placeholder.hide()
        for (year, month), day_map in grouped.items():
            grid = _MonthGrid(
                year,
                month,
                day_map,
                self._first_day,
                self._weekday_labels,
                self._body,
            )
            for cell in grid.findChildren(_DayCell):
                for dot in cell.dots:
                    dot.clicked_index.connect(self.dot_clicked.emit)
                    self._dots[dot.df_index] = dot
                ## END for dot in cell.dots....

            ## END for cell in grid.findChildren....

            self._body_layout.insertWidget(self._body_layout.count() - 1, grid)
        ## END for (year, month), day_map in grouped.items()....

        self.set_selected(previous_selected, scroll=False)

    def set_selected(self, indices: Iterable[int], scroll: bool = False) -> None:
        """Highlight dots for *indices*. Scroll only when *scroll* and exactly one index."""
        unique: List[int] = []
        seen = set()
        for df_index in indices:
            key = int(df_index)
            if key in seen:
                continue
            seen.add(key)
            unique.append(key)
        ## END for df_index in indices....

        self._selected = unique
        selected_set = set(unique)
        for df_index, dot in self._dots.items():
            dot.set_selected(df_index in selected_set)
        ## END for df_index, dot in self._dots.items()....

        self._did_scroll = False
        if scroll and len(unique) == 1:
            self._scroll_to_dot(unique[0])
        ## END if scroll and single....

    def selected_indices(self) -> List[int]:
        return list(self._selected)

    def _scroll_to_dot(self, df_index: int) -> None:
        dot = self._dots.get(df_index)
        if dot is None:
            return
        self._did_scroll = True
        self.ensureWidgetVisible(dot, 8, 8)

    def _clear_months(self) -> None:
        self._dots = {}
        # Keep placeholder (index 0) and trailing stretch; remove month grids.
        for i in range(self._body_layout.count() - 2, 0, -1):
            item = self._body_layout.takeAt(i)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            ## END if widget is not None....

        ## END for i in range....
