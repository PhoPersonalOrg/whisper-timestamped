"""Unit tests for the recordings month calendar grouping and selection scroll."""

from __future__ import annotations

import sys
import unittest
from datetime import date, datetime
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QWidget

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.gui.calendar_view import (
    CalendarEntry,
    RecordingCalendarWidget,
    group_calendar_entries,
    make_calendar_entry,
)


def _app() -> QApplication:
    existing = QApplication.instance()
    if existing is not None:
        return existing
    return QApplication([])


def _entry(
    df_index: int,
    when: str,
    name: str,
    status: str = "pending",
) -> CalendarEntry:
    entry = make_calendar_entry(df_index, when, name, status)
    assert entry is not None, f"expected dated entry for {when!r}"
    return entry


class TestGroupCalendarEntries(unittest.TestCase):
    def test_undated_entries_are_dropped_by_factory(self):
        self.assertIsNone(make_calendar_entry(0, "", "empty"))
        self.assertIsNone(make_calendar_entry(1, "not-a-date", "bad"))
        self.assertIsNone(make_calendar_entry(2, None, "none"))

    def test_groups_by_month_numbers_same_day_and_omits_empty_months(self):
        entries = [
            _entry(0, "2026-10-20 09:00:00", "later", "pending"),
            _entry(1, "2026-10-20 08:00:00", "earlier", "transcribed"),
            _entry(2, "2026-09-01 12:00:00", "september", "pending"),
            _entry(3, "2026-10-21 07:00:00", "next-day", "queued"),
        ]
        grouped = group_calendar_entries(entries)
        self.assertEqual(list(grouped.keys()), [(2026, 9), (2026, 10)])
        self.assertNotIn((2026, 8), grouped)

        oct_days = grouped[(2026, 10)]
        self.assertNotIn(date(2026, 10, 22), oct_days)
        same_day = oct_days[date(2026, 10, 20)]
        self.assertEqual([e.name for e in same_day], ["earlier", "later"])
        self.assertEqual([e.day_number for e in same_day], [1, 2])
        self.assertEqual(same_day[0].df_index, 1)
        self.assertEqual(oct_days[date(2026, 10, 21)][0].day_number, 1)
        self.assertEqual(len(grouped[(2026, 9)][date(2026, 9, 1)]), 1)

    def test_same_timestamp_breaks_ties_by_df_index(self):
        entries = [
            _entry(5, "2026-10-20 08:00:00", "b"),
            _entry(2, "2026-10-20 08:00:00", "a"),
        ]
        same_day = group_calendar_entries(entries)[(2026, 10)][date(2026, 10, 20)]
        self.assertEqual([e.df_index for e in same_day], [2, 5])
        self.assertEqual([e.day_number for e in same_day], [1, 2])


class TestRecordingCalendarSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def _calendar(self) -> RecordingCalendarWidget:
        widget = RecordingCalendarWidget()
        widget.resize(360, 220)
        widget.set_entries(
            [
                _entry(10, "2026-09-01 12:00:00", "sep", "pending"),
                _entry(20, "2026-10-20 08:00:00", "oct-a", "transcribed"),
                _entry(21, "2026-10-20 09:00:00", "oct-b", "queued"),
            ]
        )
        return widget

    def test_single_select_scrolls_to_dot(self):
        widget = self._calendar()
        visible: list[QWidget] = []

        def _spy(child: QWidget, xmargin: int = 0, ymargin: int = 0) -> None:
            visible.append(child)

        widget.ensureWidgetVisible = _spy  # type: ignore[method-assign]
        widget.set_selected([21], scroll=True)
        self.assertEqual(widget.selected_indices(), [21])
        self.assertTrue(widget._did_scroll)
        self.assertEqual(len(visible), 1)
        self.assertEqual(getattr(visible[0], "df_index", None), 21)

    def test_multi_select_highlights_without_scrolling(self):
        widget = self._calendar()
        visible: list[QWidget] = []

        def _spy(child: QWidget, xmargin: int = 0, ymargin: int = 0) -> None:
            visible.append(child)

        widget.ensureWidgetVisible = _spy  # type: ignore[method-assign]
        widget.set_selected([10, 20, 21], scroll=False)
        self.assertEqual(widget.selected_indices(), [10, 20, 21])
        self.assertFalse(widget._did_scroll)
        self.assertEqual(visible, [])

        widget.set_selected([10, 21], scroll=True)
        self.assertFalse(widget._did_scroll)
        self.assertEqual(visible, [])

        selected_dots = [
            dot.df_index
            for df_index, dot in widget._dots.items()
            if dot.property("selected")
        ]
        self.assertEqual(sorted(selected_dots), [10, 21])


if __name__ == "__main__":
    unittest.main()
