"""Unit tests for runtime job-state overlay on the recordings table model."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.gui.table_model import (
    DISPLAY_COLUMNS,
    JOB_PHASE_ROLE,
    PENDING_LIKE_STATUSES,
    PROGRESS_ROLE,
    RecordingFilterProxy,
    RecordingTableModel,
    compute_status,
    format_created_ago,
    format_created_display,
)


def _app() -> QApplication:
    existing = QApplication.instance()
    if existing is not None:
        return existing
    return QApplication([])


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["a.mkv", "b.mkv", "c.mkv"],
            "full_path": ["/tmp/a.mkv", "/tmp/b.mkv", "/tmp/c.mkv"],
            "format_id": ["rec", "rec", "rec"],
            "title": ["a", "b", "c"],
            "size_mb": [1.0, 2.0, 3.0],
            "duration_hms": ["00:01:00", "00:02:00", "00:03:00"],
            "extracted_creation_time": ["", "", ""],
            "is_duplicate": [False, False, False],
            "transcript_txt": ["", "", "/tmp/c.txt"],
        }
    )


class TestJobStateOverlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def test_queued_wins_over_pending_compute_status(self):
        model = RecordingTableModel()
        df = _sample_df()
        model.set_dataframe(df)
        self.assertEqual(compute_status(df.iloc[0]), "pending")
        self.assertEqual(model.get_row_data(0)["status"], "pending")
        model.set_job_state(0, "queued", position=1, total=2)
        self.assertEqual(model.get_row_data(0)["status"], "queued")
        self.assertEqual(model.data(model.index(0, 0), JOB_PHASE_ROLE), "queued")

    def test_transcribing_progress_role(self):
        model = RecordingTableModel()
        model.set_dataframe(_sample_df())
        model.set_job_state(1, "transcribing", progress=0.42, position=2, total=3)
        name_col = list(model._columns).index("name")
        self.assertEqual(model.data(model.index(1, 0), JOB_PHASE_ROLE), "transcribing")
        self.assertAlmostEqual(model.data(model.index(1, name_col), PROGRESS_ROLE), 0.42)

    def test_transcript_paths_clear_job_state_to_transcribed(self):
        model = RecordingTableModel()
        model.set_dataframe(_sample_df())
        model.set_job_state(0, "transcribing", progress=0.5)
        model.update_row(0, {"transcript_txt": "/tmp/a.txt"})
        self.assertIsNone(model.job_state(0))
        self.assertEqual(model.get_row_data(0)["status"], "transcribed")

    def test_update_row_without_transcripts_keeps_queued(self):
        model = RecordingTableModel()
        model.set_dataframe(_sample_df())
        model.set_job_state(0, "queued", position=1, total=1)
        model.update_row(0, {"title": "new title"})
        self.assertEqual(model.job_state(0)["phase"], "queued")
        self.assertEqual(model.get_row_data(0)["status"], "queued")

    def test_pending_filter_includes_queued_and_transcribing(self):
        model = RecordingTableModel()
        model.set_dataframe(_sample_df())
        model.set_job_state(0, "queued")
        model.set_job_state(1, "transcribing", progress=0.1)
        proxy = RecordingFilterProxy()
        proxy.setSourceModel(model)
        proxy.set_status_filter("pending")
        statuses = {
            proxy.data(proxy.index(row, 0), JOB_PHASE_ROLE)
            or model.get_row_data(proxy.mapToSource(proxy.index(row, 0)).row())["status"]
            for row in range(proxy.rowCount())
        }
        self.assertIn("queued", PENDING_LIKE_STATUSES)
        self.assertGreaterEqual(proxy.rowCount(), 2)
        self.assertTrue("queued" in statuses or "transcribing" in statuses)


def _created_sort_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["old", "empty", "newest", "bad", "mid"],
            "full_path": [
                "/tmp/old.mkv",
                "/tmp/empty.mkv",
                "/tmp/newest.mkv",
                "/tmp/bad.mkv",
                "/tmp/mid.mkv",
            ],
            "format_id": ["rec"] * 5,
            "title": ["old", "empty", "newest", "bad", "mid"],
            "size_mb": [1.0, 2.0, 3.0, 4.0, 5.0],
            "duration_hms": ["00:01:00"] * 5,
            "extracted_creation_time": [
                "2024-01-01 00:00:00",
                "",
                "2026-10-20 07:14:00",
                "not-a-date",
                "2025-06-15 12:00:00",
            ],
            "is_duplicate": [False] * 5,
            "transcript_txt": [""] * 5,
        }
    )


def _proxy_names(proxy: RecordingFilterProxy) -> list[str]:
    name_col = DISPLAY_COLUMNS.index("name")
    return [
        str(proxy.data(proxy.index(row, name_col)) or "")
        for row in range(proxy.rowCount())
    ]


class TestCreatedColumnSort(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def _sorted_proxy(self, order: Qt.SortOrder) -> RecordingFilterProxy:
        model = RecordingTableModel()
        model.set_dataframe(_created_sort_df())
        proxy = RecordingFilterProxy()
        proxy.setSourceModel(model)
        created_col = DISPLAY_COLUMNS.index("extracted_creation_time")
        proxy.sort(created_col, order)
        return proxy

    def test_descending_puts_newest_first_and_empty_last(self):
        names = _proxy_names(self._sorted_proxy(Qt.SortOrder.DescendingOrder))
        self.assertEqual(names[:3], ["newest", "mid", "old"])
        self.assertEqual(set(names[3:]), {"empty", "bad"})

    def test_ascending_puts_oldest_first_and_empty_last(self):
        names = _proxy_names(self._sorted_proxy(Qt.SortOrder.AscendingOrder))
        self.assertEqual(names[:3], ["old", "mid", "newest"])
        self.assertEqual(set(names[3:]), {"empty", "bad"})


class TestFormatCreatedDisplay(unittest.TestCase):
    def test_iso_date_weekday_and_12h_time(self):
        self.assertEqual(
            format_created_display("2026-10-07 16:56:00"),
            "2026-10-07 (Wed) 4:56pm",
        )

    def test_empty_and_unparseable_are_blank(self):
        self.assertEqual(format_created_display(""), "")
        self.assertEqual(format_created_display("not-a-date"), "")


class TestFormatCreatedAgo(unittest.TestCase):
    # Thursday, Oct 22 2026, 11:00 local
    NOW = datetime(2026, 10, 22, 11, 0, 0)

    def _ago(self, value: str) -> str:
        return format_created_ago(value, now=self.NOW)

    def test_empty_and_unparseable_are_blank(self):
        self.assertEqual(self._ago(""), "")
        self.assertEqual(self._ago("not-a-date"), "")
        self.assertEqual(format_created_ago(None, now=self.NOW), "")

    def test_under_one_hour_and_future_skew(self):
        self.assertEqual(self._ago("2026-10-22 10:30:00"), "<1 hr ago")
        self.assertEqual(self._ago("2026-10-22 11:10:00"), "<1 hr ago")

    def test_hours(self):
        self.assertEqual(self._ago("2026-10-22 08:50:00"), "~2 hr ago")
        self.assertEqual(self._ago("2026-10-21 12:00:00"), "~23 hr ago")

    def test_days_with_weekday(self):
        self.assertEqual(self._ago("2026-10-21 08:00:00"), "1 day ago (Wed)")
        self.assertEqual(self._ago("2026-10-20 10:00:00"), "2 days ago (Tue)")
        self.assertEqual(self._ago("2026-10-16 10:00:00"), "6 days ago (Fri)")

    def test_last_weekday(self):
        self.assertEqual(self._ago("2026-10-15 10:00:00"), "Last Thu")
        self.assertEqual(self._ago("2026-10-09 10:00:00"), "Last Fri")

    def test_weeks_months_years(self):
        self.assertEqual(self._ago("2026-10-08 10:00:00"), "2 weeks ago")
        self.assertEqual(self._ago("2026-09-01 10:00:00"), "7 weeks ago")
        self.assertEqual(self._ago("2026-08-20 10:00:00"), "2 months ago")
        self.assertEqual(self._ago("2025-10-22 10:00:00"), "12 months ago")
        self.assertEqual(self._ago("2024-10-22 10:00:00"), "2 years ago")


class TestAgoColumn(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def test_ago_sits_next_to_created(self):
        created_col = DISPLAY_COLUMNS.index("extracted_creation_time")
        self.assertEqual(DISPLAY_COLUMNS[created_col + 1], "ago")

    def test_ago_is_not_materialized_in_dataframe(self):
        model = RecordingTableModel()
        model.set_dataframe(_created_sort_df())
        self.assertNotIn("ago", model.get_dataframe().columns)
        model.upsert_row({"name": "x", "full_path": "/tmp/x.mkv"})
        self.assertNotIn("ago", model.get_dataframe().columns)

    def test_ago_display_and_sort_match_created(self):
        model = RecordingTableModel()
        model.set_dataframe(_created_sort_df())
        proxy = RecordingFilterProxy()
        proxy.setSourceModel(model)
        ago_col = DISPLAY_COLUMNS.index("ago")
        proxy.sort(ago_col, Qt.SortOrder.DescendingOrder)
        names = _proxy_names(proxy)
        self.assertEqual(names[:3], ["newest", "mid", "old"])
        self.assertEqual(set(names[3:]), {"empty", "bad"})
        # Rows with a timestamp render a relative label; blanks stay blank.
        top_label = str(proxy.data(proxy.index(0, ago_col)) or "")
        self.assertTrue("ago" in top_label or top_label.startswith("Last"), top_label)
        self.assertEqual(proxy.data(proxy.index(3, ago_col)) or "", "")
        self.assertEqual(proxy.data(proxy.index(4, ago_col)) or "", "")


if __name__ == "__main__":
    unittest.main()
