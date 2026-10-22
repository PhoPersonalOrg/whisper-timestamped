"""Unit tests for runtime job-state overlay on the recordings table model."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd
from PyQt6.QtWidgets import QApplication

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.gui.table_model import (
    JOB_PHASE_ROLE,
    PENDING_LIKE_STATUSES,
    PROGRESS_ROLE,
    RecordingFilterProxy,
    RecordingTableModel,
    compute_status,
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


if __name__ == "__main__":
    unittest.main()
