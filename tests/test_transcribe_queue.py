"""Unit tests for torch-free transcription queue helpers and tqdm progress."""

from __future__ import annotations

import io
import sys
import types
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm as TqdmClass

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.transcribe_queue import (
    ARM_DF_INDEX_COL,
    callback_row_index,
    coerce_row_index,
    index_in_skip_set,
    job_loop_action,
)
from whisper_timestamped.crisperwhisper.longform.progress import (
    audio_progress_bar,
    audio_progress_hook,
    set_audio_progress_callback,
)


class TestCallbackRowIndex(unittest.TestCase):
    def test_prefers_arm_df_index_column(self):
        row = pd.Series({ARM_DF_INDEX_COL: 771, "name": "CAM.mkv"})
        self.assertEqual(callback_row_index(0, row), 771)

    def test_falls_back_to_csv_index(self):
        row = pd.Series({"name": "CAM.mkv"})
        self.assertEqual(callback_row_index(3, row), 3)

    def test_coerce_none_and_nan(self):
        self.assertIsNone(coerce_row_index(None))
        self.assertIsNone(coerce_row_index(float("nan")))
        self.assertEqual(coerce_row_index("12"), 12)


class TestJobLoopAction(unittest.TestCase):
    def test_skip_indices_skips_matching_job_not_others(self):
        skip = {20}
        started = []
        skipped = []
        for row_index in (10, 20, 30):
            action = job_loop_action(False, skip, row_index)
            if action == "skip":
                skipped.append(row_index)
            elif action == "start":
                started.append(row_index)
            else:
                self.fail(f"unexpected stop for {row_index}")
            ## END if action....

        ## END for row_index in (10, 20, 30)....

        self.assertEqual(started, [10, 30])
        self.assertEqual(skipped, [20])

    def test_stop_after_current_does_not_start_later_jobs(self):
        started = []
        leftover = []
        stop = False
        for row_index in (10, 20, 30):
            action = job_loop_action(stop, set(), row_index)
            if action == "stop":
                leftover.append(row_index)
                leftover.extend([40])  # sentinel not used
                break
            started.append(row_index)
            if row_index == 10:
                stop = True
            ## END if row_index == 10....

        ## END for row_index in (10, 20, 30)....

        self.assertEqual(started, [10])
        self.assertEqual(leftover[0], 20)

    def test_skip_set_int_coercion(self):
        self.assertTrue(index_in_skip_set({5}, 5))
        self.assertTrue(index_in_skip_set({5}, np.int64(5)))
        self.assertFalse(index_in_skip_set({5}, 6))


class TestAudioProgressCallback(unittest.TestCase):
    def test_tqdm_wrapper_reports_fraction(self):
        seen = []
        set_audio_progress_callback(seen.append)
        try:
            audio = np.zeros(16000, dtype=np.float32)  # 1s → 100 frames
            with redirect_stderr(io.StringIO()):
                with audio_progress_bar(audio, verbose=False) as pbar:
                    self.assertEqual(pbar.total, 100)
                    pbar.update(25)
                    pbar.update(25)
                ## END with audio_progress_bar....

            ## END with redirect_stderr....

        finally:
            set_audio_progress_callback(None)
        self.assertIn(0.25, seen)
        self.assertEqual(seen[-1], 0.5)
        self.assertNotIn(0.0, seen)

    def test_openai_whisper_tqdm_hook_reports_fraction(self):
        fake_tqdm_ns = types.SimpleNamespace(tqdm=TqdmClass)
        fake_mod = types.ModuleType("whisper.transcribe")
        fake_mod.tqdm = fake_tqdm_ns
        prev = sys.modules.get("whisper.transcribe")
        sys.modules["whisper.transcribe"] = fake_mod
        seen = []
        try:
            with audio_progress_hook(seen.append):
                self.assertEqual(fake_mod.tqdm.tqdm.__name__, "_CallbackTqdm")
                with redirect_stderr(io.StringIO()):
                    pbar = fake_mod.tqdm.tqdm(total=100, unit="frames", disable=False)
                    pbar.update(40)
                    pbar.close()
                ## END with redirect_stderr....

            ## END with audio_progress_hook....

            self.assertIn(0.4, seen)
            self.assertIs(fake_mod.tqdm.tqdm, TqdmClass)
        finally:
            if prev is None:
                sys.modules.pop("whisper.transcribe", None)
            else:
                sys.modules["whisper.transcribe"] = prev
            set_audio_progress_callback(None)
        ## END try restore whisper.transcribe....


if __name__ == "__main__":
    unittest.main()
