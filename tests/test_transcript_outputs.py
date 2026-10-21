"""Unit tests for torch-free transcript output helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.transcript_outputs import (
    TRANSCRIPTION_PARAMS_KEY,
    _discard_transcript_outputs,
    candidate_output_dirs,
    collect_extant_status_cols,
    collect_extant_transcript_paths,
    csv_derived_output_dir,
    discover_row_transcript_updates,
    load_transcription_params,
    merge_transcript_status_columns,
    params_sidecar_path,
    write_params_sidecar,
)


class TestCollectExtant(unittest.TestCase):
    def test_collect_extant_transcript_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "session.txt").write_text("hello", encoding="utf-8")
            (output_dir / "session.words.json").write_text("{}", encoding="utf-8")
            cols = collect_extant_transcript_paths(output_dir, "session")
            self.assertTrue(cols["transcript_txt"].endswith("session.txt"))
            self.assertTrue(cols["transcript_json"].endswith("session.words.json"))
            self.assertEqual(cols["transcript_srt"], "")
            self.assertTrue(Path(cols["transcript_txt"]).is_file())
        ## END with tempfile.TemporaryDirectory....


    def test_collect_extant_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            cols = collect_extant_transcript_paths(Path(tmp), "missing")
            self.assertTrue(all(v == "" for v in cols.values()))
        ## END with tempfile.TemporaryDirectory....


class TestParamsSidecar(unittest.TestCase):
    def test_write_and_load_params_sidecar(self):
        params = {
            "backend": "crisperwhisper",
            "model_name": "medium",
            "crisper_mode": "verbatim",
            "device": "cuda",
        }
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            path = write_params_sidecar(output_dir, "session", params)
            self.assertEqual(path, params_sidecar_path(output_dir, "session"))
            self.assertTrue(path.is_file())
            loaded = load_transcription_params(output_dir, "session")
            self.assertEqual(loaded["backend"], "crisperwhisper")
            self.assertEqual(loaded["model_name"], "medium")

            status = collect_extant_status_cols(output_dir, "session")
            self.assertTrue(status["transcript_params"].endswith("session.params.json"))
            self.assertEqual(status["transcript_backend"], "crisperwhisper")
            self.assertEqual(status["transcript_model"], "medium")
            self.assertEqual(status["transcript_crisper_mode"], "verbatim")
        ## END with tempfile.TemporaryDirectory....


    def test_load_params_from_words_json_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            payload = {
                "text": "hi",
                "segments": [],
                TRANSCRIPTION_PARAMS_KEY: {
                    "backend": "openai-whisper",
                    "model_name": "medium.en",
                    "crisper_mode": "",
                },
            }
            (output_dir / "session.words.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            loaded = load_transcription_params(output_dir, "session")
            self.assertEqual(loaded["backend"], "openai-whisper")
            self.assertEqual(loaded["model_name"], "medium.en")
        ## END with tempfile.TemporaryDirectory....


    def test_discard_removes_params_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_params_sidecar(output_dir, "session", {"backend": "x"})
            (output_dir / "session.txt").write_text("hi", encoding="utf-8")
            removed = _discard_transcript_outputs(output_dir, "session")
            names = {p.name for p in removed}
            self.assertIn("session.params.json", names)
            self.assertIn("session.txt", names)
            self.assertFalse(params_sidecar_path(output_dir, "session").exists())
        ## END with tempfile.TemporaryDirectory....


class TestFilelistMergeAndDiscovery(unittest.TestCase):
    def test_csv_derived_output_dir(self):
        filelists = Path("H:/recordings/filelists/list.csv")
        self.assertEqual(
            csv_derived_output_dir(filelists),
            Path("H:/recordings/transcriptions"),
        )
        beside = Path("C:/data/list.csv")
        self.assertEqual(csv_derived_output_dir(beside), Path("C:/data/transcriptions"))


    def test_merge_transcript_status_columns_by_full_path(self):
        previous = pd.DataFrame(
            {
                "name": ["a.m4a", "b.m4a"],
                "full_path": ["C:/audio/a.m4a", "C:/audio/b.m4a"],
                "transcript_txt": ["C:/out/a.txt", ""],
                "transcript_backend": ["crisperwhisper", ""],
            }
        )
        new_df = pd.DataFrame(
            {
                "name": ["a.m4a", "b.m4a", "c.m4a"],
                "full_path": ["C:/audio/a.m4a", "C:/audio/b.m4a", "C:/audio/c.m4a"],
                "size_mb": [1.0, 2.0, 3.0],
            }
        )
        merged = merge_transcript_status_columns(new_df, previous)
        self.assertEqual(merged.at[0, "transcript_txt"], "C:/out/a.txt")
        self.assertEqual(merged.at[0, "transcript_backend"], "crisperwhisper")
        self.assertEqual(str(merged.at[1, "transcript_txt"]).strip(), "")
        self.assertEqual(str(merged.at[2, "transcript_txt"]).strip(), "")


    def test_merge_falls_back_to_name(self):
        previous = pd.DataFrame(
            {
                "name": ["keep.m4a"],
                "full_path": ["C:/old/keep.m4a"],
                "transcript_json": ["C:/out/keep.words.json"],
            }
        )
        new_df = pd.DataFrame(
            {
                "name": ["keep.m4a"],
                "full_path": ["C:/new/keep.m4a"],
            }
        )
        merged = merge_transcript_status_columns(new_df, previous)
        self.assertEqual(merged.at[0, "transcript_json"], "C:/out/keep.words.json")


    def test_discover_row_from_csv_derived_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            filelists = root / "filelists"
            transcripts = root / "transcriptions"
            filelists.mkdir()
            transcripts.mkdir()
            csv_path = filelists / "audio_file_list.csv"
            csv_path.write_text("name,full_path\n", encoding="utf-8")
            (transcripts / "take1.txt").write_text("spoken", encoding="utf-8")
            (transcripts / "take1.words.json").write_text(
                json.dumps(
                    {
                        "segments": [],
                        TRANSCRIPTION_PARAMS_KEY: {
                            "backend": "crisperwhisper",
                            "model_name": "medium",
                            "crisper_mode": "verbatim",
                        },
                    }
                ),
                encoding="utf-8",
            )
            row = {
                "name": "take1.m4a",
                "full_path": str(root / "audio" / "take1.m4a"),
                "source_csv": str(csv_path),
            }
            updates = discover_row_transcript_updates(row)
            self.assertTrue(updates["transcript_txt"].endswith("take1.txt"))
            self.assertTrue(updates["transcript_json"].endswith("take1.words.json"))
            self.assertEqual(updates["transcript_backend"], "crisperwhisper")
            self.assertEqual(updates["transcript_model"], "medium")
        ## END with tempfile.TemporaryDirectory....


    def test_candidate_output_dirs_includes_format_and_prefs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fmt_dir = root / "fmt_out"
            prefs_dir = root / "prefs_out"
            fmt_dir.mkdir()
            prefs_dir.mkdir()
            dirs = candidate_output_dirs(
                {"name": "x.m4a"},
                format_output_dir=fmt_dir,
                prefs_output_dir=prefs_dir,
            )
            resolved = {p.resolve() for p in dirs}
            self.assertIn(fmt_dir.resolve(), resolved)
            self.assertIn(prefs_dir.resolve(), resolved)
        ## END with tempfile.TemporaryDirectory....


    def test_discover_uses_stored_transcript_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "custom"
            output_dir.mkdir()
            (output_dir / "clip.txt").write_text("ok", encoding="utf-8")
            row = {
                "name": "clip.m4a",
                "transcript_txt": str(output_dir / "clip.txt"),
            }
            updates = discover_row_transcript_updates(row)
            self.assertTrue(updates["transcript_txt"].endswith("clip.txt"))
        ## END with tempfile.TemporaryDirectory....


if __name__ == "__main__":
    unittest.main()
