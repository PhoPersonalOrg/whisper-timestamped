"""Unit tests for CrisperWhisper → whisper-timestamped result mapping (no model download)."""

import unittest

from whisper_timestamped.crisper_backend import (
    map_crisper_result,
    validate_crisper_model_name,
)
from whisper_timestamped.crisperwhisper.result import (
    ChunkResult,
    TranscriptionResult,
    WordTimestamp,
)


class TestCrisperBackendMapper(unittest.TestCase):
    def test_short_clip_single_segment(self):
        result = TranscriptionResult(
            text="hello world",
            language="en",
            mode="verbatim",
            duration=1.5,
            processing_time=0.1,
            chunks=None,
            words=[
                WordTimestamp(word="hello", start=0.1, end=0.5),
                WordTimestamp(word="world", start=0.6, end=1.2),
            ],
        )
        out = map_crisper_result(result)
        self.assertEqual(out["text"], "hello world")
        self.assertEqual(out["language"], "en")
        self.assertEqual(len(out["segments"]), 1)
        seg = out["segments"][0]
        self.assertEqual(seg["id"], 0)
        self.assertEqual(seg["text"], "hello world")
        self.assertEqual([w["text"] for w in seg["words"]], ["hello", "world"])
        starts = [w["start"] for w in seg["words"]]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(seg["start"], 0.1)
        self.assertEqual(seg["end"], 1.2)

    def test_longform_two_chunks(self):
        result = TranscriptionResult(
            text="one two three four",
            language="en",
            mode="verbatim",
            duration=40.0,
            processing_time=1.0,
            chunks=[
                ChunkResult(
                    chunk_idx=0,
                    start_sec=0.0,
                    end_sec=20.0,
                    text="one two",
                ),
                ChunkResult(
                    chunk_idx=1,
                    start_sec=18.0,
                    end_sec=40.0,
                    text="three four",
                ),
            ],
            words=[
                WordTimestamp(word="one", start=0.5, end=1.0),
                WordTimestamp(word="two", start=1.2, end=1.8),
                WordTimestamp(word="three", start=20.0, end=20.5),
                WordTimestamp(word="four", start=21.0, end=21.5),
            ],
        )
        out = map_crisper_result(result)
        self.assertEqual(out["text"], "one two three four")
        self.assertEqual(len(out["segments"]), 2)
        self.assertEqual([w["text"] for w in out["segments"][0]["words"]], ["one", "two"])
        self.assertEqual(
            [w["text"] for w in out["segments"][1]["words"]], ["three", "four"]
        )
        all_starts = [
            w["start"] for seg in out["segments"] for w in seg["words"]
        ]
        self.assertEqual(all_starts, sorted(all_starts))

    def test_validate_rejects_stock_whisper_only_names(self):
        with self.assertRaises(ValueError):
            validate_crisper_model_name("tiny")
        with self.assertRaises(ValueError):
            validate_crisper_model_name("large-v2")

    def test_validate_accepts_crisper_shorthands(self):
        self.assertIn("CrisperWhisper2.0_small", validate_crisper_model_name("small"))
        self.assertIn("CrisperWhisper2.0_turbo", validate_crisper_model_name("turbo"))


if __name__ == "__main__":
    unittest.main()
