"""Unit tests for recordings format detection and parsing."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from whisper_timestamped.recording_formats import (
    detect_format,
    get_format,
    list_format_ids,
    parse_just_press_record_path,
    parse_voice_memos_filename,
)
from whisper_timestamped.recording_formats.debut import DebutFormat
from whisper_timestamped.recording_formats.ios_whisper_app import (
    IOSWhisperAppFormat,
)
from whisper_timestamped.recording_formats.just_press_record import (
    JustPressRecordFormat,
)
from whisper_timestamped.recording_formats.rec_continuous import (
    RecContinuousFormat,
)
from whisper_timestamped.recording_formats.voice_memos import (
    VoiceMemosFormat,
    load_voice_memos_metadata,
    load_voice_memos_titles,
)


class TestFormatRegistry(unittest.TestCase):
    def test_known_ids(self):
        ids = set(list_format_ids())
        self.assertEqual(
            ids,
            {
                "debut",
                "rec_continuous",
                "ios_whisper_app",
                "just_press_record",
                "voice_memos",
            },
        )

    def test_get_format(self):
        self.assertEqual(get_format("voice_memos").id, "voice_memos")
        with self.assertRaises(KeyError):
            get_format("nope")


class TestDebutAndCam(unittest.TestCase):
    def test_debut_matches_and_kwargs(self):
        fmt = DebutFormat()
        path = Path("Debut_2025-07-01T113802.mp4")
        self.assertTrue(fmt.matches_file(path))
        self.assertFalse(fmt.matches_file(Path("CAM_2025-07-01T113802.mp4")))
        kwargs = fmt.process_recordings_kwargs()
        self.assertIn("recordings_dir", kwargs)
        self.assertEqual(kwargs["video_extensions"], [".mp4", ".mkv"])
        self.assertNotIn("filelist_csv", kwargs)

    def test_cam_matches(self):
        fmt = RecContinuousFormat()
        self.assertTrue(fmt.matches_file(Path("CAM_2026-01-09T081552.mp4")))
        self.assertFalse(fmt.matches_file(Path("Debut_2026-01-09T081552.mp4")))


class TestJustPressRecord(unittest.TestCase):
    def test_parse_path(self):
        path = Path(r"H:/backups/Just Press Record/2023-08-10/16-12-39.m4a")
        parsed = parse_just_press_record_path(path)
        self.assertIsNotNone(parsed)
        name, creation = parsed
        self.assertEqual(name, "2023-08-10_16-12-39.m4a")
        self.assertEqual(creation, "2023-08-10 16:12:39")

    def test_parse_rejects_bad_stem(self):
        path = Path(r"H:/backups/Just Press Record/2023-08-10/not-a-time.m4a")
        self.assertIsNone(parse_just_press_record_path(path))

    def test_format_helpers(self):
        fmt = JustPressRecordFormat()
        path = Path("export/2023-08-10/16-12-39.m4a")
        self.assertTrue(fmt.matches_file(path))
        self.assertEqual(fmt.transcript_name(path), "2023-08-10_16-12-39.m4a")
        self.assertEqual(
            fmt.extract_creation_time(path), "2023-08-10 16:12:39"
        )

    def test_detect_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            day = root / "2023-08-10"
            day.mkdir()
            (day / "16-12-39.m4a").write_bytes(b"x")
            (day / "16-13-00.m4a").write_bytes(b"x")
            detected = detect_format(root)
            self.assertIsNotNone(detected)
            self.assertEqual(detected.id, "just_press_record")


def _write_synthetic_cloud_recordings_db(db_path: Path) -> None:
    """Minimal CloudRecordings.db for unit tests."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE ZFOLDER (
                Z_PK INTEGER PRIMARY KEY,
                ZENCRYPTEDNAME VARCHAR
            );
            CREATE TABLE ZCLOUDRECORDING (
                Z_PK INTEGER PRIMARY KEY,
                ZPATH VARCHAR,
                ZENCRYPTEDTITLE VARCHAR,
                ZCUSTOMLABELFORSORTING VARCHAR,
                ZCUSTOMLABEL VARCHAR,
                ZDATE FLOAT,
                ZDURATION FLOAT,
                ZUNIQUEID VARCHAR,
                ZFOLDER INTEGER
            );
            INSERT INTO ZFOLDER (Z_PK, ZENCRYPTEDNAME) VALUES (1, 'Offline');
            INSERT INTO ZCLOUDRECORDING (
                Z_PK, ZPATH, ZENCRYPTEDTITLE, ZCUSTOMLABELFORSORTING,
                ZCUSTOMLABEL, ZDATE, ZDURATION, ZUNIQUEID, ZFOLDER
            ) VALUES (
                1,
                '20190415 200101-4FA3EFA0.m4a',
                'Home',
                'Home',
                '2019-04-16T00:01:01Z',
                577065661.201911,
                179.3,
                '4FA3EFA0-A687-4687-A787-50F6EB45B8F7',
                NULL
            );
            INSERT INTO ZCLOUDRECORDING (
                Z_PK, ZPATH, ZENCRYPTEDTITLE, ZCUSTOMLABELFORSORTING,
                ZCUSTOMLABEL, ZDATE, ZDURATION, ZUNIQUEID, ZFOLDER
            ) VALUES (
                2,
                '20211028 230533-479E0A4C.m4a',
                '3081 Promenade Cir 2',
                '3081 Promenade Cir 2',
                NULL,
                657169533.384722,
                2229.2,
                '479E0A4C-A40F-46BA-866C-5E1EB79FA278',
                1
            );
            -- orphan row with no path should be skipped
            INSERT INTO ZCLOUDRECORDING (
                Z_PK, ZPATH, ZENCRYPTEDTITLE, ZCUSTOMLABELFORSORTING,
                ZCUSTOMLABEL, ZDATE, ZDURATION, ZUNIQUEID, ZFOLDER
            ) VALUES (
                3, NULL, 'Home 3', 'Home 3', NULL, 1.0, 1.0, 'NO-PATH', NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()
    ## END try/finally write synthetic db....


class TestVoiceMemos(unittest.TestCase):
    def test_parse_with_id(self):
        path = Path("20190415 200101-4FA3EAF0.m4a")
        dt = parse_voice_memos_filename(path)
        self.assertIsNotNone(dt)
        self.assertEqual(dt.strftime("%Y-%m-%d %H:%M:%S"), "2019-04-15 20:01:01")

    def test_parse_without_id(self):
        path = Path("20250829 151248.m4a")
        dt = parse_voice_memos_filename(path)
        self.assertIsNotNone(dt)
        self.assertEqual(dt.strftime("%Y-%m-%d %H:%M:%S"), "2025-08-29 15:12:48")

    def test_format_helpers(self):
        fmt = VoiceMemosFormat()
        path = Path("20190415 200101-4FA3EAF0.m4a")
        self.assertTrue(fmt.matches_file(path))
        self.assertEqual(fmt.transcript_name(path), path.name)
        self.assertEqual(
            fmt.extract_creation_time(path), "2019-04-15 20:01:01"
        )
        self.assertFalse(fmt.matches_file(Path("random-uuid.m4a")))

    def test_detect_audio_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "audio"
            audio.mkdir()
            (audio / "20190415 200101-4FA3EAF0.m4a").write_bytes(b"x")
            (audio / "20211028 230533-47560A4C.m4a").write_bytes(b"x")
            detected = detect_format(audio)
            self.assertIsNotNone(detected)
            self.assertEqual(detected.id, "voice_memos")

            # Root containing audio/ child
            detected_root = detect_format(Path(tmp))
            self.assertIsNotNone(detected_root)
            self.assertEqual(detected_root.id, "voice_memos")

    def test_filelist_rows_include_title_column(self):
        fmt = VoiceMemosFormat()
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "audio"
            audio.mkdir()
            (audio / "20190415 200101-4FA3EAF0.m4a").write_bytes(b"x")
            # Point format at temp root so DB lookup misses quietly.
            fmt.default_recordings_dir = audio
            fmt._voice_memos_root = Path(tmp)
            fmt._meta_cache = {}
            fmt._encoder_cache = {
                "20190415 200101-4FA3EAF0.m4a": "",
            }
            rows = fmt.build_filelist_rows(audio)
            self.assertEqual(len(rows), 1)
            self.assertIn("title", rows[0])
            self.assertEqual(rows[0]["name"], "20190415 200101-4FA3EAF0.m4a")
            self.assertEqual(rows[0]["title"], "")
            self.assertEqual(rows[0]["apple_transcript"], "")

    def test_load_metadata_from_synthetic_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = (
                Path(tmp)
                / "group.com.apple.VoiceMemos.shared"
                / "Recordings"
                / "CloudRecordings.db"
            )
            _write_synthetic_cloud_recordings_db(db)
            meta = load_voice_memos_metadata(db)
            self.assertEqual(set(meta), {
                "20190415 200101-4FA3EFA0.m4a",
                "20211028 230533-479E0A4C.m4a",
            })
            home = meta["20190415 200101-4FA3EFA0.m4a"]
            self.assertEqual(home["title"], "Home")
            self.assertEqual(home["duration_seconds"], 179.3)
            self.assertEqual(home["recorded_at_utc"], "2019-04-16 00:01:01")
            self.assertEqual(
                home["unique_id"],
                "4FA3EFA0-A687-4687-A787-50F6EB45B8F7",
            )
            self.assertEqual(home["folder"], "")

            promenade = meta["20211028 230533-479E0A4C.m4a"]
            self.assertEqual(promenade["title"], "3081 Promenade Cir 2")
            self.assertEqual(promenade["folder"], "Offline")
            # ZDATE fallback when ZCUSTOMLABEL is null
            self.assertEqual(promenade["recorded_at_utc"], "2021-10-29 03:05:33")

            titles = load_voice_memos_titles(db)
            self.assertEqual(titles["20190415 200101-4FA3EFA0.m4a"], "Home")

    def test_load_metadata_missing_db(self):
        missing = Path("/nonexistent/CloudRecordings.db")
        self.assertEqual(load_voice_memos_metadata(missing), {})
        self.assertEqual(load_voice_memos_titles(missing), {})

    def test_extra_row_fields_from_db_and_transcript(self):
        fmt = VoiceMemosFormat()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "audio"
            audio.mkdir()
            media = audio / "20190415 200101-4FA3EFA0.m4a"
            media.write_bytes(b"x")
            (root / "transcripts").mkdir()
            tx = root / "transcripts" / "Home.txt"
            tx.write_text("hello", encoding="utf-8")
            db = (
                root
                / "group.com.apple.VoiceMemos.shared"
                / "Recordings"
                / "CloudRecordings.db"
            )
            _write_synthetic_cloud_recordings_db(db)

            fmt._voice_memos_root = root
            fmt.default_recordings_dir = audio
            fmt._meta_cache = None
            fmt._encoder_cache = {media.name: "com.apple.VoiceMemos (test)"}

            fields = fmt.extra_row_fields(media)
            self.assertEqual(fields["title"], "Home")
            self.assertEqual(fields["duration_seconds"], 179.3)
            self.assertEqual(fields["recorded_at_utc"], "2019-04-16 00:01:01")
            self.assertEqual(
                fields["unique_id"],
                "4FA3EFA0-A687-4687-A787-50F6EB45B8F7",
            )
            self.assertEqual(fields["folder"], "")
            self.assertEqual(fields["apple_transcript"], str(tx))
            self.assertEqual(fields["encoder"], "com.apple.VoiceMemos (test)")

            rows = fmt.build_filelist_rows(audio)
            self.assertEqual(len(rows), 1)
            for col in (
                "title",
                "duration_seconds",
                "recorded_at_utc",
                "unique_id",
                "folder",
                "apple_transcript",
                "encoder",
            ):
                self.assertIn(col, rows[0])
            ## END for col in extra columns....


class TestIOSWhisperApp(unittest.TestCase):
    def test_matches_opaque_names(self):
        fmt = IOSWhisperAppFormat()
        self.assertTrue(fmt.matches_file(Path("ABCDEF12-3456-7890.m4a")))
        self.assertFalse(
            fmt.matches_file(Path("20190415 200101-4FA3EAF0.m4a"))
        )
        self.assertFalse(
            fmt.matches_file(Path("2023-08-10/16-12-39.m4a"))
        )

    def test_detect_flat_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "recovered_abc.m4a").write_bytes(b"x")
            (root / "uuid-like-name.caf").write_bytes(b"x")
            detected = detect_format(root)
            self.assertIsNotNone(detected)
            self.assertEqual(detected.id, "ios_whisper_app")

    def test_process_kwargs_filelist(self):
        fmt = IOSWhisperAppFormat()
        kwargs = fmt.process_recordings_kwargs(
            filelist_csv=Path("dummy.csv"),
            output_dir=Path("out"),
        )
        self.assertEqual(kwargs["filelist_csv"], Path("dummy.csv"))
        self.assertEqual(kwargs["output_dir"], Path("out"))
        self.assertNotIn("recordings_dir", kwargs)


class TestEnsureFilelist(unittest.TestCase):
    def test_voice_memos_writes_csv(self):
        fmt = VoiceMemosFormat()
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "audio"
            audio.mkdir()
            (audio / "20190415 200101-4FA3EAF0.m4a").write_bytes(b"xx")
            csv_path = Path(tmp) / "filelists" / "test.csv"
            fmt._voice_memos_root = Path(tmp)
            fmt._meta_cache = {}
            fmt._encoder_cache = {"20190415 200101-4FA3EAF0.m4a": ""}
            n = fmt.ensure_filelist_csv(csv_path, recordings_dir=audio)
            self.assertEqual(n, 1)
            self.assertTrue(csv_path.is_file())
            text = csv_path.read_text(encoding="utf-8")
            self.assertIn("full_path", text)
            self.assertIn("title", text)
            self.assertIn("duration_seconds", text)
            self.assertIn("recorded_at_utc", text)
            self.assertIn("apple_transcript", text)
            self.assertIn("encoder", text)
            self.assertIn("20190415 200101-4FA3EAF0.m4a", text)


if __name__ == "__main__":
    unittest.main()
