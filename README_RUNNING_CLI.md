# Running Via CLI

Paths below use WSL (`/mnt/h/...`). Windows / `H:/...` also work — `host_path` remaps them.

```bash
## 1) Build a filelist for a dated WhisperApp folder
uv run python scripts/iOSWhisperAppHelpers/extract_m4a_creation_times.py \
  --format ios_whisper_app \
  --audio-dir "/mnt/h/backups/2026-09-21_iPhone15Pro/WhisperApp/Audio/2026-10-07" \
  "/mnt/h/backups/2026-09-21_iPhone15Pro/WhisperApp/filelists/2026-10-07_audio_file_list.csv"

## 2) Transcribe from that filelist (--format + --filelist + --output_dir)
uv run python scripts/process_recordings.py \
  --format ios_whisper_app \
  --filelist "/mnt/h/backups/2026-09-21_iPhone15Pro/WhisperApp/filelists/2026-10-07_audio_file_list.csv" \
  --output_dir "/mnt/h/backups/2026-09-21_iPhone15Pro/WhisperApp/transcriptions"

## 3) Glob a directory instead of a filelist (--recordings_dir + --video_extensions)
uv run python scripts/process_recordings.py \
  --format ios_whisper_app \
  --recordings_dir "/mnt/h/backups/2026-09-21_iPhone15Pro/WhisperApp/Audio/2026-10-07" \
  --video_extensions ".m4a,.caf" \
  --output_dir "/mnt/h/backups/2026-09-21_iPhone15Pro/WhisperApp/transcriptions"

## 4) Format defaults only (no --filelist / --recordings_dir)
##    ios_whisper_app → newest *file_list*.csv under WhisperApp/filelists/
uv run python scripts/process_recordings.py --format ios_whisper_app

## 5) Another format (VoiceMemos filelist mode)
uv run python scripts/process_recordings.py \
  --format voice_memos \
  --filelist "/mnt/h/backups/2026-09-21_iPhone15Pro/VoiceMemos/filelists/2026-09-29_voicememos_audio_file_list.csv"
```

### `process_recordings.py` flags

| Flag | Purpose |
|------|---------|
| `--format` / `-f` | Format id: `debut`, `ios_whisper_app`, `just_press_record`, `rec_continuous`, `voice_memos` (default: `ios_whisper_app`) |
| `--filelist` | CSV with a `full_path` column; mutually exclusive with `--recordings_dir` |
| `--recordings_dir` | Directory to glob for media; mutually exclusive with `--filelist` |
| `--output_dir` | Where transcripts are written (default: format’s `default_output_dir`) |
| `--video_extensions` | Comma-separated extensions for directory mode, e.g. `.m4a,.caf` (default: format’s `media_extensions`) |

Provide **at most one** of `--filelist` or `--recordings_dir`. When neither is set, the format’s default input mode is used (filelist formats pick the newest matching CSV by mtime).
