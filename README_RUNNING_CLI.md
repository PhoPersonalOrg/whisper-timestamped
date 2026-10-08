# Running Via CLI

```bash
## build filelist
uv run python scripts/iOSWhisperAppHelpers/extract_m4a_creation_times.py \
  --format ios_whisper_app \
  --audio-dir "/mnt/h/backups/2026-09-21_iPhone15Pro/WhisperApp/Audio/2026-10-07" \
  "/mnt/h/backups/2026-09-21_iPhone15Pro/WhisperApp/filelists/2026-10-07_audio_file_list.csv"

## transcribe from that filelist
uv run python scripts/process_recordings.py \
  --format ios_whisper_app \
  --filelist "/mnt/h/backups/2026-09-21_iPhone15Pro/WhisperApp/filelists/2026-10-07_audio_file_list.csv" \
  --output_dir "/mnt/h/backups/2026-09-21_iPhone15Pro/WhisperApp/transcriptions"

## or glob a directory instead of a filelist
# uv run python scripts/process_recordings.py \
#   --format ios_whisper_app \
#   --recordings_dir "/mnt/h/backups/2026-09-21_iPhone15Pro/WhisperApp/Audio/2026-10-07" \
#   --video_extensions ".m4a,.caf"
```

`process_recordings.py` flags: `--format` / `-f`, `--filelist`, `--recordings_dir`, `--output_dir`, `--video_extensions` (comma-separated). Provide at most one of `--filelist` or `--recordings_dir`.
