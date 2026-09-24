---
name: Add Duration Columns
overview: Extend `extract_m4a_creation_times.py` to also write optional `duration (seconds)` and `duration_hms` from the same ffprobe call, leaving cells empty when duration is missing without failing the row or run.
todos:
  - id: add-duration-cols
    content: Refactor probe + add duration (seconds) and duration_hms columns
    status: completed
  - id: rerun-verify-duration
    content: Re-run on DEDUPE CSV and spot-check duration values
    status: completed
isProject: false
---

# Add Optional Duration Columns

## Change

Update [`scripts/iOSWhisperAppHelpers/extract_m4a_creation_times.py`](C:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/whisper-timestamped/scripts/iOSWhisperAppHelpers/extract_m4a_creation_times.py) only.

## Approach

1. **Refactor probe** into one `probe_format_metadata(path) -> (utc_dt | None, duration_sec | None)` that reads both `format.tags.creation_time` and `format.duration` from the same ffprobe JSON. Missing either field returns `None` for that field only (no raise).

2. **Format helpers**
   - `duration (seconds)`: float string with reasonable precision (e.g. 3 decimals), or `""` if missing
   - `duration_hms`: `HH:MM:SS` from `int(duration)` (floor), or `""` if missing

3. **Per-row independence**
   - Always append values for all three columns
   - Missing creation_time → empty `extracted_creation_time`, still try duration
   - Missing duration → empty duration columns, still write creation_time when present
   - Missing file / ffprobe failure → empty for all three; do not abort

4. **Write columns** (exact names): `extracted_creation_time`, `duration (seconds)`, `duration_hms`

5. **Re-run** on the default DEDUPE CSV and spot-check `055008C5-...` → ~`2649.813` / `00:44:09`.