---
name: Filelist process recordings
overview: Extend `process_recordings.py` to accept an extract_m4a_creation_times-style filelist CSV, transcribe each row via `full_path`, and write back `transcript_*` path columns—keeping directory-glob mode when no filelist is given.
todos:
  - id: api-filelist
    content: Add filelist_csv param, mutual exclusivity with recordings_dir, CSV load + row filtering
    status: completed
  - id: shared-loop
    content: Share transcription loop; base_name from name stem; fill transcript_* + save CSV in place
    status: completed
  - id: main-jpr
    content: Switch __main__ JPR block to filelist_csv + output_dir
    status: completed
isProject: false
---

# Filelist-driven `process_recordings`

## Goal

In [`scripts/process_recordings.py`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\scripts\process_recordings.py), support a filelist CSV (e.g. `H:\backups\2026-09-21_iPhone15Pro\Just Press Record\filelists\2026-09-29_jpr_audio_file_list.csv`) as the input source: audio from `full_path`, transcript outputs under `output_dir`, and new `transcript_*` columns written back onto that CSV.

Directory-glob mode (`recordings_dir` + extensions) stays available when no filelist is passed.

## API

Add `filelist_csv: Path | None = None` to `process_recordings`. Require exactly one of `filelist_csv` or `recordings_dir` (raise `ValueError` if both/neither).

When `filelist_csv` is set:

- Load with pandas (`pandas` already in [`pyproject.toml`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\pyproject.toml)).
- Require a `full_path` column.
- Skip rows where `is_duplicate` is truthy (when that column exists).
- Resolve each audio path with `host_path(full_path)`; skip missing files with a log line.
- **`base_name`:** `Path(name).stem` when `name` is present (JPR concatenated stem `2023-08-10_16-12-39`); else `Path(full_path).stem`.
- **`output_dir`:** if `None`, default to `filelist_csv.parent.parent / "transcriptions"` when the filelist lives under a `filelists` folder, else `filelist_csv.parent / "transcriptions"`.
- **`alias_dir`:** `output_dir.parent / "edf_video_aliases"` (same role as today’s `recordings_dir.parent / ...`).

## Processing loop

Refactor the per-file body so both modes share one loop over a list of `(audio_path, base_name, optional_row_index)`:

1. EDF alias attempt (unchanged; failures still skipped).
2. If outputs already exist → skip transcription, but still record paths into `transcript_*` for that row (so re-runs fill the CSV).
3. Else load / transcribe / `write_results` as today.
4. Map `write_results` keys to column names: `transcript_{key with '.' → '_'}` (e.g. `json` → `transcript_json`, `words.csv` → `transcript_words_csv`). Store absolute path strings (empty string if that format was not produced).

After each filelist row is handled (success, skip, or fail), update that row’s `transcript_*` cells and **save the CSV in place** so a mid-run crash keeps progress. On failure, leave `transcript_*` empty for that row (or only fill formats that exist).

Return value: keep returning the aggregated `output_files` dict; filelist mode additionally persists the CSV.

## `__main__` (JPR)

Point the Just Press Record block at the filelist instead of the empty flat `recordings_dir` glob:

```python
filelist_csv = host_path(
    r"H:/backups/2026-09-21_iPhone15Pro/Just Press Record/filelists/2026-09-29_jpr_audio_file_list.csv"
)
output_dir = host_path(r"H:/backups/2026-09-21_iPhone15Pro/Just Press Record/transcriptions")
output_files = process_recordings(
    filelist_csv=filelist_csv,
    output_dir=output_dir,
    backend="crisperwhisper",
    ...
)
```

`recordings_dir` / `video_extensions` unused in that call.

## Out of scope

- New argparse CLI (script remains `__main__`-configured).
- Changes to `extract_m4a_creation_times.py`.
- Changing default output formats from `write_results`.
