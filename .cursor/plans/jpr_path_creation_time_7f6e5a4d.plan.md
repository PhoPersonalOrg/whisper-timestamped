---
name: JPR path creation time
overview: Add a Just Press Record path parser in `extract_m4a_creation_times.py` that builds the concatenated transcript name and local `extracted_creation_time` from `YYYY-MM-DD/HH-MM-SS.ext`, then wire it into recursive file-list bootstrap and creation-time extraction (no `process_recordings.py` changes).
todos:
  - id: add-parse
    content: Add parse_just_press_record_path returning (transcript_name, extracted_creation_time) or None
    status: completed
  - id: bootstrap-rglob
    content: Update build_file_list_from_audio_dir for one-level nested audio + concatenated name
    status: completed
  - id: extract-wire
    content: Prefer JPR path time in extract_for_csv; SetFileTime via local→UTC when path-sourced
    status: completed
  - id: docs
    content: Update module/CLI description for Just Press Record path behavior
    status: completed
isProject: false
---

# Just Press Record path → extracted_creation_time

## Scope

Only [`scripts/iOSWhisperAppHelpers/extract_m4a_creation_times.py`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\scripts\iOSWhisperAppHelpers\extract_m4a_creation_times.py). Leave [`process_recordings.py`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\scripts\process_recordings.py) unchanged.

Confirmed layout: `Just Press Record/YYYY-MM-DD/HH-MM-SS.m4a` (44 date folders, 89 files in this backup).

## Parse method

Add something like `parse_just_press_record_path(path: Path) -> Optional[Tuple[str, str]]` near the other path/time helpers:

- Require parent dir name `YYYY-MM-DD` and stem `HH-MM-SS` (strict regex / `datetime.strptime`).
- **Transcript name:** `{date}_{time}{suffix}` e.g. `2023-08-10_16-12-39.m4a` (matches the comment at process_recordings L373).
- **`extracted_creation_time`:** `{date} {HH}:{MM}:{SS}` e.g. `2023-08-10 16:12:39` — same format as `utc_to_local_str` (`%Y-%m-%d %H:%M:%S`). Treat path components as already-local wall clock; do **not** run through UTC→local conversion.
- Return `None` if the hierarchy does not match (WhisperApp flat UUID paths stay unaffected).

## Wire into bootstrap + extract

1. **`build_file_list_from_audio_dir`** — collect both flat `audio_dir/*{ext}` and one-level nested `audio_dir/*/*{ext}` (covers JPR date folders without descending into `_DUP` or deeper junk). When `parse_just_press_record_path` succeeds, set CSV `name` to the concatenated transcript name; keep `full_path` as the real file path. FS `creation_time` / `modification_time` columns unchanged.

2. **`extract_for_csv`** — after `probe_format_metadata`:
   - If the resolved path parses as JPR, set `extracted_creation_time` from the path string (authoritative for this export layout).
   - Else keep today’s ffprobe → `utc_to_local_str` behavior.
   - Duration still from ffprobe only.
   - For Windows `SetFileTime`: when creation time comes from the JPR path, interpret that naive local string in `--tz`, convert to UTC, then set birth time (same side effect as probe success). Skip SetFileTime when neither probe nor path yields a time.

3. **Docs** — mention in the module docstring / argparse description that Just Press Record `date/time.ext` paths supply `extracted_creation_time` and concatenated `name` when bootstrapping.

## Out of scope

- Recursive discovery / `base_name` changes in `process_recordings.py`
- New CLI flags (reuse `--audio-dir` / `--tz`)
