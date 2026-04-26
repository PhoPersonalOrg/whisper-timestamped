---
name: Robust process_recordings per-file
overview: Extend the existing per-file try/except in process_recordings so that any failure when handling a single file (parsing filename, opening/decoding audio, transcribing, or writing) is caught, reported, and the script continues with the next file. Optionally improve error reporting and add a short summary.
todos: []
isProject: false
---

# Robust per-file error handling in process_recordings.py

## Current behavior

- In [scripts/process_recordings.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\scripts\process_recordings.py), the main loop (lines 227–269) already wraps **load_audio**, **transcribe**, and **write_results** in a single `try/except`: on any exception it prints the error and `continue`s.
- Two pieces run **outside** that try block and can still abort the whole run:
  1. **EDF filename parsing and symlink** (lines 232–236): `build_EDF_compatible_video_filename(video_file.name)` calls `parse_video_filename`, which raises `ValueError` when the filename does not match the expected date patterns ([parse_video_filename.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\whisper_timestamped\parse_video_filename.py) line 35). `edf_compatible_path.symlink_to(...)` can raise `OSError` (e.g. permissions, path length).
  2. **Output existence check** (lines 238–241): `find_extant_output_files` and the skip logic could theoretically raise (e.g. permission issues on `output_dir`).

So opening/decoding failures are already handled; parsing (filename) and symlink failures are not.

## Recommended change

**Single, minimal edit:** move the start of the existing `try` block **up** so that it wraps the entire per-file work (EDF name + symlink, output check, load, transcribe, write). Then any failure for that file is caught and the loop continues.

- **Before:** `try` starts at line 242 (just before "Loading audio...").
- **After:** `try` starts immediately after `print(f"\nProcessing: {video_file.name}")` and `base_name = video_file.stem`, so it encompasses:
  - EDF-compatible name building and symlink creation
  - `find_extant_output_files` and the skip-if-exists `continue`
  - Load audio, transcribe, write_results

**Important:** The skip logic (`if found_output_files: ... continue`) must remain inside the try. If an exception happens before that (e.g. EDF parsing), the except will run and then `continue` to the next file. If the exception happens after the skip check (e.g. during load_audio), behavior is unchanged.

No new dependencies or structural refactors required.

## Optional improvements

- **Error message:** Include exception type in the log, e.g. `print(f"  ✗ Error processing {video_file.name}: [{type(e).__name__}] {e}")`, so decoding vs parsing vs I/O is easier to distinguish.
- **Summary at end:** Track failed count (e.g. a `failed_files` list or counter incremented in the except block) and after the loop print something like "Processed N, failed K" for quick scanning.

## Out of scope

- Retries or backoff for failed files.
- Changing `write_results` (it already has per-format try/except and does not abort the script).
- Making EDF symlink optional on parse failure (possible future enhancement: catch only around EDF/symlink and still transcribe; not in this plan).
