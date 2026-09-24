---
name: Skip EDF on parse fail
overview: Make EDF-compatible alias creation best-effort in process_recordings so UUID/.m4a filenames that fail parse_video_filename still get transcribed instead of being marked failed.
todos:
  - id: isolate-edf-try
    content: Wrap EDF name + symlink in inner try/except (ValueError, OSError); warn and continue to transcription
    status: completed
isProject: false
---

# Skip EDF alias when filename doesn't match

## Problem

In [`scripts/process_recordings.py`](scripts/process_recordings.py), the per-file `try` wraps EDF alias creation and transcription together. `build_EDF_compatible_video_filename` → `parse_video_filename` raises `ValueError` for WhisperApp UUID names (and `recovered_*` names). That exception is caught as a full-file failure, so transcription never runs.

## Fix

Isolate the EDF/symlink block in its own `try/except` so parse/symlink failures are warnings only; then always continue to skip-if-exists → load → transcribe → write.

In the main loop (~lines 241–246), replace the unconditional EDF block with:

```python
try:
    edf_compatible_name = build_EDF_compatible_video_filename(video_file.name)
    print(f'\tedf_compatible_name: "{edf_compatible_name}"')
    edf_compatible_path = alias_dir / edf_compatible_name
    if not edf_compatible_path.exists():
        edf_compatible_path.symlink_to(video_file.resolve())
except (ValueError, OSError) as e:
    print(f"  ~ Skipping EDF alias for {video_file.name}: {e}")
```

Keep the outer `try/except` for real transcription/write failures only.

## Scope

- One edit in [`scripts/process_recordings.py`](scripts/process_recordings.py)
- No changes to [`whisper_timestamped/parse_video_filename.py`](whisper_timestamped/parse_video_filename.py)
- Video files with matching CAM_/Debut_ names still get aliases; audio UUID names proceed without them
