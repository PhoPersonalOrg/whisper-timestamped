---
name: REC continuous format
overview: Implement sidecar-aware CAM_* support under format id `rec_continuous_video_recorder` (label REC_continuous_video_recorder), remove the partial `debut_example` work, and keep `rec_continuous` as a backward-compatible alias.
todos:
  - id: add-debut-example-format
    content: Implement RecContinuousVideoRecorderFormat (id rec_continuous_video_recorder) with CAM_* + sidecar logic in rec_continuous.py; delete debut_example.py
    status: completed
  - id: alias-rec-continuous
    content: Keep RecContinuousFormat as alias id rec_continuous; register both in FORMATS; detect uses rec_continuous_video_recorder only
    status: completed
  - id: tests-docs-gui
    content: Update tests, README_RUNNING_CLI.md, and GUI _infer_format_id for rec_continuous_video_recorder
    status: completed
isProject: false
---

# REC_continuous_video_recorder format

## Naming (revised)

- **Canonical format id:** `rec_continuous_video_recorder` (CLI accepts `REC_continuous_video_recorder` because `get_format` lowercases)
- **Label:** `REC_continuous_video_recorder`
- **Do not** ship a `debut_example` format id
- **Alias:** `rec_continuous` points at the same implementation

## Cleanup of partial work

Already on disk from the aborted attempt:
- [debut_example.py](c:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/whisper-timestamped/whisper_timestamped/recording_formats/debut_example.py) — fold logic into `rec_continuous.py`, then delete this file
- [rec_continuous.py](c:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/whisper-timestamped/whisper_timestamped/recording_formats/rec_continuous.py) — rewrite as the full implementation (not an alias of DebutExampleFormat)
- [registry.py](c:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/whisper-timestamped/whisper_timestamped/recording_formats/registry.py) — rewire away from `debut_example`

## Implementation

### 1. Full format in `rec_continuous.py`

- Primary class `RecContinuousVideoRecorderFormat` with id `rec_continuous_video_recorder`, label `REC_continuous_video_recorder`
- Thin subclass `RecContinuousFormat` with id `rec_continuous` (alias only)
- Extensions `.mkv`/`.mp4`, `input_mode=directory`
- Defaults under `H:/ScreenRecordings/REC_continuous_video_recorder` (plus `transcriptions/`, `filelists/`)
- Match `CAM_YYYY-MM-DDTHHMMSS` stems
- Helpers: `sidecar_path_for`, `parse_cam_filename`, `read_timing_sidecar`
- `extract_creation_time`: prefer sidecar `wall_clock_start_unix` via `utc_to_local_str`; else filename wall clock
- Extra columns: `timing_sidecar`, `segment_index`, `duration_perf_seconds`, `lsl_local_clock_start`
- `build_filelist_rows` uses format creation time for `creation_time`

### 2. Registry / exports

- `_DETECT_ORDER` uses `RecContinuousVideoRecorderFormat()` in the CAM slot
- `FORMATS` includes both ids; detect returns only `rec_continuous_video_recorder`
- Delete `debut_example.py`; update `__init__.py` exports

### 3. Tests / docs / GUI

- Tests: known ids, detect CAM folder → `rec_continuous_video_recorder`, sidecar + filename fallback, alias still works
- README format list
- GUI `_infer_format_id`: check `rec_continuous_video_recorder` / `rec_continuous` before bare `debut`

## Usage

```bash
python scripts/process_recordings.py --format rec_continuous_video_recorder
python scripts/process_recordings.py -f REC_continuous_video_recorder
python scripts/process_recordings.py -f rec_continuous
```
