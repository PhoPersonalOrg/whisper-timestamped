---
name: Handle CAF audio
overview: Extend `extract_m4a_creation_times.py` so missing-CSV bootstrap and messaging include `.caf` alongside `.m4a`. Probing already works for any resolved path; CAF typically has duration but no embedded `creation_time`.
todos:
  - id: ext-constant
    content: Add AUDIO_EXTENSIONS = (".m4a", ".caf") and use it in build_file_list_from_audio_dir
    status: completed
  - id: docs
    content: Update module docstring and argparse text for m4a+caf
    status: completed
isProject: false
---

# Extend extract script for `.caf`

## Context

Only change [`extract_m4a_creation_times.py`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\scripts\iOSWhisperAppHelpers\extract_m4a_creation_times.py).

Audio dir currently has **531 `.m4a`** and **58 `.caf`**. `extract_for_csv` / `probe_format_metadata` already accept any file path via `full_path` / `name`; the hard gate is `build_file_list_from_audio_dir`, which only globs `*.m4a`.

Sample CAF probes: duration is present; format `tags` (including `creation_time`) are absent. Existing logic already leaves `extracted_creation_time` empty and still fills duration columns when available — no special CAF tag path needed.

## Changes

1. Add a module-level constant, e.g. `AUDIO_EXTENSIONS = (".m4a", ".caf")`.

2. Update [`build_file_list_from_audio_dir`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\scripts\iOSWhisperAppHelpers\extract_m4a_creation_times.py) (~L212–228):
   - Collect `*.m4a` and `*.caf` (sorted union), same row schema as today.
   - Exit message: no matching audio files (mention both extensions).

3. Update module docstring and argparse `description` / help text that say “m4a only” so they say m4a + caf (bootstrap + probe).

No CLI flags, no rename of the script, no changes to `dedupe_m4a_exports.py`.

## Expected behavior after change

- Missing CSV → bootstrap lists both extensions from `--audio-dir`.
- Existing CSV with `.caf` rows → probe as today; duration filled when ffprobe reports it; `extracted_creation_time` usually empty for CAF; Windows Date created only updated when a tag is present.