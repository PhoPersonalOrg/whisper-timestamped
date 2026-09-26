---
name: WSL path auto-convert
overview: Add a small host-path helper in process_recordings.py that maps Windows drive paths to `/mnt/<drive>/...` under WSL2 (and the reverse on native Windows), then apply it when resolving recordings_dir and output_dir so one path literal works in both environments.
todos: []
isProject: false
---

# Auto-convert Windows/WSL paths in process_recordings

## Problem

[`scripts/process_recordings.py`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\scripts\process_recordings.py) currently hardcodes both forms around lines 332–337:

```python
recordings_dir = Path(r"H:/backups/.../Audio").resolve()
recordings_dir = Path(r"/mnt/h/backups/.../Audio").resolve()  # WSL override
```

Calling `.resolve()` on a Windows drive letter under WSL before conversion is unreliable. Keep a single Windows-style literal and convert only when running in WSL2.

## Approach

Add two helpers near the top of the script (after imports):

- `_running_in_wsl()` — true if `WSL_DISTRO_NAME` / `WSL_INTEROP` is set, or `/proc/version` contains `microsoft`.
- `host_path(path)` — normalize to `Path`:
  - `H:/foo` or `H:\foo` → `/mnt/h/foo` when in WSL; leave as `H:/foo` on Windows.
  - `/mnt/h/foo` → `H:/foo` when on native Windows (`sys.platform == "win32"` and not WSL); leave as-is in WSL.
  - Other paths unchanged.
  - Convert **before** `.resolve()`.

Use it inside `process_recordings` when coercing `recordings_dir` and `output_dir` (and `model_path_root` if it looks like a Windows drive path), so every caller benefits.

Clean `__main__` to a single pair of Windows paths:

```python
recordings_dir = host_path(r"H:/backups/2026-09-21_iPhone15Pro/WhisperApp/Audio")
output_dir = host_path(r"H:/backups/2026-09-21_iPhone15Pro/WhisperApp/transcriptions")
```

Remove the duplicate `/mnt/h/...` lines.

No new dependencies; no changes outside this script.