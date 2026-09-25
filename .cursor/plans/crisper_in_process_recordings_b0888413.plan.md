---
name: Crisper in process_recordings
overview: Wire CrisperWhisper into `process_recordings.py` via new `backend` / `model_name` / `crisper_mode` args so the `__main__` WhisperApp batch can opt in without changing the default openai-whisper path for other callers.
todos: []
isProject: false
---

# Use CrisperWhisper from process_recordings.py

## What blocks it today

In [`scripts/process_recordings.py`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\scripts\process_recordings.py):

```221:225:scripts/process_recordings.py
    model_name: str = "medium.en"
    ...
    model = whisper.load_model(model_name, download_root=str(model_path_root), device=device)
```

```265:265:scripts/process_recordings.py
            result = whisper.transcribe(model, audio, language="en", vad="silero", remove_empty_words=True)
```

- No `backend=` is passed, so openai-whisper is used.
- `"medium.en"` is rejected by the Crisper backend (stock Whisper-only name). Use `"medium"` (or `small` / `turbo` / `large`).

## Changes

Add parameters to `process_recordings(...)`:

- `backend: str = "openai-whisper"`
- `model_name: str | None = None` — if unset, keep `"medium.en"` for openai-whisper / transformers; use `"medium"` when `backend="crisperwhisper"`
- `crisper_mode: str = "verbatim"`

Load and transcribe:

```python
model = whisper.load_model(
    model_name,
    download_root=str(model_path_root),
    device=device,
    backend=backend,
)
...
result = whisper.transcribe(
    model, audio, language="en", vad="silero", remove_empty_words=True,
    crisper_mode=crisper_mode,
)
```

Only assert `model_path_root.exists()` when `backend != "crisperwhisper"` (Crisper weights use the HuggingFace cache, not `F:\AITEMP\whisper_models`).

In `__main__`, call with Crisper for the current WhisperApp batch:

```python
output_files = process_recordings(
    recordings_dir=recordings_dir,
    output_dir=output_dir,
    video_extensions=video_extensions,
    backend="crisperwhisper",
    model_name="medium",
    crisper_mode="verbatim",
)
```

Return shape and `write_results` stay the same.

## Run

```powershell
uv run python scripts/process_recordings.py
```

First Crisper run downloads `nyralabs/CrisperWhisper2.0_medium` into the HF cache.