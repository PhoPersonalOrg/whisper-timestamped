---
name: Fix Crisper Mode Dropdown
overview: Replace the invalid CrisperWhisper mode option `clean` in the transcription settings dialog with `intended`, matching the backend validator.
todos:
  - id: fix-mode-combo
    content: Replace Clean/clean with Intended/intended in TranscribeConfigDialog
    status: completed
isProject: false
---

# Fix CrisperWhisper Mode Dropdown

## Problem

[`scripts/gui/dialogs.py`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\scripts\gui\dialogs.py) offers `Clean` → `"clean"`, but [`whisper_timestamped/crisper_backend.py`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\whisper_timestamped\crisper_backend.py) only accepts `"verbatim"` or `"intended"`:

```273:275:whisper_timestamped/crisper_backend.py
    if crisper_mode not in ("verbatim", "intended"):
        raise ValueError(
            f"crisper_mode must be 'verbatim' or 'intended', got {crisper_mode!r}"
```

Selecting Clean therefore fails with the error you saw.

## Fix

In `TranscribeConfigDialog` (~lines 211–213), change the second item from `Clean`/`clean` to `Intended`/`intended`:

```python
self._mode_combo.addItem("Verbatim", "verbatim")
self._mode_combo.addItem("Intended", "intended")
```

No other wiring changes: `crisper_mode()` already returns the combo’s user data, and the worker/process_recordings path already pass it through unchanged.

## Out of scope

- CLI (`--crisper_mode` already has the correct choices)
- Mapping/aliases for a legacy `"clean"` value
