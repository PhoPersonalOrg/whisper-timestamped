---
name: Transcript readable display
overview: Use `.words.json` as the canonical timestamped source and share one human-readable HTML renderer between the preview dock and the full viewer (Readable tab first), with SRT/VTT/txt fallbacks and minor enablement/refresh fixes.
todos:
  - id: transcript-format
    content: Add scripts/gui/transcript_format.py (resolve, load from json/srt/vtt/txt, HTML + plain render with sub-second times)
    status: completed
  - id: preview-wire
    content: Refactor main_window preview/copy/context-menu/post-transcribe refresh to use shared helper
    status: completed
  - id: viewer-readable
    content: Add Readable first tab to TranscriptViewerDialog with word-level detail; fix Copy for HTML tab
    status: completed
isProject: false
---

# Readable timestamped transcript display

## Source format

Prefer **`.words.json`** (`transcript_json`) everywhere structured UI is needed. It is the only default output with segment + word timings and confidence (see pipeline writers in [`scripts/process_recordings.py`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\scripts\process_recordings.py)). Fallbacks stay: segment SRT → VTT → plain TXT.

## Display shape

Shared HTML (dark-theme colors already used in preview):

- Optional header: language, segment count, total span
- Each segment:
  - Time badge with **sub-second precision** (e.g. `00:00.14 – 00:00.94`, hours only when needed)
  - Confidence when present (`96%`)
  - Segment text on the next line
- **Full viewer only:** under each segment, a muted word line with per-word `start–end` (from `segments[].words[]`)
- **Preview:** segment-level only (same styling, denser)

Copy-to-clipboard uses a matching plain-text form (`[00:00.14 – 00:00.94] text`), not HTML tags.

## Implementation

### 1. New helper — [`scripts/gui/transcript_format.py`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\scripts\gui\transcript_format.py)

- `resolve_path(row, col) -> Optional[Path]` (incl. `/mnt/X/` → `X:/` WSL rewrite already duplicated in main/viewer)
- `format_timestamp(seconds) -> str` with tenths/centiseconds
- `load_transcript(row) -> TranscriptDoc` — prefer JSON segments/words; else parse SRT/VTT; else TXT
- `render_html(doc, *, include_words: bool) -> str`
- `render_plain(doc, *, include_words: bool) -> str`
- `has_any_transcript(row) -> bool` — any non-empty `transcript_*` path

Move the load/parse/render logic currently inlined in [`main_window.py`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\scripts\gui\main_window.py) `_update_transcript_preview` (~L828–964) and `_format_time` / `_parse_timestamp_to_seconds` (~L66–88) into this module.

### 2. Preview — [`scripts/gui/main_window.py`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\scripts\gui\main_window.py)

- `_update_transcript_preview` calls `load_transcript` + `render_html(..., include_words=False)`
- Copy button uses `render_plain` (store last `TranscriptDoc` or re-load from `_current_preview_row_data`)
- Context menu “View Transcript”: enable via `has_any_transcript`, not only `transcript_txt` (~L738–743)
- After `_on_file_transcribed`, if the updated row is still selected, refresh the preview

### 3. Full viewer — [`scripts/gui/transcript_viewer.py`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\whisper-timestamped\scripts\gui\transcript_viewer.py)

- Insert a first **Readable** tab (`QTextEdit`, HTML from `render_html(..., include_words=True)`) when any parseable transcript exists
- Keep existing raw-file tabs after it (txt/srt/vtt/json/csv/…); default selection = Readable
- Copy button: if current widget is the Readable `QTextEdit`, copy `render_plain(..., include_words=True)`; else keep `toPlainText()` for raw tabs

```mermaid
flowchart LR
  row[row_data paths] --> load[load_transcript]
  load --> json[".words.json"]
  load --> srt[SRT/VTT]
  load --> txt[TXT]
  json --> html[render_html]
  srt --> html
  txt --> html
  html --> preview[Preview dock]
  html --> readable[Full viewer Readable tab]
  row --> raw[Raw file tabs]
```

## Out of scope

Changing which files the transcription pipeline writes; click-to-seek from timestamps; editing transcripts.