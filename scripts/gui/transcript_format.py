"""Load and render timestamped transcripts for the GUI preview and viewer."""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import pandas as pd

from scripts.transcript_outputs import TRANSCRIPT_PATH_COLUMNS

# Columns that may hold transcript output paths on a recording row.
_TRANSCRIPT_PATH_COLS: Sequence[str] = TRANSCRIPT_PATH_COLUMNS

# Longest-first so ".words.json" wins over ".json".
_TRANSCRIPT_SUFFIXES: Sequence[str] = (
    ".words.json",
    ".words.csv",
    ".words.vtt",
    ".words.srt",
    ".words.tsv",
    ".json",
    ".csv",
    ".vtt",
    ".srt",
    ".tsv",
    ".txt",
)


@dataclass
class WordTiming:
    text: str
    start: float
    end: float
    confidence: Optional[float] = None


@dataclass
class SegmentTiming:
    text: str
    start: float
    end: float
    confidence: Optional[float] = None
    words: List[WordTiming] = field(default_factory=list)


@dataclass
class TranscriptDoc:
    segments: List[SegmentTiming] = field(default_factory=list)
    language: Optional[str] = None
    source: str = ""  # e.g. "json", "srt", "vtt", "txt"
    plain_only: bool = False  # True when only untimed plain text is available
    plain_text: str = ""
    recording_start: Optional[datetime] = None


def resolve_path(row: Any, col: str) -> Optional[Path]:
    """Resolve a transcript_* column value to an existing local Path (WSL-aware)."""
    val = row.get(col, "") if hasattr(row, "get") else ""
    if pd.isna(val) or not str(val).strip():
        return None
    p = Path(str(val).strip())
    if not p.is_file() and str(p).startswith("/mnt/"):
        m = re.match(r"^/mnt/([a-zA-Z])/(.*)$", str(p))
        if m:
            p = Path(f"{m.group(1).upper()}:/{m.group(2)}")
        ## END if m....
    ## END if not p.is_file() and WSL path....

    return p if p.is_file() else None


def has_any_transcript(row: Any) -> bool:
    """True if any transcript_* path column is non-empty (file need not exist)."""
    for col in _TRANSCRIPT_PATH_COLS:
        val = row.get(col, "") if hasattr(row, "get") else ""
        if pd.notna(val) and str(val).strip():
            return True
        ## END if pd.notna(val) and str(val).strip()....
    ## END for col in _TRANSCRIPT_PATH_COLS....

    return False


def format_timestamp(seconds: float) -> str:
    """Format relative seconds with centisecond precision (hours only when needed)."""
    if seconds < 0:
        seconds = 0.0
    total_cs = int(round(seconds * 100))
    h = total_cs // 360_000
    rem = total_cs % 360_000
    m = rem // 6_000
    rem = rem % 6_000
    s = rem // 100
    cs = rem % 100
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}.{cs:02d}"
    return f"{m:02d}:{s:02d}.{cs:02d}"


def parse_recording_datetime(row: Any) -> Optional[datetime]:
    """Prefer extracted_creation_time, then filesystem creation_time."""
    for col in ("extracted_creation_time", "creation_time"):
        val = row.get(col, "") if hasattr(row, "get") else ""
        if pd.isna(val) or not str(val).strip():
            continue
        try:
            parsed = pd.to_datetime(val, errors="coerce")
        except (TypeError, ValueError):
            continue
        if pd.isna(parsed):
            continue
        dt = parsed.to_pydatetime()
        if getattr(dt, "tzinfo", None) is not None:
            dt = dt.replace(tzinfo=None)
        ## END if tzinfo....

        return dt
    ## END for col in extracted/creation....

    return None


def _absolute_at(recording_start: datetime, offset_seconds: float) -> datetime:
    if offset_seconds < 0:
        offset_seconds = 0.0
    return recording_start + timedelta(seconds=float(offset_seconds))


def format_clock_time(dt: datetime, *, with_seconds: bool = True) -> str:
    """iMessage-like clock: '6:48:35 PM' or '6:48 PM'."""
    if with_seconds:
        return dt.strftime("%I:%M:%S %p").lstrip("0")
    return dt.strftime("%I:%M %p").lstrip("0")


def format_day_header(dt: datetime) -> str:
    """Centered day label, e.g. 'Mon, Sep 29, 2026'."""
    return dt.strftime("%a, %b %d, %Y").replace(" 0", " ")


def format_started_at(dt: datetime) -> str:
    return f"Started {format_day_header(dt)} · {format_clock_time(dt)}"


def format_segment_time(
    offset_seconds: float,
    recording_start: Optional[datetime],
    *,
    with_seconds: bool = True,
) -> str:
    """Absolute clock when recording_start is known; else relative duration."""
    if recording_start is not None:
        return format_clock_time(
            _absolute_at(recording_start, offset_seconds),
            with_seconds=with_seconds,
        )
    ## END if recording_start is not None....

    return format_timestamp(offset_seconds)


def parse_timestamp_to_seconds(ts_str: str) -> float:
    """Parse '00:01:23,456' or '01:23.456' to float seconds."""
    s = ts_str.replace(",", ".").strip()
    # Drop WebVTT cue settings after the end time if present on a lone token
    parts = s.split(":")
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(s)
    except (ValueError, IndexError):
        return 0.0


def _confidence_pct(conf: Optional[float]) -> Optional[str]:
    if conf is None:
        return None
    try:
        return f"{int(round(float(conf) * 100))}%"
    except (TypeError, ValueError):
        return None


def _load_from_json(path: Path) -> Optional[TranscriptDoc]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None

    segments: List[SegmentTiming] = []
    for s in data.get("segments", []):
        txt = str(s.get("text", "")).strip()
        words_raw = s.get("words") or []
        words: List[WordTiming] = []
        for w in words_raw:
            wtxt = str(w.get("text", "")).strip()
            if not wtxt:
                continue
            wconf = w.get("confidence")
            words.append(
                WordTiming(
                    text=wtxt,
                    start=float(w.get("start", 0.0)),
                    end=float(w.get("end", 0.0)),
                    confidence=float(wconf) if wconf is not None else None,
                )
            )
        ## END for w in words_raw....

        if not txt and not words:
            continue
        if not txt and words:
            txt = " ".join(w.text for w in words)
        conf = s.get("confidence")
        segments.append(
            SegmentTiming(
                text=txt,
                start=float(s.get("start", 0.0)),
                end=float(s.get("end", 0.0)),
                confidence=float(conf) if conf is not None else None,
                words=words,
            )
        )
    ## END for s in data.get("segments", [])....

    lang = data.get("language")
    return TranscriptDoc(
        segments=segments,
        language=str(lang) if lang else None,
        source="json",
    )


def _parse_cue_blocks(content: str, *, skip_webvtt_header: bool) -> List[SegmentTiming]:
    segments: List[SegmentTiming] = []
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if skip_webvtt_header and lines and lines[0].upper().startswith("WEBVTT"):
            continue
        for i, ln in enumerate(lines):
            if "-->" not in ln:
                continue
            pts = ln.split("-->")
            # End time may include cue settings; take first token
            end_tok = pts[1].strip().split()[0] if pts[1].strip() else "0"
            t_s = parse_timestamp_to_seconds(pts[0])
            t_e = parse_timestamp_to_seconds(end_tok)
            txt = " ".join(lines[i + 1 :])
            if txt:
                segments.append(SegmentTiming(text=txt, start=t_s, end=t_e))
            break
        ## END for i, ln in enumerate(lines)....
    ## END for block in blocks....

    return segments


def _load_from_srt(path: Path) -> Optional[TranscriptDoc]:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    segments = _parse_cue_blocks(content, skip_webvtt_header=False)
    if not segments:
        return None
    return TranscriptDoc(segments=segments, source="srt")


def _load_from_vtt(path: Path) -> Optional[TranscriptDoc]:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    segments = _parse_cue_blocks(content, skip_webvtt_header=True)
    if not segments:
        return None
    return TranscriptDoc(segments=segments, source="vtt")


def _load_from_txt(path: Path) -> Optional[TranscriptDoc]:
    try:
        plain = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    if not plain:
        return None
    return TranscriptDoc(plain_only=True, plain_text=plain, source="txt")


def _base_name_from_transcript_file(path: Path) -> str:
    """Strip a known transcript suffix to get the recording base name."""
    name = path.name
    for suffix in _TRANSCRIPT_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
        ## END if name.endswith(suffix)....
    ## END for suffix in _TRANSCRIPT_SUFFIXES....

    return path.stem


def _iter_resolved_transcript_paths(row: Any) -> List[Path]:
    """Existing files referenced by any transcript_* column on the row."""
    paths: List[Path] = []
    seen: set[str] = set()
    for col in _TRANSCRIPT_PATH_COLS:
        p = resolve_path(row, col)
        if p is None:
            continue
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        paths.append(p)
    ## END for col in _TRANSCRIPT_PATH_COLS....

    return paths


def _sibling_path(anchor: Path, base: str, suffix: str) -> Optional[Path]:
    candidate = anchor.parent / f"{base}{suffix}"
    return candidate if candidate.is_file() else None


def _timed_candidates(row: Any) -> List[Path]:
    """Ordered candidate files for timestamped load: column paths, then siblings."""
    ordered: List[Path] = []
    seen: set[str] = set()

    def _add(p: Optional[Path]) -> None:
        if p is None:
            return
        key = str(p.resolve())
        if key in seen:
            return
        seen.add(key)
        ordered.append(p)

    # Explicit columns first (richest → poorest)
    _add(resolve_path(row, "transcript_json"))
    _add(resolve_path(row, "transcript_srt"))
    _add(resolve_path(row, "transcript_vtt"))

    # Filelists often only store transcript_txt; discover siblings beside any known file
    for known in _iter_resolved_transcript_paths(row):
        base = _base_name_from_transcript_file(known)
        _add(_sibling_path(known, base, ".words.json"))
        _add(_sibling_path(known, base, ".srt"))
        _add(_sibling_path(known, base, ".vtt"))
    ## END for known in _iter_resolved_transcript_paths(row)....

    return ordered


def _attach_recording_start(doc: Optional[TranscriptDoc], row: Any) -> Optional[TranscriptDoc]:
    if doc is None:
        return None
    doc.recording_start = parse_recording_datetime(row)
    return doc


def load_transcript(row: Any) -> Optional[TranscriptDoc]:
    """Load a TranscriptDoc: JSON → SRT → VTT (incl. siblings) → TXT."""
    for path in _timed_candidates(row):
        name = path.name.lower()
        if name.endswith(".words.json") or name.endswith(".json"):
            doc = _load_from_json(path)
            if doc is not None and doc.segments:
                return _attach_recording_start(doc, row)
            continue
        ## END if json....

        if name.endswith(".srt"):
            doc = _load_from_srt(path)
            if doc is not None and doc.segments:
                return _attach_recording_start(doc, row)
            continue
        ## END if srt....

        if name.endswith(".vtt"):
            doc = _load_from_vtt(path)
            if doc is not None and doc.segments:
                return _attach_recording_start(doc, row)
            continue
        ## END if vtt....
    ## END for path in _timed_candidates(row)....

    txt_path = resolve_path(row, "transcript_txt")
    if txt_path is not None:
        return _attach_recording_start(_load_from_txt(txt_path), row)
    ## END if txt_path is not None....

    # Last resort: sibling .txt next to any other transcript path
    for known in _iter_resolved_transcript_paths(row):
        base = _base_name_from_transcript_file(known)
        sib = _sibling_path(known, base, ".txt")
        if sib is not None:
            return _attach_recording_start(_load_from_txt(sib), row)
        ## END if sib is not None....
    ## END for known in _iter_resolved_transcript_paths(row)....

    return None


def _day_key(dt: datetime) -> Tuple[int, int, int]:
    return (dt.year, dt.month, dt.day)


def _chat_header_lines(doc: TranscriptDoc) -> List[str]:
    """Plain header lines for the top of the transcript."""
    lines: List[str] = []
    if doc.recording_start is not None:
        lines.append(format_started_at(doc.recording_start))
    ## END if doc.recording_start....

    meta: List[str] = []
    if doc.language:
        meta.append(doc.language)
    ## END if doc.language....

    if not doc.plain_only and doc.segments:
        meta.append(f"{len(doc.segments)} messages")
        if doc.recording_start is not None:
            end_dt = _absolute_at(
                doc.recording_start, max(s.end for s in doc.segments)
            )
            meta.append(f"until {format_clock_time(end_dt)}")
        else:
            span_start = doc.segments[0].start
            span_end = max(s.end for s in doc.segments)
            meta.append(
                f"{format_timestamp(span_start)} – {format_timestamp(span_end)}"
            )
        ## END if doc.recording_start....
    ## END if segments....

    if meta:
        lines.append(" · ".join(meta))
    ## END if meta....

    return lines


def _bubble_html(text: str, *, pre_escaped: bool = False) -> str:
    """Left-leaning chat bubble (QTextDocument-friendly markup)."""
    body = text if pre_escaped else html.escape(text)
    return (
        "<table width=\"100%\" cellspacing=\"0\" cellpadding=\"0\">"
        "<tr><td width=\"4\"></td><td>"
        "<div style=\"background-color: #3a3a5c; color: #e0e0ef; "
        "font-size: 13px; line-height: 1.45; padding: 10px 14px; "
        f"border-radius: 16px;\">{body}</div>"
        "</td><td width=\"18%\"></td></tr></table>"
    )


def render_html(doc: TranscriptDoc, *, include_words: bool = False) -> str:
    """Render an iMessage-like chat log HTML transcript for QTextEdit.setHtml."""
    parts: List[str] = [
        "<div style=\"font-family: 'Segoe UI', system-ui, sans-serif; "
        "padding: 8px 4px;\">"
    ]

    # Top: recording datetime + meta (centered, like a chat date stamp)
    header_lines = _chat_header_lines(doc)
    if header_lines:
        parts.append(
            "<p align=\"center\" style=\"color: #c8c8dc; font-size: 13px; "
            f"font-weight: 600; margin: 4px 0 2px 0;\">"
            f"{html.escape(header_lines[0])}</p>"
        )
        if len(header_lines) > 1:
            parts.append(
                "<p align=\"center\" style=\"color: #6e6e88; font-size: 11px; "
                f"margin: 0 0 16px 0;\">{html.escape(header_lines[1])}</p>"
            )
        else:
            parts.append("<div style=\"margin-bottom: 16px;\"></div>")
        ## END if len(header_lines) > 1....
    ## END if header_lines....

    if doc.plain_only:
        escaped = html.escape(doc.plain_text).replace("\n", "<br>")
        parts.append(_bubble_html(escaped, pre_escaped=True))
        parts.append("</div>")
        return "".join(parts)
    ## END if doc.plain_only....

    last_day: Optional[Tuple[int, int, int]] = None
    if doc.recording_start is not None:
        last_day = _day_key(doc.recording_start)
    ## END if doc.recording_start....

    for seg in doc.segments:
        # Day separator when the absolute clock crosses midnight
        if doc.recording_start is not None:
            seg_dt = _absolute_at(doc.recording_start, seg.start)
            day = _day_key(seg_dt)
            if last_day is not None and day != last_day:
                parts.append(
                    "<p align=\"center\" style=\"color: #9898b0; font-size: 11px; "
                    "font-weight: 600; margin: 18px 0 10px 0;\">"
                    f"{html.escape(format_day_header(seg_dt))}</p>"
                )
            ## END if day changed....

            last_day = day
        ## END if doc.recording_start....

        time_label = format_segment_time(seg.start, doc.recording_start)
        conf_label = _confidence_pct(seg.confidence)
        time_extra = f" · {conf_label}" if conf_label else ""
        parts.append(
            "<p align=\"center\" style=\"color: #6e6e88; font-size: 11px; "
            f"margin: 12px 0 4px 0;\">{html.escape(time_label + time_extra)}</p>"
        )
        parts.append(_bubble_html(seg.text))

        if include_words and seg.words:
            word_bits: List[str] = []
            for w in seg.words:
                w_t = format_segment_time(w.start, doc.recording_start)
                word_bits.append(
                    f"<span style=\"margin-right: 10px; white-space: nowrap;\">"
                    f"<span style=\"color: #c8c8dc;\">{html.escape(w.text)}</span> "
                    f"<span style=\"color: #6e6e88; font-size: 10px;\">"
                    f"{html.escape(w_t)}</span></span>"
                )
            ## END for w in seg.words....

            parts.append(
                "<div style=\"color: #9898b0; font-size: 11px; margin: 4px 8px 0 8px; "
                "line-height: 1.6;\">"
                + " ".join(word_bits)
                + "</div>"
            )
        ## END if include_words and seg.words....
    ## END for seg in doc.segments....

    parts.append("</div>")
    return "".join(parts)


def render_plain(doc: TranscriptDoc, *, include_words: bool = False) -> str:
    """Plain-text chat-log form matching the readable HTML layout (clipboard)."""
    lines: List[str] = []
    for h in _chat_header_lines(doc):
        lines.append(h)
    ## END for h in _chat_header_lines(doc)....

    if lines:
        lines.append("")
    ## END if lines....

    if doc.plain_only:
        lines.append(doc.plain_text)
        return "\n".join(lines).rstrip() + "\n"
    ## END if doc.plain_only....

    last_day: Optional[Tuple[int, int, int]] = None
    if doc.recording_start is not None:
        last_day = _day_key(doc.recording_start)
    ## END if doc.recording_start....

    for seg in doc.segments:
        if doc.recording_start is not None:
            seg_dt = _absolute_at(doc.recording_start, seg.start)
            day = _day_key(seg_dt)
            if last_day is not None and day != last_day:
                lines.append(f"—— {format_day_header(seg_dt)} ——")
                lines.append("")
            ## END if day changed....

            last_day = day
        ## END if doc.recording_start....

        time_label = format_segment_time(seg.start, doc.recording_start)
        conf_label = _confidence_pct(seg.confidence)
        head = time_label if not conf_label else f"{time_label}  {conf_label}"
        lines.append(head)
        lines.append(seg.text)
        if include_words and seg.words:
            word_line = "  ".join(
                f"{w.text} {format_segment_time(w.start, doc.recording_start)}"
                for w in seg.words
            )
            lines.append(f"  {word_line}")
        ## END if include_words and seg.words....

        lines.append("")
    ## END for seg in doc.segments....

    return "\n".join(lines).rstrip() + "\n"
