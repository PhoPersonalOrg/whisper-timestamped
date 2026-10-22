"""CLI folder watcher: transcribe videos once they are complete."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Set

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except Exception:  # pragma: no cover
    Observer = None
    FileSystemEventHandler = object  # type: ignore

scripts_dir = Path(__file__).parent.parent / "scripts"
if str(scripts_dir.resolve()) not in sys.path:
    sys.path.append(str(scripts_dir.resolve()))

from process_recordings import process_recordings

from whisper_timestamped.recording_formats.completeness import (
    DEFAULT_MIN_AGE_S,
    FileSnapshot,
    is_recording_complete,
    recording_path_for_event,
)
from whisper_timestamped.recording_formats.debut import DebutFormat
from whisper_timestamped.recording_formats.rec_continuous import (
    RecContinuousVideoRecorderFormat,
)

VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"]


def _cli_format_for(path: Path):
    """Best-effort format so CAM sidecars gate completeness; else Debut-like video."""
    cam = RecContinuousVideoRecorderFormat()
    if cam.matches_file(path):
        return cam
    return DebutFormat()


class Handler(FileSystemEventHandler):
    def __init__(
        self,
        recordings_dir: Path,
        output_dir: Optional[Path],
        min_age_s: float,
        cool_down_s: float,
        pending: Dict[Path, Optional[FileSnapshot]],
        lock: threading.Lock,
    ):
        self.recordings_dir = recordings_dir
        self.output_dir = output_dir
        self.min_age_s = min_age_s
        self.cool_down_s = cool_down_s
        self._pending = pending
        self._lock = lock
        self._recent: Set[Path] = set()

    def on_created(self, event):  # type: ignore[override]
        if event.is_directory:
            return
        self._note(Path(event.src_path))

    def on_modified(self, event):  # type: ignore[override]
        if event.is_directory:
            return
        self._note(Path(event.src_path))

    def _note(self, p: Path) -> None:
        media = recording_path_for_event(p)
        if media.suffix.lower() not in [e.lower() for e in VIDEO_EXTS]:
            return
        with self._lock:
            self._pending.setdefault(media, None)

    def process_ready(self) -> None:
        with self._lock:
            items = list(self._pending.items())
        for path, previous in items:
            if path in self._recent:
                with self._lock:
                    self._pending.pop(path, None)
                continue
            fmt = _cli_format_for(path)
            complete, snap = is_recording_complete(
                path, fmt, previous, min_age_s=self.min_age_s
            )
            with self._lock:
                if path in self._pending:
                    self._pending[path] = snap
            if not complete:
                continue
            self._recent.add(path)
            with self._lock:
                self._pending.pop(path, None)
            try:
                process_recordings(
                    recordings_dir=self.recordings_dir,
                    output_dir=self.output_dir,
                    video_extensions=[path.suffix.lower()],
                )
            finally:
                def _remove_later(media: Path, delay: float) -> None:
                    time.sleep(delay)
                    self._recent.discard(media)

                threading.Thread(
                    target=_remove_later, args=(path, self.cool_down_s), daemon=True
                ).start()
            ## END try process_recordings....

        ## END for path, previous in items....



def cli(argv=None):
    p = argparse.ArgumentParser(
        description="Watch a folder for new videos and transcribe them"
    )
    p.add_argument("recordings_dir", help="Folder to watch")
    p.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Where to save transcriptions (default: <recordings_dir>/transcriptions)",
    )
    p.add_argument(
        "--min-age",
        dest="min_age_s",
        type=float,
        default=DEFAULT_MIN_AGE_S,
        help="Seconds file must be unchanged before processing",
    )
    p.add_argument(
        "--cooldown",
        dest="cool_down_s",
        type=float,
        default=60.0,
        help="Prevent reprocessing same file for this many seconds",
    )

    args = p.parse_args(argv)

    recordings_dir = Path(args.recordings_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None

    if Observer is None:
        raise RuntimeError(
            'watchdog is not installed. Please install with \'pip install ".[watch]"\' '
            "or 'pip install watchdog'."
        )

    pending: Dict[Path, Optional[FileSnapshot]] = {}
    lock = threading.Lock()
    handler = Handler(
        recordings_dir, output_dir, args.min_age_s, args.cool_down_s, pending, lock
    )
    observer = Observer()
    observer.schedule(handler, recordings_dir.as_posix(), recursive=False)
    observer.start()
    print(f"Watching {recordings_dir.as_posix()} for new videos... Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
            handler.process_ready()
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    cli()
