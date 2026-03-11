import argparse
import time
from pathlib import Path
from typing import Optional, Set

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except Exception as e:  # pragma: no cover
    Observer = None
    FileSystemEventHandler = object  # type: ignore

import sys
scripts_dir = Path(__file__).parent.parent / "scripts"
if str(scripts_dir.resolve()) not in sys.path:
    sys.path.append(str(scripts_dir.resolve()))

from process_recordings import process_recordings


VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"]


def is_stable(path: Path, min_age_s: float = 10.0) -> bool:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return False
    # file must be older than min_age_s and not growing
    size1 = stat.st_size
    t0 = time.time()
    if t0 - stat.st_mtime < min_age_s:
        return False
    time.sleep(1.0)
    try:
        size2 = path.stat().st_size
    except FileNotFoundError:
        return False
    return size1 == size2


class Handler(FileSystemEventHandler):
    def __init__(self, recordings_dir: Path, output_dir: Optional[Path], min_age_s: float, cool_down_s: float):
        self.recordings_dir = recordings_dir
        self.output_dir = output_dir
        self.min_age_s = min_age_s
        self.cool_down_s = cool_down_s
        self._recent: Set[Path] = set()

    def on_created(self, event):  # type: ignore[override]
        if event.is_directory:
            return
        p = Path(event.src_path)
        if p.suffix.lower() not in [e.lower() for e in VIDEO_EXTS]:
            return
        self._maybe_process(p)

    def on_modified(self, event):  # type: ignore[override]
        if event.is_directory:
            return
        p = Path(event.src_path)
        if p.suffix.lower() not in [e.lower() for e in VIDEO_EXTS]:
            return
        self._maybe_process(p)

    def _maybe_process(self, p: Path):
        if p in self._recent:
            return
        if not is_stable(p, self.min_age_s):
            return
        self._recent.add(p)
        try:
            process_recordings(recordings_dir=self.recordings_dir, output_dir=self.output_dir, video_extensions=[p.suffix.lower()])
        finally:
            # prevent reprocessing for cool_down_s seconds
            def _remove_later(path: Path, delay: float):
                time.sleep(delay)
                self._recent.discard(path)
            import threading
            threading.Thread(target=_remove_later, args=(p, self.cool_down_s), daemon=True).start()


def cli(argv=None):
    p = argparse.ArgumentParser(description="Watch a folder for new videos and transcribe them")
    p.add_argument("recordings_dir", help="Folder to watch")
    p.add_argument("--output-dir", dest="output_dir", default=None, help="Where to save transcriptions (default: <recordings_dir>/transcriptions)")
    p.add_argument("--min-age", dest="min_age_s", type=float, default=10.0, help="Seconds file must be unchanged before processing")
    p.add_argument("--cooldown", dest="cool_down_s", type=float, default=60.0, help="Prevent reprocessing same file for this many seconds")

    args = p.parse_args(argv)

    recordings_dir = Path(args.recordings_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None

    if Observer is None:
        raise RuntimeError('watchdog is not installed. Please install with \'pip install ".[watch]"\' or \'pip install watchdog\'.')

    handler = Handler(recordings_dir, output_dir, args.min_age_s, args.cool_down_s)
    observer = Observer()
    observer.schedule(handler, recordings_dir.as_posix(), recursive=False)
    observer.start()
    print(f"Watching {recordings_dir.as_posix()} for new videos... Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    cli()


