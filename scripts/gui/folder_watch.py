"""Background watchdog thread that emits complete, transcribable recordings."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import QThread, pyqtSignal

from whisper_timestamped.recording_formats.completeness import (
    DEFAULT_MIN_AGE_S,
    FileSnapshot,
    is_recording_complete,
    path_in_watch_scope,
    recording_path_for_event,
    skip_watch_reason,
)
from whisper_timestamped.recording_formats.registry import get_format


class FolderWatchWorker(QThread):
    """Watch enabled folders; emit ``file_ready(format_id, full_path)`` when complete."""

    file_ready = pyqtSignal(str, str)
    status_message = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(
        self,
        folders: Optional[List[Dict[str, Any]]] = None,
        *,
        min_age_s: float = DEFAULT_MIN_AGE_S,
        poll_s: float = 1.0,
        emit_cooldown_s: float = 60.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._folders: List[Dict[str, Any]] = list(folders or [])
        self.min_age_s = float(min_age_s)
        self.poll_s = float(poll_s)
        self.emit_cooldown_s = float(emit_cooldown_s)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._restart = True
        self._pending: Dict[Path, Tuple[str, Optional[FileSnapshot]]] = {}
        self._emitted_at: Dict[Path, float] = {}

    def set_folders(self, folders: List[Dict[str, Any]]) -> None:
        """Replace the watch list and restart observers on the next poll."""
        with self._lock:
            self._folders = list(folders)
            self._restart = True

    def stop_watch(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
        except Exception as exc:
            self.error.emit(
                "watchdog is not installed. Install with: uv sync --extra gui"
            )
            return
        ## END try import watchdog....

        class _Handler(FileSystemEventHandler):
            def __init__(
                self,
                worker: "FolderWatchWorker",
                format_id: str,
                root: Path,
                recursive: bool,
            ) -> None:
                super().__init__()
                self.worker = worker
                self.format_id = format_id
                self.root = root
                self.recursive = recursive

            def on_created(self, event):  # type: ignore[override]
                if event.is_directory:
                    return
                self.worker._note_event(
                    Path(event.src_path), self.format_id, self.root, self.recursive
                )

            def on_modified(self, event):  # type: ignore[override]
                if event.is_directory:
                    return
                self.worker._note_event(
                    Path(event.src_path), self.format_id, self.root, self.recursive
                )

            def on_moved(self, event):  # type: ignore[override]
                if event.is_directory:
                    return
                dest = getattr(event, "dest_path", None)
                if dest:
                    self.worker._note_event(
                        Path(dest), self.format_id, self.root, self.recursive
                    )
                ## END if dest....


        observer = None
        try:
            while not self._stop.is_set():
                with self._lock:
                    restart = self._restart
                    folders = list(self._folders)
                    self._restart = False
                if restart:
                    if observer is not None:
                        observer.stop()
                        observer.join(timeout=5)
                    ## END if observer is not None....

                    observer = Observer()
                    watching = 0
                    for spec in folders:
                        if not spec.get("enabled", True):
                            continue
                        root = Path(str(spec.get("path") or ""))
                        if not root.is_dir():
                            continue
                        format_id = str(spec.get("format_id") or "").strip()
                        if not format_id:
                            continue
                        recursive = bool(spec.get("recursive", False))
                        try:
                            get_format(format_id)
                        except KeyError:
                            continue
                        handler = _Handler(self, format_id, root, recursive)
                        observer.schedule(
                            handler, root.as_posix(), recursive=recursive
                        )
                        watching += 1
                    ## END for spec in folders....

                    observer.start()
                    self.status_message.emit(f"Watching {watching} folder(s)")
                ## END if restart....

                self._flush_pending()
                self._stop.wait(self.poll_s)
            ## END while not self._stop....

        except Exception as exc:
            self.error.emit(f"Folder watch failed: {exc}")
        finally:
            if observer is not None:
                observer.stop()
                observer.join(timeout=5)
            ## END if observer is not None....


    def _note_event(
        self,
        path: Path,
        format_id: str,
        root: Path,
        recursive: bool,
    ) -> None:
        try:
            fmt = get_format(format_id)
        except KeyError:
            return
        media = recording_path_for_event(path)
        if not path_in_watch_scope(media, root, recursive):
            return
        if skip_watch_reason(media, fmt) is not None:
            return
        now = time.time()
        with self._lock:
            last = self._emitted_at.get(media)
            if last is not None and (now - last) < self.emit_cooldown_s:
                return
            if media not in self._pending:
                self._pending[media] = (format_id, None)
            ## END if media not in self._pending....


    def _flush_pending(self) -> None:
        now = time.time()
        with self._lock:
            stale = [
                path
                for path, ts in self._emitted_at.items()
                if (now - ts) >= self.emit_cooldown_s
            ]
            for path in stale:
                self._emitted_at.pop(path, None)
            ## END for path in stale....

            items = list(self._pending.items())
        ## END with self._lock....

        for path, (format_id, previous) in items:
            try:
                fmt = get_format(format_id)
            except KeyError:
                with self._lock:
                    self._pending.pop(path, None)
                continue
            ## END try get_format....

            complete, snap = is_recording_complete(
                path, fmt, previous, min_age_s=self.min_age_s
            )
            if snap is None:
                with self._lock:
                    self._pending.pop(path, None)
                continue
            ## END if snap is None....

            with self._lock:
                if path in self._pending:
                    self._pending[path] = (format_id, snap)
                ## END if path in self._pending....

            if not complete:
                continue
            with self._lock:
                self._pending.pop(path, None)
                self._emitted_at[path] = time.time()
            self.file_ready.emit(format_id, str(path))
        ## END for path, (format_id, previous) in items....
