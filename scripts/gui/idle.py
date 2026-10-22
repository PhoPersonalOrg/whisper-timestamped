"""Idle detection and auto-transcribe gating (torch-free)."""

from __future__ import annotations

import sys
from typing import Any, List, Sequence

DEFAULT_RUN_WHEN = "idle"
DEFAULT_IDLE_SECONDS = 300
DEFAULT_LOW_PRIORITY = True
IDLE_SECONDS_CHOICES = (60, 120, 300, 600, 900)
_NON_WINDOWS_IDLE_SECONDS = 10**9


def normalize_run_when(value: Any) -> str:
    """Return ``idle`` or ``always``."""
    text = str(value or "").strip().lower()
    if text == "always":
        return "always"
    return DEFAULT_RUN_WHEN


def normalize_idle_seconds(value: Any) -> int:
    """Clamp idle delay to the Watch Folders combo choices."""
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        return DEFAULT_IDLE_SECONDS
    if seconds in IDLE_SECONDS_CHOICES:
        return seconds
    return min(IDLE_SECONDS_CHOICES, key=lambda choice: abs(choice - seconds))


def last_input_idle_seconds() -> float:
    """Seconds since the last keyboard/mouse input, or a large value off Windows.

    A locked Windows session typically keeps growing this value, so it counts
    as idle. On API failure return ``0.0`` (treat as busy).
    """
    if sys.platform != "win32":
        return float(_NON_WINDOWS_IDLE_SECONDS)
    try:
        import ctypes
        from ctypes import wintypes

        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.UINT),
                ("dwTime", wintypes.DWORD),
            ]

        info = LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(LASTINPUTINFO)
        user32 = ctypes.windll.user32
        user32.GetLastInputInfo.argtypes = [ctypes.POINTER(LASTINPUTINFO)]
        user32.GetLastInputInfo.restype = wintypes.BOOL
        if not user32.GetLastInputInfo(ctypes.byref(info)):
            return 0.0
        elapsed = (
            ctypes.windll.kernel32.GetTickCount() - info.dwTime
        ) & 0xFFFFFFFF
        return float(elapsed) / 1000.0
    except (AttributeError, OSError, ValueError):
        return 0.0
    ## END try GetLastInputInfo....


def is_user_idle(
    *,
    run_when: str,
    idle_seconds: int,
    idle_for: float,
) -> bool:
    """True when auto-transcribe is allowed by the idle policy."""
    if normalize_run_when(run_when) == "always":
        return True
    return float(idle_for) >= float(normalize_idle_seconds(idle_seconds))


def should_start_auto(
    *,
    watch_enabled: bool,
    worker_running: bool,
    run_when: str,
    idle_seconds: int,
    idle_for: float,
) -> bool:
    """True when the auto-transcribe queue may start a new batch."""
    if not watch_enabled:
        return False
    if worker_running:
        return False
    return is_user_idle(
        run_when=run_when,
        idle_seconds=idle_seconds,
        idle_for=idle_for,
    )


def leftover_queued_for_requeue(
    *,
    origin: str,
    paused: bool,
    queued_indices: Sequence[int],
) -> List[int]:
    """Queued df indices to put back on the auto queue after an idle pause."""
    if str(origin) != "auto" or not paused:
        return []
    leftover: List[int] = []
    seen = set()
    for raw in queued_indices:
        idx = int(raw)
        if idx in seen:
            continue
        seen.add(idx)
        leftover.append(idx)
    ## END for raw in queued_indices....

    return leftover


def apply_current_thread_low_priority() -> None:
    """Lower the calling thread's Windows priority. No-op elsewhere."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        thread_priority_lowest = -2  # THREAD_PRIORITY_LOWEST
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentThread.restype = ctypes.c_void_p
        kernel32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
        kernel32.SetThreadPriority.restype = ctypes.c_int
        handle = kernel32.GetCurrentThread()
        kernel32.SetThreadPriority(handle, thread_priority_lowest)
    except (AttributeError, OSError, ValueError):
        return
    ## END try SetThreadPriority....
