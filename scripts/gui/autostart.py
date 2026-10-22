"""Windows login autostart for the Audio Recording Manager tray."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def launch_bat_path() -> Path:
    return repo_root() / "Launch GUI.bat"


def _startup_dir() -> Path:
    appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def shortcut_path() -> Path:
    return _startup_dir() / "AudioRecordingManager.lnk"


def fallback_bat_path() -> Path:
    return _startup_dir() / "AudioRecordingManager.bat"


def is_start_with_windows() -> bool:
    return shortcut_path().is_file() or fallback_bat_path().is_file()


def set_start_with_windows(enabled: bool) -> bool:
    """Enable or disable a Startup shortcut. Returns True on success."""
    if sys.platform != "win32":
        return False
    lnk = shortcut_path()
    bat = fallback_bat_path()
    if not enabled:
        try:
            lnk.unlink(missing_ok=True)
            bat.unlink(missing_ok=True)
            return True
        except OSError:
            return False
        ## END try unlink....

    launch = launch_bat_path()
    if not launch.is_file():
        return False
    _startup_dir().mkdir(parents=True, exist_ok=True)
    if _write_shortcut(lnk, launch):
        try:
            bat.unlink(missing_ok=True)
        except OSError:
            pass
        return True
    ## END if _write_shortcut....

    try:
        bat.write_text(
            f'@echo off\r\ncd /d "{repo_root()}"\r\ncall "{launch}" --hidden\r\n',
            encoding="utf-8",
        )
        return True
    except OSError:
        return False


def _write_shortcut(lnk: Path, launch: Path) -> bool:
    lnk_s = str(lnk).replace("'", "''")
    launch_s = str(launch).replace("'", "''")
    work_s = str(repo_root()).replace("'", "''")
    script = (
        "$ws = New-Object -ComObject WScript.Shell; "
        f"$s = $ws.CreateShortcut('{lnk_s}'); "
        f"$s.TargetPath = '{launch_s}'; "
        "$s.Arguments = '--hidden'; "
        f"$s.WorkingDirectory = '{work_s}'; "
        "$s.Save()"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            check=False,
            capture_output=True,
            timeout=20,
        )
        return completed.returncode == 0 and lnk.is_file()
    except (OSError, subprocess.TimeoutExpired):
        return False
