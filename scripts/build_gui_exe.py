"""Build a one-file Windows executable for the transcription GUI.

Usage (from repo root):

    uv run --extra gui python scripts/build_gui_exe.py

Output: dist/whisper_timestamped_gui.exe

Does not bundle Whisper/Hugging Face weights (downloaded into the user cache
on first transcribe) or ffmpeg/ffprobe (must be on PATH).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENTRY = REPO_ROOT / "scripts" / "gui_exe_entry.py"

# PyInstaller refuses to start if these obsolete stdlib backports are installed
# (enum34 is pulled by the optional realtimestt extra and shadows stdlib enum).
_PYINSTALLER_BLOCKERS = ("enum34", "typing", "pathlib")


def _uninstall_pyinstaller_blockers() -> None:
    present: list[str] = []
    for name in _PYINSTALLER_BLOCKERS:
        try:
            distribution(name)
        except PackageNotFoundError:
            continue
        present.append(name)
    ## END for name in _PYINSTALLER_BLOCKERS....

    if not present:
        return

    print(
        "Uninstalling obsolete stdlib backports that block PyInstaller: "
        + ", ".join(present)
    )
    uv = shutil.which("uv")
    if uv:
        subprocess.check_call([uv, "pip", "uninstall", *present])
    else:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "uninstall", "-y", *present]
        )


def main() -> None:
    os.chdir(REPO_ROOT)
    _uninstall_pyinstaller_blockers()

    import PyInstaller.__main__

    PyInstaller.__main__.run(
        [
            str(ENTRY),
            "--onefile",
            "--console",
            "--noconfirm",
            "--clean",
            "--name=whisper_timestamped_gui",
            f"--paths={REPO_ROOT}",
            "--collect-all=PyQt6",
            "--collect-submodules=scripts",
            "--collect-submodules=scripts.gui",
            "--collect-submodules=whisper_timestamped",
            "--collect-submodules=whisper_timestamped.recording_formats",
            "--collect-submodules=whisper_timestamped.crisperwhisper",
            "--hidden-import=scripts.process_recordings",
            "--hidden-import=scripts.transcribe_queue",
            "--hidden-import=scripts.transcript_outputs",
            "--hidden-import=scripts.iOSWhisperAppHelpers.extract_m4a_creation_times",
            "--hidden-import=sounddevice",
            "--hidden-import=soundfile",
            "--exclude-module=scripts.examples",
            "--exclude-module=scripts.test_live_whisper_app",
            "--exclude-module=scripts.transcript_to_lsl",
            "--exclude-module=scripts.build_exe",
            "--exclude-module=scripts.build_gui_exe",
            "--exclude-module=scripts.verify_setup",
            "--collect-all=torch",
            "--collect-all=torchaudio",
            "--copy-metadata=openai-whisper",
            "--copy-metadata=transformers",
            "--copy-metadata=huggingface-hub",
            f"--distpath={REPO_ROOT / 'dist'}",
            f"--workpath={REPO_ROOT / 'build'}",
            f"--specpath={REPO_ROOT}",
        ]
    )


if __name__ == "__main__":
    main()
