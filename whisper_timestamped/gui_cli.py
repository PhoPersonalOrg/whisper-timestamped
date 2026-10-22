"""Console-script wrapper for the transcription GUI.

The GUI lives under ``scripts/gui`` (outside the installable package). This
module only adds the source checkout root to ``sys.path`` and delegates to
``scripts.gui.main_window.main``. When frozen by PyInstaller, the GUI modules
are bundled and the source-checkout check is skipped.
"""

from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _running_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def cli() -> None:
    """Launch the Audio Recording Manager GUI."""
    multiprocessing.freeze_support()

    if not _running_frozen():
        repo_root = _repo_root()
        gui_dir = repo_root / "scripts" / "gui"
        if not gui_dir.is_dir():
            sys.stderr.write(
                "Transcription GUI is only available from a source checkout "
                f"(expected {gui_dir}).\n"
            )
            raise SystemExit(1)

        root_str = str(repo_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

    try:
        from scripts.gui.main_window import main
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing == "PyQt6" or missing.startswith("PyQt6."):
            sys.stderr.write(
                "PyQt6 is required for the GUI. Install and launch with:\n"
                "  uv sync --extra gui\n"
                "  uv run --extra gui whisper_timestamped_gui\n"
                "or double-click Launch GUI.bat on Windows.\n"
            )
            raise SystemExit(1) from exc
        raise

    main()
