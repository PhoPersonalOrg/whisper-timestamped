"""Entry point for running the GUI via `python -m scripts.gui` or `python scripts/gui`."""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.gui.main_window import main

if __name__ == "__main__":
    main()
