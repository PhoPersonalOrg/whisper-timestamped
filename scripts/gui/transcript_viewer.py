"""Transcript viewer dialog — tabbed view of transcription output files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
)


# Tab config: (column_name, tab_label, is_monospace)
_TAB_SPECS = [
    ("transcript_txt",       "Plain Text", False),
    ("transcript_srt",       "SRT",        True),
    ("transcript_vtt",       "VTT",        True),
    ("transcript_json",      "JSON",       True),
    ("transcript_csv",       "CSV",        True),
    ("transcript_words_csv", "Words CSV",  True),
    ("transcript_tsv",       "TSV",        True),
]


class TranscriptViewerDialog(QDialog):
    """Tabbed read-only viewer for transcript output files."""

    def __init__(self, row_data: pd.Series, parent=None) -> None:
        super().__init__(parent)
        name = str(row_data.get("name", "Unknown"))
        self.setWindowTitle(f"Transcript — {name}")
        self.setMinimumSize(700, 500)
        self.resize(900, 650)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Header
        header = QLabel(f"<b>{name}</b>")
        header.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(header)

        # Tab widget
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs, stretch=1)

        # Populate tabs for each transcript file that exists
        self._editors: Dict[str, QPlainTextEdit] = {}
        found_any = False
        for col_name, tab_label, mono in _TAB_SPECS:
            path_str = row_data.get(col_name, "")
            if pd.isna(path_str) or not str(path_str).strip():
                continue

            path = Path(str(path_str).strip())
            # Handle WSL paths — try as-is first, then convert /mnt/X/ → X:/
            if not path.is_file() and str(path).startswith("/mnt/"):
                import re
                m = re.match(r"^/mnt/([a-zA-Z])/(.*)$", str(path))
                if m:
                    path = Path(f"{m.group(1).upper()}:/{m.group(2)}")
            ## END if WSL path conversion...

            if not path.is_file():
                continue

            found_any = True
            editor = QPlainTextEdit()
            editor.setReadOnly(True)
            if mono:
                font = QFont("Cascadia Code", 11)
                font.setStyleHint(QFont.StyleHint.Monospace)
                editor.setFont(font)
            else:
                editor.setFont(QFont("Segoe UI", 11))

            # Load content (limit very large files to first 500KB)
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                if len(content) > 512_000:
                    content = content[:512_000] + "\n\n… [truncated — file exceeds 500 KB]"
                editor.setPlainText(content)
            except OSError as exc:
                editor.setPlainText(f"Error reading {path.name}: {exc}")

            self._editors[col_name] = editor
            self._tabs.addTab(editor, tab_label)
        ## END for col_name, tab_label, mono in _TAB_SPECS...

        if not found_any:
            placeholder = QPlainTextEdit()
            placeholder.setReadOnly(True)
            placeholder.setPlainText("No transcript files found for this recording.")
            self._tabs.addTab(placeholder, "—")

        # Bottom buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.clicked.connect(self._copy_current)
        btn_layout.addWidget(copy_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)


    def _copy_current(self) -> None:
        """Copy the current tab's text to the clipboard."""
        widget = self._tabs.currentWidget()
        if isinstance(widget, QPlainTextEdit):
            text = widget.toPlainText()
            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(text)
