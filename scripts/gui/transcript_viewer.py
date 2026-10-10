"""Transcript viewer dialog — readable view plus tabbed raw output files."""

from __future__ import annotations

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
    QTextEdit,
    QVBoxLayout,
)

from scripts.gui.transcript_format import (
    TranscriptDoc,
    load_transcript,
    render_html,
    render_plain,
    resolve_path,
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

        self._doc: Optional[TranscriptDoc] = load_transcript(row_data)
        self._readable_editor: Optional[QTextEdit] = None

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

        found_any = False

        # Readable tab first (structured HTML from .words.json / fallbacks)
        if self._doc is not None:
            found_any = True
            readable = QTextEdit()
            readable.setReadOnly(True)
            readable.setFont(QFont("Segoe UI", 11))
            readable.setHtml(render_html(self._doc, include_words=True))
            self._readable_editor = readable
            self._tabs.addTab(readable, "Readable")
        ## END if self._doc is not None....

        # Populate tabs for each raw transcript file that exists
        self._editors: Dict[str, QPlainTextEdit] = {}
        for col_name, tab_label, mono in _TAB_SPECS:
            path = resolve_path(row_data, col_name)
            if path is None:
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
            ## END if mono....

            # Load content (limit very large files to first 500KB)
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                if len(content) > 512_000:
                    content = content[:512_000] + "\n\n… [truncated — file exceeds 500 KB]"
                ## END if len(content) > 512_000....

                editor.setPlainText(content)
            except OSError as exc:
                editor.setPlainText(f"Error reading {path.name}: {exc}")
            ## END try read path....

            self._editors[col_name] = editor
            self._tabs.addTab(editor, tab_label)
        ## END for col_name, tab_label, mono in _TAB_SPECS....

        if not found_any:
            placeholder = QPlainTextEdit()
            placeholder.setReadOnly(True)
            placeholder.setPlainText("No transcript files found for this recording.")
            self._tabs.addTab(placeholder, "—")
        ## END if not found_any....

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
        clipboard = QApplication.clipboard()
        if clipboard is None:
            return
        ## END if clipboard is None....

        if widget is self._readable_editor and self._doc is not None:
            clipboard.setText(render_plain(self._doc, include_words=True))
            return
        ## END if readable tab....

        if isinstance(widget, QPlainTextEdit):
            clipboard.setText(widget.toPlainText())
        elif isinstance(widget, QTextEdit):
            clipboard.setText(widget.toPlainText())
        ## END if widget type....
