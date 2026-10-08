"""Dark theme QSS stylesheet for the Audio Recording Manager GUI."""

from __future__ import annotations

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont, QPalette, QColor
from PyQt6.QtCore import Qt


# -- Status colours -----------------------------------------------------------

STATUS_COLORS = {
    "transcribed": "#4ade80",   # green
    "pending":     "#facc15",   # amber
    "duplicate":   "#f87171",   # red
    "playing":     "#60a5fa",   # blue
}

# Background tints (lower opacity versions for row highlighting)
ROW_TINTS = {
    "transcribed": QColor(74, 222, 128, 28),
    "pending":     QColor(250, 204, 21, 22),
    "duplicate":   QColor(248, 113, 113, 32),
    "playing":     QColor(96, 165, 250, 36),
}

# -- Palette ------------------------------------------------------------------

_BG_DARK      = "#1e1e2e"
_BG_MID       = "#282840"
_BG_WIDGET    = "#313150"
_BG_HOVER     = "#3b3b5c"
_FG           = "#e0e0ef"
_FG_DIM       = "#9898b0"
_ACCENT       = "#7c5cfc"
_ACCENT_HOVER = "#9b80fd"
_BORDER       = "#3f3f5f"
_SELECTION_BG = "#4a3fa8"


DARK_QSS = f"""
/* ── Global ─────────────────────────────────────────────────────── */
QMainWindow, QDialog {{
    background-color: {_BG_DARK};
    color: {_FG};
}}

QWidget {{
    color: {_FG};
    font-family: "Segoe UI", "Inter", "Helvetica Neue", sans-serif;
    font-size: 13px;
}}

/* ── Menu / Toolbar ─────────────────────────────────────────────── */
QMenuBar {{
    background-color: {_BG_MID};
    border-bottom: 1px solid {_BORDER};
    padding: 2px 4px;
}}
QMenuBar::item:selected {{
    background-color: {_BG_HOVER};
    border-radius: 4px;
}}

QMenu {{
    background-color: {_BG_MID};
    border: 1px solid {_BORDER};
    border-radius: 6px;
    padding: 4px;
}}
QMenu::item {{
    padding: 6px 28px 6px 12px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background-color: {_ACCENT};
}}

QToolBar {{
    background-color: {_BG_MID};
    border-bottom: 1px solid {_BORDER};
    spacing: 6px;
    padding: 4px 8px;
}}

/* ── Buttons ────────────────────────────────────────────────────── */
QPushButton {{
    background-color: {_BG_WIDGET};
    border: 1px solid {_BORDER};
    border-radius: 6px;
    padding: 6px 16px;
    min-height: 24px;
}}
QPushButton:hover {{
    background-color: {_BG_HOVER};
    border-color: {_ACCENT};
}}
QPushButton:pressed {{
    background-color: {_ACCENT};
}}
QPushButton:disabled {{
    color: {_FG_DIM};
    border-color: {_BORDER};
}}

QPushButton#accentButton {{
    background-color: {_ACCENT};
    border: none;
    color: #ffffff;
    font-weight: 600;
}}
QPushButton#accentButton:hover {{
    background-color: {_ACCENT_HOVER};
}}

/* ── Table ──────────────────────────────────────────────────────── */
QTableView {{
    background-color: {_BG_DARK};
    alternate-background-color: {_BG_MID};
    gridline-color: {_BORDER};
    border: 1px solid {_BORDER};
    border-radius: 6px;
    selection-background-color: {_SELECTION_BG};
    selection-color: {_FG};
}}
QTableView::item {{
    padding: 4px 8px;
}}
QTableView::item:hover {{
    background-color: {_BG_HOVER};
}}

QHeaderView::section {{
    background-color: {_BG_MID};
    color: {_FG};
    border: none;
    border-right: 1px solid {_BORDER};
    border-bottom: 1px solid {_BORDER};
    padding: 6px 8px;
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
}}
QHeaderView::section:hover {{
    background-color: {_BG_HOVER};
}}

/* ── Inputs ─────────────────────────────────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {_BG_WIDGET};
    border: 1px solid {_BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    min-height: 22px;
    selection-background-color: {_ACCENT};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {_ACCENT};
}}

QComboBox {{
    background-color: {_BG_WIDGET};
    border: 1px solid {_BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    min-height: 22px;
    min-width: 100px;
}}
QComboBox:hover {{
    border-color: {_ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox QAbstractItemView {{
    background-color: {_BG_MID};
    border: 1px solid {_BORDER};
    selection-background-color: {_ACCENT};
}}

/* ── Scrollbar ──────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background-color: {_BG_DARK};
    width: 10px;
    border-radius: 5px;
}}
QScrollBar::handle:vertical {{
    background-color: {_BORDER};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background-color: {_FG_DIM};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QScrollBar:horizontal {{
    background-color: {_BG_DARK};
    height: 10px;
    border-radius: 5px;
}}
QScrollBar::handle:horizontal {{
    background-color: {_BORDER};
    border-radius: 5px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{
    background-color: {_FG_DIM};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

/* ── Tab Widget ─────────────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {_BORDER};
    border-radius: 6px;
    background-color: {_BG_DARK};
}}
QTabBar::tab {{
    background-color: {_BG_MID};
    border: 1px solid {_BORDER};
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 6px 16px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background-color: {_BG_DARK};
    border-bottom: 2px solid {_ACCENT};
}}
QTabBar::tab:hover:!selected {{
    background-color: {_BG_HOVER};
}}

/* ── Plain Text / Log ───────────────────────────────────────────── */
QPlainTextEdit, QTextEdit {{
    background-color: {_BG_WIDGET};
    border: 1px solid {_BORDER};
    border-radius: 6px;
    padding: 8px;
    font-family: "Cascadia Code", "Consolas", monospace;
    font-size: 12px;
}}

/* ── Slider ─────────────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    background-color: {_BORDER};
    height: 4px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background-color: {_ACCENT};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{
    background-color: {_ACCENT_HOVER};
}}
QSlider::sub-page:horizontal {{
    background-color: {_ACCENT};
    border-radius: 2px;
}}

/* ── Progress Bar ───────────────────────────────────────────────── */
QProgressBar {{
    background-color: {_BG_WIDGET};
    border: 1px solid {_BORDER};
    border-radius: 6px;
    text-align: center;
    height: 20px;
}}
QProgressBar::chunk {{
    background-color: {_ACCENT};
    border-radius: 5px;
}}

/* ── Status Bar ─────────────────────────────────────────────────── */
QStatusBar {{
    background-color: {_BG_MID};
    border-top: 1px solid {_BORDER};
    padding: 4px;
    font-size: 12px;
    color: {_FG_DIM};
}}

/* ── Dock Widget ────────────────────────────────────────────────── */
QDockWidget {{
    titlebar-close-icon: none;
    border: 1px solid {_BORDER};
}}
QDockWidget::title {{
    background-color: {_BG_MID};
    padding: 6px 8px;
    border-bottom: 1px solid {_BORDER};
    font-weight: 600;
}}

/* ── Group Box ──────────────────────────────────────────────────── */
QGroupBox {{
    border: 1px solid {_BORDER};
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: {_ACCENT};
}}

/* ── Label ──────────────────────────────────────────────────────── */
QLabel#dimLabel {{
    color: {_FG_DIM};
    font-size: 11px;
}}

QLabel#statusBadge {{
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
}}
"""


def apply_dark_theme(app: QApplication) -> None:
    """Apply the dark colour palette and QSS to the application."""
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(_BG_DARK))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(_FG))
    palette.setColor(QPalette.ColorRole.Base, QColor(_BG_WIDGET))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(_BG_MID))
    palette.setColor(QPalette.ColorRole.Text, QColor(_FG))
    palette.setColor(QPalette.ColorRole.Button, QColor(_BG_WIDGET))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(_FG))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(_ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(_BG_MID))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(_FG))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(_FG_DIM))

    app.setPalette(palette)
    app.setStyleSheet(DARK_QSS)

    # Prefer Segoe UI on Windows, then Inter
    font = QFont("Segoe UI", 10)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)
