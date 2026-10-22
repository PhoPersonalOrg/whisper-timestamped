"""System tray icon for the Audio Recording Manager."""

from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QMenu, QSystemTrayIcon, QWidget


class TrayController:
    """Right-click shows a context menu (including Exit); left/double-click opens the GUI."""

    def __init__(
        self,
        icon: QIcon,
        parent: QWidget,
        *,
        on_open: Callable[[], None],
        on_quit: Callable[[], None],
        on_watch_folders: Callable[[], None],
        on_auto_transcribe_toggled: Callable[[bool], None],
        on_start_with_windows_toggled: Callable[[bool], None],
        auto_transcribe: bool = True,
        start_with_windows: bool = False,
        show_autostart: bool = True,
    ) -> None:
        self._on_open = on_open
        self._on_quit = on_quit
        self._on_watch_folders = on_watch_folders
        self._on_auto_transcribe_toggled = on_auto_transcribe_toggled
        self._on_start_with_windows_toggled = on_start_with_windows_toggled
        self._show_autostart = show_autostart

        self._tray = QSystemTrayIcon(icon, parent)
        self._tray.setToolTip("Audio Recording Manager")
        self._tray.activated.connect(self._on_activated)

        self._menu = QMenu(parent)
        self._open_action = QAction("Open Audio Recording Manager", self._menu)
        self._open_action.triggered.connect(self._on_open)
        self._menu.addAction(self._open_action)

        self._auto_action = QAction("Auto-transcribe new recordings", self._menu)
        self._auto_action.setCheckable(True)
        self._auto_action.setChecked(auto_transcribe)
        self._auto_action.toggled.connect(self._on_auto_transcribe_toggled)
        self._menu.addAction(self._auto_action)

        self._watch_action = QAction("Watch folders…", self._menu)
        self._watch_action.triggered.connect(self._on_watch_folders)
        self._menu.addAction(self._watch_action)

        self._autostart_action: Optional[QAction] = None
        if show_autostart:
            self._autostart_action = QAction("Start with Windows", self._menu)
            self._autostart_action.setCheckable(True)
            self._autostart_action.setChecked(start_with_windows)
            self._autostart_action.toggled.connect(self._on_start_with_windows_toggled)
            self._menu.addAction(self._autostart_action)
        ## END if show_autostart....

        self._menu.addSeparator()
        exit_action = QAction("Exit", self._menu)
        exit_action.triggered.connect(self._on_quit)
        self._menu.addAction(exit_action)

        self._tray.setContextMenu(self._menu)
        self._tray.show()

    def is_visible(self) -> bool:
        return bool(self._tray.isVisible())

    def set_tooltip(self, text: str) -> None:
        self._tray.setToolTip(text)

    def set_auto_transcribe(self, enabled: bool) -> None:
        self._auto_action.blockSignals(True)
        self._auto_action.setChecked(enabled)
        self._auto_action.blockSignals(False)

    def set_start_with_windows(self, enabled: bool) -> None:
        if self._autostart_action is None:
            return
        self._autostart_action.blockSignals(True)
        self._autostart_action.setChecked(enabled)
        self._autostart_action.blockSignals(False)

    def hide(self) -> None:
        self._tray.hide()

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
            QSystemTrayIcon.ActivationReason.MiddleClick,
        ):
            self._on_open()
        ## END if reason....
