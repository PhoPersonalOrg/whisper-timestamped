"""Ensure only one Audio Recording Manager process is running."""

from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import QLockFile, QStandardPaths
from PyQt6.QtNetwork import QLocalServer, QLocalSocket
from PyQt6.QtWidgets import QApplication

APP_SERVER_NAME = "whisper-timestamped-AudioRecordingManager"
_LOCK_NAME = "whisper_timestamped_arm.lock"
_SHOW_MESSAGE = b"show"


class SingleInstance:
    """QLockFile + QLocalServer so a second launch raises the existing window."""

    def __init__(self) -> None:
        tmp = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.TempLocation
        )
        self._lock = QLockFile(f"{tmp}/{_LOCK_NAME}")
        self._lock.setStaleLockTime(8000)
        self._server: Optional[QLocalServer] = None

    def try_acquire(self) -> bool:
        return bool(self._lock.tryLock(200))

    def notify_existing(self) -> bool:
        socket = QLocalSocket()
        socket.connectToServer(APP_SERVER_NAME)
        if not socket.waitForConnected(800):
            return False
        socket.write(_SHOW_MESSAGE)
        socket.waitForBytesWritten(800)
        socket.flush()
        socket.disconnectFromServer()
        return True

    def listen(self, on_show: Callable[[], None]) -> None:
        QLocalServer.removeServer(APP_SERVER_NAME)
        self._server = QLocalServer()

        def _on_connection() -> None:
            if self._server is None:
                return
            incoming = self._server.nextPendingConnection()
            if incoming is None:
                return
            incoming.readyRead.connect(
                lambda sock=incoming: self._handle_payload(sock, on_show)
            )
            if incoming.bytesAvailable() > 0:
                self._handle_payload(incoming, on_show)
            ## END if bytes already available....

        self._server.newConnection.connect(_on_connection)
        self._server.listen(APP_SERVER_NAME)

    def _handle_payload(self, socket: QLocalSocket, on_show: Callable[[], None]) -> None:
        payload = bytes(socket.readAll())
        socket.disconnectFromServer()
        if _SHOW_MESSAGE in payload:
            on_show()
        ## END if show message....

    def release(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None
        ## END if self._server....

        self._lock.unlock()


def raise_window(window) -> None:
    """Bring *window* to the front (used by the second-instance handshake)."""
    window.show()
    window.raise_()
    window.activateWindow()
    app = QApplication.instance()
    if app is not None:
        app.alert(window, 0)
