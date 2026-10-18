"""AirDesk application entry point."""

from __future__ import annotations

import argparse
import atexit
import ctypes
import logging
import logging.handlers
import sys
import threading

from . import __version__
from .config import Settings, app_dir

log = logging.getLogger("airdesk")
SERVER_NAME = "AirDesk-SingleInstance"


def setup_logging() -> None:
    logs = app_dir() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    h = logging.handlers.RotatingFileHandler(logs / "airdesk.log", maxBytes=1_000_000, backupCount=3,
                                             encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(h)
    if sys.stderr is not None:
        root.addHandler(logging.StreamHandler())
    # MediaPipe/TFLite are chatty on stderr; keep our own log clean.
    logging.getLogger("absl").setLevel(logging.ERROR)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="AirDesk")
    ap.add_argument("--minimized", action="store_true", help="start in the system tray")
    ap.add_argument("--quit-after", type=float, default=0, help=argparse.SUPPRESS)  # smoke tests
    args = ap.parse_args(argv)

    setup_logging()
    log.info("AirDesk %s starting (python %s)", __version__, sys.version.split()[0])
    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("AirDesk.App")
        except Exception:
            pass

    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QIcon
    from PySide6.QtNetwork import QLocalServer, QLocalSocket
    from PySide6.QtWidgets import QApplication, QMessageBox

    app = QApplication(sys.argv[:1])
    app.setApplicationName("AirDesk")
    app.setQuitOnLastWindowClosed(False)

    # One instance only: a second launch just shows the running window
    # (and can never open a second stream on the webcam).
    probe = QLocalSocket()
    probe.connectToServer(SERVER_NAME)
    if probe.waitForConnected(300):
        probe.write(b"show")
        probe.flush()
        probe.waitForBytesWritten(300)
        log.info("another instance is running; asked it to show itself")
        return 0
    QLocalServer.removeServer(SERVER_NAME)
    server = QLocalServer()
    server.listen(SERVER_NAME)

    from .resources import asset_path
    from .runtime import Runtime
    from .ui import theme
    from .ui.keyboard_overlay import KeyboardOverlay
    from .ui.main_window import AppContext, MainWindow

    icon = asset_path("airdesk.ico")
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    settings = Settings.load()
    theme.apply(app, settings.ui.theme)
    try:
        runtime = Runtime(settings)
    except Exception as e:
        log.exception("startup failed")
        QMessageBox.critical(None, "AirDesk", f"AirDesk could not start:\n{e}")
        return 1
    overlay = KeyboardOverlay(settings.keyboard)
    runtime.set_keyboard(overlay)
    ctx = AppContext(settings, runtime, overlay)
    win = MainWindow(ctx)
    server.newConnection.connect(lambda: (server.nextPendingConnection(), win.show_normal()))

    def safe_release() -> None:
        try:
            runtime.dispatcher.release_all()
            overlay.out.release_all()
        except Exception:
            pass
    atexit.register(safe_release)

    def excepthook(exc_type, exc, tb) -> None:
        log.critical("unhandled exception - disarming", exc_info=(exc_type, exc, tb))
        try:
            runtime.emergency_stop()
        except Exception:
            pass
    sys.excepthook = excepthook
    threading.excepthook = lambda a: excepthook(a.exc_type, a.exc_value, a.exc_traceback)

    runtime.start()
    if not args.minimized:
        win.show()
    if not settings.ui.first_run_done and not args.minimized and not args.quit_after:
        from .ui.calibration import Wizard
        QTimer.singleShot(900, lambda: Wizard(ctx).exec())
    if args.quit_after:
        QTimer.singleShot(int(args.quit_after * 1000), win.quit)
    app.aboutToQuit.connect(runtime.shutdown)
    code = app.exec()
    log.info("AirDesk exited (%s)", code)
    return code
