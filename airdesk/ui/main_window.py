"""AirDesk Control Center: navigation, pages, system tray and app wiring."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QListWidget, QMainWindow, QMenu, QStackedWidget,
                               QSystemTrayIcon, QVBoxLayout, QWidget)

from .. import __version__
from ..config import Settings
from ..runtime import Runtime
from . import theme
from .keyboard_overlay import KeyboardOverlay
from .widgets import label

log = logging.getLogger(__name__)

NAV = ["Home", "Air Mouse Pro", "Air Keyboard", "Finger Selection", "Gesture Mapping", "Calibration",
       "Precision Controls", "Benchmark Laboratory", "Diagnostics", "Settings"]


class _Bridge(QObject):
    """Runtime events arrive on worker threads; re-emit on the GUI thread."""
    event = Signal(str)


@dataclass
class AppContext:
    s: Settings
    runtime: Runtime
    overlay: KeyboardOverlay
    window: "MainWindow | None" = None
    version: str = __version__
    _save_timer: QTimer | None = field(default=None, repr=False)

    def changed(self, section: str) -> None:
        if self._save_timer is None:
            self._save_timer = QTimer(singleShot=True, interval=400)
            self._save_timer.timeout.connect(lambda: self.s.save())
        self._save_timer.start()
        if section == "keyboard":
            self.overlay.apply_config(self.s.keyboard)
        elif section == "hotkeys":
            self.runtime.reload_hotkeys()
        elif section == "ui":
            theme.apply(QApplication.instance(), self.s.ui.theme)
            if self.window:
                self.window.home._sync_preview_button()
        self.runtime.apply_settings(camera_changed=section == "camera")
        if self.window:
            self.window.tray_update()

    def toggle_keyboard(self) -> None:
        self.overlay.toggle()

    def reset_settings(self) -> None:
        fresh = Settings()
        fresh.ui.first_run_done = self.s.ui.first_run_done
        for name in ("camera", "pointer", "finger", "gesture", "keyboard", "safety", "ui"):
            setattr(self.s, name, getattr(fresh, name))
        self.s.profile = "Default"
        self.runtime.s = self.s
        self.runtime.camera.cfg = self.s.camera
        self.overlay.cfg = self.s.keyboard
        self.changed("camera")
        self.changed("keyboard")
        self.changed("hotkeys")
        if self.window:
            self.window.refresh_pages()


def _tray_icon(armed: bool, paused: bool) -> QIcon:
    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(28, 30, 36))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(2, 2, 60, 60, 14, 14)
    col = QColor(255, 182, 76) if paused else (QColor(60, 207, 142) if armed else QColor(140, 146, 160))
    p.setBrush(col)
    # a stylised raised hand: palm + four fingers
    p.drawRoundedRect(18, 30, 28, 24, 8, 8)
    for i, h in enumerate((18, 24, 22, 16)):
        p.drawRoundedRect(19 + i * 7, 32 - h, 5, h + 4, 2.5, 2.5)
    p.drawRoundedRect(42, 34, 12, 6, 3, 3)
    p.end()
    return QIcon(pm)


class MainWindow(QMainWindow):
    def __init__(self, ctx: AppContext) -> None:
        super().__init__()
        from .benchmark import BenchmarkPage
        from .calibration import CalibrationPage
        from .pages import (DiagnosticsPage, FingerPage, GesturePage, HomePage, KeyboardPage, MousePage,
                            PrecisionPage, SettingsPage)
        self.ctx = ctx
        ctx.window = self
        self.setWindowTitle("AirDesk — Your Hands. Your Computer.")
        self.setWindowIcon(_tray_icon(False, False))
        self.resize(1240, 860)
        self._quitting = False

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        side = QWidget()
        side.setObjectName("Nav")
        side.setFixedWidth(220)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(10, 14, 10, 10)
        sl.addWidget(label("AirDesk", "Brand"))
        sl.addWidget(label("Your Hands. Your Computer.", "Tagline"))
        self.nav = QListWidget()
        self.nav.setObjectName("Nav")
        self.nav.addItems(NAV)
        sl.addWidget(self.nav, 1)
        self.state_lbl = label("", "Muted", wrap=True)
        sl.addWidget(self.state_lbl)
        root.addWidget(side)
        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        self.home = HomePage(ctx)
        self.pages = [self.home, MousePage(ctx), KeyboardPage(ctx), FingerPage(ctx), GesturePage(ctx),
                      CalibrationPage(ctx), PrecisionPage(ctx), BenchmarkPage(ctx), DiagnosticsPage(ctx),
                      SettingsPage(ctx)]
        for pg in self.pages:
            self.stack.addWidget(pg.widget)
        self.nav.currentRowChanged.connect(self._goto)
        self.nav.setCurrentRow(0)

        self._build_tray()
        self.bridge = _Bridge()
        self.bridge.event.connect(self._on_event)
        ctx.runtime.listeners.append(self.bridge.event.emit)
        ctx.overlay.visibility_changed.connect(lambda _v: self.tray_update())
        ctx.overlay.config_changed.connect(lambda: ctx.changed("keyboard_ui"))

        from .feedback import FeedbackMarker
        self.marker = FeedbackMarker()
        self.timer = QTimer(self, interval=33, timeout=self._tick)
        self.timer.start()
        # The click marker polls faster so "pressed" shows as soon as Windows
        # accepted the button (it reads the interaction log, not the frame).
        self.marker_timer = QTimer(self, interval=15, timeout=self._marker_tick)
        self.marker_timer.start()
        def screens_changed(*_):
            ctx.runtime.refresh_screens()
            ctx.overlay.reclamp()
        for scr in QApplication.screens():
            scr.geometryChanged.connect(screens_changed)
            scr.availableGeometryChanged.connect(screens_changed)
        QApplication.instance().screenAdded.connect(screens_changed)
        QApplication.instance().screenRemoved.connect(screens_changed)

    # ---------------------------------------------------------------- pages
    def _goto(self, i: int) -> None:
        self.stack.setCurrentIndex(i)
        self.pages[i].refresh()

    def goto(self, name: str) -> None:
        self.nav.setCurrentRow(NAV.index(name))

    def refresh_pages(self) -> None:
        for pg in self.pages:
            pg.refresh()

    def _tick(self) -> None:
        st, m = self.ctx.runtime.snapshot()
        i = self.stack.currentIndex()
        if self.isVisible() and not self.isMinimized():
            self.pages[i].on_tick(st, m)
        rt = self.ctx.runtime.engine
        txt = "PAUSED" if rt.paused else ("ARMED" if rt.armed else "DISARMED")
        col = "warn" if rt.paused else ("ok" if rt.armed else "bad")
        self.state_lbl.setText(f"<b style='color:{theme.current[col]}'>● {txt}</b><br>"
                               f"{m.capture_fps:.0f} fps · {m.latency_ms:.0f} ms")

    def _marker_tick(self) -> None:
        st, _ = self.ctx.runtime.snapshot()
        self.marker.update_from(st, self.ctx.s.ui.feedback_marker, self.ctx.runtime.engine.clicks.latest())

    # ----------------------------------------------------------------- tray
    def _build_tray(self) -> None:
        self.tray = QSystemTrayIcon(_tray_icon(False, False), self)
        menu = QMenu()
        self.a_arm = QAction("Arm gesture input", menu, triggered=self.ctx.runtime.toggle_arm)
        self.a_pause = QAction("Pause", menu, triggered=lambda: self.ctx.runtime.pause(
            not self.ctx.runtime.engine.paused))
        self.a_kb = QAction("Show Air Keyboard", menu, triggered=self.ctx.toggle_keyboard)
        self.a_drag = QAction("Legacy Drag Mode", menu, triggered=self.ctx.runtime.toggle_drag_mode)
        self.a_drag.setCheckable(True)
        a_stop = QAction("Emergency stop (Ctrl+Alt+F12)", menu, triggered=self.ctx.runtime.emergency_stop)
        a_open = QAction("Open AirDesk", menu, triggered=self.show_normal)
        a_quit = QAction("Quit AirDesk", menu, triggered=self.quit)
        for a in (self.a_arm, self.a_pause, self.a_kb, self.a_drag):
            menu.addAction(a)
        menu.addSeparator()
        menu.addAction(a_stop)
        menu.addSeparator()
        menu.addAction(a_open)
        menu.addAction(a_quit)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(lambda r: self.show_normal() if r == QSystemTrayIcon.Trigger else None)
        self.tray.show()
        self.tray_update()

    def tray_update(self) -> None:
        e = self.ctx.runtime.engine
        self.tray.setIcon(_tray_icon(e.armed, e.paused))
        state = "PAUSED" if e.paused else ("ARMED" if e.armed else "DISARMED")
        self.tray.setToolTip(f"AirDesk — {state}")
        self.a_arm.setText("Disarm gesture input" if e.armed else "Arm gesture input")
        self.a_pause.setText("Resume" if e.paused else "Pause")
        self.a_pause.setEnabled(e.armed)
        self.a_kb.setText("Hide Air Keyboard" if self.ctx.overlay.isVisible() else "Show Air Keyboard")
        hk = self.ctx.s.safety.drag_hotkey.upper().replace("+", "+")
        self.a_drag.setChecked(e.click_mode == "drag")
        self.a_drag.setText(f"Legacy Drag Mode — pinch and hold {self.ctx.s.gesture.hold_to_drag_s:.1f} s ({hk})")
        self.setWindowIcon(_tray_icon(e.armed, e.paused))

    def _on_event(self, name: str) -> None:
        if name == "toggle_keyboard":
            self.ctx.toggle_keyboard()
        elif name == "mouse_click":
            self.ctx.overlay.out.commit()
        elif name == "emergency":
            self.tray.showMessage("AirDesk", "Emergency stop: gesture input disarmed and all buttons released.",
                                  QSystemTrayIcon.Warning, 2500)
        elif name == "pause":
            self.ctx.runtime.pause(not self.ctx.runtime.engine.paused)
        if name == "click_mode":
            # The engine changed the mode (button, hotkey, or one-shot Drag
            # Mode ending): persist it and refresh the controls.
            self.ctx.changed("gesture")
        if name in ("armed", "disarmed", "paused", "resumed", "emergency", "click_mode"):
            self.tray_update()

    # -------------------------------------------------------------- window
    def show_normal(self) -> None:
        self.show()
        self.setWindowState(self.windowState() & ~Qt.WindowMinimized)
        self.raise_()
        self.activateWindow()

    def closeEvent(self, e) -> None:  # noqa: N802
        if self.ctx.s.ui.minimize_to_tray and not self._quitting:
            e.ignore()
            self.hide()
            if not getattr(self, "_told", False):
                self._told = True
                self.tray.showMessage("AirDesk is still running",
                                      "Use the tray icon to open, pause or quit. Ctrl+Alt+F12 disarms instantly.",
                                      QSystemTrayIcon.Information, 3000)
            return
        self.quit()

    def quit(self) -> None:
        self._quitting = True
        self.timer.stop()
        self.marker.hide()
        self.ctx.overlay.set_shown(False)
        self.ctx.s.save()
        self.ctx.runtime.shutdown()
        self.tray.hide()
        QApplication.quit()
