"""AirDesk runtime: owns every service and the thread lifecycles.

Threads (each with one job, none blocks another):

    airdesk-camera    CameraManager: device -> newest frame (latest-frame-wins)
    airdesk-vision    downscale + MediaPipe + InteractionEngine, per new frame
    airdesk-dispatch  MouseDispatcher: 120 Hz cursor easing + ordered button/wheel
    airdesk-hotkeys   RegisterHotKey message loop (emergency stop, arm, keyboard)
    Qt GUI thread     windows, overlay painting, tray (polls `snapshot()`)

Safety controller responsibilities live here: input starts DISARMED, the
dispatcher only injects while the engine is armed and not paused, any
exception in the vision loop triggers an emergency stop, camera stalls
feed empty frames so the engine's tracking-loss release still runs, and
`shutdown()` releases every button and latched key before exiting.
"""

from __future__ import annotations

import ctypes
import logging
import os
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from .config import Settings
from .core.actions import KeyboardSink, NullKeyboard
from .core.engine import InteractionEngine
from .core.types import EngineStatus, Finger, FrameInput, PointerMode
from .platform import win32
from .platform.dispatcher import MouseDispatcher
from .platform.hotkeys import HotkeyService
from .vision.camera import CameraManager
from .vision.tracker import HandTracker, Preview

log = logging.getLogger(__name__)


@dataclass
class Metrics:
    capture_fps: float = 0.0
    tracking_fps: float = 0.0
    infer_ms: float = 0.0
    infer_p95: float = 0.0
    latency_ms: float = 0.0       # frame arrival -> input decision (median)
    latency_p95: float = 0.0
    cpu_percent: float = 0.0      # this process, % of all logical CPUs
    camera_status: str = "stopped"
    camera_error: str = ""
    camera_mode: str = "-"
    delegate: str = "cpu"
    delegate_note: str = ""
    dispatch: dict = field(default_factory=dict)
    counters: dict = field(default_factory=dict)
    hotkey_errors: list = field(default_factory=list)


class _ProcessCpu:
    """Process CPU usage via GetProcessTimes (no psutil dependency)."""

    def __init__(self) -> None:
        self._last: tuple[float, float] | None = None
        self.n = os.cpu_count() or 1

    def sample(self) -> float:
        if not win32.IS_WINDOWS:
            return 0.0
        k = win32.kernel32
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_ulonglong)] * 4
        c, e, kt, ut = (ctypes.c_ulonglong() for _ in range(4))
        k.GetProcessTimes(k.GetCurrentProcess(), ctypes.byref(c), ctypes.byref(e),
                          ctypes.byref(kt), ctypes.byref(ut))
        busy = (kt.value + ut.value) / 1e7
        now = time.perf_counter()
        pct = 0.0
        if self._last is not None:
            dt = now - self._last[0]
            if dt > 0:
                pct = 100.0 * (busy - self._last[1]) / dt / self.n
        self._last = (now, busy)
        return pct


class _MouseTap:
    """MouseSink proxy: forwards to the dispatcher, reports clicks so the
    keyboard can end a composed word when the caret may have moved."""

    def __init__(self, inner: MouseDispatcher, on_click: Callable[[], None]) -> None:
        self.inner = inner
        self.on_click = on_click

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def click(self, button: str, count: int = 1) -> bool:
        ok = self.inner.click(button, count)
        self.on_click()
        return ok

    def button(self, button: str, down: bool) -> bool:
        ok = self.inner.button(button, down)
        if down:
            self.on_click()
        return ok


class Runtime:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.lock = threading.RLock()            # guards the engine
        self.listeners: list[Callable[[str], None]] = []
        self.screen = win32.screen_layout() if win32.IS_WINDOWS else None
        self.dispatcher = MouseDispatcher(settings.pointer.dispatch_hz, settings.pointer.smoothing_tau_ms,
                                          settings.gesture.smooth_wheel, on_physical=self._on_physical)
        self._mouse = _MouseTap(self.dispatcher, lambda: self._emit("mouse_click"))
        self.engine = InteractionEngine(settings, self.screen, self._mouse, NullKeyboard(),
                                        on_command=self._emit)
        self.camera = CameraManager(settings.camera)
        self.tracker: HandTracker | None = None
        self.hotkeys = HotkeyService()
        self.metrics = Metrics()
        self.status = EngineStatus()
        self.preview: Preview | None = None
        self.want_preview = settings.ui.show_preview
        self._stop = threading.Event()
        self._vision: threading.Thread | None = None
        self._physical: tuple[float, float] | None = None
        self._infer: deque[float] = deque(maxlen=120)
        self._lat: deque[float] = deque(maxlen=120)
        self._frame_ts: deque[float] = deque(maxlen=60)
        self._cpu = _ProcessCpu()
        self._tracker_dirty = False
        self.internal_error = ""
        self.frames_processed = 0
        self.frame_hooks: list[Callable[[FrameInput, EngineStatus], None]] = []

    # ============================================================ lifecycle
    def start(self) -> None:
        self.dispatcher.start()
        self._register_hotkeys()
        self.start_camera()

    def start_camera(self) -> None:
        if self._vision and self._vision.is_alive():
            return
        self._stop.clear()
        self.camera.start()
        self._vision = threading.Thread(target=self._vision_loop, name="airdesk-vision", daemon=True)
        self._vision.start()

    def stop_camera(self) -> None:
        """Stop capture entirely (privacy / camera hand-off). Releases input."""
        with self.lock:
            self.engine.set_armed(False, time.perf_counter())
        self._stop.set()
        if self._vision:
            self._vision.join(timeout=2.0)
        self._vision = None
        self.camera.stop()

    def shutdown(self) -> None:
        if getattr(self, "_shut", False):
            return
        self._shut = True
        m = self.metrics
        log.info("session: %d frames processed, capture %.1f fps, tracking %.1f fps, inference %.2f ms, "
                 "processing latency %.2f ms (p95 %.2f), engine counters %s", self.frames_processed,
                 m.capture_fps, m.tracking_fps, m.infer_ms, m.latency_ms, m.latency_p95, self.engine.counters)
        log.info("shutting down: releasing all input")
        try:
            with self.lock:
                self.engine.set_armed(False, time.perf_counter())
                self.engine.keyboard.release_all()
                self.dispatcher.release_all()
        finally:
            self.dispatcher.enabled = False
            self.dispatcher.stop()
            self._stop.set()
            if self._vision:
                self._vision.join(timeout=2.0)
            self.camera.stop()
            self.hotkeys.stop()
            if self.tracker:
                self.tracker.close()

    def set_keyboard(self, kb: KeyboardSink) -> None:
        with self.lock:
            self.engine.keyboard = kb

    # ============================================================== control
    def arm(self, armed: bool) -> None:
        with self.lock:
            self.engine.set_armed(armed, time.perf_counter())
            self._sync_dispatch()

    def toggle_arm(self) -> None:
        self.arm(not self.engine.armed)

    def pause(self, paused: bool) -> None:
        with self.lock:
            self.engine.set_paused(paused)
            self._sync_dispatch()

    def emergency_stop(self) -> None:
        with self.lock:
            self.engine.emergency_stop()
            self._sync_dispatch()
        self.dispatcher.release_all()
        self._emit("emergency")

    def toggle_precision(self) -> None:
        with self.lock:
            self.engine.toggle_precision()

    def toggle_drag_mode(self) -> None:
        """Instant Auto-Click <-> Drag Mode (UI button, tray, hotkey)."""
        with self.lock:
            self.engine.toggle_drag_mode()

    def set_click_mode(self, mode: str) -> None:
        with self.lock:
            self.engine.set_click_mode(mode)

    def toggle_magnifier(self) -> None:
        """Windows Magnifier for small targets: Win+Plus opens it, Win+Esc closes it."""
        if not win32.IS_WINDOWS:
            return
        self._magnifier = not getattr(self, "_magnifier", False)
        win32.key_tap(0xBB if self._magnifier else 0x1B, (0x5B,))

    def set_pointer_mode(self, mode: str) -> None:
        with self.lock:
            self.engine.pointer.set_mode(PointerMode(mode))

    def set_finger(self, name: str) -> None:
        with self.lock:
            self.engine.set_finger(Finger.from_name(name))

    def apply_settings(self, camera_changed: bool = False) -> None:
        """Push edited settings into the live services."""
        with self.lock:
            self.engine.configure(self.s)
        d = self.dispatcher
        d.hz, d.tau = self.s.pointer.dispatch_hz, self.s.pointer.smoothing_tau_ms / 1000.0
        d.smooth_wheel = self.s.gesture.smooth_wheel
        self.want_preview = self.s.ui.show_preview
        if camera_changed:
            self.camera.reconfigure(self.s.camera)
            self._tracker_dirty = True

    def refresh_screens(self) -> None:
        if not win32.IS_WINDOWS:
            return
        self.screen = win32.screen_layout()
        with self.lock:
            self.engine.set_screen(self.screen)
        self.dispatcher.refresh_screen()

    def snapshot(self) -> tuple[EngineStatus, Metrics]:
        return self.status, self.metrics

    # ============================================================== events
    def _emit(self, name: str) -> None:
        if name in ("armed", "disarmed", "paused", "resumed"):
            self._sync_dispatch()
        if name == "toggle_magnifier":
            self.toggle_magnifier()
        for cb in list(self.listeners):
            try:
                cb(name)
            except Exception:
                log.exception("listener failed for %s", name)

    def _sync_dispatch(self) -> None:
        self.dispatcher.enabled = self.engine.armed and not self.engine.paused

    def _on_physical(self, x: float, y: float) -> None:
        # Called on the dispatcher thread while it holds its own lock: only
        # record it; the vision thread applies it under the engine lock.
        self._physical = (x, y)

    def _register_hotkeys(self) -> None:
        sc = self.s.safety
        self.hotkeys = HotkeyService()
        self.hotkeys.add(sc.emergency_hotkey, self.emergency_stop)
        self.hotkeys.add(sc.arm_hotkey, self.toggle_arm)
        self.hotkeys.add(sc.keyboard_hotkey, lambda: self._emit("toggle_keyboard"))
        self.hotkeys.add(sc.precision_hotkey, self.toggle_precision)
        self.hotkeys.add(sc.drag_hotkey, self.toggle_drag_mode)
        self.hotkeys.start()
        self.metrics.hotkey_errors = list(self.hotkeys.failed)

    def reload_hotkeys(self) -> None:
        self.hotkeys.stop()
        self._register_hotkeys()

    # ========================================================= vision loop
    def _vision_loop(self) -> None:
        try:
            self.tracker = self.tracker or HandTracker(self.s.camera)
        except Exception as e:
            log.exception("hand tracker failed to start")
            self.internal_error = f"Hand tracking model failed to load: {e}"
            return
        seq = -1
        last_frame = time.perf_counter()
        last_metrics = 0.0
        while not self._stop.is_set():
            if self._tracker_dirty:
                self._tracker_dirty = False
                old, self.tracker = self.tracker, HandTracker(self.s.camera)
                old.close()
            item = self.camera.wait_frame(seq, 0.1)
            now = time.perf_counter()
            try:
                if item is None:
                    if now - last_frame > 0.15:
                        # Camera stalled/disconnected: drive tracking-loss safety.
                        with self.lock:
                            self._apply_physical(now)
                            self.status = self.engine.process(FrameInput(now, []))
                    self._update_metrics(now, force=now - last_metrics > 0.5)
                    if now - last_metrics > 0.5:
                        last_metrics = now
                    continue
                seq, t_cap, frame = item
                last_frame = now
                fi, prev = self.tracker.process(frame, t_cap, self.want_preview)
                with self.lock:
                    self._apply_physical(fi.t)
                    st = self.engine.process(fi)
                    self.status = st
                    for hook in self.frame_hooks:
                        hook(fi, st)
                done = time.perf_counter()
                self.frames_processed += 1
                self._infer.append(self.tracker.infer_ms)
                self._lat.append((done - t_cap) * 1000)
                self._frame_ts.append(done)
                st.infer_ms = self.tracker.infer_ms
                st.latency_ms = (done - t_cap) * 1000
                if prev is not None:
                    self.preview = prev
                if done - last_metrics > 0.5:
                    last_metrics = done
                    self._update_metrics(done)
            except Exception:
                log.exception("vision loop error: emergency stop")
                self.internal_error = "Internal error in gesture processing - input was disarmed"
                with self.lock:
                    self.engine.emergency_stop()
                self._sync_dispatch()
                time.sleep(0.2)

    def _apply_physical(self, t: float) -> None:
        p, self._physical = self._physical, None
        if p is not None:
            self.engine.notify_physical_mouse(p[0], p[1], t)

    def _update_metrics(self, now: float, force: bool = True) -> None:
        if not force:
            return
        m = self.metrics
        m.capture_fps = self.camera.capture_fps()
        ts = [t for t in self._frame_ts if now - t < 1.5]
        m.tracking_fps = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 2 else 0.0
        if self._infer:
            v = sorted(self._infer)
            m.infer_ms, m.infer_p95 = statistics.median(v), v[int(len(v) * 0.95) - 1]
        if self._lat:
            v = sorted(self._lat)
            m.latency_ms, m.latency_p95 = statistics.median(v), v[int(len(v) * 0.95) - 1]
        m.cpu_percent = self._cpu.sample()
        m.camera_status = self.camera.status
        m.camera_error = self.camera.error or self.internal_error
        w, h, _ = self.camera.negotiated
        m.camera_mode = f"{w}x{h} {self.s.camera.backend.upper()}" if w else "-"
        if self.tracker:
            m.delegate, m.delegate_note = self.tracker.delegate, self.tracker.delegate_note
        m.dispatch = dict(self.dispatcher.stats)
        m.counters = dict(self.engine.counters)
