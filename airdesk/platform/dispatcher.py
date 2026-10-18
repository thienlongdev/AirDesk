"""Mouse dispatcher: the single owner of injected mouse state.

Runs a 120 Hz thread (`dispatch_hz`) for cursor motion, independent of
camera rate and GUI rendering, and executes discrete commands immediately:

* **sub-frame interpolation** - camera targets arrive at ~30 Hz; the
  dispatcher eases the real cursor toward the newest target with a short
  time constant, turning 33 ms stair-steps into smooth motion at a cost of
  roughly one `smoothing_tau_ms` of extra latency;
* **immediate, synchronous buttons** (v1.1.1) - button, click, X-button and
  zoom commands are executed on the caller's thread, under the dispatcher
  lock, the moment the engine issues them: the cursor is snapped exactly
  onto its target, SendInput is called, and the call returns whether
  Windows accepted the event (with the reason in `last_error` when not).
  No other Windows API call precedes the SendInput (the button-swap setting
  is cached, the elevated-window check runs afterwards): every ctypes call
  releases the Python GIL, and getting it back while the GUI thread paints
  measured up to ~5 ms - time the button-down would otherwise wait.
  v1.1 queued them for the next 120 Hz tick (up to 8 ms later) and
  `release_all` / a disabled tick could discard queued commands without
  telling anyone. Nothing is queued now, so nothing can be dropped
  silently; wheel deltas still accumulate and flush every tick;
* **button bookkeeping** - it knows which buttons *it* holds and
  `release_all()` releases them synchronously from any thread;
* **physical-mouse detection** - if the real cursor moves away from where
  AirDesk last put it (and AirDesk did not move it), the user's physical
  mouse is active; the engine is told so and yields.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from typing import Callable

from . import win32

log = logging.getLogger(__name__)


class MouseDispatcher:
    def __init__(self, hz: int = 120, tau_ms: float = 10.0, smooth_wheel: bool = True,
                 on_physical: Callable[[float, float], None] | None = None) -> None:
        self.hz = hz
        self.tau = tau_ms / 1000.0
        self.smooth_wheel = smooth_wheel
        self.on_physical = on_physical
        self.enabled = False                       # gated by SafetyController
        self._lock = threading.RLock()
        self._target: tuple[float, float] | None = None
        self._pos: tuple[float, float] | None = None
        self._last_sent: tuple[int, int] | None = None
        self._last_sent_t = 0.0
        self._held: set[str] = set()
        self._wheel = [0.0, 0.0]
        self._mismatch = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._vs = win32.virtual_screen()
        self._self_elevated: bool | None = None
        self._swapped = bool(win32.buttons_swapped()) if hasattr(win32, "buttons_swapped") else False
        self._swapped_t = 0.0
        self.last_submit_t = 0.0                   # perf_counter right before the last SendInput(button)
        self._elevated_pids: dict[int, tuple[float, bool | None]] = {}
        self.last_error = ""                       # why the last button/click was refused
        self.last_note = ""
        self.released_reason = ""                  # why held buttons were last force-released
        self.stats = {"moves": 0, "buttons": 0, "wheel": 0, "ticks": 0, "rejected": 0,
                      "submit_ms_max": 0.0}

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="airdesk-dispatch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.release_all()
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def refresh_screen(self) -> None:
        self._vs = win32.virtual_screen()

    # -------------------------------------------------------- MouseSink API
    def move(self, x: float, y: float) -> None:
        with self._lock:
            self._target = (x, y)

    def button(self, button: str, down: bool) -> bool:
        """Press or release now. True = submitted to Windows."""
        with self._lock:
            if not self._ready():
                return False
            if down and button in self._held:
                return self._refuse(f"{button} button is already held (state mismatch)")
            if not down and button not in self._held:
                return self._refuse(f"{button} button was not held"
                                    + (f" (released earlier: {self.released_reason})" if self.released_reason else ""))
            self._snap()
            t0 = self.last_submit_t = time.perf_counter()
            ok = win32.mouse_button(button, down, self._swapped)
            self._timing(t0)
            if not ok:
                return self._refuse(f"SendInput rejected the {button} button-{'down' if down else 'up'}")
            (self._held.add if down else self._held.discard)(button)
            self.stats["buttons"] += 1
            self.last_note = ""
            if down and self._blocked_by_uipi():
                # Submitted (and tracked as held, so the matching up is still
                # sent), but Windows drops it for an elevated window.
                self.last_note = "blocked by Windows: the window under the cursor runs as administrator"
            return True

    def click(self, button: str, count: int = 1) -> bool:
        with self._lock:
            if not self._ready():
                return False
            self._snap()
            t0 = self.last_submit_t = time.perf_counter()
            ok = win32.mouse_click(button, count, self._swapped)
            self._timing(t0)
            if not ok:
                return self._refuse(f"SendInput rejected the {button} click")
            self.stats["buttons"] += count
            if self._blocked_by_uipi():
                return self._refuse("blocked by Windows: the window under the cursor runs as administrator")
            return True

    def scroll(self, dx_notches: float, dy_notches: float) -> None:
        with self._lock:
            if self.enabled:
                self._wheel[0] += dx_notches
                self._wheel[1] += dy_notches

    def xbutton(self, which: int) -> bool:
        with self._lock:
            if not self._ready():
                return False
            b = f"x{which}"
            ok = win32.mouse_button(b, True) and win32.mouse_button(b, False)
            self.stats["buttons"] += 1
            return ok

    def zoom(self, notches: int) -> bool:
        with self._lock:
            if not self._ready():
                return False
            vk_ctrl = 0x11
            already = win32.key_is_down(vk_ctrl)
            if not already:
                win32.key_event(vk_ctrl, True)
            try:
                return win32.mouse_wheel(win32.WHEEL_DELTA * int(notches))
            finally:
                if not already:
                    win32.key_event(vk_ctrl, False)

    def held_buttons(self) -> tuple[str, ...]:
        return tuple(sorted(self._held))

    def cursor_pos(self) -> tuple[float, float] | None:
        try:
            return win32.cursor_pos()
        except Exception:
            return None

    def release_all(self, reason: str = "released by AirDesk") -> None:
        """Synchronous, idempotent: never leave a button held."""
        with self._lock:
            for b in list(self._held):
                try:
                    win32.mouse_button(b, False)
                except Exception:
                    log.exception("button release failed")
            if self._held:
                self.released_reason = reason
                log.info("released held buttons: %s (%s)", ",".join(sorted(self._held)), reason)
            self._held.clear()
            self._wheel = [0.0, 0.0]

    def warp(self, x: float, y: float) -> None:
        """Place the cursor immediately (no easing)."""
        with self._lock:
            self._target = self._pos = (x, y)
            self._send_move(x, y)

    # ------------------------------------------------------------ helpers
    def _ready(self) -> bool:
        self.last_error = ""
        self.last_submit_t = 0.0
        if not self.enabled:
            return self._refuse("input is disarmed or paused")
        return True

    def _refuse(self, reason: str) -> bool:
        self.last_error = reason
        self.stats["rejected"] += 1
        log.info("mouse input refused: %s", reason)
        return False

    def _timing(self, t0: float) -> None:
        ms = (time.perf_counter() - t0) * 1000
        if ms > self.stats["submit_ms_max"]:
            self.stats["submit_ms_max"] = round(ms, 3)

    def _blocked_by_uipi(self) -> bool:
        """SendInput reports success even when User Interface Privilege
        Isolation drops the event because the target window belongs to an
        elevated process; detect that case so it is not shown as a click."""
        if not getattr(win32, "IS_WINDOWS", False) or self._last_sent is None:
            return False
        try:
            if self._self_elevated is None:
                self._self_elevated = win32.self_elevated()
            if self._self_elevated:
                return False
            hwnd = win32.window_at(*self._last_sent)
            if not hwnd:
                return False
            pid = win32.window_pid(hwnd)
            if pid == os.getpid():
                return False
            now = time.monotonic()
            cached = self._elevated_pids.get(pid)
            if cached is None or now - cached[0] > 5.0:
                cached = (now, win32.process_elevated(pid))
                self._elevated_pids[pid] = cached
            return cached[1] is True
        except Exception:
            log.debug("elevation check failed", exc_info=True)
            return False

    # ------------------------------------------------------------- thread
    def _run(self) -> None:
        period = 1.0 / max(self.hz, 30)
        last = time.perf_counter()
        next_t = last
        while not self._stop.is_set():
            now = time.perf_counter()
            dt = now - last
            last = now
            try:
                self._tick(now, dt)
            except Exception:
                log.exception("dispatcher tick failed; releasing buttons")
                self.release_all("dispatcher error")
            next_t += period
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.perf_counter()

    def _tick(self, now: float, dt: float) -> None:
        self.stats["ticks"] += 1
        with self._lock:
            if not self.enabled:
                self._target = None
                self._pos = None
                self._last_sent = None
                self._wheel = [0.0, 0.0]
                return
            self._detect_physical(now)
            if now - self._swapped_t > 2.0:          # Settings > Mouse > primary button
                self._swapped_t = now
                self._swapped = bool(win32.buttons_swapped())
            if self._target is None:
                return
            if self._pos is None:
                self._pos = self._target
            else:
                a = 1.0 - math.exp(-dt / self.tau) if self.tau > 0 else 1.0
                px = self._pos[0] + (self._target[0] - self._pos[0]) * a
                py = self._pos[1] + (self._target[1] - self._pos[1]) * a
                if math.hypot(self._target[0] - px, self._target[1] - py) < 0.5:
                    px, py = self._target
                self._pos = (px, py)
            self._send_move(*self._pos)
            self._flush_wheel()

    def _send_move(self, x: float, y: float) -> None:
        ix, iy = int(round(x)), int(round(y))
        if self._last_sent == (ix, iy):
            return
        win32.mouse_move_abs(ix, iy, self._vs)
        self._last_sent = (ix, iy)
        self._last_sent_t = time.perf_counter()
        self.stats["moves"] += 1

    def _snap(self) -> None:
        """Before any button event the cursor is exactly on its target."""
        if self._target is not None:
            self._pos = self._target
            self._send_move(*self._target)

    def _flush_wheel(self) -> None:
        for axis in (1, 0):
            acc = self._wheel[axis]
            unit = 1 if self.smooth_wheel else win32.WHEEL_DELTA
            delta = int(acc * win32.WHEEL_DELTA / unit) * unit
            if delta:
                win32.mouse_wheel(delta, horizontal=axis == 0)
                self._wheel[axis] = acc - delta / win32.WHEEL_DELTA
                self.stats["wheel"] += 1

    def _detect_physical(self, now: float) -> None:
        if self._last_sent is None or self.on_physical is None:
            return
        if now - self._last_sent_t < 0.03:   # our own move may still be in flight
            return
        x, y = win32.cursor_pos()
        if abs(x - self._last_sent[0]) > 3 or abs(y - self._last_sent[1]) > 3:
            self._mismatch += 1
            if self._mismatch >= 2:
                self._mismatch = 0
                self._last_sent = (x, y)
                self._target = self._pos = (x, y)
                if self._held:
                    self.release_all("physical mouse moved")
                self.on_physical(x, y)
        else:
            self._mismatch = 0
