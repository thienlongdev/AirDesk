"""The Air Keyboard: a floating, non-activating Windows keyboard overlay.

Focus model
-----------
The window is created with Qt.WindowDoesNotAcceptFocus + Qt.Tool and,
after Qt creates the HWND, WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | topmost
is applied with Win32 (`win32.make_noactivate`). Clicking it - with the
physical mouse or with injected gesture clicks - therefore never takes
keyboard focus from the application the user is typing into.

Gesture keys never click at all: the interaction engine hit-tests the
cursor against `hit_test()` (physical pixels, same fractional key
rectangles that are painted) and calls `press_key()`, so no mouse event
reaches the overlay or the window underneath.

Size and position
-----------------
Compact by default: 65 % of the active monitor's work-area width and 27 %
of its height at 70 % size (`keyboard_layout.keyboard_size`). The title bar
has − size% + (both dimensions), W− / W+ (width only - keys widen or narrow,
the layout redistributes), Reset and ×; the ⠿ handle moves it (pinch, move,
pinch again; or drag with the physical mouse) and the bottom-right grip
resizes (horizontal = width, vertical = size). Size, width, position and
layout persist and are clamped into the work area when monitors or DPI
change. Painted keys and gesture hit-tests use the same fractional
rectangles, so they always match at any size.

Dual-hand cursors (v1.1.3)
--------------------------
While the engine's dual-hand mode is on, `cursors()` delivers one entry per
tracked hand (keyboard-fraction position of the pointing fingertip, hovered
key, pressed flag). Each is drawn as a small vector hand - left = cool blue,
right = mint, mirrored silhouettes, index finger pointing - whose index
fingertip sits exactly on the hit-test point. Positions are eased at 60 Hz
only while a cursor is still travelling. `hit_frac()` hit-tests the same
fractional key rectangles that are painted (with a small tolerance in the
gaps between keys). A gesture move of the keyboard (⠿ handle) follows the
pinching hand via `move_by()` instead of the system cursor.

Threading
---------
`is_visible / hit_test / hover / press_key / release_all` are called from
the vision thread. They read an immutable snapshot under a lock and post
everything else to the GUI thread through queued Qt signals.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque

from PySide6.QtCore import QPoint, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QTransform
from PySide6.QtWidgets import QApplication, QWidget

from ..config import KeyboardConfig
from ..core.keyboard_layout import (MODIFIERS, SIZE_DEFAULT, SIZE_RANGE, TOOLBAR, WIDTH_DEFAULT, WIDTH_RANGE,
                                    KeyRect, clamp_rect, hit, hit_near, key_label, keyboard_size, layout_rects)
from ..platform import win32
from ..platform.keyboard_output import KeyboardOutput

log = logging.getLogger(__name__)

TOOLBAR_KEYS = set(TOOLBAR) | {"sym", "abc"}
# Hand 0 = left (cool blue), hand 1 = right (turquoise / mint).
HAND_COLORS = (QColor(84, 156, 255), QColor(46, 212, 178))
HAND_SIDES = ("left", "right")
MOD_NAMES = {"ctrl": "Ctrl", "shift": "Shift", "alt": "Alt", "win": "Win"}


_U = 100.0   # hand paths are built at 100 units (smooth curve flattening), drawn at 1/100


def _capsule(x0: float, y0: float, x1: float, y1: float, r: float) -> QPainterPath:
    path = QPainterPath()
    path.addRoundedRect(QRectF(x0 * _U, y0 * _U, (x1 - x0) * _U, (y1 - y0) * _U), r * _U, r * _U)
    return path


def hand_path(side: str) -> QPainterPath:
    """Miniature pointing hand, unit height, index fingertip at (0, 0).

    A right hand seen from the back (as the user sees their own hand
    pointing at the screen): thumb on the left, three curled fingers to the
    right of the extended index finger. The left hand is its mirror image."""
    parts = [
        _capsule(-0.085, 0.0, 0.085, 0.56, 0.085),     # index, pointing up
        _capsule(0.075, 0.30, 0.235, 0.64, 0.08),      # middle (curled knuckle)
        _capsule(0.215, 0.345, 0.37, 0.66, 0.075),     # ring
        _capsule(0.35, 0.40, 0.48, 0.68, 0.065),       # little
        _capsule(-0.105, 0.48, 0.48, 0.93, 0.14),      # palm
    ]
    thumb = _capsule(-0.075, -0.20, 0.075, 0.16, 0.075)
    parts.append(QTransform().translate(-0.14 * _U, 0.70 * _U).rotate(-38).map(thumb))
    path = parts[0]
    for q in parts[1:]:
        path = path.united(q)
    path = QTransform().scale(1 / _U, 1 / _U).map(path.simplified())
    return QTransform().scale(-1, 1).map(path) if side == "left" else path


def crease_path(side: str) -> QPainterPath:
    """Short lines between the curled fingers and along the index base."""
    path = QPainterPath()
    for x0, y0, x1, y1 in ((0.085, 0.40, 0.085, 0.52), (0.225, 0.43, 0.225, 0.55),
                           (0.36, 0.47, 0.36, 0.58), (0.0, 0.62, 0.0, 0.70)):
        path.moveTo(x0, y0)
        path.lineTo(x1, y1)
    return QTransform().scale(-1, 1).map(path) if side == "left" else path


def cuff_path(side: str) -> QPainterPath:
    path = QTransform().scale(1 / _U, 1 / _U).map(_capsule(-0.04, 0.86, 0.42, 1.0, 0.04))
    return QTransform().scale(-1, 1).map(path) if side == "left" else path


_HAND_PATHS: dict[str, tuple[QPainterPath, QPainterPath, QPainterPath]] = {}


class KeyboardOverlay(QWidget):
    # cross-thread plumbing (vision thread -> GUI thread)
    _sig_hover = Signal(int, object, float)
    _sig_press = Signal(str, int, float)      # key, hand, emit time (perf_counter)
    _sig_release = Signal()
    _sig_cursors = Signal(object)
    _sig_move_by = Signal(int, int)
    _sig_dual = Signal(bool)
    _sig_clear_mods = Signal()
    # public
    visibility_changed = Signal(bool)
    target_changed = Signal(str)          # human-readable target description
    key_typed = Signal(str)               # key id (never the produced text)
    config_changed = Signal()

    def __init__(self, cfg: KeyboardConfig, output: KeyboardOutput | None = None) -> None:
        super().__init__(None, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                         | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setWindowTitle("AirDesk Keyboard")
        self.setMouseTracking(True)
        self.cfg = cfg
        self.out = output or KeyboardOutput(language=cfg.language, tone_style=cfg.vi_tone_style,
                                            one_shot=cfg.one_shot_modifiers)
        self.out.privacy = cfg.privacy_mode
        self.out.suggestions_enabled = cfg.suggestions
        self.layout_name = cfg.layout
        self.rects: list[KeyRect] = layout_rects(self.layout_name)
        self._hover: list[tuple[str | None, float]] = [(None, 0.0), (None, 0.0)]
        self._flash: dict[str, tuple[float, int]] = {}
        # Dual-hand mode (v1.1.3): virtual hand cursors and gesture moves.
        self._dual = False
        self._cursors: dict[int, dict] = {}            # hand -> target entry from the engine
        self._shown_pos: dict[int, tuple[float, float]] = {}   # eased pixel position per hand
        self._press_t: dict[int, float] = {}
        self._move_origin: tuple[int, int] | None = None
        self.results: deque[dict] = deque(maxlen=24)  # dispatch outcome per key (diagnostics)
        # v1.1.4: ordered dispatch accounting (vision thread emits, GUI thread sends)
        self.emitted = 0
        self.processed = 0
        self.max_queue_ms = 0.0
        self._sugs: list[str] = []
        self._moving = False
        self._move_grab: QPoint | None = None
        self._resize_grab: tuple[QPoint, int, int] | None = None
        self._lock = threading.Lock()
        self._snap_visible = False
        self._snap_rect = (0, 0, 0, 0)
        self._snap_rects = self.rects
        self._target_hwnd = 0
        self._target_desc = ""
        self.warning = ""

        self._sig_hover.connect(self._on_hover, Qt.QueuedConnection)
        self._sig_press.connect(self._on_press, Qt.QueuedConnection)
        self._sig_release.connect(self.out.release_all, Qt.QueuedConnection)
        self._sig_cursors.connect(self._on_cursors, Qt.QueuedConnection)
        self._sig_move_by.connect(self._on_move_by, Qt.QueuedConnection)
        self._sig_dual.connect(self._on_dual, Qt.QueuedConnection)
        self._sig_clear_mods.connect(self._on_clear_mods, Qt.QueuedConnection)

        self._fg_timer = QTimer(self, interval=200, timeout=self._poll_foreground)
        self._move_timer = QTimer(self, interval=16, timeout=self._follow_cursor)
        self._anim_timer = QTimer(self, interval=40, timeout=self._expire_flash)
        self._cursor_timer = QTimer(self, interval=16, timeout=self._ease_cursors)
        self.setWindowOpacity(cfg.opacity)
        self._apply_geometry()

    # =============================================== KeyboardSink (any thread)
    def is_visible(self) -> bool:
        return self._snap_visible

    def hit_test(self, x: float, y: float) -> str | None:
        with self._lock:
            if not self._snap_visible:
                return None
            l, t, r, b = self._snap_rect
            rects = self._snap_rects
        if r <= l or b <= t or not (l <= x < r and t <= y < b):
            return None
        k = hit(rects, (x - l) / (r - l), (y - t) / (b - t))
        return k.key if k else None

    def hit_frac(self, fx: float, fy: float) -> str | None:
        """Hit-test a keyboard-surface fraction (dual-hand cursors)."""
        with self._lock:
            if not self._snap_visible:
                return None
            rects = self._snap_rects
        k = hit_near(rects, fx, fy)
        return k.key if k else None

    def cursors(self, data: list[dict]) -> None:
        self._sig_cursors.emit(list(data))

    def move_by(self, dx: int, dy: int) -> None:
        self._sig_move_by.emit(int(dx), int(dy))

    def set_dual(self, on: bool) -> None:
        self._sig_dual.emit(bool(on))

    def clear_modifiers(self) -> None:
        self._sig_clear_mods.emit()

    def hover(self, hand: int, key: str | None, progress: float = 0.0) -> None:
        self._sig_hover.emit(hand, key, float(progress))

    def press_key(self, key: str, hand: int = 0) -> bool:
        # Queued in emission order to the GUI thread: one ordered output
        # channel for both hands; nothing is coalesced or dropped.
        self.emitted += 1
        self._sig_press.emit(key, hand, time.perf_counter())
        return True

    @property
    def queue_depth(self) -> int:
        """Key activations emitted by the engine and not yet sent."""
        return max(self.emitted - self.processed, 0)

    def release_all(self) -> None:
        self._sig_release.emit()

    # ================================================================ public
    def toggle(self) -> None:
        self.set_shown(not self.isVisible())

    def set_shown(self, shown: bool) -> None:
        if shown == self.isVisible():
            return
        if shown:
            self._target_hwnd = win32.foreground_window() if win32.IS_WINDOWS else 0
            self.show()
        else:
            self._moving = False
            self._move_origin = None
            self.out.commit()
            self.out.clear_modifiers()      # a one-shot Ctrl never survives the keyboard closing
            self.hide()

    def apply_config(self, cfg: KeyboardConfig) -> None:
        resize = (cfg.size_pct, cfg.width_pct) != getattr(self, "_applied_size", None)
        self.cfg = cfg
        self.out.set_language(cfg.language)
        self.out.set_tone_style(cfg.vi_tone_style)
        self.out.privacy = cfg.privacy_mode
        self.out.suggestions_enabled = cfg.suggestions
        self.out.mods.one_shot = cfg.one_shot_modifiers
        self.setWindowOpacity(cfg.opacity)
        if cfg.layout != self.layout_name and self.layout_name != "symbols":
            self._set_layout(cfg.layout)
        elif resize:
            self._apply_size()
        self.update()

    def type_key(self, key: str) -> None:
        """Programmatic key press on the GUI thread (tests, benchmark)."""
        self._on_press(key, 0)

    # ================================================================ events
    def showEvent(self, e) -> None:  # noqa: N802
        super().showEvent(e)
        if win32.IS_WINDOWS:
            win32.make_noactivate(int(self.winId()))
        self._fg_timer.start()
        self._anim_timer.start()
        self._refresh_snapshot()
        self.visibility_changed.emit(True)
        self._poll_foreground()

    def hideEvent(self, e) -> None:  # noqa: N802
        super().hideEvent(e)
        self._hover = [(None, 0.0), (None, 0.0)]     # no stale hover when shown again
        self._cursors.clear()
        self._shown_pos.clear()
        self._cursor_timer.stop()
        self._move_grab = self._resize_grab = None
        self._fg_timer.stop()
        self._anim_timer.stop()
        self._move_timer.stop()
        self._refresh_snapshot()
        self.visibility_changed.emit(False)

    def moveEvent(self, e) -> None:  # noqa: N802
        super().moveEvent(e)
        self._refresh_snapshot()

    def resizeEvent(self, e) -> None:  # noqa: N802
        super().resizeEvent(e)
        self._refresh_snapshot()

    def mousePressEvent(self, e) -> None:  # noqa: N802
        """Physical mouse / touch use: press keys, drag the handle, resize."""
        pos = e.position().toPoint()
        if pos.x() > self.width() - 18 and pos.y() > self.height() - 18:
            self._resize_grab = (e.globalPosition().toPoint(), self.cfg.width_pct, self.cfg.size_pct)
            return
        k = hit(self.rects, pos.x() / max(self.width(), 1), pos.y() / max(self.height(), 1))
        if k is None:
            return
        if k.key == "move":
            self._move_grab = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            return
        self._on_press(k.key, 0)

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        g = e.globalPosition().toPoint()
        if self._move_grab is not None:
            self.move(g - self._move_grab)
        elif self._resize_grab is not None:
            start, w0, s0 = self._resize_grab
            dx, dy = g.x() - start.x(), g.y() - start.y()
            area = self._work_area()
            base_w, base_h = keyboard_size(area.width(), area.height(), self.layout_name, s0, 100)
            self.cfg.size_pct = _clamp(int(s0 * (1 + dy / max(base_h, 1))), SIZE_RANGE)
            self.cfg.width_pct = _clamp(int(w0 * (1 + dx / max(base_w, 1))), WIDTH_RANGE)
            self._apply_size(keep="topleft")

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if self._move_grab is not None or self._resize_grab is not None:
            self._move_grab = self._resize_grab = None
            self._save_geometry()

    # ============================================================ GUI slots
    def _on_hover(self, hand: int, key, progress: float) -> None:
        if self._hover[hand] != (key, progress):
            self._hover[hand] = (key, progress)
            self.update()

    def _on_cursors(self, data: list) -> None:
        seen = set()
        W, H = max(self.width(), 1), max(self.height(), 1)
        for d in data:
            i = int(d.get("hand", 0)) % 2
            seen.add(i)
            self._cursors[i] = d
            px = (d["x"] * W, d["y"] * H)
            old = self._shown_pos.get(i)
            if old is None or math.dist(old, px) > 0.25 * W:
                self._shown_pos[i] = px                 # (re)appeared: no glide across the keys
            if d.get("pressed") and not self._press_t.get(i):
                self._press_t[i] = time.monotonic()
            elif not d.get("pressed"):
                self._press_t[i] = 0.0
        hover = tuple((i, d.get("key"), bool(d.get("pressed"))) for i, d in sorted(self._cursors.items()))
        for i in list(self._cursors):
            if i not in seen:
                self._cursors.pop(i, None)
                self._shown_pos.pop(i, None)
                hover = None
        if self._cursors and not self._cursor_timer.isActive():
            self._cursor_timer.start()
        if hover != getattr(self, "_last_hover_sig", None) or hover is None:
            self._last_hover_sig = hover
            self.update()                 # hovered / pressed key changed: full repaint (rare)

    def _cursor_rect(self, x: float, y: float) -> QRect:
        """Pixels a hand cursor at (x, y) can touch (hand, shadow, press ring)."""
        size = self._cursor_size()
        return QRect(int(x - 0.55 * size) - 6, int(y - 0.35 * size) - 6, int(1.1 * size) + 12, int(1.45 * size) + 12)

    def _cursor_size(self) -> float:
        key_h = min(r.h for r in self.rects if r.key not in TOOLBAR_KEYS) * max(self.height(), 1)
        return min(max(key_h * 0.85, 20.0), 48.0)

    def _ease_cursors(self) -> None:
        """60 Hz easing toward the latest 30 Hz targets; stops once settled."""
        W, H = max(self.width(), 1), max(self.height(), 1)
        moving = False
        now = time.monotonic()
        for i, d in self._cursors.items():
            tx, ty = d["x"] * W, d["y"] * H
            x0, y0 = x, y = self._shown_pos.get(i, (tx, ty))
            if abs(tx - x) > 0.3 or abs(ty - y) > 0.3:
                x, y = x + (tx - x) * 0.55, y + (ty - y) * 0.55
                moving = True
            else:
                x, y = tx, ty
            self._shown_pos[i] = (x, y)
            pt = self._press_t.get(i) or 0.0
            if (x, y) != (x0, y0) or (pt and now - pt < 0.2):
                # Only the pixels around this cursor (old and new position)
                # are repainted - not the whole keyboard.
                self.update(self._cursor_rect(x0, y0).united(self._cursor_rect(x, y)))
        if not moving and not self._cursors:
            self._cursor_timer.stop()

    def _on_dual(self, on: bool) -> None:
        self._dual = on
        if not on:
            self._cursors.clear()
            self._shown_pos.clear()
            if self._moving and self._move_origin is not None:
                self._moving = False
                self._move_origin = None
                self._save_geometry()
            self.update()

    def _on_move_by(self, dx: int, dy: int) -> None:
        if not (self._moving and self._move_origin is not None and win32.IS_WINDOWS):
            return
        win32.move_window_noactivate(int(self.winId()), self._move_origin[0] + dx, self._move_origin[1] + dy)
        self._refresh_snapshot()

    def _on_clear_mods(self) -> None:
        if self.out.mods.armed():
            self.out.clear_modifiers()
            self.update()

    def _on_press(self, key: str, hand: int, t_emit: float = 0.0) -> None:
        t0 = time.perf_counter()
        queue_ms = (t0 - t_emit) * 1000 if t_emit else 0.0
        if t_emit:
            self.processed += 1
            self.max_queue_ms = max(self.max_queue_ms, queue_ms)
        self._flash[f"{key}"] = (time.monotonic(), hand)
        if self._moving:
            self._moving = False
            self._move_origin = None
            self._move_timer.stop()
            self._save_geometry()
            self.update()
            return
        res = None
        if key in TOOLBAR_KEYS:
            self._toolbar(key)
        else:
            self._check_target()
            res = self.out.press(key)
            if not res.ok and res.message:
                self.warning = res.message
            self.key_typed.emit(key)
            self._sugs = self.out.suggestions()
        self.results.append({"t": time.perf_counter(), "hand": HAND_SIDES[hand % 2] if self._dual else "pointer",
                             "key": key, "ok": res.ok if res else True,
                             "kind": res.kind if res else "toolbar", "combo": res.combo if res else "",
                             "message": res.message if res else "",
                             "queue_ms": round(queue_ms, 2), "send_ms": round((time.perf_counter() - t0) * 1000, 2)})
        self.update()

    def _toolbar(self, key: str) -> None:
        if key == "move":
            self._moving = True
            if self._dual and win32.IS_WINDOWS:
                # Dual-hand mode: the pinching hand moves the keyboard
                # (move_by); the system cursor is not involved.
                r = win32.window_rect(int(self.winId()))
                self._move_origin = (r.left, r.top)
            else:
                self._move_timer.start()
        elif key == "hide":
            self.set_shown(False)
        elif key == "lang":
            order = ["en", "vi", "raw"]
            self.cfg.language = order[(order.index(self.cfg.language) + 1) % 3]
            self.out.set_language(self.cfg.language)
            self._sugs = []
            self.config_changed.emit()
        elif key == "layout":
            self._set_layout("compact" if self.layout_name == "full" else "full")
            self.cfg.layout = self.layout_name
            self.config_changed.emit()
        elif key == "sym":
            self._set_layout("symbols")
        elif key == "abc":
            self._set_layout(self.cfg.layout)
        elif key in ("size-", "size+"):
            self.cfg.size_pct = _clamp(self.cfg.size_pct + (10 if key == "size+" else -10), SIZE_RANGE)
            self._apply_size()
            self._save_geometry()
        elif key in ("width-", "width+"):
            self.cfg.width_pct = _clamp(self.cfg.width_pct + (10 if key == "width+" else -10), WIDTH_RANGE)
            self._apply_size()
            self._save_geometry()
        elif key == "reset":
            self.reset_geometry()
        elif key.startswith("sug"):
            i = int(key[3:])
            if i < len(self._sugs):
                self._check_target()
                self.out.accept_suggestion(self._sugs[i])
                self.key_typed.emit(key)
                self._sugs = []

    def _set_layout(self, name: str) -> None:
        self.layout_name = name
        self.rects = layout_rects(name)
        self._apply_size()
        self._refresh_snapshot()
        self.update()

    def _follow_cursor(self) -> None:
        """Gesture move mode: the move handle follows the cursor."""
        if not self._moving or not win32.IS_WINDOWS:
            return
        cx, cy = win32.cursor_pos()
        handle = next((r for r in self.rects if r.key == "move"), None)
        rect = win32.window_rect(int(self.winId()))
        if handle is None or rect.width <= 0:
            return
        ox = int((handle.x + handle.w / 2) * rect.width)
        oy = int((handle.y + handle.h / 2) * rect.height)
        win32.move_window_noactivate(int(self.winId()), cx - ox, cy - oy)
        self._refresh_snapshot()

    def _expire_flash(self) -> None:
        now = time.monotonic()
        if any(now - t > 0.15 for t, _h in self._flash.values()):
            self._flash = {k: v for k, v in self._flash.items() if now - v[0] <= 0.15}
            self.update()
        if self.out.mods.expire(now, self.cfg.modifier_timeout_s):
            self.update()                  # an armed modifier nobody used: cleared

    # ============================================================== target
    def _poll_foreground(self) -> None:
        if not win32.IS_WINDOWS:
            return
        fg = win32.foreground_window()
        if not fg or fg == int(self.winId()) or fg == self._target_hwnd:
            return
        self._target_hwnd = fg
        self.out.commit()          # never continue a word in another window
        self._sugs = []
        self._describe_target()

    def _check_target(self) -> None:
        if win32.IS_WINDOWS and win32.foreground_window() != self._target_hwnd:
            self._poll_foreground()

    def _describe_target(self) -> None:
        hwnd = self._target_hwnd
        pid = win32.window_pid(hwnd)
        name = win32.process_name(pid) or "unknown"
        self.warning = ""
        elevated = win32.process_elevated(pid)
        if elevated and not win32.self_elevated():
            self.warning = f"{name} runs as administrator: Windows blocks injected input (UIPI)."
        self.out.sensitive = win32.is_password_field(hwnd)
        if self.out.sensitive:
            self.warning = "Password field: suggestions disabled."
        self._target_desc = name
        self.target_changed.emit(name)
        self.update()

    # ============================================================ geometry
    def reset_geometry(self) -> None:
        """Compact default size, bottom centre of the active monitor."""
        self.cfg.size_pct, self.cfg.width_pct = SIZE_DEFAULT, WIDTH_DEFAULT
        self.cfg.x = self.cfg.y = -1
        self._apply_geometry()
        self.config_changed.emit()

    def _work_area(self) -> QRect:
        """Work area (logical px) of the monitor the keyboard is on, else the
        monitor under the cursor (the one the user is working on)."""
        scr = None
        if self.isVisible() or (self.cfg.x >= 0 and self.cfg.y >= 0):
            c = self.frameGeometry().center() if self.isVisible() else QPoint(self.cfg.x, self.cfg.y)
            scr = QApplication.screenAt(c)
        scr = scr or QApplication.screenAt(QCursor.pos()) or QApplication.primaryScreen()
        return scr.availableGeometry()

    def _apply_size(self, keep: str = "bottomcentre") -> None:
        """Resize to the configured size/width, keeping the window anchored
        (its bottom centre by default) and fully inside the work area."""
        area = self._work_area()
        w, h = keyboard_size(area.width(), area.height(), self.layout_name, self.cfg.size_pct,
                             self.cfg.width_pct)
        g = self.frameGeometry()
        if keep == "topleft":
            x, y = g.x(), g.y()
        else:
            x, y = g.x() + (g.width() - w) // 2, g.y() + g.height() - h
        x, y = clamp_rect(x, y, w, h, (area.x(), area.y(), area.width(), area.height()))
        self.setGeometry(x, y, w, h)
        self._applied_size = (self.cfg.size_pct, self.cfg.width_pct)
        self.update()

    def _apply_geometry(self) -> None:
        area = self._work_area()
        w, h = keyboard_size(area.width(), area.height(), self.layout_name, self.cfg.size_pct,
                             self.cfg.width_pct)
        if self.cfg.x >= 0 and self.cfg.y >= 0 and QApplication.screenAt(QPoint(self.cfg.x, self.cfg.y)):
            x, y = self.cfg.x, self.cfg.y
        else:
            x, y = area.x() + (area.width() - w) // 2, area.y() + area.height() - h - 16
        x, y = clamp_rect(x, y, w, h, (area.x(), area.y(), area.width(), area.height()))
        self.setGeometry(x, y, w, h)
        self._applied_size = (self.cfg.size_pct, self.cfg.width_pct)

    def reclamp(self) -> None:
        """Monitor / work-area / DPI change: re-derive the size and keep the
        keyboard on a visible screen."""
        self._apply_geometry()
        self._refresh_snapshot()

    def _save_geometry(self) -> None:
        g = self.frameGeometry()
        area = self._work_area()
        x, y = clamp_rect(g.x(), g.y(), g.width(), g.height(), (area.x(), area.y(), area.width(), area.height()))
        if (x, y) != (g.x(), g.y()):
            self.move(x, y)          # never leave keys off-screen
        self.cfg.x, self.cfg.y = x, y
        self.config_changed.emit()

    def _refresh_snapshot(self) -> None:
        vis = self.isVisible()
        rect = (0, 0, 0, 0)
        if vis and win32.IS_WINDOWS:
            r = win32.window_rect(int(self.winId()))
            rect = (r.left, r.top, r.right, r.bottom)
        with self._lock:
            self._snap_visible = vis
            self._snap_rect = rect
            self._snap_rects = self.rects

    # ============================================================== paint
    def paintEvent(self, e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        bg = QPainterPath()
        bg.addRoundedRect(QRectF(0, 0, W, H), 14, 14)
        p.fillPath(bg, QColor(24, 26, 32, 235))
        p.setPen(QPen(QColor(255, 255, 255, 30), 1))
        p.drawPath(bg)

        # Fonts follow the key rows (pixel sizes, so DPI does not double-scale);
        # the title bar gets its own size from its shorter row.
        key_h = min(r.h for r in self.rects if r.key not in TOOLBAR_KEYS) * H
        bar_h = min(r.h for r in self.rects if r.key in TOOLBAR_KEYS) * H
        font, small, bar = QFont("Segoe UI"), QFont("Segoe UI"), QFont("Segoe UI")
        font.setPixelSize(max(9, int(key_h * 0.36)))
        small.setPixelSize(max(8, int(key_h * 0.25)))
        bar.setPixelSize(max(8, int(bar_h * 0.42)))
        big_bar = QFont(bar)
        big_bar.setPixelSize(max(10, int(bar_h * 0.62)))
        big_bar.setBold(True)
        now = time.monotonic()
        mods = self.out.mods.levels
        armed = self.out.mods.armed()
        dirty = QRectF(e.rect())
        for r in self.rects:
            box = QRectF(r.x * W, r.y * H, r.w * W, r.h * H)
            if not box.intersects(dirty):
                continue                 # partial repaint (cursor easing): untouched key
            path = QPainterPath()
            path.addRoundedRect(box, 7, 7)
            toolbar = r.key in TOOLBAR_KEYS
            fill = QColor(40, 43, 52) if toolbar else QColor(52, 56, 66)
            if r.key in MODIFIERS and mods.get(r.key):
                fill = QColor(52, 112, 206) if mods[r.key] == 1 else QColor(36, 84, 160)
            if r.key == "caps" and self.out.caps:
                fill = QColor(46, 98, 180)
            if r.key == "move" and self._moving:
                fill = QColor(46, 98, 180)
            ft, fh = self._flash.get(r.key, (-1.0, 0))
            if now - ft < 0.15:
                fill = QColor(HAND_COLORS[fh % 2]).darker(115) if self._dual else QColor(90, 150, 240)
            p.fillPath(path, fill)
            if r.key in MODIFIERS and mods.get(r.key):
                # Armed modifier: a bright bar along the bottom edge.
                p.fillRect(QRectF(box.x() + 5, box.bottom() - 4, box.width() - 10, 2.5), QColor(150, 200, 255))
            for hand, (hk, prog) in enumerate(self._hover):
                if hk == r.key:
                    p.setPen(QPen(HAND_COLORS[hand], 2.2))
                    p.drawPath(path)
                    if prog > 0:
                        bar = QRectF(box.x() + 4, box.bottom() - 5, (box.width() - 8) * min(prog, 1), 3)
                        p.fillRect(bar, HAND_COLORS[hand])
            label = self._label(r.key)
            p.setPen(QColor(235, 238, 245) if not toolbar else QColor(170, 178, 192))
            if toolbar:
                p.setFont(big_bar if r.key in ("size-", "size+", "hide") else bar)
            else:
                p.setFont(small if len(label) > 2 else font)
            p.drawText(box, Qt.AlignCenter, label)

        if armed:
            # Armed one-shot modifiers: a pill in the title bar (over the
            # suggestion slots - the next key is a shortcut, not a word).
            sl = [r for r in self.rects if r.key in ("sug0", "sug1", "sug2")]
            if sl:
                area = QRectF(sl[0].x * W, sl[0].y * H, (sl[-1].x + sl[-1].w - sl[0].x) * W, sl[0].h * H)
                p.setFont(bar)
                txt = " + ".join(MOD_NAMES[m] for m in armed) + " + …   next key · tap again to cancel"
                tw = min(p.fontMetrics().horizontalAdvance(txt) + 24, area.width())
                pill = QRectF(area.x() + 4, area.y() + 3, tw, area.height() - 6)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(52, 112, 206))
                p.drawRoundedRect(pill, pill.height() / 2, pill.height() / 2)
                p.setPen(QColor(240, 246, 255))
                p.drawText(pill, Qt.AlignCenter, txt)
        if self.warning or self._target_desc:
            p.setFont(small)
            p.setPen(QColor(255, 190, 90) if self.warning else QColor(140, 150, 165))
            txt = self.warning or f"→ {self._target_desc}"
            p.drawText(QRectF(10, H - 16, W - 30, 14), Qt.AlignLeft | Qt.AlignVCenter, txt)
        # resize grip
        p.setPen(QPen(QColor(255, 255, 255, 60), 1.5))
        for i in (5, 10):
            p.drawLine(W - 4, H - i - 2, W - i - 2, H - 4)
        if self._dual and self.cfg.hand_cursors:
            for i, d in self._cursors.items():
                x, y = self._shown_pos.get(i, (d["x"] * W, d["y"] * H))
                self._paint_hand(p, x, y, i, key_h, d)
        p.end()

    def _paint_hand(self, p: QPainter, x: float, y: float, hand: int, key_h: float, d: dict) -> None:
        """One miniature hand cursor; its index fingertip is (x, y) - the
        exact point that is hit-tested."""
        side = HAND_SIDES[hand % 2]
        if side not in _HAND_PATHS:
            _HAND_PATHS[side] = (hand_path(side), cuff_path(side), crease_path(side))
        body, cuff, crease = _HAND_PATHS[side]
        accent = HAND_COLORS[hand % 2]
        size = self._cursor_size()
        pt = self._press_t.get(hand) or 0.0
        age = time.monotonic() - pt if pt else 9.0
        press = max(0.0, 1.0 - age / 0.18) if pt else 0.0
        if d.get("pressed"):
            press = max(press, 0.35)
        p.save()
        p.setOpacity(0.45 if d.get("lost") else 0.95)
        # fingertip ring: where the key is pressed
        if press > 0:
            ring = QColor(accent)
            ring.setAlpha(int(90 + 150 * press))
            p.setPen(QPen(ring, 2.0))
            p.setBrush(Qt.NoBrush)
            rr = size * (0.16 + 0.10 * press)
            p.drawEllipse(QRectF(x - rr, y - rr, 2 * rr, 2 * rr))
        k = size * (1.0 - 0.10 * press)
        tr = QTransform().translate(x, y).scale(k, k)
        shadow = QTransform().translate(x + 1.2, y + 2.0).scale(k, k)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 80))
        p.drawPath(shadow.map(body))
        grad = QLinearGradient(x, y, x, y + k)
        grad.setColorAt(0.0, QColor(250, 252, 255))
        grad.setColorAt(1.0, QColor(accent).lighter(165))
        outline = QPen(QColor(18, 22, 30, 220), 1.6)
        outline.setCosmetic(True)
        p.setPen(outline)
        p.setBrush(grad)
        p.drawPath(tr.map(body))
        if size >= 24:
            cp = QPen(QColor(40, 48, 62, 140), 1.0)
            cp.setCosmetic(True)
            p.setPen(cp)
            p.setBrush(Qt.NoBrush)
            p.drawPath(tr.map(crease))
        edge = QPen(accent, 1.1)
        edge.setCosmetic(True)
        p.setPen(edge)
        p.setBrush(accent)
        p.drawPath(tr.map(cuff))
        if size >= 26:
            f = QFont("Segoe UI")
            f.setPixelSize(max(7, int(size * 0.16)))
            f.setBold(True)
            p.setFont(f)
            p.setPen(QColor(12, 16, 24))
            c = tr.map(cuff).boundingRect()
            p.drawText(c, Qt.AlignCenter, "L" if side == "left" else "R")
        p.restore()

    def _label(self, key: str) -> str:
        if key.startswith("sug"):
            i = int(key[3:])
            return self._sugs[i] if i < len(self._sugs) else ""
        if key == "lang":
            return {"en": "EN", "vi": "VI", "raw": "IME"}[self.cfg.language]
        if key == "sizepct":
            return f"{self.cfg.size_pct}%" + ("" if self.cfg.width_pct == 100 else f" W{self.cfg.width_pct}")
        if key == "space":
            return {"en": "English", "vi": "Tiếng Việt (Telex)", "raw": "System IME"}[self.cfg.language]
        return key_label(key, self.out.shift, self.out.caps, self.cfg.language)


def _clamp(v: int, rng: tuple[int, int]) -> int:
    return min(max(int(v), rng[0]), rng[1])
