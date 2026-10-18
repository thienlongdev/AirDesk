"""On-screen gesture feedback next to the cursor (lightweight, optional).

A tiny click-through, non-activating, topmost window. Since v1.1.1 every
click-related state is taken from the interaction log (`core/clicklog.py`),
i.e. from what was actually submitted to Windows - never from the
recognizer's intention alone. The states look different on purpose:

* **Aim** (Smart Aim Lock) - a small, faint crosshair: the cursor is held
  where you aimed; *no mouse button has been pressed*. Since v1.1.2 it
  looks nothing like a click and gives way to the click result at once.
* **Click completed** - one green ring with a check mark, briefly: both
  ordered button events were submitted (Instant Auto-Click: right at the
  pinch).
* **Pressed** - a solid green ring: Windows accepted the button-down
  (classic mode, drags). While dragging, the ring follows the cursor.
* **Accepted, waiting** - a dashed ring (thumb-pointer presses, where the
  release decides between left and right click).
* **Drag Mode hold** - a dashed ring filling up toward the 1.2 s hold;
  nothing is pressed until it is full. A "Drag Mode" badge beside the
  cursor shows the mode is on.
* **Pinky Drag** (v1.1.3) - *Drag Ready*: a mint ring filling over the
  0.2 s acquisition (nothing pressed yet); *Dragging*: the solid ring
  follows the cursor (button down accepted by Windows); *Released*: a
  brief badge once the button-up was submitted; *Cancelled*: amber cross
  when the pose was dropped before 0.2 s (nothing was sent).
* **Not clicked** - an amber (cancelled) or red (rejected) ring with a
  cross and a short reason, e.g. "Tracking lost" or "Blocked by Windows".
* a progress arc while the **🤟 keyboard gesture** is held, a small
  "Scroll" badge in **Scroll Mode**, and a brief text hint when the hand
  is too far, too close, poorly tracked or leaving the camera view.

SendInput accepting an event proves Windows received it, not that every
application acted on it; the controlled check (scripts/check_click_reliability.py)
verifies real target activation.

It never receives input (WS_EX_TRANSPARENT via Qt.WindowTransparentForInput
- verified with real clicks through the marker) and never takes focus.
Positions come in physical pixels and are applied with SetWindowPos.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..platform import win32

RING = 64          # logical px: the ring area, centred on the point
WIDTH = 230        # logical px: ring + room for a short reason text
DONE_S = 0.35      # how long "click completed" stays visible
FAIL_S = 1.4       # how long a cancelled / rejected outcome stays visible

AIM = QColor(76, 154, 255)
DOWN = QColor(60, 207, 142)
CANCEL = QColor(255, 176, 64)
REJECT = QColor(240, 84, 84)
TEXT = QColor(235, 238, 245)


def short_reason(reason: str) -> str:
    r = (reason or "").lower()
    for key, text in (("tracking lost", "Tracking lost"), ("administrator", "Blocked: admin window"),
                      ("disarm", "Input disarmed"), ("paused", "Paused"), ("emergency", "Emergency stop"),
                      ("physical mouse", "Physical mouse moved"), ("style", "Click style changed"),
                      ("sendinput", "Windows rejected the input"), ("zoom", "Zoom gesture"),
                      ("held too long", "Held too long"), ("debounce", "Too soon after last click"),
                      ("drag hold", "Released before drag"), ("click mode", "Click mode changed"),
                      ("not held", "Button already released"), ("hand changed", "Hand changed"),
                      ("keyboard opened", "Air Keyboard opened"), ("pinky", "Drag cancelled")):
        if key in r:
            return text
    return (reason[:34] + "…") if len(reason) > 35 else (reason or "Not clicked")


def feedback_state(st, it, now: float, enabled: bool = True):
    """(kind, point, progress, text) for the latest status + interaction.

    `it` is the newest `Interaction` (or None). Priority: an active press,
    then Aim Lock, then a recent outcome flash, then gesture badges.
    """
    if not (enabled and st.armed):
        return "", None, 0.0, ""
    if it is not None and not it.done and st.tracking:
        if it.phase == "dragging":
            return "drag", st.cursor, 0.0, ""
        if it.phase == "pressed":
            return "down", it.pos, 0.0, ""
        if it.phase == "accepted":
            if it.kind == "drag" and it.slot == "pinky":  # Pinky Drag acquisition: nothing pressed yet
                return "dragready", it.pos, getattr(st, "drag_progress", 0.0), "Drag Ready"
            if it.kind == "drag":                       # Drag Mode: holding toward the threshold
                return "dragwait", it.pos, getattr(st, "drag_progress", 0.0), ""
            return "pending", it.pos, 0.0, ""
    if it is not None and it.done and it.wall_end:
        # A finished click is the message that matters: it shows before
        # (and instead of) a new aim indicator.
        age = now - it.wall_end
        if it.phase == "completed" and age < DONE_S and it.kind != "drag":
            return "done", it.pos, 0.0, ""
        if it.phase == "completed" and it.kind == "drag" and it.slot == "pinky" and age < 0.6:
            return "released", st.cursor or it.pos, 0.0, "Released"
        if it.phase in ("cancelled", "rejected") and age < FAIL_S:
            return ("cancel" if it.phase == "cancelled" else "reject"), it.pos, 0.0, short_reason(it.reason)
    if st.aim_lock is not None and st.tracking:
        return "aim", st.aim_lock, 0.0, ""
    if st.tracking and (0.0 < st.progress < 1.0 or st.gesture.startswith("Keyboard")
                        or st.gesture.startswith("🤟")):
        return "kb", st.cursor, st.progress, ""
    if st.tracking and st.state == "SCROLLING":
        return "scroll", st.cursor, 0.0, ""
    if st.tracking and getattr(st, "hint", ""):
        return "hint", st.cursor, 0.0, st.hint
    if st.tracking and getattr(st, "click_mode", "") == "drag":
        return "dragmode", st.cursor, 0.0, "Drag Mode"
    return "", None, 0.0, ""


class FeedbackMarker(QWidget):
    def __init__(self) -> None:
        super().__init__(None, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                         | Qt.WindowDoesNotAcceptFocus | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.NoFocus)
        self.resize(WIDTH, RING)
        self.kind = ""
        self.progress = 0.0
        self.text = ""
        self._last: tuple | None = None

    def update_from(self, st, enabled: bool, it=None) -> None:
        """Called from a GUI timer with the latest EngineStatus and the
        newest interaction-log record."""
        kind, at, prog, text = feedback_state(st, it, time.perf_counter(), enabled)
        key = (kind, at, round(prog, 2), text)
        if key == self._last:
            return
        self._last = key
        if not kind or at is None:
            if self.isVisible():
                self.hide()
            return
        self.kind, self.progress, self.text = kind, prog, text
        dpr = self.devicePixelRatioF() or 1.0
        half = int(RING * dpr / 2)
        x, y = at
        if kind in ("kb", "scroll", "hint", "dragmode"):
            x, y = x + half, y + half          # beside the cursor, not on the target
        if not self.isVisible():
            self.show()
            if win32.IS_WINDOWS:
                win32.make_noactivate(int(self.winId()))
        if win32.IS_WINDOWS:
            win32.move_window_noactivate(int(self.winId()), int(x - half), int(y - half))
        else:
            self.move(int(x - half), int(y - half))
        self.update()

    def paintEvent(self, e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = RING / 2
        r = QRectF(c - 14, c - 14, 28, 28)
        k = self.kind
        if k == "aim":
            # Small faint crosshair ticks: the aim is held, nothing pressed.
            # Deliberately unlike any click marker.
            aim = QColor(AIM)
            aim.setAlpha(170)
            for pen in (QPen(QColor(0, 0, 0, 90), 3), QPen(aim, 1.4)):
                p.setPen(pen)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    p.drawLine(int(c + dx * 5), int(c + dy * 5), int(c + dx * 9), int(c + dy * 9))
        elif k == "dragwait":
            p.setPen(QPen(QColor(0, 0, 0, 140), 5))
            p.drawEllipse(r)
            pen = QPen(AIM, 2.0)
            pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.drawEllipse(r)
            p.setPen(QPen(DOWN, 3.5))
            p.drawArc(r, 90 * 16, -int(360 * 16 * max(self.progress, 0.02)))
        elif k == "dragready":
            p.setPen(QPen(QColor(0, 0, 0, 140), 5))
            p.drawEllipse(r)
            pen = QPen(DOWN, 1.6)
            pen.setStyle(Qt.DotLine)
            p.setPen(pen)
            p.drawEllipse(r)
            p.setPen(QPen(DOWN, 3.5))
            p.drawArc(r, 90 * 16, -int(360 * 16 * max(self.progress, 0.02)))
            self._tag(p, c, DOWN, self.text)
        elif k == "released":
            p.setPen(QPen(QColor(0, 0, 0, 120), 5))
            p.drawEllipse(r)
            p.setPen(QPen(DOWN, 2.4))
            p.drawEllipse(r)
            self._tag(p, c, DOWN, self.text)
        elif k in ("hint", "dragmode"):
            p.setFont(QFont("Segoe UI", 8, QFont.Bold))
            w = min(p.fontMetrics().horizontalAdvance(self.text) + 16, WIDTH - c + 18)
            box = QRectF(c - 22, c - 10, w, 20)
            p.setBrush(QColor(24, 26, 32, 225))
            p.setPen(QPen(CANCEL if k == "hint" else DOWN, 1.5))
            p.drawRoundedRect(box, 6, 6)
            p.setPen(TEXT)
            p.drawText(box, Qt.AlignCenter, self.text)
        elif k in ("down", "drag", "done", "pending"):
            p.setPen(QPen(QColor(0, 0, 0, 150), 6))
            p.drawEllipse(r)
            pen = QPen(DOWN if k != "pending" else AIM, 3.2)
            if k == "pending":
                pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.drawEllipse(r)
            if k == "done":
                self._badge(p, c, DOWN, "✓")
        elif k in ("cancel", "reject"):
            col = CANCEL if k == "cancel" else REJECT
            p.setPen(QPen(QColor(0, 0, 0, 150), 6))
            p.drawEllipse(r)
            p.setPen(QPen(col, 3.0))
            p.drawEllipse(r)
            self._badge(p, c, col, "✕")
            if self.text:
                p.setFont(QFont("Segoe UI", 8, QFont.Bold))
                box = QRectF(c + 22, c - 10, WIDTH - c - 24, 20)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(24, 26, 32, 225))
                p.drawRoundedRect(box, 5, 5)
                p.setPen(col)
                p.drawText(box.adjusted(6, 0, -4, 0), Qt.AlignVCenter | Qt.AlignLeft, self.text)
        elif k == "kb":
            p.setPen(QPen(QColor(0, 0, 0, 120), 5))
            p.drawEllipse(r)
            p.setPen(QPen(AIM, 3.5))
            p.drawArc(r, 90 * 16, -int(360 * 16 * max(self.progress, 0.02)))
            p.setPen(TEXT)
            p.setFont(QFont("Segoe UI Emoji", 8))
            p.drawText(r, Qt.AlignCenter, "🤟")
        elif k == "scroll":
            box = QRectF(c - 22, c - 10, 44, 20)
            p.setBrush(QColor(24, 26, 32, 220))
            p.setPen(QPen(AIM, 1.5))
            p.drawRoundedRect(box, 6, 6)
            p.setPen(TEXT)
            p.setFont(QFont("Segoe UI", 7, QFont.Bold))
            p.drawText(box, Qt.AlignCenter, "⇕ Scroll")
        p.end()

    @staticmethod
    def _tag(p: QPainter, c: float, col: QColor, text: str) -> None:
        """Short state text to the right of the ring."""
        p.setFont(QFont("Segoe UI", 8, QFont.Bold))
        box = QRectF(c + 22, c - 10, min(p.fontMetrics().horizontalAdvance(text) + 14, WIDTH - c - 24), 20)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(24, 26, 32, 225))
        p.drawRoundedRect(box, 5, 5)
        p.setPen(col)
        p.drawText(box, Qt.AlignCenter, text)

    @staticmethod
    def _badge(p: QPainter, c: float, col: QColor, glyph: str) -> None:
        b = QRectF(c + 7, c + 7, 15, 15)
        p.setPen(Qt.NoPen)
        p.setBrush(col)
        p.drawEllipse(b)
        p.setPen(QColor(16, 18, 22))
        p.setFont(QFont("Segoe UI", 8, QFont.Bold))
        p.drawText(b, Qt.AlignCenter, glyph)
