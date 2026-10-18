"""Control-center pages: Home, Air Mouse Pro, Air Keyboard, Finger Selection,
Gesture Mapping, Precision Controls, Diagnostics and Settings.
(Calibration and Benchmark Laboratory live in their own modules.)"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QFormLayout, QGridLayout, QHBoxLayout,
                               QHeaderView, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton,
                               QTableWidget, QTableWidgetItem, QWidget)

from ..config import app_dir
from ..core.actions import GESTURE_LABELS, POSE_ACTIONS, SLOT_ACTIONS
from ..core.filters import FILTER_NAMES
from ..core.gestures import STYLE_NAMES, slot_contacts
from ..core.types import Finger
from ..platform import autostart, win32
from ..platform.hotkeys import parse_hotkey
from ..vision.camera import MODES
from .widgets import Bars, Binder, Card, PreviewWidget, Sparkline, group, label, page

if TYPE_CHECKING:
    from .main_window import AppContext

ACTION_LABELS = {
    "left_click_drag": "Left click (legacy Drag Mode: hold to drag)", "left_click": "Left click",
    "right_click": "Right click", "right_click_drag": "Right click (legacy Drag Mode: hold to drag)",
    "middle_click": "Middle click",
    "double_click": "Double click", "triple_click": "Triple click", "back": "Browser back",
    "forward": "Browser forward", "toggle_keyboard": "Show/hide Air Keyboard",
    "toggle_precision": "Toggle precision mode", "toggle_magnifier": "Toggle Windows Magnifier",
    "scroll": "Scroll", "clutch": "Clutch (freeze cursor)", "zoom": "Zoom (Ctrl + wheel)",
    "pause": "Pause / resume", "none": "— nothing —",
}
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Little"]
CLICK_MODES = {
    "instant": "Instant Auto-Click",
    "drag": "Legacy Drag Mode",
    "classic": "Classic (click on release)",
}
CLICK_MODE_HELP = {
    "instant": "<b>Instant Auto-Click</b> (default): pinch once → the click completes immediately "
               "(button down + up together). Holding the pinch does nothing more; open your fingers "
               "before the next click. To drag, use <b>Pinky Drag</b> (only the little finger up, "
               "{pinky:.1f} s) — no mode switch.",
    "drag": "<b>Legacy Drag Mode</b> (v1.1.2; Pinky Drag replaces it): aim, pinch and hold <b>{hold:.1f} s</b> "
            "— the ring fills, then the button goes down and the cursor follows your hand; release to drop. "
            "Releasing earlier does nothing (no click). {oneshot}",
    "classic": "<b>Classic</b> (v1.1.1 behaviour): the button goes down at the pinch and up when you open "
               "your fingers, so the click completes on release; holding {hold:.1f} s turns it into a drag. "
               "It cannot complete a click at the pinch because it must first tell a click from a drag.",
}


def _opts(values) -> list[tuple[str, str]]:
    return [(v, ACTION_LABELS.get(v, v)) for v in values]


class BasePage:
    widget: QWidget

    def __init__(self, ctx: "AppContext") -> None:
        self.ctx = ctx
        self.binder = Binder(ctx.changed)

    def on_tick(self, status, metrics) -> None:
        pass

    def refresh(self) -> None:
        self.binder.refresh()


# ====================================================================== Home
class HomePage(BasePage):
    def __init__(self, ctx: "AppContext") -> None:
        super().__init__(ctx)
        self.widget, lay = page("Home", "Point with any finger. Pinch to click. Your physical mouse and "
                                "keyboard always keep working, and Ctrl+Alt+F12 disarms instantly.")
        bar = QHBoxLayout()
        self.arm = QPushButton("ARM GESTURE INPUT")
        self.arm.setObjectName("Arm")
        self.arm.clicked.connect(ctx.runtime.toggle_arm)
        self.pause = QPushButton("Pause")
        self.pause.clicked.connect(lambda: ctx.runtime.pause(not ctx.runtime.engine.paused))
        self.kb = QPushButton("Air Keyboard")
        self.kb.clicked.connect(ctx.toggle_keyboard)
        self.prec = QPushButton("Precision")
        self.prec.clicked.connect(ctx.runtime.toggle_precision)
        self.drag = QPushButton("Legacy Drag Mode")
        self.drag.setCheckable(True)
        self.drag.setToolTip("Not needed for dragging since v1.1.3 - use Pinky Drag (only the little finger "
                             "up, 0.2 s). Legacy: the next pinch-and-hold drags (Ctrl+Alt+F8); Instant "
                             "Auto-Click comes back after one drag unless one-shot is off on Gesture Mapping.")
        self.drag.clicked.connect(lambda _c: ctx.runtime.toggle_drag_mode())
        stop = QPushButton("Emergency stop")
        stop.setObjectName("Danger")
        stop.clicked.connect(ctx.runtime.emergency_stop)
        for w in (self.arm, self.pause, self.kb, self.prec, self.drag):
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(stop)
        lay.addLayout(bar)

        grid = QGridLayout()
        grid.setSpacing(10)
        names = ["Camera", "Input", "Active hand", "Active finger", "Pointer mode", "Gesture",
                 "Capture FPS", "Tracking FPS", "Processing latency", "Mouse buttons", "Air Keyboard",
                 "Click style", "Click mode", "Last click", "Pinky Drag", "Recognized pose"]
        self.cards = {n: Card(n) for n in names}
        for i, n in enumerate(names):
            grid.addWidget(self.cards[n], i // 4, i % 4)
        lay.addLayout(grid)

        head = QHBoxLayout()
        head.addWidget(label("Camera preview", "H2"))
        head.addStretch(1)
        self.toggle_prev = QPushButton()
        self.toggle_prev.clicked.connect(self._toggle_preview)
        head.addWidget(self.toggle_prev)
        lay.addLayout(head)
        self.preview = PreviewWidget()
        self.preview.setMinimumHeight(300)
        lay.addWidget(self.preview, 1)
        self.hint = label("", "Muted", wrap=True)
        lay.addWidget(self.hint)
        self._sync_preview_button()

    def _toggle_preview(self) -> None:
        self.ctx.s.ui.show_preview = not self.ctx.s.ui.show_preview
        self.ctx.changed("ui")
        self._sync_preview_button()

    def _sync_preview_button(self) -> None:
        on = self.ctx.s.ui.show_preview
        self.toggle_prev.setText("Hide preview" if on else "Show preview")
        self.preview.setVisible(on)

    def on_tick(self, st, m) -> None:
        rt, c = self.ctx.runtime, self.cards
        armed = st.armed or rt.engine.armed
        self.arm.setText("ARMED — click to disarm" if armed else "ARM GESTURE INPUT")
        self.arm.setStyleSheet(f"background: {'#2fae73' if armed else '#4c9aff'}; color: white;")
        self.pause.setText("Resume" if rt.engine.paused else "Pause")
        cam = {"running": ("Running", "ok"), "opening": ("Opening…", "warn"),
               "reconnecting": ("Reconnecting", "warn"), "error": ("Error", "bad"),
               "stopped": ("Stopped", "muted")}.get(m.camera_status, (m.camera_status, None))
        c["Camera"].set(cam[0], cam[1])
        c["Input"].set("PAUSED" if rt.engine.paused else ("ARMED" if armed else "DISARMED"),
                       "warn" if rt.engine.paused else ("ok" if armed else "bad"))
        c["Active hand"].set(st.hand if st.tracking else "—")
        c["Active finger"].set(f"{st.finger} ({st.finger_mode})" if st.tracking else f"— ({st.finger_mode})")
        c["Pointer mode"].set(st.pointer_mode.capitalize())
        c["Gesture"].set(st.gesture if st.tracking else "No hand")
        c["Capture FPS"].set(f"{m.capture_fps:.1f}")
        c["Tracking FPS"].set(f"{m.tracking_fps:.1f}")
        c["Processing latency"].set(f"{m.latency_ms:.1f} ms")
        c["Mouse buttons"].set(", ".join(st.buttons) or "none", "warn" if st.buttons else None)
        c["Air Keyboard"].set("Visible" if self.ctx.overlay.isVisible() else "Hidden")
        gc = self.ctx.s.gesture
        c["Click style"].set(f"{STYLE_NAMES.get(gc.click_style, gc.click_style)} · "
                             f"{gc.pinch_sensitivity.capitalize()}")
        mode = rt.engine.click_mode
        c["Click mode"].set(CLICK_MODES.get(mode, mode), "warn" if mode == "drag" else None)
        if self.drag.isChecked() != (mode == "drag"):
            self.drag.setChecked(mode == "drag")
        self.drag.setText("Legacy Drag Mode ON" if mode == "drag" else "Legacy Drag Mode")
        drag = {"ready": ("Drag Ready…", "warn"), "dragging": ("Dragging (button down)", "warn"),
                "released": ("Released ✓", "ok"), "cancelled": ("Cancelled (nothing sent)", None)}
        c["Pinky Drag"].set(*drag.get(st.drag, ("Ready: little finger only, "
                                                f"{self.ctx.s.gesture.pinky_drag_s:.1f} s"
                                                if self.ctx.s.gesture.pinky_drag else "Off", None)))
        c["Recognized pose"].set(("Air Keyboard · two hands" if st.dual else st.pose) if st.tracking else "—")
        last = rt.engine.clicks.latest()
        if last is not None:
            ok = last.phase == "completed"
            c["Last click"].set(f"{'✓' if ok else '✕'} {last.button} {last.kind} · {last.clock}",
                                "ok" if ok else ("warn" if last.phase == "cancelled" else
                                                 "bad" if last.phase == "rejected" else None))
        else:
            c["Last click"].set("—")
        if self.ctx.s.ui.show_preview and self.widget.isVisible():
            pc = self.ctx.s.pointer
            box = (pc.box_cx, pc.box_cy, pc.box_w, pc.box_h) if st.pointer_mode == "direct" else None
            self.preview.set_frame(rt.preview, rt.engine.hands._last_wrist, st.finger if st.tracking else None,
                                   box, pc.anchor_blend, st.gesture)
        msg = m.camera_error or ("; ".join(m.hotkey_errors) and "Hotkey conflict: " + "; ".join(m.hotkey_errors))
        self.hint.setText(msg or st.message or (st.hint and f"Camera: {st.hint}") or
                          f"Gestures: pinch = click (completes at once) · only the little finger up "
                          f"(thumb tucked) {gc.pinky_drag_s:.1f} s = Pinky Drag, curl it to drop · "
                          f"V pose (index + middle up) = Scroll Mode · 🤟 "
                          f"(thumb, index, little out) {gc.keyboard_gesture_s:.1f} s = show/hide Air Keyboard · "
                          "fist = clutch.  Hotkeys: Ctrl+Alt+F11 arm/disarm · Ctrl+Alt+F12 emergency stop · "
                          "Ctrl+Alt+F10 keyboard · Ctrl+Alt+F9 precision · Ctrl+Alt+F8 legacy Drag Mode")


# ============================================================= Air Mouse Pro
class MousePage(BasePage):
    def __init__(self, ctx: "AppContext") -> None:
        super().__init__(ctx)
        self.widget, lay = page("Air Mouse Pro", "Real Windows mouse input (SendInput) with three pointer "
                                "modes. Changes apply immediately.")
        b, pc, gc = self.binder, ctx.s.pointer, ctx.s.gesture
        g, f = group("Pointer mode", lay)
        b.combo(f, "Mode", pc, "mode", [("direct", "Direct — hand position maps to the screen"),
                                        ("relative", "Relative trackpad — hand motion moves the cursor"),
                                        ("precision", "Precision — low gain, heavy smoothing")], "pointer")
        b.combo(f, "Target monitor", pc, "monitor", self._monitor_opts(), "pointer",
                tip="Direct mode maps the control box onto this area")
        b.number(f, "Direct gain", pc, "direct_gain", 0.5, 2.5, 0.05, 2, section="pointer")
        b.number(f, "Relative gain (slow)", pc, "relative_gain_low", 0.2, 4, 0.1, 1, section="pointer")
        b.number(f, "Relative gain (fast)", pc, "relative_gain_high", 0.5, 8, 0.1, 1, section="pointer")
        g, f = group("Smoothing", lay)
        b.combo(f, "Filter", pc, "filter", [(n, {"one_euro": "One Euro (recommended)",
                                                 "kalman": "Adaptive Kalman", "ema": "EMA (baseline)",
                                                 "none": "None (raw)"}[n]) for n in FILTER_NAMES], "pointer")
        b.number(f, "Jitter cutoff (One Euro min cutoff)", pc, "min_cutoff", 0.1, 5, 0.05, 2, "Hz", "pointer",
                 "Lower = steadier when still, more lag")
        b.number(f, "Speed response (One Euro beta)", pc, "beta", 0, 40, 0.5, 1, section="pointer",
                 tip="Higher = less lag during fast motion")
        b.number(f, "Kalman prediction", pc, "kalman_prediction_ms", 0, 60, 5, 0, "ms", "pointer")
        b.number(f, "Cursor easing (120 Hz dispatcher)", pc, "smoothing_tau_ms", 0, 40, 1, 0, "ms", "pointer")
        b.number(f, "Dispatch rate", pc, "dispatch_hz", 60, 240, 10, 0, "Hz", "pointer")
        b.number(f, "Physical mouse takes over for", pc, "yield_to_physical_s", 0.2, 5, 0.1, 1, "s", "pointer")
        g, f = group("Scroll Mode (V pose: index + middle up, ring + little curled)", lay)
        f.addRow(label("Make the V pose: the cursor freezes and the status shows Scroll Ready. Move the hand up "
                       "or down to scroll (further = faster), clearly sideways for horizontal scrolling. "
                       "Drop the V to exit; the cursor continues from where it froze. Middle click is a "
                       "separate pinch.", "Muted", wrap=True))
        b.combo(f, "Style", gc, "scroll_style", [("joystick", "Joystick — displacement sets speed"),
                                                ("direct", "Direct — hand motion drags the page")], "gesture")
        b.number(f, "Speed", gc, "scroll_speed", 2, 60, 1, 0, "notches/s", "gesture")
        b.number(f, "Dead zone", gc, "scroll_dead_zone", 0.02, 0.5, 0.01, 2, section="gesture")
        b.number(f, "Acceleration (1 = linear)", gc, "scroll_accel", 1.0, 3.0, 0.1, 1, section="gesture",
                 tip="Higher = fine control near the start, fast scrolling at full deflection")
        b.number(f, "Horizontal needs x more sideways motion", gc, "scroll_h_bias", 1.0, 4.0, 0.1, 1,
                 section="gesture")
        b.check(f, "Natural (reversed) scrolling", gc, "natural_scroll", "gesture")
        b.check(f, "Lock to the first axis moved (vertical or horizontal)", gc, "scroll_axis_lock", "gesture")
        b.check(f, "Smooth sub-notch wheel deltas", gc, "smooth_wheel", "gesture")
        g, f = group("Supported mouse actions", lay)
        f.addRow(label("Move · left/right/middle click · double & triple click · click-and-hold · drag-and-drop "
                       "(text selection, window move/resize, file drag) · vertical & horizontal scroll · "
                       "browser back/forward (X1/X2 buttons) · Ctrl+wheel zoom · clutch · precision mode · "
                       "multi-monitor. Bind gestures on the Gesture Mapping page.", "Muted", wrap=True))

    def _monitor_opts(self):
        opts = [("all", "All monitors (virtual desktop)"), ("primary", "Primary monitor")]
        if self.ctx.runtime.screen:
            for i, m in enumerate(self.ctx.runtime.screen.monitors):
                opts.append((str(i), f"Monitor {i + 1}: {m.width}×{m.height} at ({m.left}, {m.top})"))
        return opts


# ============================================================== Air Keyboard
class KeyboardPage(BasePage):
    def __init__(self, ctx: "AppContext") -> None:
        super().__init__(ctx)
        self.widget, lay = page("Air Keyboard", "A floating Windows keyboard that never takes focus from the "
                                "app you are typing into. Both hands type: each gets its own small hand cursor "
                                "(left = blue, right = mint); pinch to press the key under the fingertip.")
        row = QHBoxLayout()
        self.btn = QPushButton("Show keyboard")
        self.btn.setObjectName("Primary")
        self.btn.clicked.connect(ctx.toggle_keyboard)
        reset = QPushButton("Reset size and position")
        reset.clicked.connect(self._reset_geometry)
        row.addWidget(self.btn)
        row.addWidget(reset)
        row.addStretch(1)
        lay.addLayout(row)
        self.target = label("", "Muted", wrap=True)
        lay.addWidget(self.target)
        b, kc = self.binder, ctx.s.keyboard
        g, f = group("Layout and language", lay)
        b.combo(f, "Layout", kc, "layout", [("full", "Full (function keys, arrows, shortcuts)"),
                                            ("compact", "Compact")], "keyboard")
        b.combo(f, "Language", kc, "language", [
            ("en", "English (Unicode)"), ("vi", "Tiếng Việt — AirDesk Telex"),
            ("raw", "System IME (send raw keys; use when Unikey/Windows Vietnamese is active)")], "keyboard")
        b.combo(f, "Vietnamese tone placement", kc, "vi_tone_style",
                [("classic", "Classic (hòa, thủy)"), ("modern", "Modern (hoà, thuỷ)")], "keyboard")
        b.check(f, "Word suggestions", kc, "suggestions", "keyboard")
        b.check(f, "Privacy mode (no suggestions; nothing typed is ever stored either way)", kc,
                "privacy_mode", "keyboard")
        g, f = group("Selection", lay)
        b.combo(f, "Select keys by", kc, "selection", [("pinch", "Pinch"), ("dwell", "Dwell (hold still)"),
                                                       ("both", "Pinch or dwell")], "keyboard")
        b.number(f, "Dwell time", kc, "dwell_ms", 250, 2500, 50, 0, "ms", "keyboard")
        b.number(f, "Dwell tolerance", kc, "dwell_tolerance_px", 4, 60, 1, 0, "px", "keyboard")
        b.check(f, "Hold pinch to repeat Backspace / arrows / space", kc, "repeat_backspace", "keyboard")
        b.check(f, "Legacy second-hand typing (only when Dual-Hand is off)", kc, "two_hand_typing", "keyboard")
        g, f = group("Dual-Hand Air Keyboard (v1.1.3)", lay)
        b.check(f, "Both hands type, each with its own hand cursor (no desktop mouse while open)", kc,
                "dual_hand", "keyboard")
        b.check(f, "Show the miniature left / right hand cursors", kc, "hand_cursors", "keyboard")
        b.combo(f, "Pinch that types", kc, "kb_click_style", [
            ("same", "Same as the mouse click style (Normal / Pinch Free)"),
            ("pinch_pointer", "Thumb + index (Normal)"),
            ("pinch_free", "Thumb + middle — the index fingertip stays still (Pinch Free)")], "keyboard")
        b.check(f, "One-shot Ctrl / Shift / Alt (tap = next key only; tap again = cancel)", kc,
                "one_shot_modifiers", "keyboard")
        b.number(f, "Clear an unused modifier after", kc, "modifier_timeout_s", 2, 60, 1, 0, "s", "keyboard")
        b.number(f, "Held pinch repeats Backspace / Del / arrows after", kc, "key_repeat_delay_s", 0.3, 2.0, 0.05,
                 2, "s", "keyboard")
        f.addRow(label("Shortcuts: tap Ctrl with either hand, then C with either hand → a real Ctrl+C (Ctrl down, "
                       "C down, C up, Ctrl up). Several modifiers can be armed (Ctrl+Shift+S). Win+L, Ctrl+Esc, "
                       "Alt+Tab and Alt+Esc are never sent from the Air Keyboard.", "Muted", wrap=True))
        self.hands = label("", "Muted", wrap=True)
        f.addRow("Live", self.hands)
        g, f = group("Size and appearance", lay)
        b.number(f, "Size", kc, "size_pct", 40, 130, 10, 0, "%", "keyboard",
                 "Overall size (both dimensions). Default 70 %")
        b.number(f, "Width", kc, "width_pct", 60, 150, 10, 0, "%", "keyboard",
                 "Narrower or wider keys without changing the height. Default 100 %")
        b.number(f, "Opacity", kc, "opacity", 0.4, 1.0, 0.02, 2, section="keyboard")
        lay.addWidget(label("Show/hide: hold the 🤟 pose about 0.4 s — thumb, index and little finger out, "
                            "middle and ring curled; relax the hand before repeating — or Ctrl+Alt+F10, or the "
                            "button above.", "Muted", wrap=True))
        lay.addWidget(label("Title bar: ⠿ Air Keyboard = move (pinch it, move, pinch again — or drag it with the "
                            "mouse) · suggestions · EN/VI/IME · Layout · − 70% + size · W− / W+ width · Reset · "
                            "× hide. The bottom-right grip resizes with the physical mouse (sideways = width, "
                            "up/down = size). Size, width and position are remembered.", "Muted", wrap=True))
        lay.addStretch(1)

    def _reset_geometry(self) -> None:
        self.ctx.overlay.reset_geometry()
        self.binder.refresh()
        self.ctx.changed("keyboard")

    def on_tick(self, st, m) -> None:
        vis = self.ctx.overlay.isVisible()
        self.btn.setText("Hide keyboard" if vis else "Show keyboard")
        o = self.ctx.overlay
        self.target.setText((f"Typing into: {o._target_desc}" if o._target_desc else "") +
                            (f"  —  {o.warning}" if o.warning else ""))
        warn = ("  (Settings → Hands to track is set to One: only one hand can type - choose Two.)"
                if self.ctx.s.camera.num_hands < 2 and self.ctx.s.keyboard.dual_hand else "")
        self.hands.setText((_hands_text(st, o) if vis else "Open the keyboard to see both hand cursors here.") + warn)


# ========================================================= Finger Selection
class FingerPage(BasePage):
    def __init__(self, ctx: "AppContext") -> None:
        super().__init__(ctx)
        self.widget, lay = page("Finger Selection", "Any of the five fingers can drive the pointer. In Auto mode "
                                "AirDesk follows the finger you point with; switching keeps the cursor in place.")
        b, fc = self.binder, ctx.s.finger
        g, f = group("Mode", lay)
        b.combo(f, "Finger mode", fc, "mode", [("auto", "Auto — follow the pointing finger"),
                                              ("manual", "Manual — always the chosen finger"),
                                              ("lock", "Lock — keep the current finger until changed")], "finger")
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        self.fbtn = QButtonGroup(self.widget)
        for i, n in enumerate(FINGER_NAMES):
            btn = QPushButton(n)
            btn.setCheckable(True)
            self.fbtn.addButton(btn, i)
            h.addWidget(btn)
        self.fbtn.idClicked.connect(lambda i: (ctx.runtime.set_finger(FINGER_NAMES[i]), ctx.changed("finger")))
        f.addRow("Chosen finger", row)
        b.combo(f, "Hand", fc, "hand", [("auto", "Automatic (keep following the same hand)"),
                                        ("right", "Right hand"), ("left", "Left hand")], "finger")
        b.combo(f, "Two hands", fc, "two_hand_mode", [
            ("off", "Off — one hand points and clicks"),
            ("action_hand", "Control + action hand — one hand points, the other pinches to click")], "finger")
        g, f = group("Switching stability", lay)
        b.number(f, "Switch delay", fc, "switch_delay_ms", 50, 1000, 10, 0, "ms", "finger",
                 "The new finger must be the clear candidate this long")
        b.number(f, "Switch confidence", fc, "switch_confidence", 0.3, 0.95, 0.05, 2, section="finger")
        b.number(f, "Freeze when active finger curls below", fc, "curl_freeze", 0.0, 0.7, 0.05, 2,
                 section="finger", tip="Prevents the cursor following a finger that is folding away")
        g, f = group("Live finger analysis", lay)
        self.contacts = label("", "Muted", wrap=True)
        f.addRow(self.contacts)
        self.ext = Bars(FINGER_NAMES)
        self.conf = Bars(FINGER_NAMES)
        f.addRow("Extension", self.ext)
        f.addRow("Pointer confidence", self.conf)
        lay.addStretch(1)

    def refresh(self) -> None:
        super().refresh()
        i = FINGER_NAMES.index(self.ctx.s.finger.finger.capitalize())
        self.fbtn.button(i).setChecked(True)

    def on_tick(self, st, m) -> None:
        self.ext.set(st.extension, st.finger)
        self.conf.set(st.finger_confidence, st.finger)
        if st.tracking and st.finger in FINGER_NAMES:
            sc = slot_contacts(Finger.from_name(st.finger), self.ctx.s.gesture.click_style)
            txt = ", ".join(f"{k}: {v.replace('pinch:', 'thumb + ').replace('thumb_press', 'thumb press on index')}"
                            for k, v in sc.items())
            self.contacts.setText(f"Active: {st.hand} hand, {st.finger} finger.  Contacts → {txt}")
        else:
            self.contacts.setText("Show your hand to the camera to see live finger analysis.")


# ========================================================== Gesture Mapping
STYLE_HELP = {
    "pinch_pointer": "<b>Normal</b> (recommended): the pointer finger may take part in clicking — "
                     "thumb + pointer = left click. Smart Aim Lock keeps the click on the spot you aimed "
                     "at while the finger bends toward the thumb.",
    "pinch_free": "<b>Pinch Free</b>: the pointer finger stays free for aiming; another finger pairing does "
                  "the clicking (index pointer: thumb + middle = left, + ring = right, + little = middle).",
}
GUIDE = [
    ("Point", "Extend one finger (any of the five) — the cursor follows it"),
    ("Click", "Pinch once — the click completes immediately, no need to wait or release first "
              "(the table above shows which finger). Open your fingers before the next click"),
    ("Pinky Drag", "Aim with your pointing finger, then curl it and leave only the little finger up "
                   "(thumb tucked). Hold 0.2 s: \"Drag Ready\" fills, the button goes down where you were "
                   "aiming, and the hand moves the item. Curl the little finger (or point again) to drop. "
                   "No mode switch — clicks keep working as before"),
    ("Little-finger pointer", "Pointing with the little finger needs the thumb out (🤙-like). Little finger "
                              "alone with the thumb tucked is Pinky Drag. With the little finger chosen as "
                              "the pointer (Manual / Lock), Pinky Drag is off"),
    ("Legacy drag", "Legacy Drag Mode (Home button, tray or Ctrl+Alt+F8): pinch and hold 1.2 s, move, "
                    "release"),
    ("Double click", "Two pinches in a row (open the fingers in between)"),
    ("Scroll Mode", "V pose: index + middle up, ring + little curled. Move the hand up/down "
                    "(or clearly sideways) to scroll; drop the V to exit"),
    ("Air Keyboard", "🤟 — thumb, index and little finger out, middle and ring curled; hold about 0.4 s "
                     "(progress ring), relax the hand, repeat to hide. Ctrl+Alt+F10 also works. While it "
                     "is open both hands type with their own hand cursors; one pinch = one key; tap Ctrl, "
                     "then a letter, for shortcuts"),
    ("Clutch", "Closed fist freezes the cursor so you can reposition your hand"),
    ("Back / Forward", "Open-palm swipe right / left"),
    ("Zoom", "Pinch with both hands and move them apart / together"),
    ("Thumb pointer", "Thumb press (one contact, so the release decides): release before 0.8 s = left "
                      "click, between 0.8 and 1.2 s = right click, hold 1.2 s = drag"),
    ("Click feedback", "Small blue crosshair = aim held (nothing pressed) · green ring with ✓ = click "
                       "completed · dotted ring filling + \"Drag Ready\" = Pinky Drag starting (nothing "
                       "pressed yet) · solid green ring = button held (dragging) · \"Released\" = button "
                       "up · amber/red ✕ = not clicked / cancelled, with the reason"),
]
CONTACT_NAMES = {"index": "Thumb + index", "middle": "Thumb + middle", "ring": "Thumb + ring",
                 "little": "Thumb + little"}


class GesturePage(BasePage):
    def __init__(self, ctx: "AppContext") -> None:
        super().__init__(ctx)
        self.widget, lay = page("Gesture Mapping", "Click style is a setting you choose here — AirDesk never "
                                "switches it from your hand pose. Neither style needs an open palm or a fist.")
        b, gc = self.binder, ctx.s.gesture
        g, f = group("Click style", lay)
        self.style_lbl = label("", "H2")
        f.addRow(self.style_lbl)
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        self.style_btn = QButtonGroup(self.widget)
        self._styles = ["pinch_pointer", "pinch_free"]
        for i, text in enumerate(("Normal (recommended)", "Pinch Free")):
            btn = QPushButton(text)
            btn.setCheckable(True)
            self.style_btn.addButton(btn, i)
            h.addWidget(btn)
        h.addStretch(1)
        self.style_btn.idClicked.connect(self._set_style)
        f.addRow(row)
        self.style_help = label("", "Muted", wrap=True)
        f.addRow(self.style_help)
        self.live = label("", "Muted", wrap=True)
        f.addRow(self.live)
        self.table = QTableWidget(4, 2)
        self.table.setHorizontalHeaderLabels(["Physical contact", "Mouse action"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setFixedHeight(4 * 32 + 40)
        f.addRow(self.table)

        g, f = group("Click mode", lay)
        f.addRow(label("Separate from the click style above: the style decides <i>which</i> finger "
                       "contact means which button; the mode decides <i>what happens</i> when a pinch "
                       "is recognised.", "Muted", wrap=True))
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        self.mode_btn = QButtonGroup(self.widget)
        self._modes = list(CLICK_MODES)
        for i, m in enumerate(self._modes):
            btn = QPushButton(CLICK_MODES[m] + (" (default)" if m == "instant" else ""))
            btn.setCheckable(True)
            self.mode_btn.addButton(btn, i)
            h.addWidget(btn)
        h.addStretch(1)
        self.mode_btn.idClicked.connect(lambda i: ctx.runtime.set_click_mode(self._modes[i]))
        f.addRow(row)
        self.mode_help = label("", "Muted", wrap=True)
        f.addRow(self.mode_help)
        b.check(f, "Legacy Drag Mode is one-shot — back to Instant Auto-Click after one drag", gc,
                "drag_one_shot", "gesture")

        g, f = group("Pinky Drag (v1.1.3) — click and drag without switching modes", lay)
        b.check(f, "Little finger only (thumb tucked) = drag", gc, "pinky_drag", "gesture")
        b.number(f, "Pinky Drag hold", gc, "pinky_drag_s", 0.1, 1.0, 0.05, 2, "s", "gesture",
                 "The pose must be held this long before the button goes down. Default 0.2 s")
        f.addRow(label("Priority: emergency stop / disarm → ending a drag (always releases the button) → "
                       "🤟, V scroll, fist → Pinky Drag → pointing and clicks. A pinch never waits for Pinky "
                       "Drag: the two poses cannot overlap (Pinky Drag needs the thumb tucked).",
                       "Muted", wrap=True))

        g, f = group("Pinch recognition", lay)
        b.combo(f, "Sensitivity", gc, "pinch_sensitivity", [
            ("fast", "Fast — about 50 ms"),
            ("balanced", "Balanced — about 55–60 ms (default)"),
            ("stable", "Stable — about 105 ms, more evidence for every contact"),
            ("custom", "Custom — use the values under Timing")], "gesture",
            "Time from finger contact to a recognised pinch (measured on synthetic hands at 30 fps)")
        b.number(f, "Legacy Drag Mode hold", gc, "hold_to_drag_s", 0.5, 3.0, 0.05, 2, "s", "gesture",
                 "Legacy Drag Mode / Classic: pinch held this long starts a drag. Default 1.2 s")
        b.check(f, "Smart Aim Lock — keep the click where I aimed while my fingers close", gc, "aim_lock",
                "gesture")
        b.check(f, "🤟 pose toggles the Air Keyboard", gc, "keyboard_gesture", "gesture")
        b.number(f, "🤟 hold time", gc, "keyboard_gesture_s", 0.2, 2.0, 0.05, 2, "s", "gesture",
                 "Default 0.4 s")

        g, f = group("Gesture guide", lay)
        for name, text in GUIDE:
            f.addRow(name, label(text, "Muted", wrap=True))

        g, f = group("Advanced bindings", lay)
        for key in ("pinch_primary", "pinch_secondary", "pinch_tertiary"):
            self._dict_combo(f, key, SLOT_ACTIONS)
        for key, opts in POSE_ACTIONS.items():
            self._dict_combo(f, key, opts)
        g, f = group("Timing and thresholds", lay)
        b.number(f, "Custom: frames to confirm a pinch", gc, "press_frames", 1, 6, 1, section="gesture")
        b.number(f, "Custom: contact evidence after crossing", gc, "pinch_confirm_ms", 0, 200, 5, 0, "ms",
                 "gesture")
        b.number(f, "Custom: frames to confirm a release", gc, "release_frames", 1, 6, 1, section="gesture")
        b.number(f, "Frames to confirm a pose", gc, "pose_frames", 2, 10, 1, section="gesture")
        b.number(f, "Thumb long-press = right click", gc, "long_press_s", 0.4, 2.0, 0.05, 2, "s", "gesture")
        b.check(f, "Thumb pointer: long press gives right click (thumb drag then unavailable)", gc,
                "long_press_right_click_thumb", "gesture")
        b.number(f, "Swipe distance", gc, "swipe_min_dist", 0.6, 4.0, 0.1, 1, "hands", "gesture")
        b.number(f, "Swipe cooldown", gc, "swipe_cooldown_s", 0.3, 3.0, 0.1, 1, "s", "gesture")
        b.number(f, "Zoom step", gc, "zoom_step", 0.04, 0.5, 0.01, 2, section="gesture")
        b.number(f, "Tracking-loss grace", gc, "tracking_loss_grace_s", 0.05, 1.0, 0.05, 2, "s", "gesture")
        b.number(f, "Auto-disarm when no hand for (0 = never)", gc, "auto_disarm_s", 0, 600, 5, 0, "s",
                 "gesture")
        lay.addStretch(1)
        self._refresh_style()

    def _set_style(self, i: int) -> None:
        gc = self.ctx.s.gesture
        if gc.click_style != self._styles[i]:
            gc.click_style = self._styles[i]
            self.ctx.changed("gesture")
        self._refresh_style()

    def _refresh_style(self, pointer: str | None = None, active: str | None = None) -> None:
        gc = self.ctx.s.gesture
        st = gc.click_style
        self.style_btn.button(self._styles.index(st) if st in self._styles else 0).setChecked(True)
        self.style_lbl.setText(f"Click Style: {STYLE_NAMES.get(st, st)}")
        self.style_help.setText(STYLE_HELP.get(st, ""))
        rows = contact_rows(pointer or self.ctx.s.finger.finger.capitalize(), st, gc)
        self.table.setRowCount(len(rows))
        for r, (contact, act) in enumerate(rows):
            for c, text in enumerate((contact, act)):
                item = QTableWidgetItem(text)
                if active and contact.startswith(active):
                    item.setBackground(Qt.darkCyan)
                self.table.setItem(r, c, item)

    def refresh(self) -> None:
        super().refresh()
        self._refresh_style()

    def on_tick(self, st, m) -> None:
        pointer = st.finger if (st.tracking and st.finger in FINGER_NAMES) else None
        active = None
        if pointer and st.gesture.startswith(("Press", "Click", "Drag", "Key")):
            sc = slot_contacts(Finger.from_name(pointer), self.ctx.s.gesture.click_style)
            slot = next((k for k in ("primary", "secondary", "tertiary") if f"({k})" in st.gesture), None)
            c = sc.get(slot or "", "")
            active = CONTACT_NAMES.get(c.split(":")[-1]) if c.startswith("pinch:") else None
        key = (pointer, active, self.ctx.s.gesture.click_style)
        if key != getattr(self, "_last_key", None):
            self._last_key = key
            self._refresh_style(pointer, active)
        gc = self.ctx.s.gesture
        mkey = (gc.click_mode, gc.hold_to_drag_s, gc.drag_one_shot, gc.pinky_drag_s)
        if mkey != getattr(self, "_last_mode", None):
            self._last_mode = mkey
            m = gc.click_mode if gc.click_mode in self._modes else "instant"
            self.mode_btn.button(self._modes.index(m)).setChecked(True)
            self.mode_help.setText(CLICK_MODE_HELP[m].format(
                hold=gc.hold_to_drag_s, pinky=gc.pinky_drag_s, oneshot="One-shot: Instant Auto-Click returns after the drop."
                if gc.drag_one_shot else "Stays on until you switch it off (Ctrl+Alt+F8)."))
        if st.tracking:
            buttons = ", ".join(st.buttons) or "none"
            self.live.setText(f"Live: pointer finger <b>{pointer or '—'}</b> · gesture <b>{st.gesture}</b>"
                              f" · buttons held: {buttons}")
        else:
            self.live.setText("Live: show your hand to the camera to test contacts here.")

    def _dict_combo(self, form: QFormLayout, key: str, options) -> None:
        d = self.ctx.s.gesture.bindings
        cb = QComboBox()
        for v in options:
            cb.addItem(ACTION_LABELS.get(v, v), v)

        def load() -> None:
            cb.blockSignals(True)
            cb.setCurrentIndex(max(cb.findData(d.get(key, "none")), 0))
            cb.blockSignals(False)

        def store(_i) -> None:
            d[key] = cb.currentData()
            self.ctx.changed("gesture")
        load()
        self.binder._refreshers.append(load)
        cb.currentIndexChanged.connect(store)
        form.addRow(GESTURE_LABELS[key], cb)


def _hands_text(st, overlay) -> str:
    """Per virtual hand: tracked, state, key under the fingertip, keys typed."""
    if not st.vhands:
        return "no hand"
    parts = []
    for v in st.vhands:
        if not v["tracked"]:
            parts.append(f"{v['side']}: not tracked")
            continue
        parts.append(f"{v['side']}: {v['state']}" + (f" over [{v['key']}]" if v["key"] else "")
                     + f" · {v['keys']} keys" + (f" · last [{v['last_key']}]" if v["last_key"] else ""))
    mods = overlay.out.mods.armed() if overlay is not None else []
    return " | ".join(parts) + (f" | armed: {'+'.join(m.capitalize() for m in mods)}" if mods else "")


def _hand_row(v: dict | None) -> str:
    """One keyboard hand (v1.1.4): MediaPipe detection, track, lifecycle,
    cursor, pinch, key target, last key, timings."""
    if not v:
        return "-"
    det = f"MediaPipe: {'yes' if v.get('detected') else 'no'}" + (f" ({v['label']})" if v.get("detected") else "")
    parts = [det, f"track #{v['track']}" if v.get("track") else "no track", v.get("life", "-")
             + (" (identity provisional)" if v.get("provisional") else "")]
    if v.get("since_ms") is not None and not v.get("detected"):
        parts.append(f"last seen {v['since_ms']} ms ago")
    parts.append(f"cursor {v.get('cursor', '-')}")
    parts.append(f"pinch {v.get('pinch', '-')}")
    if v.get("key"):
        parts.append(f"over [{v['key']}]")
    parts.append(f"{v.get('keys', 0)} keys" + (f", last [{v['last_key']}]" if v.get("last_key") else ""))
    if v.get("rec_ms") is not None:
        parts.append(f"contact->key {v['rec_ms']:.0f} ms median")
    if v.get("reacqs"):
        parts.append(f"reacquired {v['reacqs']}x" + (f", ready {v['reacq_ms']:.0f} ms / {v['reacq_frames']} frames"
                                                    if v.get("reacq_ms") is not None else ""))
    return " · ".join(parts)


def _identity_text(st) -> str:
    """Was a hand missing from MediaPipe's output, or not associated?"""
    d = st.dual_diag or {}
    if not d:
        return "-"
    obs = ", ".join(f"{o['label']} {o['score']:.2f} -> {o['to']}" for o in d.get("obs", [])) or "none"
    c = d.get("counters", {})
    pol = {1: "as delivered", -1: "inverted", 0: "ignored (unreliable)"}.get(d.get("polarity"), "?")
    agree = d.get("label_agree")
    return (f"MediaPipe hands {d.get('mp_hands', 0)} [{obs}] · controllers assigned {d.get('assigned', 0)} · "
            f"labels {pol}" + (f" (agree {100 * agree:.0f}% of {d.get('label_samples')} samples)" if agree is not None
                               else "") +
            f" · new tracks {c.get('new_tracks', 0)}, reacquired {c.get('reacquired', 0)}, "
            f"identity corrections {c.get('identity_corrections', 0)}, ambiguous updates {c.get('ambiguous_updates', 0)}")


def _rejects_text(st) -> str:
    c = (st.dual_diag or {}).get("counters", {})
    if not c:
        return "-"
    return (f"keys {c.get('keys', 0)} · simultaneous pairs {c.get('simultaneous_pairs', 0)} · duplicates suppressed "
            f"{c.get('duplicates_suppressed', 0)} · blocked (ambiguous hand) {c.get('uncertain_blocked', 0)} · "
            f"pinch outside keys {c.get('misses', 0)} · stale press prevented {c.get('stale_press_prevented', 0)}")


def _queue_text(overlay) -> str:
    if overlay is None:
        return "-"
    last = overlay.results[-1] if overlay.results else None
    return (f"depth {overlay.queue_depth} · {overlay.emitted} emitted / {overlay.processed} sent · "
            f"max wait {overlay.max_queue_ms:.1f} ms" +
            (f" · last: wait {last.get('queue_ms', 0):.1f} ms, send {last.get('send_ms', 0):.1f} ms" if last else ""))


def _events_text(st) -> str:
    ev = (st.dual_diag or {}).get("events", [])
    return " | ".join(f"{e['t']:.2f}s {e['side']}: {e['event']}" + (f" ({e['detail']})" if e["detail"] else "")
                      for e in ev[-3:]) or "-"


def _dispatch_text(overlay) -> str:
    """Last keyboard dispatches: hand, key, outcome (never the produced text)."""
    if overlay is None or not overlay.results:
        return "-"
    out = []
    for r in list(overlay.results)[-4:]:
        what = r["combo"] or r["key"]
        out.append(f"{r['hand']} {what} {'✓' if r['ok'] else '✕ ' + (r['message'] or r['kind'])}")
    return " · ".join(out)


def contact_rows(pointer: str, style: str, gc) -> list[tuple[str, str]]:
    """(physical contact, resulting action) rows for the Gesture Mapping table."""
    finger = Finger.from_name(pointer)
    if finger is Finger.THUMB:
        return [("Thumb press on the index side", ACTION_LABELS.get(gc.bindings.get("pinch_primary"), "")),
                ("Thumb press held 0.8 s", "Right click" if gc.long_press_right_click_thumb else "—")]
    by_contact = {c.split(":")[1]: slot for slot, c in slot_contacts(finger, style).items()}
    rows = []
    for fname in ("index", "middle", "ring", "little"):
        slot = by_contact.get(fname)
        act = ACTION_LABELS.get(gc.bindings.get(f"pinch_{slot}", "none"), "") if slot else "not assigned"
        mine = fname == finger.name.lower()
        if mine and style == "pinch_free":
            act = "not assigned (pointer stays free)"
        rows.append((CONTACT_NAMES[fname] + (" (pointer)" if mine else ""), act))
    return rows


# ======================================================= Precision Controls
class PrecisionPage(BasePage):
    def __init__(self, ctx: "AppContext") -> None:
        super().__init__(ctx)
        self.widget, lay = page("Precision Controls", "Predictable assistance only: nothing ever snaps the "
                                "cursor to a different target. Every aid can be switched off.")
        b, pc, gc, fc = self.binder, ctx.s.pointer, ctx.s.gesture, ctx.s.finger
        g, f = group("Precision mode (Ctrl+Alt+F9 or a bound gesture)", lay)
        b.number(f, "Precision gain", pc, "precision_gain", 0.05, 1.0, 0.05, 2, section="pointer")
        b.number(f, "Precision jitter cutoff", pc, "precision_min_cutoff", 0.05, 3, 0.05, 2, "Hz", "pointer")
        b.number(f, "Precision speed response", pc, "precision_beta", 0, 20, 0.5, 1, section="pointer")
        g, f = group("Click accuracy", lay)
        b.number(f, "Stationary dead-zone", pc, "deadzone_px", 0, 8, 0.5, 1, "px", "pointer",
                 "Cursor ignores sub-pixel tremor while the hand is still")
        b.check(f, "Smart Aim Lock (anchor the click as soon as a pinch starts closing)", gc, "aim_lock",
                "gesture")
        b.number(f, "Aim Lock: max look-back", gc, "aim_max_rewind_ms", 50, 400, 10, 0, "ms", "gesture",
                 "The anchor is never older than this")
        b.number(f, "Aim Lock: closing speed that arms it", gc, "aim_close_speed", 0.5, 5.0, 0.1, 1,
                 "hands/s", "gesture")
        b.number(f, "Aim Lock: hand motion that cancels it (re-aim)", gc, "aim_cancel_move", 0.1, 1.0, 0.05, 2,
                 section="gesture")
        b.number(f, "Aim Lock: release when the contact stops closing for", gc, "aim_stall_s", 0.1, 0.6, 0.05,
                 2, "s", "gesture", "No pinch follows: the cursor moves again. Default 0.2 s")
        b.number(f, "Contact: confirm a settled contact within", gc, "contact_settle_ratio", 1.0, 1.6, 0.05, 2,
                 "x threshold", "gesture",
                 "A contact that closed in and stopped this close counts as a pinch. 1.0 = off. Default 1.35")
        b.number(f, "Fallback click rewind (Aim Lock off)", pc, "click_rewind_ms", 0, 200, 5, 0, "ms",
                 "pointer", "A click lands where you aimed just before the pinch moved your hand")
        b.number(f, "Settle after release", pc, "release_settle_ms", 0, 400, 10, 0, "ms", "pointer")
        b.number(f, "Double-click stickiness", gc, "sticky_double_click_px", 0, 40, 1, 0, "px", "gesture")
        b.number(f, "Pointer anchor (tip → knuckle blend)", pc, "anchor_blend", 0, 0.8, 0.05, 2,
                 section="pointer", tip="Blending toward the knuckle reduces pinch-induced drift")
        g, f = group("Pinch thresholds (hand-size units)", lay)
        for name in ("index", "middle", "ring", "little"):
            self._pair(f, f"{name.capitalize()} engage / release", fc.pinch_thresholds[name])
        self._pair(f, "Thumb press engage / release", fc.thumb_press)
        b.number(f, "Min extension of a pinched finger", fc, "pinch_min_extension", 0, 0.6, 0.05, 2,
                 section="finger")
        b.number(f, "... or min fingertip reach (deep pinches)", fc, "pinch_min_reach", 0.2, 0.8, 0.01, 2,
                 section="finger", tip="Fingertip-to-knuckle distance; lower accepts deeper pinches")
        b.check(f, "Show a marker at the pending click location (and keyboard-gesture progress)",
                ctx.s.ui, "feedback_marker", "ui")
        g, f = group("Accessibility", lay)
        b.check(f, "Dwell-to-click (hold the cursor still to left-click)", gc, "dwell_click", "gesture")
        b.number(f, "Dwell-click time", gc, "dwell_click_s", 0.4, 3.0, 0.1, 1, "s", "gesture")
        mag = QPushButton("Toggle Windows Magnifier (Win + Plus / Win + Esc)")
        mag.clicked.connect(lambda: ctx.runtime.toggle_magnifier())
        f.addRow(mag)
        lay.addStretch(1)

    def _pair(self, form: QFormLayout, text: str, values: list[float]) -> None:
        from PySide6.QtWidgets import QDoubleSpinBox
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        spins = []
        for i in range(2):
            sp = QDoubleSpinBox()
            sp.setRange(0.05, 1.2)
            sp.setSingleStep(0.01)
            sp.setDecimals(2)
            sp.setValue(values[i])

            def store(v, i=i, sp=sp) -> None:
                values[i] = float(v)
                if values[1] < values[0] + 0.04:      # keep hysteresis
                    values[1] = values[0] + 0.04
                    spins[1].blockSignals(True)
                    spins[1].setValue(values[1])
                    spins[1].blockSignals(False)
                self.ctx.changed("finger")
            sp.valueChanged.connect(store)
            spins.append(sp)
            h.addWidget(sp)
        h.addStretch(1)

        def load() -> None:
            for i, sp in enumerate(spins):
                sp.blockSignals(True)
                sp.setValue(values[i])
                sp.blockSignals(False)
        self.binder._refreshers.append(load)
        form.addRow(text, row)


# =============================================================== Diagnostics
class DiagnosticsPage(BasePage):
    def __init__(self, ctx: "AppContext") -> None:
        super().__init__(ctx)
        self.widget, lay = page("Diagnostics", "Live measurements from the running pipeline. Camera FPS, "
                                "inference FPS, cursor dispatch rate and latency are different things and are "
                                "reported separately.")
        grid = QGridLayout()
        self.lat = Sparkline("Processing latency (frame arrival → input decision)", "ms")
        self.inf = Sparkline("MediaPipe inference", "ms")
        self.cap = Sparkline("Capture FPS", "fps", y_max=40)
        self.trk = Sparkline("Tracking FPS", "fps", y_max=40)
        for i, w in enumerate((self.lat, self.inf, self.cap, self.trk)):
            grid.addWidget(w, i // 2, i % 2)
        lay.addLayout(grid)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(420)
        lay.addWidget(self.table)
        row = QHBoxLayout()
        b1 = QPushButton("Open log folder")
        b1.clicked.connect(lambda: os.startfile(str(app_dir() / "logs")))
        b2 = QPushButton("Copy diagnostics")
        b2.clicked.connect(self._copy)
        b3 = QPushButton("Export gesture log")
        b3.setToolTip("Saves gesture/click metadata only (timings, distances, buttons, positions) - "
                      "no camera frames, no typed text, no window contents")
        b3.clicked.connect(self._export)
        row.addWidget(b1)
        row.addWidget(b2)
        row.addWidget(b3)
        row.addStretch(1)
        lay.addLayout(row)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(160)
        lay.addWidget(label("Recent log (never contains typed text)", "H2"))
        lay.addWidget(self.log)
        self._gpu = "measuring…"
        self._gpu_t = 0.0
        self._log_t = 0.0
        self._rows: dict[str, int] = {}

    def _set(self, k: str, v: str) -> None:
        if k not in self._rows:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(k))
            self.table.setItem(r, 1, QTableWidgetItem(v))
            self._rows[k] = r
        else:
            it = self.table.item(self._rows[k], 1)
            if it.text() != v:
                it.setText(v)

    def _poll_gpu(self) -> None:
        smi = shutil.which("nvidia-smi")
        if not smi:
            self._gpu = "not measurable (nvidia-smi not found)"
            return
        try:
            out = subprocess.run([smi, "--query-gpu=name,utilization.gpu,memory.used",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
            name, util, mem = [x.strip() for x in out.splitlines()[0].split(",")]
            self._gpu = f"{util}% system-wide, {mem} MiB used ({name}); AirDesk inference runs on CPU"
        except Exception as e:
            self._gpu = f"unavailable ({type(e).__name__})"

    def on_tick(self, st, m) -> None:
        self.lat.push(m.latency_ms)
        self.inf.push(m.infer_ms)
        self.cap.push(m.capture_fps)
        self.trk.push(m.tracking_fps)
        now = time.monotonic()
        if now - self._gpu_t > 3:
            self._gpu_t = now
            threading.Thread(target=self._poll_gpu, daemon=True).start()
        rt = self.ctx.runtime
        d = m.dispatch
        rows = {
            "Camera": f"{m.camera_status} · {m.camera_mode} · driver fps {rt.camera.negotiated[2]:.0f}",
            "Camera error": m.camera_error or "none",
            "Capture FPS": f"{m.capture_fps:.1f}",
            "Tracking (inference) FPS": f"{m.tracking_fps:.1f}",
            "Inference median / p95": f"{m.infer_ms:.2f} / {m.infer_p95:.2f} ms",
            "Processing latency median / p95": f"{m.latency_ms:.2f} / {m.latency_p95:.2f} ms "
                                               "(excludes camera exposure/USB transfer and display)",
            "Inference input": f"{self.ctx.s.camera.inference_width}px wide, delegate {m.delegate.upper()}"
                               + (f" — {m.delegate_note}" if m.delegate_note else ""),
            "AirDesk CPU": f"{m.cpu_percent:.1f}% of {os.cpu_count()} logical CPUs",
            "GPU": self._gpu,
            "Engine state": st.state,
            "Hands visible": str(st.hands_seen),
            "Cursor dispatch": f"{rt.dispatcher.hz} Hz target · {d.get('ticks', 0)} ticks · "
                               f"{d.get('moves', 0)} moves · {d.get('buttons', 0)} button events · "
                               f"{d.get('wheel', 0)} wheel events",
            "Injection enabled": str(rt.dispatcher.enabled),
            "Held buttons": ", ".join(st.buttons) or "none",
            "Click pipeline": self._click_counters(rt),
            "Counters": ", ".join(f"{k} {v}" for k, v in m.counters.items()),
            "Pinch distances": ", ".join(f"{k} {v:.2f}" for k, v in st.pinch.items()) or "-",
            "Click mode / style": f"{st.click_mode} · {STYLE_NAMES.get(st.click_style, st.click_style)} · "
                                  f"pointer {st.finger}",
            "Nearest contact": _contact_text(st.contact) if st.tracking else "-",
            "Aim Lock": (f"holding: {st.aim_reason} · {st.candidate_ms:.0f} ms" if st.aim_lock else "off"),
            "Press / rearm": (f"{st.state} · {st.candidate_ms:.0f} ms · {st.rearm}"
                              if st.state == "PRESS_PENDING" else st.rearm),
            "Camera hint": st.hint or "none",
            "Recognized pose": f"{st.hand} hand · {st.pose}" if st.tracking else "-",
            "Pinky Drag": (f"{st.drag or 'idle'}" + (f" · {st.candidate_ms:.0f} ms" if st.drag == "ready" else "")
                           + f" · buttons held: {', '.join(st.buttons) or 'none'}"),
            "Air Keyboard hands": _hands_text(st, self.ctx.overlay) if st.dual else "dual-hand mode off",
            "Keyboard dispatch": _dispatch_text(self.ctx.overlay),
            "Keyboard LEFT hand": _hand_row(next((v for v in st.vhands if v["side"] == "left"), None))
                                  if st.dual else "-",
            "Keyboard RIGHT hand": _hand_row(next((v for v in st.vhands if v["side"] == "right"), None))
                                   if st.dual else "-",
            "Keyboard hand identity": _identity_text(st) if st.dual else "-",
            "Keyboard actions / rejections": _rejects_text(st) if st.dual else "-",
            "Keyboard event queue": _queue_text(self.ctx.overlay),
            "Keyboard hand events (newest last)": _events_text(st) if st.dual else "-",
            "Monitors": " | ".join(f"{r.width}×{r.height}@({r.left},{r.top})"
                                   for r in (rt.screen.monitors if rt.screen else [])),
            "AirDesk elevated": str(win32.self_elevated()),
            "Hotkeys": "OK" if not m.hotkey_errors else "; ".join(m.hotkey_errors),
        }
        for i, it in enumerate(reversed(rt.engine.clicks.recent(6))):
            rows[f"Interaction {i + 1} (newest first)"] = _describe(it)
        for i, ev in enumerate(reversed(rt.engine.clicks.events(4))):
            rows[f"Gesture event {i + 1} (newest first)"] = _describe_event(ev)
        for k, v in rows.items():
            self._set(k, v)
        if now - self._log_t > 2:
            self._log_t = now
            try:
                lines = (app_dir() / "logs" / "airdesk.log").read_text(encoding="utf-8").splitlines()[-120:]
                txt = "\n".join(lines)
                if txt != self.log.toPlainText():
                    self.log.setPlainText(txt)
                    self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
            except OSError:
                pass

    @staticmethod
    def _click_counters(rt) -> str:
        c = rt.engine.clicks.snapshot_counters()
        d = rt.dispatcher.stats
        return (f"{c['accepted']} presses accepted · {c['down_submitted']} button-downs and "
                f"{c['up_submitted']} button-ups submitted · {c['completed']} completed · "
                f"{c['cancelled']} cancelled · {c['rejected']} rejected · {c['drags']} drags · "
                f"SendInput max {d.get('submit_ms_max', 0):.2f} ms")

    def _export(self) -> None:
        try:
            path = export_gesture_log(self.ctx.runtime.engine, self.ctx.s, self.ctx.version)
        except OSError as e:
            QMessageBox.warning(self.widget, "AirDesk", f"Could not save the gesture log: {e}")
            return
        QMessageBox.information(self.widget, "AirDesk", f"Gesture log saved (metadata only):\n{path}")

    def _copy(self) -> None:
        from PySide6.QtWidgets import QApplication
        rows = [f"{self.table.item(r, 0).text()}: {self.table.item(r, 1).text()}"
                for r in range(self.table.rowCount())]
        QApplication.clipboard().setText("\n".join(rows))


# ================================================================== Settings
class SettingsPage(BasePage):
    def __init__(self, ctx: "AppContext") -> None:
        super().__init__(ctx)
        self.widget, lay = page("Settings", f"Stored in {app_dir()}. No webcam images or typed text are "
                                "ever saved.")
        b, cc, uc, sc = self.binder, ctx.s.camera, ctx.s.ui, ctx.s.safety
        g, f = group("Camera", lay)
        self.cam_combo = QComboBox()
        self.cam_combo.addItem(f"Camera {cc.index}", cc.index)
        self.cam_combo.currentIndexChanged.connect(self._pick_camera)
        scan = QPushButton("Scan cameras")
        scan.clicked.connect(self._scan)
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.cam_combo, 1)
        h.addWidget(scan)
        f.addRow("Device", row)
        b.combo(f, "Backend", cc, "backend", [("msmf", "Media Foundation (recommended, measured 30 FPS)"),
                                              ("dshow", "DirectShow"), ("any", "OpenCV default")], "camera")
        self.mode = QComboBox()
        for w, hh in MODES:
            self.mode.addItem(f"{w} × {hh}", (w, hh))
        self.mode.setCurrentIndex(max(self.mode.findData((cc.width, cc.height)), 0))
        self.mode.currentIndexChanged.connect(self._pick_mode)
        f.addRow("Capture resolution", self.mode)
        b.number(f, "Capture FPS", cc, "fps", 15, 60, 5, 0, "fps", "camera", slider=False)
        b.combo(f, "Inference width", cc, "inference_width",
                [(320, "320 px (fastest)"), (480, "480 px"), (640, "640 px (recommended)"),
                 (960, "960 px (far from camera)")], "camera")
        b.combo(f, "MediaPipe delegate", cc, "delegate", [("cpu", "CPU (XNNPACK)"),
                                                          ("gpu", "GPU if supported (falls back to CPU)")],
                "camera")
        b.check(f, "Mirror camera (selfie view)", cc, "mirror", "camera")
        b.combo(f, "Hands to track", cc, "num_hands", [(1, "One"), (2, "Two (needed for zoom, two-hand "
                                                                  "modes and the dual-hand keyboard)")],
                "camera")
        b.number(f, "Detection confidence", cc, "min_detection_confidence", 0.2, 0.95, 0.05, 2, section="camera")
        b.number(f, "Tracking confidence", cc, "min_tracking_confidence", 0.2, 0.95, 0.05, 2, section="camera")
        g, f = group("Safety and hotkeys", lay)
        self.hk: dict[str, QLineEdit] = {}
        for name, text in (("emergency_hotkey", "Emergency stop"), ("arm_hotkey", "Arm / disarm"),
                           ("keyboard_hotkey", "Show/hide keyboard"), ("precision_hotkey", "Precision mode"),
                           ("drag_hotkey", "Legacy Drag Mode on/off")):
            e = QLineEdit(getattr(sc, name))
            e.editingFinished.connect(lambda n=name, e=e: self._hotkey(n, e))
            self.hk[name] = e
            f.addRow(text, e)
        f.addRow(label("Input always starts DISARMED. The emergency hotkey works even when hand tracking "
                       "fails, and pressing it releases every mouse button and latched key.", "Muted", True))
        self.auto = QPushButton()
        self.auto.clicked.connect(self._toggle_autostart)
        f.addRow("Start with Windows", self.auto)
        g, f = group("Appearance and behaviour", lay)
        b.combo(f, "Theme", uc, "theme", [("dark", "Dark"), ("light", "Light")], "ui")
        b.check(f, "Show camera preview on Home", uc, "show_preview", "ui")
        b.check(f, "Closing the window keeps AirDesk running in the tray", uc, "minimize_to_tray", "ui")
        row = QHBoxLayout()
        reset = QPushButton("Reset all settings to defaults")
        reset.clicked.connect(self._reset)
        folder = QPushButton("Open data folder")
        folder.clicked.connect(lambda: os.startfile(str(app_dir())))
        row.addWidget(reset)
        row.addWidget(folder)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addWidget(label(f"AirDesk {ctx.version} · Python {sys.version.split()[0]} · "
                            "MediaPipe Tasks HandLandmarker · PySide6", "Muted"))
        lay.addStretch(1)
        self._sync_autostart()

    def _pick_camera(self) -> None:
        idx = self.cam_combo.currentData()
        if idx is not None and idx != self.ctx.s.camera.index:
            self.ctx.s.camera.index = int(idx)
            self.ctx.changed("camera")

    def _pick_mode(self) -> None:
        w, h = self.mode.currentData()
        self.ctx.s.camera.width, self.ctx.s.camera.height = w, h
        self.ctx.changed("camera")

    def _scan(self) -> None:
        from ..vision.camera import enumerate_cameras
        rt = self.ctx.runtime
        rt.stop_camera()                      # never two streams on one device
        try:
            cams = enumerate_cameras(self.ctx.s.camera.backend)
        finally:
            rt.start_camera()
        self.cam_combo.blockSignals(True)
        self.cam_combo.clear()
        for c in cams:
            self.cam_combo.addItem(f"{c.index}: {c.name} ({c.width}×{c.height})", c.index)
        if not cams:
            self.cam_combo.addItem("No camera found", None)
        i = self.cam_combo.findData(self.ctx.s.camera.index)
        self.cam_combo.setCurrentIndex(max(i, 0))
        self.cam_combo.blockSignals(False)

    def _hotkey(self, name: str, edit: QLineEdit) -> None:
        spec = edit.text().strip().lower()
        try:
            parse_hotkey(spec)
        except ValueError as e:
            QMessageBox.warning(self.widget, "Hotkey", f"Invalid hotkey: {e}")
            edit.setText(getattr(self.ctx.s.safety, name))
            return
        if spec != getattr(self.ctx.s.safety, name):
            setattr(self.ctx.s.safety, name, spec)
            self.ctx.changed("hotkeys")

    def _sync_autostart(self) -> None:
        on = autostart.is_enabled()
        self.auto.setText("Enabled — click to disable" if on else "Disabled — click to enable")

    def _toggle_autostart(self) -> None:
        if autostart.is_enabled():
            autostart.disable()
        else:
            ok = QMessageBox.question(
                self.widget, "Start with Windows",
                "Start AirDesk automatically when you sign in to Windows?\n\nIt will start DISARMED and "
                "minimized to the tray; gesture input still has to be armed by you.\n\n"
                "This adds an entry under HKEY_CURRENT_USER\\...\\Run (no administrator rights).")
            if ok == QMessageBox.Yes:
                if not autostart.enable():
                    QMessageBox.warning(self.widget, "Start with Windows",
                                        "Only the packaged AirDesk.exe can be registered for autostart.")
        self.ctx.s.safety.autostart = autostart.is_enabled()
        self.ctx.changed("safety")
        self._sync_autostart()

    def _reset(self) -> None:
        if QMessageBox.question(self.widget, "Reset", "Reset every setting to its default?") != QMessageBox.Yes:
            return
        self.ctx.reset_settings()

    def refresh(self) -> None:
        super().refresh()
        for name, e in self.hk.items():
            e.setText(getattr(self.ctx.s.safety, name))
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(max(self.mode.findData((self.ctx.s.camera.width, self.ctx.s.camera.height)), 0))
        self.mode.blockSignals(False)


def _contact_text(c: dict) -> str:
    if not c:
        return "no mapped contact near the thumb"
    t = (f"{c['pair']} ({c['slot']}) {c['d']:.3f} hand scales · engage {c['engage']:.2f} · "
         f"{c['ratio']:.2f}x threshold")
    return t + (f" · {c['note']}" if c.get("note") else "")


def _describe_event(ev: dict) -> str:
    parts = [f"{ev['id']} {ev['time']}", ev["text"]]
    if ev.get("pair"):
        parts.append(f"{ev['pair']} closest {ev.get('closest')}x threshold")
    if ev.get("age_ms") is not None:
        parts.append(f"after {ev['age_ms']} ms")
    if ev.get("note"):
        parts.append(ev["note"])
    return " · ".join(parts)


def export_gesture_log(engine, settings, version: str) -> str:
    """Owner acceptance / bug reports: gesture and click metadata only."""
    import json
    from dataclasses import asdict
    out_dir = app_dir() / "bench_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"gesture-log-{time.strftime('%Y%m%d-%H%M%S')}.json"
    data = {
        "airdesk": version,
        "exported": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "metadata only: no camera frames, typed text or window contents",
        "settings": {"gesture": {k: v for k, v in asdict(settings.gesture).items() if k != "bindings"},
                     "finger": asdict(settings.finger), "pointer_mode": settings.pointer.mode},
        "counters": engine.clicks.snapshot_counters(),
        "engine_counters": dict(engine.counters),
        "interactions": [asdict(i) for i in engine.clicks.recent(64)],
        "gesture_events": engine.clicks.events(64),
        # v1.1.4: dual-hand keyboard identity / lifecycle metadata (no key ids, no text)
        "dual_keyboard": _dual_export(engine),
    }
    path.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    return str(path)


def _dual_export(engine) -> dict:
    d = engine.dual
    hands = []
    for v in d.status():
        hands.append({k: v[k] for k in ("side", "track", "life", "detected", "label", "evidence", "cursor", "pinch",
                                        "keys", "rec_ms", "reacqs", "reacq_ms", "reacq_frames")})
    diag = d.diag()
    return {"active": engine._dual_on, "hands": hands, "polarity": diag["polarity"],
            "label_agree": diag["label_agree"], "label_samples": diag["label_samples"],
            "counters": diag["counters"], "events": list(d.events)}


def _describe(it) -> str:
    """One interaction-log record for Diagnostics (no typed content)."""
    rec = it.recognition_ms()
    down = it.confirm_to_down_ms()
    parts = [f"#{it.id} {it.clock} {it.button} {it.kind} at {it.pos[0]},{it.pos[1]}", it.phase]
    if it.mode or it.evidence:
        parts.append(f"{it.mode or '-'} · confirmed by {it.evidence or '-'}")
    if rec is not None:
        parts.append(f"recognised {rec:.0f} ms after contact")
    if down is not None:
        parts.append(f"down submitted {down:.2f} ms after acceptance" + ("" if it.down_ok else " (FAILED)"))
    if it.wall_up is not None and it.wall_down is not None:
        parts.append(f"held {1000 * (it.wall_up - it.wall_down):.0f} ms")
    if it.reason:
        parts.append(it.reason)
    return " · ".join(parts)
