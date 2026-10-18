"""Interaction engine: the explicit state machine between gestures and output.

Per camera frame:  hands -> active hand -> features -> active finger ->
gesture primitives -> state machine -> MouseSink / KeyboardSink.

States (exactly one active):
    DISARMED        no output of any kind
    PAUSED          armed but suspended (tray, hotkey or palm-hold)
    IDLE            armed, waiting for a hand
    POINTER         cursor follows the active finger (hover, keyboard hover).
                    Sub-modes, not separate states: *Smart Aim Lock* (a
                    thumb contact is closing: cursor anchored where the
                    closing began; released as soon as the contact stops
                    closing in without a pinch) and the *keyboard gesture*
                    (🤟 pose: cursor held, toggle after a 0.4 s hold)
    PRESS_PENDING   a contact confirmed at the anchored aim point, cursor
                    frozen. What happens depends on the click mode
                    (`click_mode`, independent of the click style):
                    *instant* (default) - one complete click (down + up,
                    one SendInput batch) right now; holding does nothing
                    more; a new click needs the contact to re-open.
                    *drag* - nothing is sent until hold_to_drag_s (1.2 s):
                    then button down = drag; an earlier release = nothing.
                    *classic* (v1.1.1) - button down now, up at release,
                    held >= hold_to_drag_s = drag with the same down.
                    Thumb pointer (long-press right click on): the button
                    waits for the release in every mode - < long_press_s =
                    left click, long_press_s..hold_to_drag_s = right click,
                    >= hold_to_drag_s = left drag
    DRAGGING        button held, cursor follows the rigid knuckle centroid.
                    Entered from Drag Mode / classic holds, or (v1.1.3, no mode
                    switch) from **Pinky Drag**: only the little finger up,
                    thumb tucked, held pinky_drag_s (0.2 s) - the button goes
                    down at the point aimed before the pose change; curling
                    the little finger (or extending another finger) drops
    KEYBOARD        Dual-Hand Air Keyboard open (v1.1.3): each hand drives its
                    own virtual hand cursor over the keyboard
                    (`dual_keyboard.py`); no desktop mouse input at all
    KEY_PRESS       contact engaged over the Air Keyboard (no mouse events)
    SCROLLING       scroll pose: joystick/direct wheel output, cursor frozen
    ZOOMING         both hands pinching: Ctrl+wheel from hand separation
    CLUTCH          fist: cursor frozen, hand can be repositioned
    TRACKING_LOST   hand gone longer than the grace period; buttons released

Invariants enforced here (and covered by tests):
  * a mouse button is pressed at most once per gesture and is always
    released on gesture end, tracking loss, disarm, pause or emergency stop;
  * at most one action fires per physical gesture;
  * the active finger never changes while a contact/scroll/drag is active;
  * no state change ever teleports the cursor (see pointer.py).
"""

from __future__ import annotations

import enum
import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Callable

from ..config import Settings
from .actions import BUTTON_ACTIONS, DRAG_ACTIONS, KeyboardSink, MouseSink, NullKeyboard
from .clicklog import ClickLog, Interaction
from .dual_keyboard import DualKeyboard
from .gestures import STYLE_NAMES, GestureFrame, GestureRecognizer, Hysteresis, slot_contacts
from .hand_model import EXTENDED_ON, analyze, ily_pose, pinky_forming, pinky_held, pinky_pose, pointer_candidate
from .pointer import PointerEngine
from .screen import ScreenLayout
from .selection import FingerSelector, HandSelector
from .types import (EngineStatus, Finger, FingerMode, FrameInput, HandFeatures,
                    HandPreference, PointerMode)

log = logging.getLogger(__name__)

REPEAT_KEYS = {"backspace", "delete", "left", "right", "up", "down", "space"}


class State(str, enum.Enum):
    DISARMED = "DISARMED"
    PAUSED = "PAUSED"
    IDLE = "IDLE"
    POINTER = "POINTER"
    PRESS_PENDING = "PRESS_PENDING"
    DRAGGING = "DRAGGING"
    KEY_PRESS = "KEY_PRESS"
    SCROLLING = "SCROLLING"
    ZOOMING = "ZOOMING"
    CLUTCH = "CLUTCH"
    TRACKING_LOST = "TRACKING_LOST"
    KEYBOARD = "KEYBOARD"


ENGAGED = {State.PRESS_PENDING, State.DRAGGING, State.KEY_PRESS, State.SCROLLING,
           State.ZOOMING, State.CLUTCH}


@dataclass(slots=True)
class _Press:
    slot: str
    action: str
    t0: float
    knuckle0: tuple[float, float]
    scale: float
    pos: tuple[float, float]
    key: str | None = None
    consumed: bool = False
    next_repeat: float = 0.0
    down: str | None = None        # mouse button this press holds down (sent once)
    deferred: bool = False         # button sent on release / drag start instead of at confirm
    it: Interaction | None = None  # interaction log record (diagnostics + feedback)
    clicked: bool = False          # instant click already completed (waiting for the release)
    drag_wait: bool = False        # Drag Mode: holding toward hold_to_drag_s, nothing sent yet
    follow: bool = False           # instant click held long / hand moved: cursor follows again
    misses: int = 0                # Pinky Drag: consecutive frames without the pose


@dataclass(slots=True)
class _Pinky:
    """Pinky Drag acquisition: the pose is held, nothing is pressed yet."""
    t0: float
    anchor: tuple[float, float]
    it: Interaction | None
    misses: int = 0


@dataclass(slots=True)
class _Aim:
    """Smart Aim Lock: the pending click location while a pinch forms."""
    anchor: tuple[float, float]
    t0: float
    knuckle0: tuple[float, float]
    scale: float
    ratio0: float                  # contact distance / engage when the lock armed
    best: float = math.inf         # closest approach so far
    best_t: float = 0.0            # when it was last improved
    slot: str | None = None        # contact that armed it


class InteractionEngine:
    def __init__(self, settings: Settings, screen: ScreenLayout, mouse: MouseSink,
                 keyboard: KeyboardSink | None = None,
                 on_command: Callable[[str], None] | None = None) -> None:
        self.s = settings
        self.screen = screen
        self.mouse = mouse
        self.keyboard: KeyboardSink = keyboard or NullKeyboard()
        self.on_command = on_command or (lambda name: None)

        self.hands = HandSelector()
        self.fingers = FingerSelector()
        self.pointer = PointerEngine(settings.pointer, screen)
        self.rec = GestureRecognizer(settings.gesture, settings.finger)
        self.rec_other = GestureRecognizer(settings.gesture, settings.finger)
        self._zoom_a = Hysteresis(0.26, 0.38)
        self._zoom_b = Hysteresis(0.26, 0.38)
        self.configure(settings)

        self.state = State.DISARMED
        self.armed = False
        self.paused = False
        self._finger: Finger | None = None
        self._press: _Press | None = None
        self._lost_since: float | None = None
        self._armed_at = -1e9
        self._settle_until = 0.0
        self._yield_until = -1e9
        self._last_click: tuple[float, tuple[float, float]] | None = None
        self._last_down: tuple[float, tuple[float, float]] | None = None
        self._press_block_until = 0.0      # after scroll/zoom exits only (not after clicks)
        self._last_hold_t = -1e9           # last frame the cursor was held (press, settle, lock)
        self.clicks = ClickLog()
        self._scroll_origin: tuple[float, float] | None = None
        self._scroll_prev: tuple[float, float] = (0.0, 0.0)
        self._scroll_axis: str | None = None
        self._scroll_t = 0.0
        self._zoom_ref = 0.0
        self._hover_key: str | None = None
        self._hover_t = 0.0
        self._hover_pos: tuple[float, float] = (0.0, 0.0)
        self._dwell_fired = False
        self._dwell_click_pos: tuple[float, float] | None = None
        self._dwell_click_t = 0.0
        self._palm_hold_t: float | None = None
        self._palm_hold_fired = False
        self._prev_mode: PointerMode | None = None
        self._other_ptr: tuple[float, float] | None = None
        self._other_key: str | None = None
        self._aim: _Aim | None = None
        self._aim_latched = False          # lock cancelled: wait for the contact to re-open
        self._aim_latch_ratio = math.inf
        self._hint = ""
        self._hint_cand = ""
        self._hint_t = 0.0
        self._gf: GestureFrame | None = None
        self._other_feats: HandFeatures | None = None
        self._last_instant: tuple[float, str] | None = None
        self._knuckles: deque[tuple[float, float, float]] = deque(maxlen=30)
        self._kb_t: float | None = None    # 🤟 keyboard gesture hold start
        self._kb_fired = False
        self._kb_cooldown = -1e9
        self._kb_progress = 0.0
        self._pinky: _Pinky | None = None  # Pinky Drag candidate (0.2 s acquisition)
        self._pinky_block = False          # after a drop / cancel: the pose must be released first
        self._point_ok_t = -1e9            # last frame the active finger was steadily pointing
        self._tip_rel: tuple[float, float] | None = None
        self.drag_state = ""               # ready | dragging | released | cancelled (diagnostics)
        self._drag_state_t = 0.0
        self._drag_progress = 0.0
        self._pose = "-"
        self.dual = DualKeyboard(self)
        self._dual_on = False
        self.hands_seen = 0
        self.gesture_label = "-"
        self._hand_label = "-"
        self.message = ""
        self.counters = {"clicks": 0, "drags": 0, "scrolls": 0, "keys": 0,
                         "finger_switches": 0, "tracking_losses": 0}

    # ================================================================ config
    def configure(self, settings: Settings, screen: ScreenLayout | None = None) -> None:
        self.s = settings
        if screen is not None:
            self.screen = screen
        f = settings.finger
        self.hands.preference = HandPreference(f.hand)
        self.fingers.mode = FingerMode(f.mode)
        self.fingers.manual_finger = Finger.from_name(f.finger)
        self.fingers.switch_delay_s = f.switch_delay_ms / 1000.0
        self.fingers.switch_confidence = f.switch_confidence
        if self.fingers.mode is FingerMode.MANUAL:
            self.fingers.current = self.fingers.manual_finger
        self.pointer.configure(settings.pointer, self.screen)
        style_changed = getattr(self, "_style", None) not in (None, settings.gesture.click_style)
        mode_changed = getattr(self, "_mode_seen", None) not in (None, settings.gesture.click_mode)
        self._style = settings.gesture.click_style
        self._mode_seen = settings.gesture.click_mode
        self.rec.configure(settings.gesture, settings.finger)
        self.rec_other.configure(settings.gesture, settings.finger)
        if style_changed or mode_changed:
            self._on_style_change("click style changed" if style_changed else "click mode changed")

    @property
    def click_mode(self) -> str:
        m = self.s.gesture.click_mode
        return m if m in ("instant", "drag", "classic") else "instant"

    def set_click_mode(self, mode: str) -> None:
        """Instant Auto-Click / Drag Mode / classic. Pending presses are
        cancelled safely (nothing is clicked by the switch)."""
        if mode not in ("instant", "drag", "classic") or mode == self.s.gesture.click_mode:
            return
        self.s.gesture.click_mode = mode
        self._mode_seen = mode
        self._on_style_change("click mode changed")
        log.info("click mode: %s", mode)
        self.on_command("click_mode")

    def toggle_drag_mode(self) -> None:
        self.set_click_mode("instant" if self.click_mode == "drag" else "drag")

    def _on_style_change(self, reason: str = "click style changed") -> None:
        """Normal <-> Pinch Free, or a click-mode change: drop unresolved
        presses without emitting a click, release anything held, and start
        the new mapping clean."""
        self.rec.cancel_press()
        self.rec_other.cancel_press()
        if self._press is not None or self.state in (State.PRESS_PENDING, State.DRAGGING, State.KEY_PRESS):
            self._end_interaction(reason)
            self.mouse.release_all()
            self._press = None
            if self.state in (State.PRESS_PENDING, State.DRAGGING, State.KEY_PRESS):
                self.state = State.POINTER
        self._cancel_aim(latch=False)
        self._cancel_pinky(reason)
        if self.pointer.holding:
            self.pointer.release_hold()
        log.info("click style: %s", STYLE_NAMES.get(self._style, self._style))

    def set_screen(self, screen: ScreenLayout) -> None:
        self.screen = screen
        self.pointer.configure(self.s.pointer, screen)

    # ================================================================ control
    def set_armed(self, armed: bool, t: float) -> None:
        if armed == self.armed:
            return
        self.armed = armed
        if armed:
            self._armed_at = t
            self.paused = False
            pos = self.mouse.cursor_pos()
            if pos is not None:
                self.pointer.sync_cursor(*pos)
            self.state = State.IDLE
            log.info("ARMED")
        else:
            self._safe_release("input disarmed")
            self._exit_dual(t, "input disarmed")
            self.state = State.DISARMED
            log.info("DISARMED")
        self.on_command("armed" if armed else "disarmed")

    def set_paused(self, paused: bool) -> None:
        if paused == self.paused:
            return
        self.paused = paused
        if paused:
            self._safe_release("paused")
            self._exit_dual(0.0, "paused")
            if self.armed:
                self.state = State.PAUSED
        elif self.armed:
            self.state = State.IDLE
            pos = self.mouse.cursor_pos()
            if pos is not None:
                self.pointer.sync_cursor(*pos)
        self.on_command("paused" if paused else "resumed")

    def emergency_stop(self) -> None:
        """Hotkey/tray: release everything and disarm immediately."""
        self._safe_release("emergency stop")
        self._exit_dual(0.0, "emergency stop")
        self.keyboard.release_all()
        self.armed = False
        self.paused = False
        self.state = State.DISARMED
        self.on_command("disarmed")
        log.warning("EMERGENCY STOP")

    def toggle_precision(self) -> None:
        if self.pointer.mode is PointerMode.PRECISION:
            self.pointer.set_mode(self._prev_mode or PointerMode.DIRECT)
        else:
            self._prev_mode = self.pointer.mode
            self.pointer.set_mode(PointerMode.PRECISION)
        self.on_command("pointer_mode")

    def set_finger(self, finger: Finger) -> None:
        self.fingers.manual_finger = finger
        self.s.finger.finger = finger.name.lower()
        self.fingers.force(finger)

    def notify_physical_mouse(self, x: float, y: float, t: float) -> None:
        """The physical mouse moved: it always wins for a moment."""
        self._yield_until = t + self.s.pointer.yield_to_physical_s
        self._cancel_pinky("physical mouse moved", t)
        if self.state in (State.DRAGGING, State.PRESS_PENDING, State.SCROLLING, State.ZOOMING):
            self._safe_release("physical mouse moved")
            self.state = State.POINTER
        self.pointer.sync_cursor(x, y)

    # ================================================================ frame
    def process(self, frame: FrameInput) -> EngineStatus:
        t = frame.t
        w, h = frame.frame_size
        aspect = (w / h) if w and h else 16 / 9
        self.rec.aspect = self.rec_other.aspect = aspect
        if self._dual_check(t):
            return self._process_dual(frame, t, aspect)
        control, other = self.hands.select(frame.hands, t)
        self.hands_seen = len(frame.hands)
        if (control is not None and not self.hands.continued
                and (self._pinky is not None or self._pinky_dragging())):
            # Ownership of the controlling hand changed: never hand a held
            # button (or a pending Pinky Drag) over to another hand.
            self._safe_release("controlling hand changed (button released)")
            self.drag_state, self._drag_state_t = "cancelled", t

        if control is None:
            self._on_lost(t)
            return self._status(t, None, len(frame.hands))

        self._hand_label = control.handedness
        dw = self.s.gesture.contact_depth_weight
        feats = analyze(control, aspect, dw)
        dragging = self.state is State.DRAGGING
        pinky_ok = self._pinky_available()
        self.fingers.reserve_pinky = pinky_ok
        finger = self.fingers.update(feats, t, blocked=self.state in ENGAGED)
        raw = self._source(feats, finger, dragging)
        if self._finger is not None and finger is not self._finger and not dragging:
            self.pointer.change_source(self._source(feats, self._finger, False), raw)
            self.counters["finger_switches"] += 1
        self._finger = finger

        if self._lost_since is not None:
            self._lost_since = None
            if self.state is State.TRACKING_LOST:
                self.state = State.POINTER

        self._knuckles.append((t, *feats.knuckle_center))
        if finger is not None:
            # Last moment the pointer finger was steadily pointing: extended,
            # and its tip not moving relative to the knuckles (a finger that
            # starts folding moves the cursor before its extension drops).
            sx, sy = self._source(feats, finger, False)
            sc = max(feats.hand_scale, 1e-3)
            rel = ((sx - feats.knuckle_center[0]) * aspect / sc, (sy - feats.knuckle_center[1]) / sc)
            steady = self._tip_rel is not None and math.dist(rel, self._tip_rel) < 0.06
            self._tip_rel = rel
            if steady and feats.extension[finger] >= EXTENDED_ON:
                self._point_ok_t = t
        action_hand = self.s.finger.two_hand_mode == "action_hand"
        # Pinky Drag pose (its acquisition, or a pinky drag): no new thumb
        # contact may start and the curled fingers are no Clutch fist - for
        # this hand only, and only while that pose is actually present.
        pinky_now = pinky_ok and (pinky_pose(feats) or (self._pinky is not None and pinky_forming(feats))
                                  or (self._pinky_dragging() and pinky_held(feats)))
        gf = self.rec.update(feats, t, finger, dragging=dragging,
                             allow_fist=finger is not Finger.THUMB and not pinky_now, slots=not action_hand,
                             block_new=self.fingers.switching or pinky_now, confidence=control.score)
        other_feats = analyze(other, aspect, dw) if other is not None else None
        self._other_feats = other_feats
        if action_hand:
            if other_feats is not None:
                agf = self.rec_other.update(other_feats, t, None, style="pinch_free",
                                            dragging=dragging, allow_fist=False)
                gf.slot, gf.slot_depth = agf.slot, agf.slot_depth
            else:
                self.rec_other.reset()
                gf.slot = None

        self._gf = gf
        self._pose = _pose_name(feats, gf, pinky_ok)
        self._update_hint(control, feats, t)
        self._update_palm_hold(gf, t)

        if not self.armed or self.paused:
            self.pointer.update(raw, t)  # keep filters warm for a seamless arm
            # Show what is recognised (no action is taken while disarmed).
            self.gesture_label = (f"Pinch ({gf.slot})" if gf.slot else "🤟 Keyboard pose" if gf.ily
                                  else "Scroll pose" if gf.scroll_pose else "Fist" if gf.fist
                                  else "Pinch forming" if gf.closing
                                  else "Open palm" if gf.open_palm else "Point")
            return self._status(t, feats, len(frame.hands), gf)
        if t - self._armed_at < 0.3 or t < self._yield_until:
            self.pointer.update(raw, t)
            if t < self._yield_until:
                self.message = "Physical mouse in use"
            return self._status(t, feats, len(frame.hands), gf)
        if self.state in (State.IDLE, State.DISARMED, State.PAUSED):
            self.state = State.POINTER

        handler = {
            State.POINTER: self._in_pointer,
            State.PRESS_PENDING: self._in_press,
            State.DRAGGING: self._in_drag,
            State.KEY_PRESS: self._in_key_press,
            State.SCROLLING: self._in_scroll,
            State.ZOOMING: self._in_zoom,
            State.CLUTCH: self._in_clutch,
        }.get(self.state)
        if handler is not None:
            handler(feats, gf, raw, t, other_feats)
        if self.pointer.holding:
            self._last_hold_t = t
        if (self.s.keyboard.two_hand_typing and not action_hand
                and other_feats is not None and self.keyboard.is_visible()):
            self._other_hand_typing(other_feats, t)
        elif self._other_key is not None:
            self._other_key = None
            self.keyboard.hover(1, None)
        return self._status(t, feats, len(frame.hands), gf)

    # ================================================================ states
    def _in_pointer(self, feats: HandFeatures, gf: GestureFrame, raw, t: float, other) -> None:
        b = self.s.gesture.bindings
        if self._keyboard_gesture(gf, raw, t):
            return
        if self._pinky_update(feats, gf, raw, t):
            return
        if gf.scroll_pose and b.get("two_finger_pose") == "scroll":
            self._enter_scroll(feats, t)
            return
        if other is not None and b.get("two_hand_pinch") == "zoom" and self._zoom_check(feats, other):
            self._enter_zoom(feats, other)
            return
        if gf.fist and b.get("fist") == "clutch":
            self._cancel_aim(latch=False)
            self.pointer.hold()
            self.state = State.CLUTCH
            self.gesture_label = "Clutch"
            return
        finger = self._finger
        curling = (finger is not None and finger is not Finger.THUMB
                   and feats.extension[finger] < self.s.finger.curl_freeze)
        if gf.slot is not None and t >= self._press_block_until:
            # A pinch while the pointer folds away *and another finger is
            # clearly taking over as the pointer* is a finger switch, not a
            # click (Auto finger mode only). The pointer merely bending in
            # sympathy with a middle-finger pinch (right click, Pinch Free
            # left click) never blocks the click - v1.1 blocked it whenever
            # the pointer's extension fell below curl_freeze. In Normal style
            # the pointer *is* the pinched finger and bends by design.
            own = slot_contacts(finger, self.s.gesture.click_style).get(gf.slot) == \
                f"pinch:{finger.name.lower()}" if finger is not None else False
            switching = (self.fingers.mode is FingerMode.AUTO
                         and pointer_candidate(feats) not in (None, finger))
            if not (curling and switching) or own:
                self._enter_press(feats, gf, t)
                return
        if gf.tap is not None and t >= self._press_block_until and not (curling and self.fingers.mode
                                                                         is FingerMode.AUTO
                                                                         and pointer_candidate(feats)
                                                                         not in (None, finger)):
            # Quick tap (contact already re-opened): press and release now,
            # at the anchored aim point - one complete click.
            self._enter_press(feats, gf, t, slot=gf.tap)
            p = self._press
            if p is not None:
                if self.state is State.PRESS_PENDING:
                    self._release_action(p, 0.0, t)
                self._end_press(t)
                if self.gesture_label.startswith("Click"):
                    self.gesture_label += " · tap"
            return
        if self._aim_update(feats, gf, raw, t):
            return
        if gf.swipe:
            self._run_action(b.get(f"swipe_{gf.swipe}", "none"), t)

        if finger is not None and feats.extension[finger] < self.s.finger.curl_freeze:
            # The active finger (thumb included) is folding away (pose change toward another
            # pointer finger, or tracking jitter): a curled finger is not
            # aiming, so hold the cursor instead of following it down.
            self.pointer.hold()
            self.pointer.update(raw, t)
            self.gesture_label = "Pose change"
            return
        if self.pointer.holding and t >= self._settle_until:
            self.pointer.release_hold()
        out = self.pointer.update(raw, t)
        if out is None:
            return
        if out.moved:
            self.mouse.move(out.x, out.y)
        self.gesture_label = "Point"
        self._keyboard_hover((out.x, out.y), t)
        self._dwell_click((out.x, out.y), t)

    # ---------------------------------------------------------- pinky drag
    def _pinky_available(self) -> bool:
        """Pinky Drag needs the desktop pointer (not the dual-hand keyboard)
        and a pointer finger other than the little finger: with the little
        finger chosen as the pointer (Manual / Lock, or Auto after pointing
        with it, thumb out) the pinky-only pose is pointing, not dragging."""
        if not self.s.gesture.pinky_drag or self._dual_on:
            return False
        f = self.fingers
        if f.mode is FingerMode.MANUAL and f.manual_finger is Finger.LITTLE:
            return False
        if f.mode is FingerMode.LOCK and (f.current or f.manual_finger) is Finger.LITTLE:
            return False
        return self._finger is not Finger.LITTLE

    def _pinky_dragging(self) -> bool:
        return self.state is State.DRAGGING and self._press is not None and self._press.slot == "pinky"

    def _pinky_update(self, feats: HandFeatures, gf: GestureFrame, raw, t: float) -> bool:
        """Pinky Drag acquisition (POINTER state). Returns True while it owns
        the frame. Runs beside the click pipeline, never in front of it: a
        pinch never waits for it, and it only starts on the exclusive
        pinky-only pose (the thumb tucked, so no thumb contact is forming)."""
        g = self.s.gesture
        c = self._pinky
        if c is None:
            if self._pinky_block:
                if not pinky_forming(feats):
                    self._pinky_block = False       # pose released: a new Pinky Drag may start
                return False
            if not self._pinky_available() or self._press is not None or not pinky_pose(feats):
                return False
            anchor = self._pinky_anchor(t)
            if anchor is None:
                return False
            if self.keyboard.is_visible() and self.keyboard.hit_test(*anchor) is not None:
                return False                        # never a desktop drag on the Air Keyboard
            self._cancel_aim(latch=False)
            it = self.clicks.begin("left", "drag", anchor, t, "pinky")
            it.evidence, it.mode = "pinky", "pinky"
            c = self._pinky = _Pinky(t, anchor, it)
            self.pointer.hold(anchor, firm=True)    # anchored while the pose is acquired
            self.mouse.move(*anchor)
            self.drag_state, self._drag_state_t = "ready", t
        self.pointer.update(raw, t)                 # filter stays warm; output stays on the anchor
        if pinky_forming(feats):
            c.misses = 0
        else:
            c.misses += 1
            if c.misses >= 2:
                self._cancel_pinky(f"pinky drag cancelled: pose released before {g.pinky_drag_s:.1f} s", t)
                return False
        held = t - c.t0
        self._drag_progress = min(held / max(g.pinky_drag_s, 0.05), 1.0)
        if held >= g.pinky_drag_s - 1e-6 and c.misses == 0:
            self._start_pinky_drag(feats, t)
        else:
            self.gesture_label = f"Drag Ready {int(self._drag_progress * 100)}%"
        return True

    def _pinky_anchor(self, t: float) -> tuple[float, float] | None:
        """Where the drag begins: the cursor at the last moment the pointer
        finger was really pointing - before it bent into the palm for the
        pinky-only pose (the bend itself must not move the target)."""
        if t - self._point_ok_t <= 1.0:
            pos = self.pointer.position_at(self._point_ok_t)
            if pos is not None:
                return pos
        return self.pointer.cursor

    def _start_pinky_drag(self, feats: HandFeatures, t: float) -> None:
        c = self._pinky
        assert c is not None
        self._pinky = None
        p = self._press = _Press("pinky", "left_click_drag", t, feats.knuckle_center, feats.hand_scale,
                                 c.anchor, it=c.it)
        self.mouse.move(*c.anchor)                  # move, then down: Windows ordering
        if not self._send_button(c.it, "left", True):
            p.consumed = True
            self._press = None
            self._pinky_block = True
            self.drag_state, self._drag_state_t = "cancelled", t
            self.gesture_label = f"Drag rejected ({c.it.reason if c.it else 'input refused'})"
            return
        p.down = "left"
        self._last_down = (t, c.anchor)
        self.clicks.dragging(c.it)
        # The drag follows the rigid knuckle centroid (unchanged by the curled
        # fingers), re-based so the held cursor continues without a jump.
        assert self._finger is not None
        self.pointer.change_source(self._source(feats, self._finger, False), feats.knuckle_center)
        self.pointer.release_hold()
        self.state = State.DRAGGING
        self.counters["drags"] += 1
        self.drag_state, self._drag_state_t = "dragging", t
        self.gesture_label = "Dragging (pinky)"

    def _cancel_pinky(self, reason: str, t: float | None = None) -> None:
        c = self._pinky
        if c is None:
            return
        self.clicks.end(c.it, "cancelled", reason)
        self._pinky = None
        self._pinky_block = True                    # released below once the pose is gone
        self._drag_progress = 0.0
        self.drag_state, self._drag_state_t = "cancelled", t if t is not None else self._drag_state_t
        if t is not None:
            self._settle_until = t + self.s.pointer.release_settle_ms / 1000.0
        self.gesture_label = "Drag cancelled"

    # ------------------------------------------------------- dual keyboard
    def _dual_check(self, t: float) -> bool:
        kc = self.s.keyboard
        # Dwell-only typing stays on the single desktop pointer (dwell needs
        # one steady cursor; pinch typing is what the dual-hand mode adds).
        want = (self.armed and not self.paused and kc.dual_hand and kc.selection != "dwell"
                and self.keyboard.is_visible())
        if want and not self._dual_on:
            self._enter_dual(t)
        elif not want and self._dual_on:
            self._exit_dual(t, "Air Keyboard closed")
        return self._dual_on

    def _enter_dual(self, t: float) -> None:
        """Keyboard OFF -> ON: release anything held, freeze the desktop
        cursor where it is (no click, no jump) and give both hands their
        own keyboard cursors."""
        self._safe_release("Air Keyboard opened")
        self._cancel_pinky("Air Keyboard opened", t)
        self.pointer.hold(firm=True)
        self.rec.require_open()
        if self._hover_key is not None:
            self._hover_key = None
            self.keyboard.hover(0, None)
        if self._other_key is not None:
            self._other_key = None
            self.keyboard.hover(1, None)
        self._dual_on = True
        self.state = State.KEYBOARD
        self.dual.start(t)
        log.info("dual-hand Air Keyboard on")

    def _exit_dual(self, t: float, reason: str) -> None:
        """Keyboard ON -> OFF: cursors gone, modifiers cleared, pending keys
        cancelled; the desktop pointer resumes from where it was frozen
        (continuity offset, no jump) and a pinch or 🤟 still held must be
        released before it clicks or toggles."""
        if not self._dual_on:
            return
        self._dual_on = False
        self.dual.stop(reason)
        self.rec.require_open()
        g = self.s.gesture
        self._press_block_until = max(self._press_block_until, t + 0.25)
        self._settle_until = t + self.s.pointer.release_settle_ms / 1000.0
        self._kb_fired = True
        self._kb_t, self._kb_progress = None, 0.0
        self._kb_cooldown = max(self._kb_cooldown, t + g.keyboard_gesture_cooldown_s)
        self._pinky_block = True
        if self.pointer.holding:
            self.pointer.release_hold()
        if self.state is State.KEYBOARD:
            self.state = (State.POINTER if self.armed and not self.paused
                          else State.PAUSED if self.armed else State.DISARMED)
        log.info("dual-hand Air Keyboard off (%s)", reason)

    def _process_dual(self, frame: FrameInput, t: float, aspect: float) -> EngineStatus:
        control, _other = self.hands.select(frame.hands, t)
        self.hands_seen = len(frame.hands)
        dw = self.s.gesture.contact_depth_weight
        feats_all = [analyze(o, aspect, dw) for o in frame.hands]
        feats = None
        if control is not None:
            feats = next(f for o, f in zip(frame.hands, feats_all) if o is control)
            self._hand_label = control.handedness
            finger = self.fingers.update(feats, t, blocked=True)
            self._finger = finger
            # The desktop cursor stays frozen; its filter keeps tracking so
            # the hand-over back to the desktop pointer has no jump.
            self.pointer.update(self._source(feats, finger, False), t)
            self._lost_since = None
            self._pose = "🤟" if ily_pose(feats) else "Point"
        self.gesture_label = self.dual.update(frame.hands, feats_all, t, aspect)
        return self._status(t, feats, len(frame.hands), None)

    # --------------------------------------------------------- aim lock
    def _aim_update(self, feats: HandFeatures, gf: GestureFrame, raw, t: float) -> bool:
        """Smart Aim Lock. Returns True while the cursor is anchored.

        Arms on the *onset* of a closing thumb contact (before the pinch is
        confirmed) and anchors the cursor where it was when the closing
        began, so the finger flexion of the pinch never moves the click.
        It only preserves the aim - a confirmed pinch (handled before this)
        always wins. Released (cursor resumes without a jump) when the
        contact re-opens, the hand clearly moves to re-aim, the contact
        stops closing in for `aim_stall_s` without becoming a pinch, or
        `aim_max_hold_s` passes (v1.1.1 held it up to 0.6 s regardless).
        """
        g = self.s.gesture
        if self._aim_latched and (gf.approach > g.aim_release_ratio or not gf.closing and gf.opening
                                  or gf.approach < self._aim_latch_ratio - 0.25):
            self._aim_latched = False
        if (self._aim is None and g.aim_lock and gf.closing and not self._aim_latched
                and t >= self._press_block_until):
            self._start_aim(feats, gf, t)
        a = self._aim
        if a is None:
            return False
        if gf.approach < a.best - 0.06:          # still closing in (not the filter tail)
            a.best, a.best_t = gf.approach, t
        moved = self._hand_motion(feats, a.knuckle0, a.scale)
        if gf.opening or gf.approach > max(g.aim_release_ratio, a.ratio0 * 1.2):
            self._cancel_aim(latch=False, reason="fingers opened again", t=t)
            return False
        if moved > g.aim_cancel_move:
            self._cancel_aim(latch=True, reason="hand moved: re-aiming", t=t)
            return False
        if t - a.best_t > g.aim_stall_s or t - a.t0 > g.aim_max_hold_s:
            # The contact stopped closing in (or took too long) without a
            # confirmed pinch: follow the hand again right away.
            why = "contact stopped closing" if t - a.best_t > g.aim_stall_s else "no pinch in time"
            self._cancel_aim(latch=True, reason=f"{why} at {gf.approach:.2f}x the contact threshold", t=t)
            return False
        self.pointer.update(raw, t)           # keep the filter warm; output stays held
        self.gesture_label = "Aiming"
        self._keyboard_hover(a.anchor, t)
        return True

    def _start_aim(self, feats: HandFeatures, gf: GestureFrame, t: float) -> None:
        g = self.s.gesture
        # Never anchor before the cursor was last free: rewinding into a
        # previous press or its settle hold would put this click on the
        # previous click's pixel, not where the visible cursor was aimed.
        t_anchor = max(gf.closing_onset, t - g.aim_max_rewind_ms / 1000.0, self._last_hold_t)
        # Never anchor at a moment when the whole hand was elsewhere: a fast
        # aiming movement that ends in a pinch must click where it ended.
        kx, ky = feats.knuckle_center
        scale = max(feats.hand_scale, 1e-3)
        newer = t
        for ht, x, y in reversed(self._knuckles):
            if ht < t_anchor:
                break
            if math.hypot((x - kx) * self.rec.aspect, y - ky) / scale > g.aim_hand_guard:
                t_anchor = newer        # first moment the hand was already here
                break
            newer = ht
        pos = self.pointer.position_at(t_anchor) or self.pointer.cursor
        if pos is None:
            return
        self._aim = _Aim(pos, t, feats.knuckle_center, feats.hand_scale, gf.approach,
                         gf.approach, t, gf.near_slot)
        self.pointer.hold(pos)
        self.mouse.move(*pos)

    def _cancel_aim(self, latch: bool, reason: str = "", t: float | None = None) -> None:
        if self._aim is None:
            return
        a = self._aim
        if reason:
            gf = self._gf
            pair = _pair_name(self._finger, self.s.gesture.click_style, a.slot)
            self.clicks.note("aim_released", f"Aim Lock released without a click: {reason}",
                             pair=pair, closest=round(a.best, 2),
                             d=round(gf.near_d, 3) if gf and math.isfinite(gf.near_d) else None,
                             engage=round(gf.near_engage, 3) if gf else None,
                             age_ms=round(((t or a.t0) - a.t0) * 1000), note=gf.note if gf else "")
        self._aim = None
        self._aim_latched = latch
        self._aim_latch_ratio = self._gf.approach if (latch and self._gf) else math.inf
        if self.state is State.POINTER and self.pointer.holding:
            self.pointer.release_hold()

    def _hand_motion(self, feats: HandFeatures, knuckle0: tuple[float, float], scale: float) -> float:
        dx = (feats.knuckle_center[0] - knuckle0[0]) * self.rec.aspect
        dy = feats.knuckle_center[1] - knuckle0[1]
        return math.hypot(dx, dy) / max(scale, 1e-3)

    # ---------------------------------------------------- keyboard gesture
    def _keyboard_gesture(self, gf: GestureFrame, raw, t: float) -> bool:
        """The 🤟 pose held for keyboard_gesture_s toggles the Air Keyboard
        once; the pose must be released before it can fire again. Only
        reachable from POINTER (never while pressing or dragging)."""
        g = self.s.gesture
        if not g.keyboard_gesture:
            return False
        if not (gf.ily or gf.ily_raw):
            self._kb_t, self._kb_fired, self._kb_progress = None, False, 0.0
            return False
        self._cancel_aim(latch=False)
        self.pointer.hold()
        self.pointer.update(raw, t)
        if self._kb_fired:
            self.gesture_label = "Keyboard toggled - relax your hand"
            return True
        if not gf.ily:
            self._kb_t, self._kb_progress = None, 0.0
            self.gesture_label = "🤟 Keyboard pose"
            return True
        if self._kb_t is None:
            self._kb_t = t
        self._kb_progress = min((t - self._kb_t) / max(g.keyboard_gesture_s, 0.1), 1.0)
        self.gesture_label = f"Keyboard gesture {int(self._kb_progress * 100)}%"
        if self._kb_progress >= 1.0 and t >= self._kb_cooldown:
            self._kb_fired = True
            self._kb_cooldown = t + g.keyboard_gesture_cooldown_s
            self.counters["keyboard_toggles"] = self.counters.get("keyboard_toggles", 0) + 1
            self.on_command("toggle_keyboard")
            self.gesture_label = "Keyboard toggled - relax your hand"
        return True

    # -------------------------------------------------------------- press
    def _enter_press(self, feats: HandFeatures, gf: GestureFrame, t: float, slot: str | None = None) -> None:
        g = self.s.gesture
        slot = slot or gf.slot or "primary"
        action = g.bindings.get(f"pinch_{slot}", "none")
        if self._aim is not None:
            pos = self._aim.anchor           # Smart Aim Lock: the pre-pinch aim point
            self._aim = None
        else:
            # Fallback (lock disabled or never armed): rewind past the
            # confirmation span to just before the pinch moved the hand.
            onset = t - (self.s.pointer.click_rewind_ms + self.rec._confirm_s * 1000.0) / 1000.0
            pos = self.pointer.position_at(onset) or self.pointer.cursor or (0.0, 0.0)
        # Sticky double/triple click: a rapid second press lands on exactly
        # the same pixel as the first - only inside Windows' double-click
        # time measured from the previous button-down (as Windows does), so
        # a later click on a neighbouring control is never pulled back.
        if self._last_down is not None:
            lt, lp = self._last_down
            if t - lt < g.double_click_s and math.dist(lp, pos) <= g.sticky_double_click_px:
                pos = lp
        self.pointer.hold(pos)
        p = self._press = _Press(slot, action, t, feats.knuckle_center, feats.hand_scale, pos)

        key = self.keyboard.hit_test(*pos) if self.keyboard.is_visible() else None
        if key is not None and slot == "primary":
            p.key = key
            if self.s.keyboard.selection in ("pinch", "both"):
                self.keyboard.press_key(key, 0)
                self.counters["keys"] += 1
            p.next_repeat = t + 0.5
            self.state = State.KEY_PRESS
            self.gesture_label = f"Key {key}"
            return
        if key is not None:
            # Secondary contacts over the keyboard are swallowed.
            p.consumed = True
        self.mouse.move(*pos)
        self.state = State.PRESS_PENDING
        self.gesture_label = f"Press ({slot})"
        button = BUTTON_ACTIONS.get(action)
        if p.consumed or button is None:
            return
        mode = self.click_mode
        p.it = self.clicks.begin(button, "click", pos, t, slot, gf.cross_t)
        p.it.evidence, p.it.mode = gf.evidence or ("tap" if gf.tap else "threshold"), mode
        # Thumb pointer long-press right click, and a possible two-hand zoom,
        # need the release to decide: the button goes out then (or at the
        # drag threshold).
        p.deferred = self._thumb_long_press(action) or self._zoom_possible()
        if p.deferred:
            return
        if mode == "drag" and action in DRAG_ACTIONS:
            # Drag Mode: nothing is pressed until the hold completes, so an
            # early release clicks nothing.
            p.drag_wait = True
            p.it.kind = "drag"
            self.gesture_label = f"Drag Mode: hold {g.hold_to_drag_s:.1f} s"
            return
        if mode != "classic":
            # Instant Auto-Click: the complete click goes out now (button
            # down + up in one ordered SendInput batch); the pinch only has
            # to re-open before the next one.
            self._instant_click(p, button, t)
            return
        # Classic (v1.1.1): button down now, up at the release.
        if self._send_button(p.it, button, True):
            p.down = button
            self._last_down = (t, pos)
        else:
            p.consumed = True                # nothing went down: nothing to complete
            self.gesture_label = f"Click rejected ({p.it.reason})"

    def _zoom_possible(self) -> bool:
        """Two-hand zoom is bound and the second hand is pinching too: the
        release (or the zoom) must decide, so the click waits."""
        o = self._other_feats
        if self.s.gesture.bindings.get("two_hand_pinch") != "zoom" or o is None:
            return False
        if self.click_mode == "classic":
            return True                       # v1.1.1 behaviour: any second hand defers
        e = self.s.finger.pinch_thresholds.get("index", [0.26, 0.36])[0]
        return o.pinch[Finger.INDEX] < e * self.s.gesture.aim_near_ratio

    def _instant_click(self, p: _Press, button: str, t: float) -> None:
        g = self.s.gesture
        p.consumed = p.clicked = True
        last = self._last_instant
        self._last_instant = (t, p.slot)
        if last is not None and last[1] == p.slot and t - last[0] < g.instant_min_interval_s:
            # Faster than a human pinch cycle: a tracking flicker re-engaged
            # the same contact. Never a second click.
            self.clicks.end(p.it, "cancelled", "debounced: same contact re-engaged too fast "
                                               f"({(t - last[0]) * 1000:.0f} ms)")
            self.gesture_label = "Click ✓ (open fingers to click again)"
            return
        if self._last_click is not None and t - self._last_click[0] < g.click_debounce_s:
            self.clicks.end(p.it, "cancelled", "debounced: too soon after the previous click")
            self.gesture_label = "Click ignored (too soon)"
            return
        ok = self.mouse.click(button, 1) is not False
        reason = getattr(self.mouse, "last_error", "") if not ok else ""
        self.clicks.submitted(p.it, True, ok, reason or ("" if ok else "refused by the input dispatcher"),
                              getattr(self.mouse, "last_submit_t", None) or None)
        if not ok:
            self.gesture_label = f"Click rejected ({p.it.reason if p.it else reason})"
            return
        self.clicks.submitted(p.it, False, True)
        self.counters["clicks"] += 1
        double = (self._last_click is not None and self._last_click[1] == p.pos
                  and t - self._last_click[0] < g.double_click_s)
        self._last_click = (t, p.pos)
        self._last_down = (t, p.pos)
        self.gesture_label = ("Double click" if double else "Click") + f" ✓ ({button})"

    def _send_button(self, it: Interaction | None, button: str, down: bool) -> bool:
        """One button event through the sink; the outcome (SendInput result,
        or the dispatcher's reason for refusing) goes to the interaction log."""
        ok = self.mouse.button(button, down) is not False
        reason = getattr(self.mouse, "last_error", "") if not ok else getattr(self.mouse, "last_note", "")
        self.clicks.submitted(it, down, ok, reason or ("" if ok else "refused by the input dispatcher"),
                              getattr(self.mouse, "last_submit_t", None) or None)
        return ok

    def _thumb_long_press(self, action: str) -> bool:
        g = self.s.gesture
        return (self._finger is Finger.THUMB and g.long_press_right_click_thumb
                and action == "left_click_drag")

    def _in_press(self, feats: HandFeatures, gf: GestureFrame, raw, t: float, other) -> None:
        p = self._press
        assert p is not None
        out = self.pointer.update(raw, t)
        g = self.s.gesture
        held = t - p.t0
        if p.clicked and gf.slot == p.slot:
            # Instant click done; the contact is still closed. Keep the cursor
            # on the clicked pixel, but never frozen for long: after
            # instant_hold_follow_s or a whole-hand move it follows again.
            # No further click until the contact re-opens.
            if not p.follow and (held >= g.instant_hold_follow_s
                                 or self._hand_motion(feats, p.knuckle0, p.scale) > g.aim_cancel_move):
                p.follow = True
                self.pointer.release_hold()
            elif p.follow and out is not None and out.moved:
                self.mouse.move(out.x, out.y)
            return
        if gf.slot != p.slot:
            self._release_action(p, held, t)
            self._end_press(t)
            return
        if (p.down is None and other is not None and g.bindings.get("two_hand_pinch") == "zoom"
                and self._zoom_check(feats, other)):
            self.clicks.end(p.it, "cancelled", "second hand joined: zoom")
            self._press = None   # second hand joined: this is a zoom, not a click
            self._enter_zoom(feats, other)
            return
        if p.consumed:
            return
        if p.action in DRAG_ACTIONS and held >= g.hold_to_drag_s:
            self._start_drag(feats, p, t)
        elif p.drag_wait:
            self.gesture_label = f"Drag Mode: hold {max(g.hold_to_drag_s - held, 0):.1f} s"
        elif p.deferred and self._thumb_long_press(p.action):
            self.gesture_label = ("Thumb press - release = left click" if held < g.long_press_s
                                  else "Release = right click · keep holding to drag")
        elif p.down is not None:
            self.gesture_label = f"Press ({p.slot}) - hold {max(g.hold_to_drag_s - held, 0):.1f} s to drag" \
                if p.action in DRAG_ACTIONS else f"Press ({p.slot})"

    def _release_action(self, p: _Press, held: float, t: float) -> None:
        if p.clicked:
            return                            # instant click: completed at the pinch
        if p.drag_wait:
            self.clicks.end(p.it, "cancelled", f"released before the {self.s.gesture.hold_to_drag_s:.1f} s "
                                               "drag hold (Drag Mode: no click)")
            self.gesture_label = "Drag cancelled"
            return
        if p.down is not None:
            # Button went down at confirm: one button-up at the same anchored
            # pixel completes exactly one click (or a double click when the
            # previous one was moments ago at the same pixel).
            self.mouse.move(*p.pos)
            ok = self._send_button(p.it, p.down, False)
            if not p.consumed and ok:
                self.counters["clicks"] += 1
                double = (self._last_click is not None and self._last_click[1] == p.pos
                          and t - self._last_click[0] < self.s.gesture.double_click_s)
                self._last_click = (t, p.pos)
                self.gesture_label = ("Double click" if double else "Click") + f" ({p.down})"
            elif not ok:
                self.gesture_label = "Click incomplete"
            p.down = None
            return
        if p.consumed:
            return
        a = p.action
        if a in BUTTON_ACTIONS:
            if self._thumb_long_press(a):
                # Thumb pointer: the hold length picks the button.
                button = "left" if held < self.s.gesture.long_press_s else "right"
                if button == "right" and p.it is not None:
                    p.it.button, p.it.kind = "right", "long_press"
                self._click(button, 1, t, p.pos, p.it)
            elif held <= 1.2 + self.s.gesture.hold_to_drag_s:
                self._click(BUTTON_ACTIONS[a], 1, t, p.pos, p.it)
            else:
                self.clicks.end(p.it, "cancelled", "held too long for a click")
        elif held <= 1.2:
            self._run_action(a, t, p.pos)

    def _run_action(self, action: str, t: float, pos: tuple[float, float] | None = None) -> None:
        if action in ("left_click", "left_click_drag"):
            self._click("left", 1, t, pos)
        elif action in ("right_click", "right_click_drag"):
            self._click("right", 1, t, pos)
        elif action == "middle_click":
            self._click("middle", 1, t, pos)
        elif action == "double_click":
            self._click("left", 2, t, pos)
        elif action == "triple_click":
            self._click("left", 3, t, pos)
        elif action == "back":
            self.mouse.xbutton(1)
            self.gesture_label = "Back"
        elif action == "forward":
            self.mouse.xbutton(2)
            self.gesture_label = "Forward"
        elif action == "toggle_precision":
            self.toggle_precision()
        elif action in ("toggle_keyboard", "pause", "toggle_magnifier"):
            self.on_command(action)

    def _click(self, button: str, count: int, t: float, pos, it: Interaction | None = None,
               kind: str = "click") -> None:
        """A complete click sent at once (deferred presses, bindings, dwell)."""
        if self._last_click is not None and t - self._last_click[0] < self.s.gesture.click_debounce_s:
            self.clicks.end(it, "cancelled", "debounced: too soon after the previous click")
            return
        at = pos or self.pointer.cursor or (0.0, 0.0)
        it = it or self.clicks.begin(button, kind, at, t)
        if pos is not None:
            self.mouse.move(*pos)
        ok = self.mouse.click(button, count) is not False
        self.clicks.submitted(it, True, ok, getattr(self.mouse, "last_error", "") if not ok else "",
                              getattr(self.mouse, "last_submit_t", None) or None)
        if not ok:
            self.gesture_label = "Click rejected"
            return
        self.clicks.submitted(it, False, True)
        self.counters["clicks"] += 1
        self._last_click = (t, at)
        self._last_down = (t, at)
        self.gesture_label = {1: "Click", 2: "Double click", 3: "Triple click"}[count] + f" ({button})"

    def _end_press(self, t: float) -> None:
        p = self._press
        if p is not None and p.it is not None and not p.it.done:
            self.clicks.end(p.it, "cancelled", "no button event was sent")
        self._press = None
        # The cursor stays put a moment so the opening hand does not jerk it;
        # a new pinch may start right away (quick double clicks).
        self._settle_until = t + self.s.pointer.release_settle_ms / 1000.0
        self.state = State.POINTER

    def _start_drag(self, feats: HandFeatures, p: _Press, t: float) -> None:
        """Hold threshold reached: the button that went down at confirm is
        now a drag (no second button-down); the cursor follows the rigid
        knuckle centroid from the anchored point, without a jump."""
        button = DRAG_ACTIONS[p.action]
        self.mouse.move(*p.pos)
        if p.down != button:                 # deferred press: first and only button-down
            if p.it is not None:
                p.it.button = button
            if not self._send_button(p.it, button, True):
                p.consumed = True
                self.gesture_label = "Drag rejected"
                return
            p.down = button
        p.drag_wait = False
        self.clicks.dragging(p.it)
        assert self._finger is not None
        self.pointer.change_source(self._source(feats, self._finger, False), feats.knuckle_center)
        self.pointer.release_hold()
        self.state = State.DRAGGING
        self.counters["drags"] += 1
        self.gesture_label = f"Drag ({button})"

    def _in_drag(self, feats: HandFeatures, gf: GestureFrame, raw, t: float, other) -> None:
        p = self._press
        assert p is not None
        button = DRAG_ACTIONS[p.action]
        if p.slot == "pinky":
            # Pinky Drag continues while the little finger stays up; a couple
            # of frames without it (curled, or another finger extended) drop.
            p.misses = 0 if pinky_held(feats) else p.misses + 1
            ended = p.misses >= max(1, self.s.gesture.pinky_release_frames)
        else:
            ended = gf.slot != p.slot
        if ended:
            rewind = t - self.s.pointer.click_rewind_ms / 1000.0
            pos = self.pointer.position_at(rewind) or self.pointer.cursor or p.pos
            self.mouse.move(*pos)
            self._send_button(p.it, button, False)
            p.down = None
            self.pointer.hold(pos)
            assert self._finger is not None
            self.pointer.change_source(feats.knuckle_center, self._source(feats, self._finger, False))
            self._end_press(t)
            self.gesture_label = "Drop"
            if p.slot == "pinky":
                # Back to pointing + Instant Auto-Click. A contact closed at
                # the drop must open before it clicks; the pose must be
                # released before another Pinky Drag.
                self._pinky_block = True
                self.rec.require_open()
                self.drag_state, self._drag_state_t = "released", t
                self.gesture_label = "Released"
            elif self.click_mode == "drag" and self.s.gesture.drag_one_shot:
                self.set_click_mode("instant")       # one-shot Drag Mode
            return
        out = self.pointer.update(raw, t)
        if out is not None and out.moved:
            self.mouse.move(out.x, out.y)

    def _in_key_press(self, feats: HandFeatures, gf: GestureFrame, raw, t: float, other) -> None:
        p = self._press
        assert p is not None
        self.pointer.update(raw, t)
        if gf.slot != p.slot:
            self._end_press(t)
            return
        if (p.key in REPEAT_KEYS and self.s.keyboard.repeat_backspace
                and self.s.keyboard.selection in ("pinch", "both") and t >= p.next_repeat):
            self.keyboard.press_key(p.key, 0)
            p.next_repeat = t + 0.09

    # ------------------------------------------------------------- scroll
    def _enter_scroll(self, feats: HandFeatures, t: float) -> None:
        """V pose (index + middle up, ring + little curled): Scroll Mode.
        The cursor freezes; hand displacement from where the pose began
        drives the wheel (joystick) or moves it directly."""
        self._cancel_aim(latch=False)
        self.pointer.hold()
        self._scroll_origin = feats.knuckle_center
        self._scroll_prev = (0.0, 0.0)
        self._scroll_axis = None
        self._scroll_t = t
        self.state = State.SCROLLING
        self.counters["scrolls"] += 1
        self.gesture_label = "Scroll Ready"

    def _in_scroll(self, feats: HandFeatures, gf: GestureFrame, raw, t: float, other) -> None:
        g = self.s.gesture
        self.pointer.update(raw, t)
        if not gf.scroll_pose:
            # Exit: the cursor resumes from where it was frozen (no jump) and
            # the hand returning from the V pose cannot click right away.
            self.pointer.release_hold()
            self._settle_until = self._press_block_until = t + self.s.pointer.release_settle_ms / 1000.0
            self.state = State.POINTER
            self.gesture_label = "Point"
            return
        assert self._scroll_origin is not None
        scale = max(feats.hand_scale, 1e-3)
        dx = (feats.knuckle_center[0] - self._scroll_origin[0]) * self.rec.aspect / scale
        dy = (feats.knuckle_center[1] - self._scroll_origin[1]) / scale
        if g.scroll_axis_lock:
            if self._scroll_axis is None and max(abs(dx), abs(dy)) > g.scroll_dead_zone:
                # Vertical unless the motion is clearly sideways.
                self._scroll_axis = "x" if abs(dx) > g.scroll_h_bias * abs(dy) else "y"
            if self._scroll_axis == "y":
                dx = 0.0
            elif self._scroll_axis == "x":
                dy = 0.0
        dt = min(max(t - self._scroll_t, 1e-3), 0.1)
        self._scroll_t = t
        sign = -1.0 if g.natural_scroll else 1.0
        if g.scroll_style == "direct":
            sx = (dx - self._scroll_prev[0]) * g.scroll_direct_gain
            sy = (dy - self._scroll_prev[1]) * g.scroll_direct_gain
        else:
            sx = _deflect(dx, g.scroll_dead_zone, g.scroll_full, g.scroll_accel) * g.scroll_speed * dt
            sy = _deflect(dy, g.scroll_dead_zone, g.scroll_full, g.scroll_accel) * g.scroll_speed * dt
        self._scroll_prev = (dx, dy)
        # Hand up (dy < 0) scrolls content up (positive wheel) unless natural.
        wheel_y = -sy * sign
        wheel_x = sx * sign
        if wheel_x or wheel_y:
            self.mouse.scroll(wheel_x, wheel_y)
            self.gesture_label = "Scrolling " + ("Up" if wheel_y > 0 else "Down" if wheel_y < 0
                                                 else "Right" if wheel_x > 0 else "Left")
        else:
            self.gesture_label = "Scroll Ready"

    # --------------------------------------------------------------- zoom
    def _zoom_check(self, a: HandFeatures, b: HandFeatures) -> bool:
        e, r = self.s.finger.pinch_thresholds.get("index", [0.26, 0.36])
        self._zoom_a.engage = self._zoom_b.engage = e
        self._zoom_a.release = self._zoom_b.release = r
        return self._zoom_a.update(a.pinch[Finger.INDEX]) & self._zoom_b.update(b.pinch[Finger.INDEX])

    def _hands_apart(self, a: HandFeatures, b: HandFeatures) -> float:
        ax = (a.tips[Finger.INDEX][0] + a.tips[Finger.THUMB][0]) / 2 * self.rec.aspect
        ay = (a.tips[Finger.INDEX][1] + a.tips[Finger.THUMB][1]) / 2
        bx = (b.tips[Finger.INDEX][0] + b.tips[Finger.THUMB][0]) / 2 * self.rec.aspect
        by = (b.tips[Finger.INDEX][1] + b.tips[Finger.THUMB][1]) / 2
        return math.hypot(ax - bx, ay - by)

    def _enter_zoom(self, a: HandFeatures, b: HandFeatures) -> None:
        self.pointer.hold()
        self._zoom_ref = self._hands_apart(a, b)
        self.state = State.ZOOMING
        self.gesture_label = "Zoom"

    def _in_zoom(self, feats: HandFeatures, gf: GestureFrame, raw, t: float, other) -> None:
        self.pointer.update(raw, t)
        if other is None or not self._zoom_check(feats, other):
            self.pointer.release_hold()
            self._settle_until = self._press_block_until = t + self.s.pointer.release_settle_ms / 1000.0
            self.state = State.POINTER
            return
        d = self._hands_apart(feats, other)
        step = self.s.gesture.zoom_step
        if self._zoom_ref > 1e-3:
            ratio = d / self._zoom_ref
            if ratio >= 1 + step:
                self.mouse.zoom(1)
                self._zoom_ref = d
                self.gesture_label = "Zoom in"
            elif ratio <= 1 - step:
                self.mouse.zoom(-1)
                self._zoom_ref = d
                self.gesture_label = "Zoom out"

    def _in_clutch(self, feats: HandFeatures, gf: GestureFrame, raw, t: float, other) -> None:
        self.pointer.update(raw, t)
        if not gf.fist:
            self.pointer.release_hold()
            self.state = State.POINTER

    # ------------------------------------------------------------ keyboard
    def _keyboard_hover(self, pos: tuple[float, float], t: float) -> None:
        if not self.keyboard.is_visible():
            if self._hover_key is not None:
                self._hover_key = None
                self.keyboard.hover(0, None)
            return
        kc = self.s.keyboard
        key = self.keyboard.hit_test(*pos)
        if key != self._hover_key:
            self._hover_key, self._hover_t, self._hover_pos = key, t, pos
            self._dwell_fired = False
            self.keyboard.hover(0, key, 0.0)
            return
        if key is None:
            return
        progress = 0.0
        if kc.selection in ("dwell", "both") and not self._dwell_fired:
            if math.dist(pos, self._hover_pos) > kc.dwell_tolerance_px:
                self._hover_t, self._hover_pos = t, pos
            progress = (t - self._hover_t) / max(kc.dwell_ms / 1000.0, 0.1)
            if progress >= 1.0:
                self.keyboard.press_key(key, 0)
                self.counters["keys"] += 1
                self._dwell_fired = True
                progress = 1.0
        self.keyboard.hover(0, key, progress)

    def _other_hand_typing(self, feats: HandFeatures, t: float) -> None:
        """Second hand types with its index finger (direct mapped, no cursor)."""
        tip = feats.tips[Finger.INDEX]
        if self._other_ptr is None:
            self._other_ptr = tip
        a = 0.45
        self._other_ptr = (self._other_ptr[0] + a * (tip[0] - self._other_ptr[0]),
                           self._other_ptr[1] + a * (tip[1] - self._other_ptr[1]))
        nx, ny = self.pointer._box_norm(*self._other_ptr)
        r = self.screen.target_rect(self.s.pointer.monitor)
        x = r.left + min(max(nx, 0), 1) * (r.width - 1)
        y = r.top + min(max(ny, 0), 1) * (r.height - 1)
        key = self.keyboard.hit_test(x, y)
        gf = self.rec_other.update(feats, t, Finger.INDEX, style="pinch_pointer", allow_fist=False)
        if gf.slot == "primary" and key is not None and self._other_key != "__pressed__":
            self.keyboard.press_key(key, 1)
            self.counters["keys"] += 1
            self._other_key = "__pressed__"
        elif gf.slot is None and self._other_key == "__pressed__":
            self._other_key = None
        if self._other_key != "__pressed__":
            self._other_key = key
        self.keyboard.hover(1, key, 0.0)

    def _dwell_click(self, pos: tuple[float, float], t: float) -> None:
        g = self.s.gesture
        if not g.dwell_click or (self.keyboard.is_visible() and self._hover_key):
            return
        if self._dwell_click_pos is None or math.dist(pos, self._dwell_click_pos) > 12:
            self._dwell_click_pos, self._dwell_click_t = pos, t
            return
        if self._dwell_click_t > 0 and t - self._dwell_click_t >= g.dwell_click_s:
            self._click("left", 1, t, pos, kind="dwell")
            self._dwell_click_t = -1.0  # one click per stop; re-armed by moving

    def _update_palm_hold(self, gf: GestureFrame, t: float) -> None:
        action = self.s.gesture.bindings.get("open_palm_hold", "none")
        if action == "none" or not self.armed or not gf.open_palm:
            self._palm_hold_t, self._palm_hold_fired = None, False
            return
        if self._palm_hold_t is None:
            self._palm_hold_t = t
        elif not self._palm_hold_fired and t - self._palm_hold_t >= 1.2:
            self._palm_hold_fired = True
            if action == "pause":
                self.set_paused(not self.paused)
            else:
                self.on_command(action)

    # --------------------------------------------------------------- hint
    def _update_hint(self, obs, feats: HandFeatures, t: float) -> None:
        """Brief camera-condition hint, shown only after it persists. The
        limits are image-space estimates (hand scale = wrist to middle
        knuckle / frame height), not a measured calibration."""
        xy = obs.image[:, :2]
        if feats.hand_scale < 0.07:
            h = "Hand too far - move closer"
        elif feats.hand_scale > 0.55:
            h = "Hand too close - move back"
        elif obs.score < 0.6:
            h = "Tracking weak - add light"
        elif float(xy.min()) < 0.01 or float(xy.max()) > 0.99:
            h = "Keep whole hand in view"
        else:
            h = ""
        if h != self._hint_cand:
            self._hint_cand, self._hint_t = h, t
        if not h:
            self._hint = ""
        elif t - self._hint_t >= 0.7:
            self._hint = h

    # ------------------------------------------------------------- safety
    def _on_lost(self, t: float) -> None:
        if self._lost_since is None:
            self._lost_since = t
        lost = t - self._lost_since
        grace = self.s.gesture.tracking_loss_grace_s
        if self.state in ENGAGED or self.state is State.POINTER:
            if lost >= grace:
                self._safe_release("tracking lost (button released)")
                self.pointer.reset()
                self.rec.reset()
                self.rec_other.reset()
                self.state = State.TRACKING_LOST
                self.counters["tracking_losses"] += 1
                self.gesture_label = "Tracking lost"
        if self._hover_key is not None:
            self._hover_key = None
            self.keyboard.hover(0, None)
        if self.armed and self.s.gesture.auto_disarm_s > 0 and lost >= self.s.gesture.auto_disarm_s:
            self.set_armed(False, t)

    def _end_interaction(self, reason: str) -> None:
        p = self._press
        if p is not None and p.it is not None and not p.it.done:
            self.clicks.end(p.it, "cancelled", reason)

    def _safe_release(self, reason: str = "released") -> None:
        self._end_interaction(reason)
        self._cancel_pinky(reason)
        if self._pinky_dragging():
            self.drag_state = "cancelled"
        self.mouse.release_all()
        self._press = None
        self._aim = None
        self._kb_t, self._kb_progress = None, 0.0
        if self.pointer.holding:
            self.pointer.release_hold()
        if self.state not in (State.DISARMED, State.PAUSED):
            self.state = State.IDLE if self.armed else State.DISARMED

    # ------------------------------------------------------------- helpers
    def _source(self, feats: HandFeatures, finger: Finger, dragging: bool) -> tuple[float, float]:
        if dragging:
            return feats.knuckle_center
        tip, mcp = feats.tips[finger], feats.mcps[finger]
        a = self.s.pointer.anchor_blend
        return tip[0] * (1 - a) + mcp[0] * a, tip[1] * (1 - a) + mcp[1] * a

    def _status(self, t: float, feats: HandFeatures | None, n: int,
                gf: GestureFrame | None = None) -> EngineStatus:
        st = EngineStatus(
            t=t, armed=self.armed, paused=self.paused, state=self.state.value,
            finger=self._finger.label if (self._finger and feats) else "-",
            finger_mode=self.fingers.mode.value, pointer_mode=self.pointer.mode.value,
            gesture=self.gesture_label if feats else "-", buttons=self.mouse.held_buttons(),
            keyboard_hover=self._hover_key,
            cursor=(int(self.pointer.cursor[0]), int(self.pointer.cursor[1])) if self.pointer.cursor else None,
            tracking=feats is not None, hands_seen=n, message=self.message,
            click_style=self.s.gesture.click_style, progress=self._kb_progress,
            click_mode=self.click_mode, hint=self._hint if feats is not None else "")
        p = self._press
        if self._aim is not None:
            st.aim_lock = (int(self._aim.anchor[0]), int(self._aim.anchor[1]))   # no button pressed
            pair = _pair_name(self._finger, self.s.gesture.click_style, self._aim.slot)
            st.aim_reason = f"{pair} closing ({gf.approach:.2f}x threshold)" if gf else pair + " closing"
            st.candidate_ms = round((t - self._aim.t0) * 1000)
        elif p is not None and self.state is State.PRESS_PENDING:
            st.press_at = (int(p.pos[0]), int(p.pos[1]))
            st.candidate_ms = round((t - p.t0) * 1000)
            if p.drag_wait:
                st.drag_progress = min((t - p.t0) / max(self.s.gesture.hold_to_drag_s, 0.1), 1.0)
        if self._pinky is not None:
            st.press_at = (int(self._pinky.anchor[0]), int(self._pinky.anchor[1]))
            st.drag_progress = self._drag_progress
            st.candidate_ms = round((t - self._pinky.t0) * 1000)
        st.drag = self.drag_state if (self.drag_state in ("ready", "dragging")
                                      or t - self._drag_state_t < 1.5) else ""
        st.pose = self._pose if feats is not None else "-"
        st.dual = self._dual_on
        if self._dual_on:
            st.vhands = self.dual.status()
            st.dual_diag = self.dual.diag()
        if p is not None and (p.clicked or (p.it is not None and p.it.done)):
            st.rearm = "open fingers to click again"
        if gf is not None and gf.near_slot is not None and math.isfinite(gf.near_d):
            st.contact = {"pair": _pair_name(self._finger, self.s.gesture.click_style, gf.near_slot),
                          "slot": gf.near_slot, "d": round(gf.near_d, 3), "engage": round(gf.near_engage, 3),
                          "ratio": round(gf.near_d / max(gf.near_engage, 1e-6), 2), "note": gf.note}
        if feats is not None:
            st.hand = self._hand_label
            st.extension = {f.label: round(v, 2) for f, v in feats.extension.items()}
            st.finger_confidence = {f.label: round(v, 2) for f, v in self.fingers.confidence.items()}
            if gf is not None:
                st.pinch = {k: round(v, 3) for k, v in gf.contacts.items()}
        self.message = ""
        return st


def _pose_name(feats: HandFeatures, gf: GestureFrame, pinky_ok: bool) -> str:
    """Recognized hand pose (diagnostics)."""
    if gf.slot:
        return f"Pinch ({gf.slot})"
    if gf.ily or gf.ily_raw:
        return "🤟"
    if pinky_pose(feats):
        return "Pinky only (drag)" if pinky_ok else "Pinky only (pointer)"
    if gf.scroll_pose:
        return "V (scroll)"
    if gf.fist:
        return "Fist"
    if gf.open_palm:
        return "Open palm"
    return "Pinch forming" if gf.closing else "Point"


def _pair_name(finger: Finger | None, style: str, slot: str | None) -> str:
    """'thumb+index' for a slot under the current mapping (diagnostics)."""
    if slot is None:
        return "-"
    c = slot_contacts(finger, style).get(slot, "")
    return "thumb press" if c == "thumb_press" else ("thumb+" + c.split(":")[1]) if c else "-"


def _deflect(d: float, dead: float, full: float, accel: float = 1.0) -> float:
    if abs(d) < dead:
        return 0.0
    mag = min((abs(d) - dead) / max(full - dead, 1e-3), 1.0)
    return math.copysign(mag ** max(accel, 0.2), d)
