"""v1.1.3 Dual-Hand Air Keyboard: two virtual hand cursors, per-hand pinch typing,
shortcuts across hands, identity, hand-over to and from the desktop pointer."""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import RecMouse

from airdesk.config import Settings
from airdesk.core.dual_keyboard import EDGE
from airdesk.core.engine import InteractionEngine, State
from airdesk.core.hand_model import analyze
from airdesk.core.keyboard_layout import hit_near, layout_rects
from airdesk.core.screen import ScreenLayout
from airdesk.core.types import Finger
from airdesk.sim.synthetic import Pose, frame, observation

IDX, MID, LIT, THB = Finger.INDEX, Finger.MIDDLE, Finger.LITTLE, Finger.THUMB


class DualKb:
    """KeyboardSink with the real full layout and the v1.1.3 extensions."""

    def __init__(self):
        self.visible = True
        self.rects = layout_rects("full")
        self.pressed: list[tuple[str, int]] = []
        self.cursor_log: list[list] = []
        self.dual_calls: list[bool] = []
        self.cleared = 0
        self.moves: list[tuple[int, int]] = []

    def is_visible(self):
        return self.visible

    def hit_test(self, x, y):           # desktop pointer path (screen px): bottom 30 % of the screen
        return "a" if self.visible and y > 760 else None

    def hit_frac(self, fx, fy):
        k = hit_near(self.rects, fx, fy)
        return k.key if k else None

    def hover(self, hand, key, progress=0.0):
        pass

    def press_key(self, key, hand=0):
        self.pressed.append((key, hand))
        return True

    def release_all(self):
        pass

    def cursors(self, data):
        self.cursor_log.append(data)

    def set_dual(self, on):
        self.dual_calls.append(on)

    def clear_modifiers(self):
        self.cleared += 1

    def move_by(self, dx, dy):
        self.moves.append((dx, dy))

    def centre(self, key):
        r = next(r for r in self.rects if r.key == key)
        return r.x + r.w / 2, r.y + r.h / 2


class Hand:
    """One synthetic hand whose index fingertip cursor is aimed at a key."""

    def __init__(self, sim: "Dual", side: str, wrist_x: float):
        self.sim, self.side = sim, side
        self.x, self.y = wrist_x, 0.72
        self.pose = Pose.pointing(IDX)

    def obs(self, pose=None, **kw):
        return observation(pose or self.pose, self.x, self.y, handedness="Left" if self.side == "left" else "Right",
                           mirror=self.side == "left", **kw)

    def aim(self, key):
        """Place the wrist so the (blended) index source maps onto the key centre."""
        s = self.sim.s
        fx, fy = self.sim.kb.centre(key)
        nx, ny = (fx + EDGE) / (1 + 2 * EDGE), (fy + EDGE) / (1 + 2 * EDGE)
        p = s.pointer
        ix = p.box_cx + (nx - 0.5) * p.box_w / p.direct_gain
        iy = p.box_cy + (ny - 0.5) * p.box_h / p.direct_gain
        f = analyze(self.obs(Pose.pointing(IDX)), 16 / 9)
        a = p.anchor_blend
        sx = f.tips[IDX][0] * (1 - a) + f.mcps[IDX][0] * a
        sy = f.tips[IDX][1] * (1 - a) + f.mcps[IDX][1] * a
        self.x += ix - sx
        self.y += iy - sy


class Dual:
    def __init__(self, settings=None, style="pinch_pointer"):
        self.s = settings or Settings()
        self.s.gesture.click_style = style
        self.mouse = RecMouse()
        self.kb = DualKb()
        self.commands: list[str] = []
        self.engine = InteractionEngine(self.s, ScreenLayout.single(1920, 1080), self.mouse, self.kb,
                                        on_command=self.commands.append)
        self.t = 0.0
        self.engine.set_armed(True, self.t)
        self.L = Hand(self, "left", 0.36)
        self.R = Hand(self, "right", 0.64)

    def step(self, left=None, right=None, n=1, use_left=True, use_right=True, **kw):
        st = None
        for _ in range(n):
            self.t += 1 / 30
            hs = []
            if use_left:
                hs.append(self.L.obs(left, **kw))
            if use_right:
                hs.append(self.R.obs(right, **kw))
            st = self.engine.process(frame(self.t, *hs))
        return st

    def settle(self, n=12, **kw):
        return self.step(n=n, **kw)

    def pinch(self, hand: str, finger=IDX, hold=2, **kw):
        """Natural pinch (closing, contact, re-opening) on one hand while the
        other hand keeps pointing."""
        for a in (0.35, 0.7, 1.0):
            self.step(**{hand: Pose.closing(finger, a)}, **kw)
        self.step(**{hand: Pose.closing(finger, 1.0)}, n=hold, **kw)
        for a in (0.6, 0.2, 0.0):
            self.step(**{hand: Pose.closing(finger, a)}, **kw)
        self.step(n=4, **kw)


def keys(d: Dual):
    return [k for k, _h in d.kb.pressed]


def test_keyboard_mode_never_injects_desktop_mouse_input():
    d = Dual()
    d.settle()
    assert d.engine.state is State.KEYBOARD and d.kb.dual_calls == [True]
    d.L.aim("q")
    d.R.aim("p")
    d.settle()
    d.pinch("left")
    d.pinch("right")
    assert keys(d) == ["q", "p"]
    assert d.mouse.events == []                       # no move, button, click or wheel


@pytest.mark.parametrize("style", ["pinch_pointer", "pinch_free"])
def test_each_hand_types_alone(style):
    finger = IDX if style == "pinch_pointer" else MID
    for side in ("left", "right"):
        d = Dual(style=style)
        hand = d.L if side == "left" else d.R
        word = "hello" if side == "right" else "sad"
        out = []
        for ch in word:
            hand.aim(ch)
            d.step(n=8, use_left=side == "left", use_right=side == "right")
            for a in (0.35, 0.7, 1.0):
                d.step(**{side: Pose.closing(finger, a)}, use_left=side == "left", use_right=side == "right")
            d.step(**{side: Pose.closing(finger, 1.0)}, n=2, use_left=side == "left", use_right=side == "right")
            for a in (0.6, 0.2, 0.0):
                d.step(**{side: Pose.closing(finger, a)}, use_left=side == "left", use_right=side == "right")
            d.step(n=3, use_left=side == "left", use_right=side == "right")
        out = [k for k, h in d.kb.pressed]
        assert out == list(word), (side, style, out)
        assert {h for _k, h in d.kb.pressed} == {0 if side == "left" else 1}
        assert d.mouse.events == []


def test_both_hands_type_in_order_and_simultaneous_presses_both_arrive():
    d = Dual()
    d.L.aim("x")
    d.R.aim("y")
    d.settle()
    # Same frames: the left pinch crosses its threshold first.
    seq = [(0.35, 0.0), (0.7, 0.35), (1.0, 0.7), (1.0, 1.0), (1.0, 1.0), (0.6, 0.6), (0.2, 0.2), (0.0, 0.0)]
    for la, ra in seq:
        d.step(left=Pose.closing(IDX, la), right=Pose.closing(IDX, ra))
    d.settle(n=4)
    assert d.kb.pressed == [("x", 0), ("y", 1)]
    # Exactly the same frame on both hands: both keys, one each.
    d.kb.pressed.clear()
    d.L.aim("f")
    d.R.aim("j")
    d.settle()
    for a in (0.35, 0.7, 1.0, 1.0, 1.0, 0.6, 0.2, 0.0):
        d.step(left=Pose.closing(IDX, a), right=Pose.closing(IDX, a))
    d.settle(n=4)
    assert sorted(d.kb.pressed) == [("f", 0), ("j", 1)]


def test_one_pinch_is_one_key_and_rearm_needs_opening():
    d = Dual()
    d.R.aim("k")
    d.settle()
    d.pinch("right", hold=45)                          # 1.5 s held: still one key
    assert keys(d) == ["k"]
    d.pinch("right")
    assert keys(d) == ["k", "k"]


def test_backspace_repeats_only_after_the_hold_delay():
    d = Dual()
    d.R.aim("backspace")
    d.settle()
    d.pinch("right", hold=12)                          # ~0.5 s: below the 0.7 s repeat delay
    assert keys(d) == ["backspace"]
    d.pinch("right", hold=40)
    assert keys(d).count("backspace") > 2


@pytest.mark.parametrize("mod_hand,key_hand", [("left", "right"), ("right", "left"), ("left", "left")])
def test_shortcut_across_hands(mod_hand, key_hand):
    d = Dual()
    for hand, key in ((mod_hand, "ctrl"), (key_hand, "c")):
        (d.L if hand == "left" else d.R).aim(key)
        d.settle(n=10)
        d.pinch(hand)
    assert keys(d) == ["ctrl", "c"]
    assert [h for _k, h in d.kb.pressed] == [0 if mod_hand == "left" else 1, 0 if key_hand == "left" else 1]


def test_both_hands_on_the_same_key_in_the_same_instant_is_one_key():
    d = Dual()
    d.L.aim("g")
    d.R.aim("g")
    d.R.x += 0.12                                     # (wrists apart; fingertips on the same key)
    d.R.aim("g")
    d.settle()
    for a in (0.35, 0.7, 1.0, 1.0, 0.6, 0.2, 0.0):
        d.step(left=Pose.closing(IDX, a), right=Pose.closing(IDX, a))
    d.settle(n=4)
    assert keys(d) == ["g"]
    assert d.engine.dual.counters["duplicates_suppressed"] == 1


def test_one_hand_leaving_keeps_the_other_typing_and_lost_hands_clear_modifiers():
    d = Dual()
    d.L.aim("ctrl")
    d.R.aim("v")
    d.settle()
    d.pinch("left")
    d.step(n=20, use_left=False)                       # left hand leaves the camera
    st = d.step(use_left=False)
    assert [v["side"] for v in st.vhands if v["tracked"]] == ["right"]
    for a in (0.35, 0.7, 1.0, 1.0, 0.6, 0.2, 0.0):
        d.step(right=Pose.closing(IDX, a), use_left=False)
    assert keys(d) == ["ctrl", "v"]
    cleared = d.kb.cleared
    d.step(n=40, use_left=False, use_right=False)      # both hands gone > 1 s
    assert d.kb.cleared == cleared + 1
    assert d.kb.cursor_log[-1] == []


def test_identity_survives_hands_crossing_without_wrong_keys():
    d = Dual()
    d.L.aim("a")
    d.R.aim("l")
    d.settle()
    rng = np.random.default_rng(1)
    # The hands sweep past each other; MediaPipe's labels flip midway.
    for k in range(40):
        d.L.x += 0.008
        d.R.x -= 0.008
        lo = d.L.obs(noise=0.001, rng=rng)
        ro = d.R.obs(noise=0.001, rng=rng)
        if 15 <= k <= 25:
            lo.handedness, ro.handedness = "Right", "Left"
        d.t += 1 / 30
        st = d.engine.process(frame(d.t, ro, lo))
    sides = {v["side"]: v for v in st.vhands}
    # Continuity kept the identities: the left cursor followed the left hand
    # all the way to the right of the keyboard, and nothing was typed.
    assert sides["left"]["pos"][0] > sides["right"]["pos"][0]
    assert d.kb.pressed == []


def test_uncertain_identity_never_types():
    d = Dual()
    d.L.aim("g")
    d.R.aim("h")
    d.R.x, d.R.y = d.L.x + 0.002, d.L.y               # wrists on top of each other: ambiguous
    d.settle()
    for a in (0.35, 0.7, 1.0, 1.0, 1.0, 0.6, 0.2, 0.0):
        d.step(left=Pose.closing(IDX, a))
    assert d.kb.pressed == []


def test_open_with_a_held_pinch_types_nothing_and_close_clicks_nothing():
    d = Dual()
    d.kb.visible = False
    d.step(right=Pose.closing(IDX, 1.0), n=20, use_left=False)       # desktop: one click
    clicks0 = len(d.mouse.of("click"))
    d.kb.visible = True                                                # keyboard opens mid-pinch
    d.R.aim("u")
    d.step(right=Pose.closing(IDX, 1.0), n=20, use_left=False)
    assert d.kb.pressed == []                                         # the held pinch must re-open first
    d.step(n=6, use_left=False)
    d.step(right=Pose.closing(IDX, 1.0), n=6, use_left=False)
    assert keys(d) == ["u"]
    d.kb.visible = False                                               # closes while pinched
    st = d.step(right=Pose.closing(IDX, 1.0), n=20, use_left=False)
    assert len(d.mouse.of("click")) == clicks0 and d.mouse.held == set()
    # (the synthetic Normal pinch also reads as a fist once its contact is
    # waiting to re-open: Clutch = cursor frozen, never a click)
    assert st.state in ("POINTER", "CLUTCH") and d.kb.dual_calls[-1] is False
    assert d.kb.cursor_log[-1] == []


def test_desktop_pointer_resumes_without_a_jump():
    d = Dual()
    d.kb.visible = False
    d.step(n=25, use_left=False)
    before = d.engine.pointer.cursor
    d.kb.visible = True
    d.R.x += 0.1                                       # hand moves while typing
    d.step(n=20, use_left=False)
    assert d.engine.pointer.cursor == before            # desktop cursor frozen
    d.kb.visible = False
    d.mouse.events.clear()
    for _ in range(15):
        d.R.x += 0.004
        d.step(use_left=False)
    pts = [before] + [(e[1], e[2]) for e in d.mouse.of("move")]
    assert len(pts) > 5 and max(math.dist(a, b) for a, b in zip(pts, pts[1:])) < 30


def test_ily_closes_the_keyboard_from_either_hand_once():
    for side in ("left", "right"):
        d = Dual()
        d.settle()
        d.step(**{side: Pose.ily()}, n=20)
        assert d.commands.count("toggle_keyboard") == 1, side
        d.kb.visible = False
        d.step(**{side: Pose.ily()}, n=30)              # still held after closing: no re-open
        assert d.commands.count("toggle_keyboard") == 1, side


def test_ily_that_opened_the_keyboard_must_be_released_first():
    d = Dual()
    d.kb.visible = False
    d.step(n=12, use_left=False)
    d.step(right=Pose.ily(), n=16, use_left=False)
    assert d.commands.count("toggle_keyboard") == 1
    d.kb.visible = True                                 # the overlay opened
    d.step(right=Pose.ily(), n=40, use_left=False)
    assert d.commands.count("toggle_keyboard") == 1     # held pose: no immediate close


def test_disarm_and_emergency_stop_leave_keyboard_mode():
    for how in ("disarm", "emergency"):
        d = Dual()
        d.settle()
        if how == "disarm":
            d.engine.set_armed(False, d.t)
        else:
            d.engine.emergency_stop()
        assert d.kb.dual_calls[-1] is False and d.kb.cursor_log[-1] == [], how
        d.step(right=Pose.closing(IDX, 1.0), n=10)
        assert d.kb.pressed == [] and d.mouse.events == [], how


def test_no_pinky_drag_or_scroll_on_the_desktop_while_typing():
    d = Dual()
    d.settle()
    pinky = Pose(extended=set(), curls={IDX: 1.0, LIT: 0.0})
    d.step(right=pinky, n=20)
    for _ in range(20):
        d.R.y -= 0.01
        d.step(right=Pose.two_finger())
    assert d.mouse.events == [] and d.engine.state is State.KEYBOARD


def test_dual_mode_off_keeps_the_single_pointer_keyboard():
    st = Settings()
    st.keyboard.dual_hand = False
    d = Dual(st)
    d.step(n=10)
    assert d.engine.state is not State.KEYBOARD and d.kb.dual_calls == []


def test_gesture_move_follows_the_pinching_hand_and_ends_without_a_jump():
    d = Dual()
    d.R.aim("move")
    d.settle()
    d.pinch("right")
    assert keys(d) == ["move"]
    d.R.x += 0.05
    d.step(n=6)
    assert d.kb.moves and d.kb.moves[-1][0] > 0
    pos_before = next(v for v in d.engine.dual.status() if v["side"] == "right")["pos"]
    d.pinch("right")                                   # any key ends the move
    assert keys(d) == ["move", "move"]
    pos_after = next(v for v in d.engine.dual.status() if v["side"] == "right")["pos"]
    assert math.dist(pos_before, pos_after) < 0.05
