"""v1.1.3 Pinky Drag: little finger only (thumb tucked) held 0.2 s = drag, no mode switch."""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import Sim

from airdesk.config import Settings
from airdesk.core.engine import State
from airdesk.core.hand_model import analyze, pinky_pose
from airdesk.core.types import Finger
from airdesk.sim.synthetic import Pose, observation

IDX, MID, RING, LIT, THB = Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.LITTLE, Finger.THUMB


def transition(c: float) -> Pose:
    """Index pointing (c=0) -> pinky only (c=1): the index folds into the
    palm while the little finger rises; thumb tucked throughout."""
    return Pose(extended=set(), curls={IDX: c, LIT: 1.0 - c})


PINKY = transition(1.0)


def pinky_in(s: Sim, frames: int = 4, **kw) -> None:
    for k in range(frames):
        s.step(transition((k + 1) / frames), **kw)


def pinky_out(s: Sim, frames: int = 4, **kw) -> None:
    for k in range(frames):
        s.step(transition(1 - (k + 1) / frames), **kw)


def downs(s: Sim) -> list:
    return [e for e in s.mouse.events if e[0] == "button" and e[2]]


def ups(s: Sim) -> list:
    return [e for e in s.mouse.events if e[0] == "button" and not e[2]]


def test_pose_is_exclusive():
    def feats(pose):
        return analyze(observation(pose, world=True), 16 / 9, 0.5)
    assert pinky_pose(feats(PINKY))
    for other in (Pose.pointing(IDX), Pose(extended={LIT, THB}), Pose.fist(), Pose.ily(), Pose.horns(),
                  Pose.two_finger(), Pose.open_palm(), Pose(extended={IDX}, pinch=IDX)):
        assert not pinky_pose(feats(other)), other


def test_pinky_drag_starts_after_0_2_s_at_the_aimed_point_and_drops_once():
    s = Sim()
    s.settle(n=25)
    aimed = tuple(round(v) for v in s.engine.pointer.cursor)
    s.mouse.events.clear()
    t_pose = None
    t_down = None
    for k in range(16):
        st = s.step(transition(min(1.0, (k + 1) / 4)))
        if t_pose is None and st.drag == "ready":
            t_pose = s.t
        if t_down is None and downs(s):
            t_down = s.t
    assert t_pose is not None and t_down is not None
    assert 0.2 - 1e-6 <= t_down - t_pose <= 0.2 + 1 / 30 + 1e-6, t_down - t_pose
    # The button went down exactly where the index was aiming (no teleport to
    # the pinky, no click first).
    assert len(downs(s)) == 1 and s.mouse.of("click") == []
    start = s.mouse._pos_before(s.mouse.events.index(downs(s)[0]))
    assert math.dist(start, aimed) <= 3, (start, aimed)
    assert s.engine.state is State.DRAGGING
    s.step(PINKY, n=12, dx=0.004)                     # the hand drags the target
    pinky_out(s)
    s.settle(n=6)
    assert len(downs(s)) == 1 and len(ups(s)) == 1
    drags = s.mouse.drags()
    assert len(drags) == 1 and drags[0][0] == "left"
    assert drags[0][2][0] - drags[0][1][0] > 40      # moved right with the hand
    xs = [e[1] for e in s.mouse.of("move")]
    assert max(abs(a - b) for a, b in zip(xs, xs[1:])) < 40   # no jump at start or drop
    assert s.engine.state is State.POINTER


def test_cursor_stays_anchored_during_acquisition():
    s = Sim()
    s.settle(n=25)
    aimed = s.engine.pointer.cursor
    pinky_in(s)
    for _ in range(3):                               # < 0.2 s, still acquiring
        s.step(PINKY)
        assert math.dist(s.engine.pointer.cursor, aimed) <= 3
    assert downs(s) == []


@pytest.mark.parametrize("hold_frames", [0, 1, 3])
def test_pinky_released_before_0_2_s_is_cancelled_with_no_input(hold_frames):
    # (the last two transition frames already show the pose: 2..5 frames of
    # pose = 67..167 ms, all below the 0.2 s threshold)
    s = Sim()
    s.settle(n=25)
    s.mouse.events.clear()
    pinky_in(s)
    s.step(PINKY, n=hold_frames)
    pinky_out(s)
    s.settle(n=10)
    assert s.mouse.of("button") == [] and s.mouse.of("click") == []
    assert s.engine.clicks.latest().phase == "cancelled"


@pytest.mark.parametrize("how", ["tracking", "disarm", "emergency", "other_hand", "physical"])
def test_drag_always_releases_the_button(how):
    s = Sim()
    s.settle(n=25)
    pinky_in(s)
    s.step(PINKY, n=10, dx=0.002)
    assert s.mouse.held == {"left"}
    if how == "tracking":
        s.step(n=12, hand=False)
    elif how == "disarm":
        s.engine.set_armed(False, s.t)
    elif how == "emergency":
        s.engine.emergency_stop()
    elif how == "other_hand":
        s.x += 0.35                                  # a different hand, far away, takes over
        s.step(PINKY, n=2)
    elif how == "physical":
        s.engine.notify_physical_mouse(10, 10, s.t)
    assert s.mouse.held == set() and len(ups(s)) == 1, how


def test_short_tracking_dropout_keeps_the_drag():
    s = Sim()
    s.settle(n=25)
    pinky_in(s)
    s.step(PINKY, n=10, dx=0.002)
    s.step(n=2, hand=False)                          # 67 ms dropout < 0.25 s grace
    s.step(PINKY, n=5, dx=0.002)
    assert s.mouse.held == {"left"} and len(downs(s)) == 1 and ups(s) == []
    s.step(n=12, hand=False)                         # > grace: released
    assert s.mouse.held == set() and len(ups(s)) == 1


def test_long_drag_stays_held_and_drops_once():
    s = Sim()
    s.settle(n=25)
    pinky_in(s)
    rng = np.random.default_rng(5)
    for k in range(240):                             # 8 s, slow wandering, landmark noise
        s.step(PINKY, dx=0.0015 * math.sin(k / 20), dy=0.001 * math.cos(k / 25), noise=0.0012, rng=rng,
               world=True)
    assert s.mouse.held == {"left"} and len(downs(s)) == 1
    pinky_out(s)
    s.settle(n=5)
    assert len(ups(s)) == 1 and s.mouse.held == set()


def test_fast_and_repeated_pinky_drags_with_noise_never_click():
    s = Sim()
    rng = np.random.default_rng(9)
    kw = dict(noise=0.0012, rng=rng, world=True)
    s.step(Pose.pointing(IDX), n=25, **kw)
    s.mouse.events.clear()
    for cycle in range(10):
        pinky_in(s, frames=2 + cycle % 3, **kw)
        s.step(PINKY, n=8, dx=0.003 * (1 if cycle % 2 else -1), **kw)
        pinky_out(s, frames=2 + cycle % 2, **kw)
        s.step(Pose.pointing(IDX), n=10, **kw)
    assert s.mouse.of("click") == []
    assert len(downs(s)) == 10 and len(ups(s)) == 10 and len(s.mouse.drags()) == 10


def test_pinky_pose_never_becomes_clutch_scroll_or_keyboard_toggle():
    s = Sim()
    s.settle(n=25)
    labels, states = set(), set()
    for _ in range(3):
        pinky_in(s, frames=6)
        for _ in range(10):
            st = s.step(PINKY)
            labels.add(st.gesture)
            states.add(st.state)
        pinky_out(s, frames=6)
        s.settle(n=8)
    assert "CLUTCH" not in states and "SCROLLING" not in states
    assert s.commands.count("toggle_keyboard") == 0
    assert not any(lbl.startswith("Keyboard") for lbl in labels)


def test_ordinary_pointing_and_relaxed_hands_never_drag():
    s = Sim()
    rng = np.random.default_rng(2)
    relaxed = Pose(extended={IDX}, curls={MID: 0.55, RING: 0.5, LIT: 0.35})
    for k in range(300):
        s.step(relaxed if k % 100 < 50 else Pose.pointing(IDX), dx=0.002 * math.sin(k / 9),
               noise=0.0015, rng=rng, world=True)
    assert s.mouse.of("button") == [] and s.mouse.of("click") == []


def test_instant_click_events_identical_with_pinky_drag_on_or_off():
    """Pinky Drag must not change a single event (or frame) of the v1.1.2
    Instant Auto-Click path: same pinch sequences, Normal and Pinch Free,
    left and right clicks, with Pinky Drag on and off."""
    def run(pinky: bool, style: str) -> list:
        st = Settings()
        st.gesture.pinky_drag = pinky
        st.gesture.click_style = style
        s = Sim(st)
        rng = np.random.default_rng(4)
        kw = dict(noise=0.001, rng=rng, world=True)
        s.step(Pose.pointing(IDX), n=20, **kw)
        main = IDX if style == "pinch_pointer" else MID
        sec = MID if style == "pinch_pointer" else RING
        for k in range(12):
            f = main if k % 3 else sec
            for a in (0.35, 0.7, 1.0):
                s.step(Pose.closing(f, a), **kw)
            s.step(Pose.closing(f, 1.0), n=2 + k % 3, **kw)
            for a in (0.6, 0.2, 0.0):
                s.step(Pose.closing(f, a), **kw)
            s.step(Pose.pointing(IDX), n=6, dx=0.003, **kw)
        return s.mouse.events
    for style in ("pinch_pointer", "pinch_free"):
        on, off = run(True, style), run(False, style)
        assert on == off, style
        assert len([e for e in on if e[0] == "click"]) == 12, style


def test_little_finger_pointer_keeps_working():
    # Auto: pointing with the little finger, thumb out -> little pointer, no drag.
    s = Sim()
    st = s.step(Pose(extended={LIT, THB}), n=25)
    assert st.finger == "Little"
    s.step(Pose(extended={LIT, THB}), n=20, dx=0.003)
    assert s.mouse.of("button") == []
    # Manual little finger: the pinky-only pose is pointing; Pinky Drag is off.
    cfg = Settings()
    cfg.finger.mode = "manual"
    cfg.finger.finger = "little"
    s2 = Sim(cfg)
    s2.step(PINKY, n=30, dx=0.003)
    assert s2.mouse.of("button") == [] and s2.engine.state is State.POINTER


def test_pinky_drag_can_be_disabled():
    cfg = Settings()
    cfg.gesture.pinky_drag = False
    s = Sim(cfg)
    s.settle(n=25)
    pinky_in(s)
    s.step(PINKY, n=15, dx=0.003)
    assert s.mouse.of("button") == []


def test_feedback_follows_the_drag_lifecycle():
    from airdesk.ui.feedback import feedback_state
    s = Sim()
    s.settle(n=25)
    pinky_in(s)
    seen = []
    for _ in range(10):
        st = s.step(PINKY)
        seen.append(feedback_state(st, s.engine.clicks.latest(), 0.0)[0])
    assert "dragready" in seen and "drag" in seen
    assert seen.index("dragready") < seen.index("drag")
    # Ready never claims a pressed button.
    st = s.step(PINKY)
    assert st.drag == "dragging" and s.mouse.held == {"left"}
    pinky_out(s)
    st = s.settle(n=2)
    it = s.engine.clicks.latest()
    assert it.phase == "completed" and st.drag == "released"
    assert feedback_state(st, it, it.wall_end + 0.1)[0] == "released"
