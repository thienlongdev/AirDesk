"""v1.1.4 Dual-Hand Air Keyboard: independent hand lifecycles, reacquisition
without the other hand, evidence-based identity, independent two-hand
typing, same-frame ordering and the diagnostics that tell a MediaPipe miss
from an association failure.

Every test checks the keys that reached the keyboard sink (and, for
shortcuts, the chord a real `KeyboardOutput` produced), not only internal
state. Tests marked "v1.1.3 defect" fail on the v1.1.3 code."""

from __future__ import annotations

import json

import pytest
from test_v113_dual import IDX, Dual, keys
from test_v113_shortcuts import Rec

from airdesk.core.dual_keyboard import TRACK_GRACE_S
from airdesk.platform.keyboard_output import VK, KeyboardOutput
from airdesk.sim.synthetic import Pose, frame

CYCLE = (0.35, 0.7, 1.0, 1.0, 1.0, 0.6, 0.2, 0.0)


class Rig(Dual):
    """`Dual` with per-hand presence and MediaPipe label control."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.labels = {"left": None, "right": None}       # None = correct label

    def obs(self, side, pose=None):
        h = self.L if side == "left" else self.R
        o = h.obs(pose)
        if self.labels[side] is not None:
            o.handedness = self.labels[side]
        return o

    def go(self, left=None, right=None, n=1, show=("left", "right"), order=("left", "right")):
        st = None
        for _ in range(n):
            self.t += 1 / 30
            poses = {"left": left, "right": right}
            hs = [self.obs(s, poses[s]) for s in order if s in show]
            st = self.engine.process(frame(self.t, *hs))
        return st

    def type(self, side, key, show=("left", "right"), settle=8):
        (self.L if side == "left" else self.R).aim(key)
        self.go(n=settle, show=show)
        for a in CYCLE:
            self.go(**{side: Pose.closing(IDX, a)}, show=show)
        self.go(n=3, show=show)

    def side_status(self, side):
        return next(v for v in self.engine.dual.status() if v["side"] == side)


def hands_of(d):
    return [h for _k, h in d.kb.pressed]


# ------------------------------------------------------------ reacquisition
@pytest.mark.parametrize("label_mode", ["correct", "wrong_first_frames", "wrong_first_6_frames"])
@pytest.mark.parametrize("side", ["right", "left"])
def test_hand_returns_alone_and_types_without_the_other_hand(side, label_mode):
    """Owner's P0 (v1.1.3 defect for the wrong-label cases): one hand types,
    leaves the camera, comes back alone and must get its own controller and
    cursor again - the other hand is never shown."""
    d = Rig()
    only = (side,)
    idx = 0 if side == "left" else 1
    # Right hand typing left-half keys (and the reverse): the return happens
    # on the "wrong" half of the image, so position cannot carry it.
    first, second = ("ploki", "jumpy") if side == "left" else ("sweat", "grade")
    for ch in first:
        d.type(side, ch, show=only)
    assert keys(d) == list(first) and set(hands_of(d)) == {idx}
    d.go(n=45, show=())                                   # 1.5 s out of view
    assert d.side_status(side)["life"] == "retired" and not d.side_status(side)["tracked"]
    if label_mode != "correct":
        # A hand re-entering at the image edge: MediaPipe's label is wrong at first.
        d.labels[side] = "Right" if side == "left" else "Left"
        d.go(n=3 if label_mode == "wrong_first_frames" else 6, show=only)
        d.labels[side] = None
    d.go(n=6, show=only)
    st = d.side_status(side)
    assert st["tracked"] and st["detected"] and st["reacqs"] == 1
    assert d.side_status("left" if side == "right" else "right")["reacqs"] == 0
    assert d.kb.cursor_log[-1] and d.kb.cursor_log[-1][0]["hand"] == idx      # its own cursor is back
    assert not d.side_status("left" if side == "right" else "right")["tracked"]
    for ch in second:
        d.type(side, ch, show=only)
    assert keys(d) == list(first + second)
    assert set(hands_of(d)) == {idx}                       # every key from the same controller


def test_reacquisition_is_ready_on_the_first_frame_and_measured():
    d = Rig()
    d.type("right", "k", show=("right",))
    d.go(n=30, show=())
    d.R.aim("j")                                                        # returns pointing at j
    d.go(show=("right",))
    st = d.side_status("right")
    assert st["reacq_frames"] == 1 and st["reacq_ms"] == 0.0          # open fingers: typing allowed at once
    assert st["life"] in ("reacquiring", "tracking")
    d.type("right", "j", show=("right",), settle=1)
    assert keys(d) == ["k", "j"]


def test_returning_hand_with_a_closed_pinch_types_nothing_until_it_opens():
    d = Rig()
    d.type("right", "k", show=("right",))
    d.go(n=30, show=())
    d.R.aim("p")
    d.go(right=Pose.closing(IDX, 1.0), n=15, show=("right",))           # comes back already pinched
    assert keys(d) == ["k"]
    assert d.engine.dual.counters["stale_press_prevented"] == 1
    d.go(right=Pose.closing(IDX, 0.0), n=3, show=("right",))
    for a in CYCLE:
        d.go(right=Pose.closing(IDX, a), show=("right",))
    assert keys(d) == ["k", "p"]


def test_brief_dropout_keeps_the_track_and_the_held_pinch():
    """MediaPipe misses the hand for 100 ms while it holds a pinch: same
    track, no second key, the next pinch types normally."""
    d = Rig()
    d.R.aim("k")
    d.go(n=8)
    for a in (0.35, 0.7, 1.0, 1.0):
        d.go(right=Pose.closing(IDX, a))
    tid = d.side_status("right")["track"]
    d.go(right=Pose.closing(IDX, 1.0), n=3, show=("left",))
    assert d.side_status("right")["life"] == "temporarily lost"
    d.go(right=Pose.closing(IDX, 1.0), n=5)
    assert d.side_status("right")["track"] == tid and keys(d) == ["k"]
    for a in (0.6, 0.2, 0.0) + CYCLE:
        d.go(right=Pose.closing(IDX, a))
    assert keys(d) == ["k", "k"]


def test_a_hand_lost_longer_than_the_grace_starts_a_new_track():
    d = Rig()
    d.go(n=5)
    tid = d.side_status("left")["track"]
    d.go(n=int(TRACK_GRACE_S * 30) + 3, show=("right",))
    assert d.side_status("left")["life"] == "retired"
    d.go(n=2)
    assert d.side_status("left")["track"] != tid and d.side_status("left")["reacqs"] == 1


@pytest.mark.parametrize("leaver", ["right", "left"])
def test_one_hand_leaves_and_returns_while_the_other_keeps_typing(leaver):
    d = Rig()
    stay = "left" if leaver == "right" else "right"
    d.type(stay, "a" if stay == "left" else "l")
    d.type(leaver, "s" if leaver == "left" else "k")
    d.type(stay, "d" if stay == "left" else "o", show=(stay,))       # the other hand is out of view
    d.go(n=20, show=(stay,))
    d.type(stay, "f" if stay == "left" else "i", show=(stay,))
    d.type(leaver, "w" if leaver == "left" else "j")                 # back, typing at once
    d.type(stay, "e" if stay == "left" else "u")
    exp_l = ["a", "d", "f", "e"] if stay == "left" else ["s", "w"]
    exp_r = ["l", "o", "i", "u"] if stay == "right" else ["k", "j"]
    assert [k for k, h in d.kb.pressed if h == 0] == exp_l
    assert [k for k, h in d.kb.pressed if h == 1] == exp_r


def test_both_hands_leave_and_reenter_together():
    d = Rig()
    d.type("left", "a")
    d.type("right", "l")
    d.go(n=40, show=())
    d.go(n=2, order=("right", "left"))                                  # both back in the same frame
    assert d.side_status("left")["tracked"] and d.side_status("right")["tracked"]
    d.type("left", "q")
    d.type("right", "p")
    assert d.kb.pressed == [("a", 0), ("l", 1), ("q", 0), ("p", 1)]


# ---------------------------------------------------------------- identity
def test_inverted_labels_pair_follows_left_right_order_and_learns_polarity():
    d = Rig()
    d.labels = {"left": "Right", "right": "Left"}                      # MediaPipe convention inverted
    d.go(n=40)
    st = {v["side"]: v for v in d.engine.dual.status()}
    assert st["left"]["pos"][0] < st["right"]["pos"][0]
    assert d.engine.dual.polarity == -1
    d.type("left", "a")
    d.type("right", "l")
    assert d.kb.pressed == [("a", 0), ("l", 1)]
    # A lone return far from where it was lost: the learned polarity reads
    # the inverted "Left" label as the right hand.
    d.go(n=40, show=("left",))
    d.R.x = 0.8
    d.go(n=3, show=("left",))
    d.go(n=2)
    assert d.side_status("right")["tracked"]
    d.type("right", "p")
    assert d.kb.pressed[-1] == ("p", 1)


def test_the_other_hand_entering_where_a_hand_was_lost_keeps_its_own_label():
    """The right hand types on the left half and leaves; the LEFT hand then
    comes in at that very spot with a confident label: it is the left hand."""
    d = Rig()
    for ch in "sweat":
        d.type("right", ch, show=("right",))
    d.go(n=36, show=())
    d.L.x, d.L.y = d.R.x, d.R.y
    d.go(n=2, show=("left",))
    assert d.side_status("left")["tracked"] and not d.side_status("right")["tracked"]
    for ch in "grade":
        d.type("left", ch, show=("left",))
    assert [h for _k, h in d.kb.pressed[-5:]] == [0] * 5


def test_lone_hand_with_the_wrong_side_is_corrected_when_the_other_hand_appears():
    """v1.1.3 defect: a lone right hand given the left side (inverted label,
    nothing remembered) kept it forever."""
    d = Rig()
    d.labels = {"left": "Right", "right": "Left"}
    d.go(n=10, show=("right",))
    d.go(n=10)                                                           # the left hand joins, clearly on the left
    st = {v["side"]: v for v in d.engine.dual.status()}
    assert st["left"]["pos"][0] < st["right"]["pos"][0]
    assert d.engine.dual.counters["identity_corrections"] >= 1
    d.type("right", "o")
    d.type("left", "w")
    assert d.kb.pressed == [("o", 1), ("w", 0)]


def test_mirror_off_uses_the_users_left_right_order():
    d = Rig()
    d.s.camera.mirror = False
    # Raw camera image: the user's right hand appears on the image LEFT; the
    # tracker already delivers the user's anatomical labels.
    d.L.x, d.R.x = 0.64, 0.36
    d.go(n=10, order=("right", "left"))
    st = {v["side"]: v for v in d.engine.dual.status()}
    assert st["right"]["pos"][0] < st["left"]["pos"][0]
    d.type("right", "q")
    d.type("left", "p")
    assert d.kb.pressed == [("q", 1), ("p", 0)]


def test_touching_hands_block_only_new_keys_briefly_then_both_type():
    d = Rig()
    d.L.aim("g")
    d.R.aim("h")
    d.R.x, d.R.y = d.L.x + 0.002, d.L.y
    d.go(n=10)
    assert d.engine.dual.counters["ambiguous_updates"] > 0
    d.R.aim("k")
    d.L.aim("a")                                                         # apart again
    d.go(n=6)                                                            # 0.2 s
    d.type("left", "a", settle=0)
    d.type("right", "k", settle=0)
    assert keys(d) == ["a", "k"]


# ------------------------------------------------------- independent input
@pytest.mark.parametrize("holder", ["left", "right"])
def test_a_held_pinch_never_blocks_the_other_hand(holder):
    d = Rig()
    typer = "right" if holder == "left" else "left"
    hk, word = ("x", "ok") if holder == "left" else ("p", "as")
    (d.L if holder == "left" else d.R).aim(hk)
    d.go(n=8)
    for a in (0.35, 0.7, 1.0):
        d.go(**{holder: Pose.closing(IDX, a)})
    for ch in word + word[0]:
        (d.R if typer == "right" else d.L).aim(ch)
        d.go(**{holder: Pose.closing(IDX, 1.0)}, n=4)
        for a in CYCLE:
            d.go(**{holder: Pose.closing(IDX, 1.0), typer: Pose.closing(IDX, a)})
    assert keys(d) == [hk] + list(word + word[0])                       # typed while the other held
    for a in (0.6, 0.2, 0.0):
        d.go(**{holder: Pose.closing(IDX, a)})
    d.go(n=3)
    for a in CYCLE:                                                      # both rearm independently
        d.go(left=Pose.closing(IDX, a), right=Pose.closing(IDX, a))
    assert sorted(keys(d)[-2:]) == sorted([hk, word[0]])


def test_same_frame_keys_both_arrive_in_a_deterministic_order():
    for _ in range(2):
        d = Rig()
        d.L.aim("f")
        d.R.aim("j")
        d.go(n=8)
        for a in CYCLE:
            d.go(left=Pose.closing(IDX, a), right=Pose.closing(IDX, a), order=("right", "left"))
        assert d.kb.pressed == [("f", 0), ("j", 1)]
        assert d.engine.dual.counters["simultaneous_pairs"] == 1


@pytest.mark.parametrize("mod_side", ["left", "right"])
def test_modifier_pinched_in_the_same_frame_goes_first_and_makes_a_real_chord(mod_side):
    """v1.1.3 defect (mod_side=right): the left 'c' went out before the
    right Ctrl in the same frame, so Ctrl+C became a plain 'c'."""
    d = Rig()
    rec = Rec()
    out = KeyboardOutput(rec, one_shot=True)
    d.kb.press_key = lambda key, hand=0: (d.kb.pressed.append((key, hand)), out.press(key))[-1]
    key_side = "left" if mod_side == "right" else "right"
    (d.L if mod_side == "left" else d.R).aim("ctrl")
    (d.L if key_side == "left" else d.R).aim("c")
    d.go(n=8)
    for a in CYCLE:
        d.go(left=Pose.closing(IDX, a), right=Pose.closing(IDX, a))
    assert keys(d) == ["ctrl", "c"]
    assert rec.calls == [("tap", ord("C"), (VK["ctrl"],))] and out.mods.armed() == []


def test_rapid_overlapping_two_hand_typing_loses_and_duplicates_nothing():
    d = Rig()
    left, right = "asdfg", "hjklp"
    seq = list(CYCLE) + [0.0, 0.0]
    for lk, rk in zip(left, right):
        d.L.aim(lk)
        d.R.aim(rk)
        d.go(n=2)
        for i in range(len(seq) + 3):                                   # right lags 3 frames (overlapping)
            la = seq[i] if i < len(seq) else 0.0
            ra = seq[i - 3] if 3 <= i < len(seq) + 3 else 0.0
            d.go(left=Pose.closing(IDX, la), right=Pose.closing(IDX, ra))
    assert [k for k, h in d.kb.pressed if h == 0] == list(left)
    assert [k for k, h in d.kb.pressed if h == 1] == list(right)
    assert len(d.kb.pressed) == 10


def test_hand_disappearing_during_the_other_hands_key_loses_nothing():
    d = Rig()
    rec = Rec()
    out = KeyboardOutput(rec, one_shot=True)
    d.kb.press_key = lambda key, hand=0: (d.kb.pressed.append((key, hand)), out.press(key))[-1]
    d.type("left", "ctrl")
    d.R.aim("c")
    d.go(n=8)
    for k, a in enumerate(CYCLE):                                        # the left hand vanishes mid-pinch
        d.go(right=Pose.closing(IDX, a), show=("right",) if k >= 2 else ("left", "right"))
    assert keys(d) == ["ctrl", "c"] and rec.calls[-1] == ("tap", ord("C"), (VK["ctrl"],))
    d.type("right", "v", show=("right",))
    assert keys(d)[-1] == "v" and rec.calls[-1] == ("text", "v")         # one-shot Ctrl cleared, right still typing


def test_a_single_lost_hand_does_not_clear_the_other_hands_armed_modifier():
    d = Rig()
    cleared = d.kb.cleared
    d.type("right", "ctrl")
    d.go(n=30, show=("right",))                                          # left hand retired
    assert d.kb.cleared == cleared


# ------------------------------------------------------------- diagnostics
def test_diagnostics_tell_a_mediapipe_miss_from_an_association(tmp_path, monkeypatch):
    monkeypatch.setenv("AIRDESK_HOME", str(tmp_path))
    from airdesk.ui.pages import _hand_row, _identity_text, export_gesture_log
    d = Rig()
    d.type("right", "k", show=("right",))
    d.go(n=30, show=())
    st = d.go(n=1, show=())
    row = {v["side"]: v for v in st.vhands}
    assert not row["right"]["detected"] and row["right"]["life"] == "retired"
    assert st.dual_diag["mp_hands"] == 0                                  # MediaPipe reported nothing
    assert "MediaPipe: no" in _hand_row(row["right"])
    st = d.go(n=1, show=("right",))
    row = {v["side"]: v for v in st.vhands}
    assert row["right"]["detected"] and st.dual_diag["obs"][0]["to"] == "right"
    assert "-> right" in _identity_text(st) and "MediaPipe: yes" in _hand_row(row["right"])
    events = [e["event"] for e in st.dual_diag["events"]]
    assert events[-2:] == ["retired", "reacquiring"]
    data = json.loads(open(export_gesture_log(d.engine, d.s, "test"), encoding="utf-8").read())
    dk = data["dual_keyboard"]
    assert dk["counters"]["reacquired"] == 1 and dk["hands"][1]["reacqs"] == 1
    assert "last_key" not in json.dumps(dk) and '"k"' not in json.dumps(dk)   # no key ids / text


def test_owner_acceptance_v114_window_builds(monkeypatch):
    from types import SimpleNamespace

    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from airdesk.ui.benchmark import OwnerAcceptanceV114
    from airdesk.ui.keyboard_overlay import KeyboardOverlay
    from airdesk.config import KeyboardConfig, Settings
    d = Rig()
    ov = KeyboardOverlay(KeyboardConfig(), output=KeyboardOutput(Rec()))     # never real SendInput
    ctx = SimpleNamespace(s=Settings(), overlay=ov, runtime=SimpleNamespace(engine=d.engine))
    w = OwnerAcceptanceV114(ctx)
    assert len(w.STAGES) == 12 and w.name == "owner_acceptance_v114"
    ov.press_key("a", 1)
    assert ov.queue_depth == 1
    QApplication.processEvents()
    assert ov.queue_depth == 0 and ov.results[-1]["hand"] in ("left", "right", "pointer")
    w.close()
    ov.close()


def test_a_hand_darting_next_to_the_pinching_hand_neither_loses_nor_moves_its_key():
    """v1.1.3 defect: the other hand's quick move made the association
    'ambiguous', which blocked BOTH hands for 0.25 s (and disabled the aim
    hold) - the pinching hand's key was lost (or typed a row lower). Now
    the key is deferred and typed at the aimed key once resolved."""
    import numpy as np
    d = Rig()
    rng = np.random.default_rng(3)
    for _ in range(4):
        d.L.aim("d")
        d.R.aim("k")
        d.go(n=10)
        x0 = d.R.x
        n0 = len(d.kb.pressed)
        for f in range(13):
            if f < 2:
                d.R.x = x0 - 0.14 * (0.6, 1.0)[f]
            a = CYCLE[f - 3] if 3 <= f < 3 + len(CYCLE) else 0.0
            d.t += 1 / 30
            d.engine.process(frame(d.t, d.L.obs(Pose.closing(IDX, a), noise=0.001, rng=rng),
                                   d.R.obs(noise=0.001, rng=rng)))
        assert d.kb.pressed[n0:] == [("d", 0)]
        d.R.x = x0
        d.go(n=6)
    assert d.engine.dual.counters["ambiguous_updates"] > 0


def test_pinch_that_ends_while_hands_stay_ambiguous_types_nothing():
    d = Rig()
    d.L.aim("g")
    d.R.x, d.R.y = d.L.x + 0.002, d.L.y
    d.go(n=10)
    for a in CYCLE:
        d.go(left=Pose.closing(IDX, a))
    assert d.kb.pressed == [] and d.engine.dual.counters["uncertain_blocked"] >= 1
