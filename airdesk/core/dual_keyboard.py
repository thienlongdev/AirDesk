"""Dual-Hand Air Keyboard (v1.1.3, identity and lifecycles reworked in v1.1.4).

While the Air Keyboard is open (and ``keyboard.dual_hand`` is on) the
interaction engine hands every camera frame to `DualKeyboard` instead of
the desktop pointer. Nothing here touches the Windows mouse: the system
cursor stays where it was, no button or wheel event is injected, and each
hand drives its own *virtual* cursor, hit-tested against the keyboard's
own key rectangles (fractions of the keyboard surface, the same ones that
are painted).

Two controllers, ``left`` and ``right`` (the user's anatomical hands), each
own everything about their hand: track id, lifecycle, landmark filter,
cursor, its own unchanged v1.1.2 `GestureRecognizer` (pinch confirmation,
quick taps, adaptive release, rearm), aim hold, pressed key and counters.
No state is shared between them, so a pinch held by one hand never delays,
blocks or rearms the other.

Lifecycle (per controller, v1.1.4)::

    detected / reacquiring -> tracking -> temporarily lost -> (back) tracking
                                                \\-> retired (after 0.3 s) -> reacquiring ...

* **association** - observations are matched to the live controllers by
  predicted wrist position (constant velocity, gate grows while a hand is
  briefly missing). A hand that is not matched is *temporarily lost* (cursor
  dimmed, pinch latch kept) for 0.3 s, then *retired*: its state is dropped
  and its last position is remembered for reacquisition.
* **identity of a new track** - never a single MediaPipe label. Evidence:
  MediaPipe handedness (score-weighted, with a polarity learned from
  well-separated two-hand frames, because the label convention depends on
  the image that reaches the model), where each hand was last lost
  (memory), the side of the image (user's perspective, mirroring-aware),
  and - when both hands are present - their left/right order. A lone new
  track stays *provisional* (its colour may still be corrected while it
  is idle) until the evidence is confident or 1 s passed; a forming pair is
  re-checked for 0.5 s. The other hand is never needed for reacquisition.
* **ambiguity** - only a genuinely ambiguous association (hands touching
  or crossing, costs within the swap margin) blocks *new* key starts, on
  the hands involved, for 0.15 s. Identities are never swapped by labels
  mid-gesture.
* **mapping** - the index fingertip (blended toward its knuckle by
  ``pointer.anchor_blend``, filtered with the desktop pointer's filter
  settings) is mapped through the user's calibrated control box onto the
  keyboard surface, with a small margin so the edge keys stay reachable.
  The drawn fingertip of the hand cursor *is* the hit-test point.
* **pinch** - the confirmed primary contact activates the key under the
  fingertip; a light per-hand aim hold keeps the cursor on the key while
  the fingers close. A fresh track must open a contact that is already
  closed when it appears (no stale press after a return).
* **🤟** - the keyboard toggle still works on either hand (closes it).

Activations of both hands in one frame are ordered deterministically -
modifiers first, then by threshold crossing, then left before right - and
go out through the single ordered keyboard dispatch path
(`KeyboardSink.press_key` -> queued to the overlay's GUI thread ->
`KeyboardOutput` -> SendInput). Both hands landing on the same key within
80 ms is one activation.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .filters import PointFilter, make_filter
from .gestures import GestureRecognizer
from .keyboard_layout import MODIFIERS
from .types import Finger, HandFeatures, HandObservation

if TYPE_CHECKING:  # pragma: no cover
    from .engine import InteractionEngine

SIDES = ("left", "right")
HAND_INDEX = {"left": 0, "right": 1}
OTHER = {"left": "right", "right": "left"}
SIGN = {"left": -1.0, "right": 1.0}
TRACK_GRACE_S = 0.3       # a hand missing this long is retired (its cursor disappears)
MATCH_RADIUS = 0.25       # wrist continuity radius (normalized image units)
GATE_GROWTH = 1.0         # ... grows while a hand is briefly missing (image units / s)
SWAP_MARGIN = 0.012       # assignment cost gap (beyond 1.5x the better cost) below which identity is ambiguous
UNCERTAIN_S = 0.15        # no new key on an ambiguously associated hand for this long
VEL_ALPHA = 0.5           # wrist velocity smoothing (prediction)
EDGE = 0.04               # control box maps onto the keyboard plus this margin
REPEAT_KEYS = {"backspace", "del", "left", "right", "up", "down"}
REPEAT_S = 0.09
SAME_KEY_S = 0.08         # both hands on the same key within this = one activation
PRESS_FLASH_S = 0.18
BOTH_LOST_CLEAR_S = 1.0   # both hands gone this long: armed modifiers are cleared

# Identity evidence (positive = the user's right hand).
# A confident label (|0.9|) outweighs the memory (<= 0.8); the memory carries the
# first, unreliable frames of a hand re-entering at the image edge.
W_LABEL, W_MEMORY, W_POSITION, W_ORDER = 1.0, 0.8, 0.5, 3.0
PAIR_SEP = 0.06           # user-x separation from which the left/right order counts
MEMORY_RADIUS = 0.25      # a hand reappearing near where a side was lost ...
MEMORY_TAU_S = 20.0       # ... is probably that side (fades over ~20 s)
LABEL_TAU_S = 0.3         # per-track handedness evidence smoothing
CONFIDENT = 0.6           # identity evidence that ends the provisional phase
RELABEL_MIN = 0.3         # opposite evidence that corrects a provisional identity
PROVISIONAL_MAX_S = 1.0
PAIR_CHECK_S = 0.5        # a forming pair is re-checked this long
SWAP_FIX = 1.0            # pair evidence against the current sides that corrects them
POLARITY_MIN = 30.0       # label/geometry samples before the label polarity is learned
POLARITY_SEP = 0.12       # user-x separation of a pair used as a polarity sample


@dataclass
class VirtualHand:
    side: str
    rec: GestureRecognizer
    filt: PointFilter
    wrist: tuple[float, float] | None = None
    vel: tuple[float, float] = (0.0, 0.0)        # smoothed wrist velocity (image units / s) for prediction
    knuckle: tuple[float, float] | None = None
    seen: float = -1e9
    pos: tuple[float, float] | None = None       # shown / hit-tested fingertip (keyboard fraction)
    raw: tuple[float, float] | None = None       # mapped point before the continuity offset
    offset: list[float] = field(default_factory=lambda: [0.0, 0.0])
    history: deque = field(default_factory=lambda: deque(maxlen=24))
    raw_hist: deque = field(default_factory=lambda: deque(maxlen=24))
    closing_t: float | None = None               # when the current pinch began closing (rewound)
    hold: tuple[float, float] | None = None      # aim hold / press point
    aim_t0: float = 0.0
    aim_best: float = math.inf
    aim_best_t: float = 0.0
    aim_knuckle: tuple[float, float] = (0.0, 0.0)
    aim_scale: float = 1.0
    aim_latched: bool = False
    key: str | None = None                       # key under the fingertip
    pinch: bool = False                          # primary/secondary contact engaged
    pressed: str | None = None                   # key this pinch activated
    deferred: tuple[str, float] | None = None    # (key, crossing) confirmed while the association was ambiguous
    next_repeat: float = 0.0
    release_t: float = -1e9
    flash_t: float = -1e9
    uncertain_until: float = -1e9
    ily_t: float | None = None
    ily_block: bool = False
    state: str = "lost"
    last_key: str = ""
    keys: int = 0
    # v1.1.4: track lifecycle and identity evidence
    track_id: int = 0
    life: str = "absent"     # absent | detected | reacquiring | tracking | temporarily lost | retired
    born: float = -1e9
    frames: int = 0
    detected: bool = False                       # an observation was assigned this frame
    obs_label: str = ""
    obs_score: float = 0.0
    lab_ev: float = 0.0                          # smoothed label evidence (+ = right), polarity applied
    mem_ev: float = 0.0                          # memory evidence at acquisition
    identity: float = 0.0                        # last identity evidence (+ = right)
    provisional: bool = False
    why: str = ""
    ready_t: float | None = None                 # first frame this track could type
    reacqs: int = 0
    reacq_ms: float | None = None                # last reacquisition: first observation -> ready to type
    reacq_frames: int | None = None
    rec_ms: deque = field(default_factory=lambda: deque(maxlen=20))   # contact -> activation per key

    @property
    def active(self) -> bool:
        return self.pos is not None

    @property
    def idle(self) -> bool:
        """No pinch, aim hold or 🤟 in progress: the identity may be corrected."""
        return not self.pinch and self.hold is None and self.ily_t is None


class DualKeyboard:
    def __init__(self, engine: "InteractionEngine") -> None:
        self.e = engine
        self.hands = {s: self._new_hand(s) for s in SIDES}
        self._move: tuple[str, float, float, tuple[float, float]] | None = None
        self._last_act: dict[str, tuple[float, str]] = {}
        self._lost_cleared = False
        self._last_pub: list | None = None
        self._memory: dict[str, tuple[tuple[float, float], float]] = {}   # side -> (wrist, last seen)
        self._next_id = 1
        self._pair_check_until = -1e9
        self._ambiguous = False
        self._t = 0.0
        # Learned handedness-label polarity (+1 MediaPipe label as delivered,
        # -1 inverted, 0 uninformative): weighted agree / total samples.
        self._lab_agree = 0.0
        self._lab_n = 0.0
        self.polarity = 1
        self.frame_diag: dict = {"mp_hands": 0, "obs": [], "assigned": 0}
        self.events: deque[dict] = deque(maxlen=40)
        self.counters = {"keys": 0, "duplicates_suppressed": 0, "uncertain_blocked": 0, "misses": 0,
                         "ambiguous_updates": 0, "new_tracks": 0, "reacquired": 0,
                         "identity_corrections": 0, "stale_press_prevented": 0, "simultaneous_pairs": 0}

    def _new_hand(self, side: str) -> VirtualHand:
        s = self.e.s
        p = s.pointer
        return VirtualHand(side, GestureRecognizer(s.gesture, s.finger),
                           make_filter(p.filter, min_cutoff=p.min_cutoff, beta=p.beta))

    # ------------------------------------------------------------ lifecycle
    def start(self, t: float) -> None:
        self.hands = {s: self._new_hand(s) for s in SIDES}
        self._move = None
        self._last_act.clear()
        self._lost_cleared = False
        self._last_pub = None
        self._memory.clear()
        self._pair_check_until = -1e9
        self._set_dual(True)

    def stop(self, reason: str = "") -> None:
        """Keyboard closed / disarmed: cursors removed, pinches cancelled,
        armed modifiers cleared. Nothing is ever held down (every key is a
        complete tap), so there is no key state to release."""
        for h in self.hands.values():
            self._drop(h)
            h.life = "absent"
        self._move = None
        kb = self.e.keyboard
        self._publish(0.0, force=True)
        for i in (0, 1):
            kb.hover(i, None)
        clear = getattr(kb, "clear_modifiers", None)
        if clear is not None:
            clear()
        self._set_dual(False)

    def _set_dual(self, on: bool) -> None:
        f = getattr(self.e.keyboard, "set_dual", None)
        if f is not None:
            f(on)

    # --------------------------------------------------------------- frame
    def update(self, hands: list[HandObservation], feats: list[HandFeatures], t: float,
               aspect: float) -> str:
        self._t = t
        assign, fresh = self._associate(hands, t)
        self.frame_diag = {
            "mp_hands": len(hands), "assigned": len(assign),
            "obs": [{"label": o.handedness, "score": round(o.score, 2),
                     "to": next((s for s, i in assign.items() if i == k), "-")} for k, o in enumerate(hands)]}
        acts: list[tuple] = []
        for side in SIDES:
            h = self.hands[side]
            i = assign.get(side)
            if i is None:
                h.detected = False
                if h.wrist is not None:
                    if t - h.seen > TRACK_GRACE_S:
                        self._retire(h)
                    else:
                        if h.life != "temporarily lost":
                            self._event(t, side, "temporarily lost", "")
                        h.life, h.state = "temporarily lost", "lost"
                continue
            obs = hands[i]
            if side in fresh:
                self._fresh(h, obs, t, fresh[side])
            elif h.life == "temporarily lost":
                self._event(t, side, "resumed", f"after {1000 * (t - h.seen):.0f} ms (same track #{h.track_id})")
                h.life = "tracking" if not h.provisional else ("reacquiring" if h.reacqs else "detected")
            if h.wrist is not None and t > h.seen:
                dt = t - h.seen
                v = ((obs.wrist[0] - h.wrist[0]) / dt, (obs.wrist[1] - h.wrist[1]) / dt)
                h.vel = (h.vel[0] + VEL_ALPHA * (v[0] - h.vel[0]), h.vel[1] + VEL_ALPHA * (v[1] - h.vel[1]))
                a = dt / (dt + LABEL_TAU_S)
                h.lab_ev += a * (self._label_term(obs) - h.lab_ev)
            h.wrist, h.seen = obs.wrist, t
            h.detected, h.obs_label, h.obs_score = True, obs.handedness, obs.score
            h.frames += 1
            act = self._hand(h, obs, feats[i], t, aspect)
            if act is not None:
                acts.append(act)
            self._readiness(h, t)
        self._identity_review(hands, assign, t)
        if assign:
            self._lost_cleared = False
        elif (not self._lost_cleared
              and t - max((h.seen for h in self.hands.values()), default=-1e9) >= BOTH_LOST_CLEAR_S):
            # No hand seen for 1 s: never leave Ctrl armed for hands that are gone.
            self._lost_cleared = True
            clear = getattr(self.e.keyboard, "clear_modifiers", None)
            if clear is not None:
                clear()
        # Same frame, both hands: modifiers first (Ctrl + C pinched together is
        # always Ctrl+C), then the earlier threshold crossing, then left.
        if len(acts) > 1:
            self.counters["simultaneous_pairs"] += 1
        for _prio, tc, _o, side, key, new in sorted(acts):
            self._activate(side, key, t, tc if new else None)
        self._publish(t)
        return self._label()

    def _fresh(self, h: VirtualHand, obs: HandObservation, t: float, why: str) -> None:
        """New track (a hand appearing, returning, or a jump too large to be
        the same hand): no history, and a pinch or 🤟 already held at this
        moment must be released before it can type or toggle."""
        h.rec.reset()
        h.rec.require_open()
        h.filt.reset()
        h.history.clear()
        h.raw_hist.clear()
        h.offset = [0.0, 0.0]
        h.raw = h.pos = h.hold = h.knuckle = h.wrist = None
        h.vel = (0.0, 0.0)
        h.pinch, h.pressed, h.aim_latched, h.deferred = False, None, False, None
        h.ily_t, h.ily_block = None, True       # a held 🤟 (the one that opened the keyboard) must re-form
        h.key = None
        h.uncertain_until = -1e9
        if self._move is not None and self._move[0] == h.side:
            self._move = None
        returning = h.side in self._memory
        h.mem_ev = self._mem_term(obs.wrist, t)
        h.track_id, self._next_id = self._next_id, self._next_id + 1
        h.born, h.frames, h.ready_t = t, 0, None
        h.lab_ev = self._label_term(obs)
        h.provisional, h.why = True, why
        h.life = "reacquiring" if returning else "detected"
        self.counters["new_tracks"] += 1
        if returning:
            h.reacqs += 1
            self.counters["reacquired"] += 1
        self._event(t, h.side, h.life, f"track #{h.track_id}: {why}; MediaPipe {obs.handedness} {obs.score:.2f}")

    def _drop(self, h: VirtualHand) -> None:
        h.rec.reset()
        h.filt.reset()
        h.history.clear()
        h.pos = h.raw = h.hold = h.wrist = None
        h.pinch, h.pressed, h.key, h.deferred = False, None, None, None
        h.ily_t = None
        h.detected = False
        h.state = "lost"

    def _retire(self, h: VirtualHand) -> None:
        """Missing longer than the grace period: the track ends. Its last
        position is remembered (reacquisition evidence); a key is never
        held (every key is a complete tap) and armed modifiers stay armed
        for the other hand."""
        self._memory[h.side] = (h.wrist, h.seen)
        self._event(h.seen + TRACK_GRACE_S, h.side, "retired",
                    f"track #{h.track_id} last seen at ({h.wrist[0]:.2f}, {h.wrist[1]:.2f})")
        if self._move is not None and self._move[0] == h.side:
            self._move = None
        self._drop(h)
        h.life, h.provisional = "retired", False

    def _readiness(self, h: VirtualHand, t: float) -> None:
        """Reacquisition latency: first observation of the track -> it can
        type (identity usable, no stale contact left to re-open)."""
        if h.ready_t is not None:
            return
        primary = h.rec._mapping.get("primary")
        if t < h.uncertain_until or (primary is not None and primary in h.rec._need_open):
            if h.frames == 1 and primary in h.rec._need_open:
                self.counters["stale_press_prevented"] += 1
                self._event(t, h.side, "stale press prevented", "contact already closed on arrival: open first")
            return
        h.ready_t = t
        if h.life == "reacquiring":
            h.reacq_ms, h.reacq_frames = round(1000 * (t - h.born), 1), h.frames
        if not h.provisional:
            h.life = "tracking"

    # ------------------------------------------------------------ identity
    def _label_term(self, o: HandObservation) -> float:
        """MediaPipe handedness as evidence (+ = right), score-weighted, with
        the learned polarity applied."""
        conf = min(max(2.0 * o.score - 1.0, 0.0), 1.0)
        return (1.0 if o.handedness == "Right" else -1.0) * conf * self.polarity

    def _user_x(self, w: tuple[float, float]) -> float:
        """Horizontal position from the user's point of view (0 = their left)."""
        return w[0] if self.e.s.camera.mirror else 1.0 - w[0]

    def _pos_term(self, w: tuple[float, float]) -> float:
        return min(max((self._user_x(w) - 0.5) * 2.5, -1.0), 1.0)

    def _mem_term(self, w: tuple[float, float], t: float, sides=SIDES) -> float:
        ev = 0.0
        for side in sides:
            m = self._memory.get(side)
            if m is not None:
                ev += SIGN[side] * math.exp(-math.dist(m[0], w) / MEMORY_RADIUS) * math.exp(-(t - m[1]) / MEMORY_TAU_S)
        return ev

    def _evidence(self, o: HandObservation, t: float, sides=SIDES) -> float:
        return (W_LABEL * self._label_term(o) + W_MEMORY * self._mem_term(o.wrist, t, sides)
                + W_POSITION * self._pos_term(o.wrist))

    def _order(self, right_w, left_w) -> float:
        dx = self._user_x(right_w) - self._user_x(left_w)
        return 0.0 if abs(dx) < PAIR_SEP else math.copysign(1.0, dx)

    def _gate(self, h: VirtualHand, t: float) -> float:
        return MATCH_RADIUS + GATE_GROWTH * min(max(t - h.seen, 0.0), TRACK_GRACE_S)

    def _associate(self, hands: list[HandObservation], t: float) -> tuple[dict[str, int], dict[str, str]]:
        """side -> observation index for this frame, and the sides that start
        a new track (with the reason). Continuity first; new tracks get their
        side from evidence, never from one frame's label alone."""
        idx = sorted(range(len(hands)), key=lambda k: -hands[k].score)[:2]
        live = {s: h for s, h in self.hands.items() if h.wrist is not None and t - h.seen <= TRACK_GRACE_S}
        self._ambiguous = False
        cost = {(s, i): math.dist(self._predict(h, t), hands[i].wrist) for s, h in live.items() for i in idx}
        valid = {k: d for k, d in cost.items() if d <= self._gate(live[k[0]], t)}
        options: list[dict[str, int]] = [{}]
        for s in live:
            nxt = []
            for o in options:
                nxt.append(o)
                nxt.extend({**o, s: i} for i in idx if (s, i) in valid and i not in o.values())
            options = nxt

        def total(o: dict[str, int]) -> float:
            return sum(valid[(s, i)] for s, i in o.items())

        options.sort(key=lambda o: (-len(o), total(o)))
        best = options[0]
        if best:
            alt = next((o for o in options[1:] if len(o) == len(best) and o != best), None)
            c1 = total(best)
            if alt is not None and total(alt) - c1 < SWAP_MARGIN + 1.5 * c1:
                if len(best) == 1 and len(live) == 2 and set(alt) != set(best):
                    # One observation, close to both hands: the accumulated
                    # handedness evidence of each track breaks the tie.
                    (s0, i0), = best.items()
                    s1 = next(iter(alt))
                    lt = self._label_term(hands[i0])
                    if lt and abs(live[s1].lab_ev - live[s0].lab_ev) > 0.5 and \
                            (live[s1].lab_ev - live[s0].lab_ev) * lt > 0:
                        best = alt
                # Hands touching or crossing: the involved hands start no new key.
                self._ambiguous = True
                self.counters["ambiguous_updates"] += 1
                for s in best:
                    if t >= self.hands[s].uncertain_until:
                        self._event(t, s, "ambiguous association", "no new key for 0.15 s")
                    self.hands[s].uncertain_until = t + UNCERTAIN_S
        assign = dict(best)
        fresh: dict[str, str] = {}
        free_obs = [i for i in idx if i not in assign.values()]
        free_sides = [s for s in SIDES if s not in assign]
        if len(free_obs) == 2 and len(free_sides) == 2:
            a, b = hands[free_obs[0]], hands[free_obs[1]]
            # Score of "a = left, b = right"; the swap scores the negative.
            score = (self._evidence(b, t) - self._evidence(a, t)) + W_ORDER * self._order(b.wrist, a.wrist)
            right, left = (free_obs[1], free_obs[0]) if score >= 0 else (free_obs[0], free_obs[1])
            assign["left"], assign["right"] = left, right
            why = f"two hands appeared (pair evidence {abs(score):.1f})"
            fresh["left"] = fresh["right"] = why
            self._pair_check_until = t + PAIR_CHECK_S
        elif free_obs and len(free_sides) == 2:
            i = free_obs[0]
            ev = self._evidence(hands[i], t)
            side = "right" if ev >= 0 else "left"
            assign[side] = i
            fresh[side] = f"identity evidence {ev:+.2f} (label/memory/position)"
        elif free_obs and len(free_sides) == 1:
            side = free_sides[0]
            i = free_obs[0]
            other = OTHER[side]
            why = "free side"
            if other in assign:
                # A pair is forming: check the existing hand's side against the
                # pair's evidence (corrects a lone hand that got the wrong side).
                why = "the other hand is tracked"
                fix = self._pair_fix(hands, assign[other], other, i, t)
                if fix:
                    i_old = assign.pop(other)
                    self._swap_sides(t, "pair formation: " + fix)
                    assign[side] = i_old              # the existing hand now carries `side`
                    side = other                      # the new hand takes the freed side
                    why = "pair formation corrected the sides"
                self._pair_check_until = t + PAIR_CHECK_S
            assign[side] = i
            fresh[side] = why
        return assign, fresh

    def _pair_score(self, right_o: HandObservation, left_o: HandObservation,
                    right_ev: float, left_ev: float) -> tuple[float, float]:
        """Evidence for "right_o is the right hand, left_o the left" (+), and
        the left/right-order part of it (-1 / 0 / +1)."""
        order = self._order(right_o.wrist, left_o.wrist)
        return right_ev - left_ev + W_ORDER * order, order

    def _pair_fix(self, hands, i_other: int, other: str, i_new: int, t: float) -> str:
        """Evidence that the existing hand (`other`, observation `i_other`)
        has the wrong side, given a new hand `i_new`. Labels alone never
        swap an existing track: the hands must be clearly in the opposite
        left/right order, the total evidence against the current sides, and
        the existing hand idle. Returns the reason, or "" to keep."""
        h = self.hands[other]
        if not h.idle:
            return ""
        o_old, o_new = hands[i_other], hands[i_new]
        ev_old = W_LABEL * h.lab_ev + W_POSITION * self._pos_term(o_old.wrist)
        ev_new = self._evidence(o_new, t, sides=(OTHER[other],))   # (the live side's memory is stale)
        if other == "right":
            cur, order = self._pair_score(o_old, o_new, ev_old, ev_new)
        else:
            cur, order = self._pair_score(o_new, o_old, ev_new, ev_old)
        return f"evidence {cur:+.1f} against the current sides" if order < 0 and cur <= -SWAP_FIX else ""

    _SIDE_HISTORY = ("keys", "last_key", "reacqs", "reacq_ms", "reacq_frames", "rec_ms", "flash_t")

    def _swap_sides(self, t: float, why: str) -> None:
        """Identity correction: the two controllers exchange their side.
        Each keeps its own hand's track (filter, recognizer, evidence,
        cursor); the per-side history (keys, reacquisitions, timings) stays
        with the side."""
        L, R = self.hands["left"], self.hands["right"]
        for f in self._SIDE_HISTORY:
            a, b = getattr(L, f), getattr(R, f)
            setattr(L, f, b)
            setattr(R, f, a)
        L.side, R.side = "right", "left"
        self.hands = {"left": R, "right": L}
        for h in (L, R):
            if not h.provisional:
                continue
            # A new track counts as a reacquisition of the side it really has.
            was, now = h.life == "reacquiring", h.side in self._memory
            old = self.hands[OTHER[h.side]]              # now at the side this track came from
            if was:                                      # (its count moves with the correction)
                old.reacqs = max(old.reacqs - 1, 0)
                self.counters["reacquired"] -= 1
            if now:
                h.reacqs += 1
                self.counters["reacquired"] += 1
            h.life = "reacquiring" if now else "detected"
            if now and h.ready_t is not None:
                h.reacq_ms, h.reacq_frames = round(1000 * (h.ready_t - h.born), 1), h.frames
        if self._move is not None:
            self._move = (OTHER[self._move[0]], *self._move[1:])
        self.counters["identity_corrections"] += 1
        self._event(t, "both", "identity corrected", why)

    def _identity_review(self, hands: list[HandObservation], assign: dict[str, int], t: float) -> None:
        """Per frame, after the hands were processed: finish or correct a
        provisional identity, re-check a forming pair, learn the label
        polarity from well-separated pairs."""
        both = "left" in assign and "right" in assign
        if both and not self._ambiguous:
            lw, rw = hands[assign["left"]].wrist, hands[assign["right"]].wrist
            dx = self._user_x(rw) - self._user_x(lw)
            if t < self._pair_check_until and self.hands["left"].idle and self.hands["right"].idle:
                L, R = self.hands["left"], self.hands["right"]
                cur, order = self._pair_score(hands[assign["right"]], hands[assign["left"]],
                                              W_LABEL * R.lab_ev + W_POSITION * self._pos_term(rw),
                                              W_LABEL * L.lab_ev + W_POSITION * self._pos_term(lw))
                if order < 0 and cur <= -SWAP_FIX:
                    self._swap_sides(t, f"pair check: evidence {cur:+.1f} against the current sides")
                    return
            if dx >= POLARITY_SEP:
                # Geometry and assignment agree: a sample of how MediaPipe's
                # label relates to the user's hands in this camera setup.
                for side in SIDES:
                    agree = hands[assign[side]].handedness.lower() == side
                    self._lab_n = self._lab_n * 0.995 + 1.0
                    self._lab_agree = self._lab_agree * 0.995 + (1.0 if agree else 0.0)
                if self._lab_n >= POLARITY_MIN:
                    r = self._lab_agree / self._lab_n
                    pol = 1 if r >= 0.6 else -1 if r <= 0.4 else 0
                    if pol != self.polarity:
                        self._event(t, "both", "label polarity", f"{pol:+d} (labels agree {100 * r:.0f}%)")
                        self.polarity = pol
            for h in self.hands.values():
                if h.provisional:
                    h.provisional = False
                    if h.ready_t is not None:
                        h.life = "tracking"
            return
        for side in SIDES:
            h = self.hands[side]
            if not h.provisional or side not in assign:
                continue
            o = hands[assign[side]]
            ev = W_LABEL * h.lab_ev + W_MEMORY * h.mem_ev + W_POSITION * self._pos_term(o.wrist)
            h.identity = ev
            other = self.hands[OTHER[side]]
            other_free = OTHER[side] not in assign and (other.wrist is None or t - other.seen > TRACK_GRACE_S)
            if SIGN[side] * ev <= -RELABEL_MIN and h.idle and other_free:
                self._swap_sides(t, f"provisional {side} hand: evidence {ev:+.2f}")
                h.provisional = SIGN[h.side] * ev < CONFIDENT
                break
            if SIGN[side] * ev >= CONFIDENT or t - h.born >= PROVISIONAL_MAX_S:
                h.provisional = False
                if h.ready_t is not None:
                    h.life = "tracking"

    @staticmethod
    def _predict(h: VirtualHand, t: float) -> tuple[float, float]:
        dt = min(max(t - h.seen, 0.0), 0.1)
        return h.wrist[0] + h.vel[0] * dt, h.wrist[1] + h.vel[1] * dt

    # ---------------------------------------------------------------- hand
    def _hand(self, h: VirtualHand, obs: HandObservation, feats: HandFeatures, t: float,
              aspect: float) -> tuple | None:
        e, s = self.e, self.e.s
        g, kc = s.gesture, s.keyboard
        style = kc.kb_click_style if kc.kb_click_style in ("pinch_pointer", "pinch_free") else g.click_style
        uncertain = t < h.uncertain_until
        h.rec.aspect = aspect
        # The recognizer always runs (its closing onset drives the aim hold);
        # an ambiguous association defers the key instead (below).
        gf = h.rec.update(feats, t, Finger.INDEX, style=style, allow_fist=False, confidence=obs.score)

        # Fingertip -> keyboard fraction (calibrated control box, keyboard margin).
        a = s.pointer.anchor_blend
        tip, mcp = feats.tips[Finger.INDEX], feats.mcps[Finger.INDEX]
        fx, fy = h.filt(tip[0] * (1 - a) + mcp[0] * a, tip[1] * (1 - a) + mcp[1] * a, t)
        nx, ny = e.pointer._box_norm(fx, fy)
        kx, ky = -EDGE + nx * (1 + 2 * EDGE), -EDGE + ny * (1 + 2 * EDGE)
        kn = feats.knuckle_center
        if h.knuckle is not None and (h.offset[0] or h.offset[1]):
            # The continuity offset (after a gesture move) fades with whole-
            # hand motion only - a finger flexing for a pinch does not count.
            bw, bh = max(s.pointer.box_w, 1e-3), max(s.pointer.box_h, 1e-3)
            k = math.exp(-math.hypot((kn[0] - h.knuckle[0]) / bw, (kn[1] - h.knuckle[1]) / bh) / 0.25)
            h.offset = [h.offset[0] * k, h.offset[1] * k]
        h.knuckle = kn
        h.raw = (kx, ky)
        h.raw_hist.append((t, kx, ky))
        pos = (min(max(kx + h.offset[0], -EDGE), 1 + EDGE), min(max(ky + h.offset[1], -EDGE), 1 + EDGE))
        h.history.append((t, *pos))

        mv = self._move
        if mv is not None and mv[0] == h.side:
            # Moving the keyboard with this hand: its cursor stays on the
            # handle while the window follows the hand.
            r = e.screen.target_rect(s.pointer.monitor) if e.screen is not None else None
            if r is not None:
                mover = getattr(e.keyboard, "move_by", None)
                if mover is not None:
                    mover(int(round((nx - mv[1]) * r.width)), int(round((ny - mv[2]) * r.height)))
            h.hold = mv[3]

        # 🤟 on either hand toggles (closes) the keyboard.
        if gf.ily_raw or gf.ily:
            h.state = "🤟"
            h.hold = h.hold or pos
            if not h.ily_block and gf.ily and not h.pinch:
                h.ily_t = h.ily_t if h.ily_t is not None else t
                prog = (t - h.ily_t) / max(g.keyboard_gesture_s, 0.1)
                e._kb_progress = min(prog, 1.0)
                if prog >= 1.0 and t >= e._kb_cooldown:
                    h.ily_block = True
                    e._kb_cooldown = t + g.keyboard_gesture_cooldown_s
                    e.counters["keyboard_toggles"] = e.counters.get("keyboard_toggles", 0) + 1
                    e.on_command("toggle_keyboard")
            h.pos = h.hold
            h.key = self._hit(h.pos)
            return None
        if h.ily_t is not None or h.ily_block:
            h.ily_t, h.ily_block = None, False
            e._kb_progress = 0.0
            if not h.pinch and mv is None:
                h.hold = None

        act: tuple | None = None
        slot = gf.slot
        if slot is None and not h.pinch:
            h.closing_t = (h.closing_t or max(gf.closing_onset, t - g.aim_max_rewind_ms / 1000.0)
                           if gf.closing else None)
            # The aim hold also runs while the association is ambiguous: a key
            # deferred by it then lands where the closing began, not a row lower.
            self._aim(h, gf, feats, pos, t)
        if slot is not None and not h.pinch:
            # New confirmed contact: one activation for this pinch.
            h.pinch = True
            at = h.hold or pos
            h.hold = at
            key = self._hit(at)
            if slot != "primary":
                h.state = "pinch (not a key contact)"
            elif key is None:
                self.counters["misses"] += 1
                h.state = "pinch outside the keys"
            elif uncertain:
                # Hands touching / crossing: typed only if the association is
                # resolved while this pinch is still held.
                h.deferred = (key, gf.cross_t if gf.cross_t is not None else t)
                h.state = "uncertain hand - key deferred"
            else:
                h.pressed = key
                h.next_repeat = t + kc.key_repeat_delay_s
                act = self._act(h, key, gf.cross_t if gf.cross_t is not None else t)
        elif slot is not None and h.pinch:
            if h.deferred is not None and not uncertain:
                key, tc = h.deferred
                h.deferred, h.pressed = None, key
                h.next_repeat = t + kc.key_repeat_delay_s
                act = self._act(h, key, tc)
            elif (h.pressed in REPEAT_KEYS and kc.repeat_backspace and t >= h.next_repeat):
                h.next_repeat = t + REPEAT_S
                act = self._act(h, h.pressed, t, new=False)
        elif slot is None and h.pinch:
            if h.deferred is not None:
                self.counters["uncertain_blocked"] += 1
                self._event(t, h.side, "key not typed", "pinch ended while the hands were ambiguous")
                h.deferred = None
            h.pinch, h.pressed = False, None
            h.release_t = t
            if mv is None or mv[0] != h.side:
                h.hold = None
        if gf.tap == "primary" and not h.pinch and act is None and uncertain:
            self.counters["uncertain_blocked"] += 1
            h.hold = None
        elif gf.tap == "primary" and not h.pinch and act is None:
            # Quick tap (re-opened before the confirmation span): one key.
            at = h.hold or pos
            key = self._hit(at)
            if key is not None:
                act = self._act(h, key, gf.cross_t if gf.cross_t is not None else t)
            h.hold = None
            h.release_t = t

        h.pos = h.hold or pos
        h.key = self._hit(h.pos)
        if act is None and not h.pinch:
            h.state = ("uncertain hand" if uncertain else "aiming" if h.hold else
                       "rearm: open fingers" if h.rec._need_open else "hover")
        elif h.pinch and h.pressed:
            h.state = f"pressed {h.pressed}"
        return act

    @staticmethod
    def _act(h: VirtualHand, key: str, cross_t: float, new: bool = True) -> tuple:
        """(priority, crossing time, hand, side, key, new pinch?) - the sort
        key of the frame's activations: modifiers first."""
        return (0 if key in MODIFIERS else 1, cross_t, HAND_INDEX[h.side], h.side, key, new)

    def _aim(self, h: VirtualHand, gf, feats: HandFeatures, pos, t: float) -> None:
        """Per-hand aim hold: while a contact closes, the cursor stays where
        the closing began (the flexing finger would otherwise slide it onto
        the key below)."""
        g = self.e.s.gesture
        if h.aim_latched and (gf.opening or gf.approach > g.aim_release_ratio):
            h.aim_latched = False
        if self._move is not None and self._move[0] == h.side:
            return
        if h.hold is None:
            if g.aim_lock and gf.closing and not h.aim_latched:
                t_anchor = max(gf.closing_onset, t - g.aim_max_rewind_ms / 1000.0, h.release_t)
                at = pos
                for ht, x, y in reversed(h.history):
                    at = (x, y)
                    if ht <= t_anchor:
                        break
                h.hold = at
                h.aim_t0, h.aim_best, h.aim_best_t = t, gf.approach, t
                h.aim_knuckle, h.aim_scale = feats.knuckle_center, max(feats.hand_scale, 1e-3)
            return
        if gf.approach < h.aim_best - 0.06:
            h.aim_best, h.aim_best_t = gf.approach, t
        moved = math.hypot((feats.knuckle_center[0] - h.aim_knuckle[0]) * h.rec.aspect,
                           feats.knuckle_center[1] - h.aim_knuckle[1]) / h.aim_scale
        if gf.opening or gf.approach > g.aim_release_ratio:
            h.hold = None
        elif (moved > g.aim_cancel_move or t - h.aim_best_t > g.aim_stall_s
              or t - h.aim_t0 > g.aim_max_hold_s):
            h.hold, h.aim_latched = None, True

    def _hit(self, pos: tuple[float, float] | None) -> str | None:
        if pos is None:
            return None
        f = getattr(self.e.keyboard, "hit_frac", None)
        return f(*pos) if f is not None else None

    # ---------------------------------------------------------- activation
    def _activate(self, side: str, key: str, t: float, cross_t: float | None = None) -> None:
        h = self.hands[side]
        last = self._last_act.get(key)
        if last is not None and last[1] != side and t - last[0] < SAME_KEY_S:
            # Both hands landed on the same key in the same instant: one key.
            self.counters["duplicates_suppressed"] += 1
            self._event(t, side, "duplicate suppressed", "both hands on the same key within 80 ms")
            return
        self._last_act[key] = (t, side)
        mv = self._move
        if mv is not None:
            # Any key ends a gesture move (the overlay does the same on its
            # side); the moving hand's cursor continues without a jump.
            mh = self.hands[mv[0]]
            if mh.raw is not None:
                # Reference = where the moving hand mapped before its ending
                # pinch flexed the finger (the aim anchor), else now.
                ref = mh.raw
                if mv[0] == side and mh.closing_t is not None:
                    for ht, x, y in reversed(mh.raw_hist):
                        ref = (x, y)
                        if ht <= mh.closing_t:
                            break
                mh.offset = [mv[3][0] - ref[0], mv[3][1] - ref[1]]
            mh.hold = None
            self._move = None
        elif key == "move" and h.raw is not None:
            nx = (h.raw[0] + EDGE) / (1 + 2 * EDGE)
            ny = (h.raw[1] + EDGE) / (1 + 2 * EDGE)
            self._move = (side, nx, ny, h.hold or h.pos or h.raw)
        self.e.keyboard.press_key(key, HAND_INDEX[side])
        self.e.counters["keys"] += 1
        self.counters["keys"] += 1
        h.keys += 1
        h.flash_t, h.last_key = t, key
        if cross_t is not None:                 # contact -> activation (repeats excluded)
            h.rec_ms.append(round(1000 * max(t - cross_t, 0.0), 1))

    # -------------------------------------------------------------- output
    def _publish(self, t: float, force: bool = False) -> None:
        kb = self.e.keyboard
        data = []
        for side in SIDES:
            h = self.hands[side]
            if h.pos is None:
                continue
            data.append({"hand": HAND_INDEX[side], "side": side, "x": round(h.pos[0], 4),
                         "y": round(h.pos[1], 4), "key": h.key,
                         "pressed": bool(h.pinch) or t - h.flash_t < PRESS_FLASH_S,
                         "lost": t - h.seen > 0.12})
        if force or data != self._last_pub:
            prev = {d["hand"]: d["key"] for d in (self._last_pub or [])}
            self._last_pub = data
            f = getattr(kb, "cursors", None)
            if f is not None:
                f(data)
            now = {d["hand"]: d["key"] for d in data}
            for i in (0, 1):
                if now.get(i) != prev.get(i):
                    kb.hover(i, now.get(i))

    def _event(self, t: float, side: str, what: str, detail: str) -> None:
        self.events.append({"t": round(t, 3), "side": side, "event": what, "detail": detail})

    def status(self) -> list[dict]:
        out = []
        now = self._t
        for side in SIDES:
            h = self.hands[side]
            cursor = ("hidden" if h.pos is None else "dimmed (lost)" if h.life == "temporarily lost"
                      else "shown")
            out.append({"side": side, "tracked": h.active, "state": h.state, "key": h.key or "",
                        "pressed": h.pressed or "", "last_key": h.last_key, "keys": h.keys,
                        "pos": (round(h.pos[0], 3), round(h.pos[1], 3)) if h.pos else None,
                        "track": h.track_id, "life": h.life, "detected": h.detected,
                        "label": f"{h.obs_label} {h.obs_score:.2f}" if h.obs_label else "",
                        "evidence": round(h.lab_ev, 2), "provisional": h.provisional,
                        "since_ms": round(1000 * (now - h.seen)) if h.seen > -1e8 else None,
                        "cursor": cursor,
                        "pinch": ("held, key deferred (hands ambiguous)" if h.deferred else "held" if h.pinch
                                  else "rearm (open first)" if h.rec._need_open else "open"),
                        "rec_ms": statistics.median(h.rec_ms) if h.rec_ms else None,
                        "reacqs": h.reacqs, "reacq_ms": h.reacq_ms, "reacq_frames": h.reacq_frames})
        return out

    def diag(self) -> dict:
        """Frame-level identity diagnostics (no key ids, no text)."""
        r = (self._lab_agree / self._lab_n) if self._lab_n else None
        return {**self.frame_diag, "polarity": self.polarity,
                "label_agree": None if r is None else round(r, 2), "label_samples": round(self._lab_n),
                "counters": dict(self.counters), "events": list(self.events)[-12:]}

    def _label(self) -> str:
        parts = []
        for side in SIDES:
            h = self.hands[side]
            if h.active:
                parts.append(f"{side[0].upper()}: {h.state}" + (f" [{h.key}]" if h.key else ""))
        return "Air Keyboard · " + (" · ".join(parts) if parts else "show a hand")
