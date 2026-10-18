"""Active-hand and active-finger selection.

`HandSelector` keeps following the same physical hand from frame to frame
(wrist continuity first, handedness preference second) so a second hand
entering the view never steals the pointer. Idea credited to AirCursor's
`HandSelector` (MIT, https://github.com/FarzamFattahi/aircursor @ b44af9f);
extended here with a control/action hand split for two-hand mode.

`FingerSelector` implements the five-finger pointer policy:
AUTO (pose-driven with temporal confidence + hysteresis), MANUAL, LOCK.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .hand_model import pinky_pose, pointer_candidate
from .types import FINGERS, Finger, FingerMode, HandFeatures, HandObservation, HandPreference


@dataclass(slots=True)
class HandSelector:
    preference: HandPreference = HandPreference.AUTO
    match_radius: float = 0.22      # normalized image units
    grace_s: float = 0.35           # keep identity through short dropouts
    _last_wrist: tuple[float, float] | None = field(default=None, init=False)
    _last_seen: float = field(default=-1e9, init=False)
    # False when the control hand was (re)acquired rather than continued
    # from the previous frame: a drag must never pass to another hand.
    continued: bool = field(default=False, init=False)

    def reset(self) -> None:
        self._last_wrist = None
        self._last_seen = -1e9

    def select(self, hands: list[HandObservation], t: float
               ) -> tuple[HandObservation | None, HandObservation | None]:
        """Return (control_hand, other_hand)."""
        self.continued = False
        if not hands:
            if t - self._last_seen > self.grace_s:
                self._last_wrist = None
            return None, None

        candidates = hands
        if self.preference is not HandPreference.AUTO:
            want = "Left" if self.preference is HandPreference.LEFT else "Right"
            preferred = [h for h in hands if h.handedness == want]
            if preferred:
                candidates = preferred
            elif self._last_wrist is None:
                # Preferred hand absent and nothing to continue: no control hand.
                return None, hands[0]

        chosen: HandObservation | None = None
        if self._last_wrist is not None:
            near = min(candidates, key=lambda h: math.dist(self._last_wrist, h.wrist))
            if math.dist(self._last_wrist, near.wrist) <= self.match_radius:
                chosen = near
                self.continued = True
        if chosen is None:
            chosen = max(candidates, key=lambda h: h.score)

        self._last_wrist = chosen.wrist
        self._last_seen = t
        other = next((h for h in hands if h is not chosen), None)
        return chosen, other


@dataclass(slots=True)
class FingerSelector:
    """Decides which fingertip drives the pointer.

    AUTO mode accumulates per-finger confidence with an exponential moving
    average of "this frame's unambiguous pointing candidate". A switch
    happens only when another finger has been the candidate continuously
    for `switch_delay_s` AND its confidence exceeds `switch_confidence`;
    ambiguous poses (open palm, V-sign, half-curl) keep the current finger.
    Switching is blocked entirely while a button/scroll gesture is active.
    """

    mode: FingerMode = FingerMode.AUTO
    manual_finger: Finger = Finger.INDEX
    switch_delay_s: float = 0.25
    switch_confidence: float = 0.6
    confidence_tau_s: float = 0.18
    current: Finger | None = field(default=None, init=False)
    confidence: dict[Finger, float] = field(default_factory=lambda: {f: 0.0 for f in FINGERS}, init=False)
    _candidate: Finger | None = field(default=None, init=False)
    _candidate_since: float = field(default=0.0, init=False)
    _last_t: float | None = field(default=None, init=False)
    switches: int = field(default=0, init=False)
    # v1.1.3: the exclusive pinky-only pose (thumb tucked) is Pinky Drag, not
    # a little-finger pointer; little-finger pointing with the thumb out is.
    reserve_pinky: bool = False

    def reset(self) -> None:
        self.confidence = {f: 0.0 for f in FINGERS}
        self._candidate = None
        self._last_t = None
        if self.mode is not FingerMode.LOCK:
            self.current = None

    @property
    def switching(self) -> bool:
        """Another long finger is currently the unambiguous pointing pose:
        a finger switch is under way, so no new pinch may start."""
        return (self.mode is FingerMode.AUTO and self._candidate is not None
                and self._candidate is not self.current and self._candidate is not Finger.THUMB)

    def force(self, finger: Finger) -> None:
        """Explicit user switch (UI, hotkey, LOCK/MANUAL selection)."""
        self.current = finger

    def update(self, feats: HandFeatures, t: float, *, blocked: bool = False) -> Finger:
        """Feed one frame; returns the active finger (never None)."""
        if self.mode is FingerMode.MANUAL:
            self.current = self.manual_finger
            return self.current

        dt = 1 / 30 if self._last_t is None else min(max(t - self._last_t, 1e-3), 0.2)
        self._last_t = t
        cand = pointer_candidate(feats)
        if cand is Finger.LITTLE and self.reserve_pinky and pinky_pose(feats):
            cand = None
        a = dt / (dt + self.confidence_tau_s)
        for f in FINGERS:
            target = 1.0 if cand is f else 0.0
            self.confidence[f] += a * (target - self.confidence[f])

        if cand is not self._candidate:
            self._candidate = cand
            self._candidate_since = t

        if self.mode is FingerMode.LOCK:
            if self.current is None:
                self.current = self.manual_finger
            return self.current

        if self.current is None:
            # First acquisition: take the pose's finger, else the default.
            self.current = cand if cand is not None else self.manual_finger
            return self.current

        if (not blocked and cand is not None and cand is not self.current
                and t - self._candidate_since >= self.switch_delay_s
                and self.confidence[cand] >= self.switch_confidence):
            self.current = cand
            self.switches += 1
        return self.current
