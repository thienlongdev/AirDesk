"""Gesture recognition: hand features -> debounced gesture primitives.

This layer knows nothing about mice, keyboards or Windows. It turns noisy
per-frame geometry into clean, debounced primitives that the interaction
state machine consumes:

* **contact slots** ``primary`` / ``secondary`` / ``tertiary`` - which
  physical contact backs each slot depends on the active pointer finger
  and the click style (see `slot_contacts`). Each contact has its own
  low-passed distance and hysteresis detector (engage < release). When
  several contacts engage in the same frame (one physical squeeze often
  brings two fingertips near the thumb) only the *deepest* one, relative
  to its own threshold, wins - one physical action, one command. A press
  is confirmed after a short evidence span measured from the interpolated
  threshold crossing (Fast / Balanced / Stable presets). Release is
  *adaptive*: a contact also releases once it re-opens clearly beyond the
  deepest point of this press, so a relaxed, small opening still ends the
  click (v1.1 kept the button down until a fixed, wide threshold).
* **settled contact** (v1.1.2) - real MediaPipe fingertip landmarks often
  stop a little *above* the engage threshold when the pads touch (the
  landmarks sit at the nail ends; depth is estimated). v1.1.1 then held
  Aim Lock with no button, pinch after pinch. A contact that converged
  from an open hand, stopped moving within ``contact_settle_ratio`` of
  its threshold and is unambiguously the nearest mapped contact is
  confirmed too. Re-arming needs a real re-opening (``contact_settle_drop``
  above the settled distance), so a held or jittering contact never
  clicks twice.
* **deep contact** (v1.1.2) - a contact that arrived from the near zone
  and is already far below its threshold is confirmed on that frame (no
  extra evidence span), unless another mapped contact is closing in too.
* **pinch onset** (``closing``) - a mapped contact converging on the thumb,
  reported *before* any press is confirmed so the engine can anchor the
  cursor (Smart Aim Lock).
* **poses** ``scroll`` (index + middle extended) and ``fist``, each
  requiring N stable frames to enter and to leave.
* **🤟 keyboard pose** - thumb, index and little finger extended, middle and
  ring curled (Air Keyboard toggle). The thumb is far from every fingertip
  in this pose, so it cannot overlap a pinch; new contacts are blocked only
  while the pose itself is present (no global click delay).
* **swipes** - fast horizontal open-palm motion, with a cooldown.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from ..config import FingerConfig, GestureConfig, pinch_timing
from .filters import LowPass1D
from .hand_model import EXTENDED_ON, action_fingers, ily_pose
from .types import NON_THUMB, Finger, HandFeatures

SLOTS = ("primary", "secondary", "tertiary")
STYLE_NAMES = {"pinch_pointer": "Normal", "pinch_free": "Pinch Free"}

# Contact convergence slower than this (hand scales / s) is not "closing".
_CLOSING_MIN_SPEED = 0.5
_OPENING_SPEED = 1.0
# Adaptive release: an engaged contact (not dragging) releases once it is
# this far above the deepest distance of the press (hand scales), but never
# below engage + _RELEASE_MIN_OVER, and at the latest at its fixed release
# threshold.
_RELEASE_GAP = 0.12
_RELEASE_MIN_OVER = 0.04
# A press confirmed from a settled contact releases only this far above its
# deepest point (it may sit above the fixed release threshold's margin).
_SETTLE_RELEASE_GAP = 0.08


def slot_contacts(pointer: Finger | None, style: str) -> dict[str, str]:
    """Map slot name -> contact id ("pinch:<finger>" or "thumb_press").

    Index pointer: Normal = thumb+index / +middle / +ring;
    Pinch Free = thumb+middle / +ring / +little (the index never clicks).
    """
    if pointer is Finger.THUMB:
        return {"primary": "thumb_press"}
    free = action_fingers(pointer)
    if style == "pinch_pointer" and pointer is not None:
        order = [pointer] + free
    else:
        order = free
    return {slot: f"pinch:{f.name.lower()}" for slot, f in zip(SLOTS, order)}


class Hysteresis:
    """Engages below `engage`, releases above `release` (release > engage)."""

    __slots__ = ("engage", "release", "engaged")

    def __init__(self, engage: float, release: float) -> None:
        self.engage, self.release = engage, release
        self.engaged = False

    def update(self, d: float, release_extra: float = 0.0) -> bool:
        if self.engaged:
            if d > self.release + release_extra:
                self.engaged = False
        elif d < self.engage:
            self.engaged = True
        return self.engaged


@dataclass(slots=True)
class GestureFrame:
    slot: str | None = None            # debounced pressed slot
    slot_depth: float = 0.0
    contacts: dict[str, float] = field(default_factory=dict)
    scroll_pose: bool = False
    fist: bool = False
    open_palm: bool = False
    swipe: str | None = None           # "left" | "right"
    # Smart Aim Lock signals (pinch *onset*, before any contact is confirmed)
    closing: bool = False              # a mapped contact is converging on the thumb
    closing_onset: float = 0.0         # when that convergence began
    approach: float = math.inf         # nearest mapped contact / its engage threshold
    opening: bool = False              # the nearest contact is moving apart again
    cross_t: float | None = None       # threshold crossing of the confirmed press
    tap: str | None = None             # quick tap: contact re-opened just before confirmation
    # 🤟 keyboard pose (Air Keyboard toggle)
    ily: bool = False                  # pose held (debounced)
    ily_raw: bool = False              # pose present this frame
    # v1.1.2 diagnostics / evidence
    settled: bool = False              # the confirmed press came from a settled contact
    evidence: str = ""                 # how the press was confirmed: threshold | settled | deep
    near_slot: str | None = None       # nearest mapped contact (ungated) this frame
    near_d: float = math.inf           # its filtered distance (hand scales)
    near_engage: float = 0.0           # its engage threshold
    note: str = ""                     # why a near contact was not confirmed (ambiguous, ...)


class _Stable:
    """Boolean that must hold for N frames to switch on and M to switch off."""

    __slots__ = ("on_frames", "off_frames", "value", "_count")

    def __init__(self, on_frames: int, off_frames: int) -> None:
        self.on_frames, self.off_frames = on_frames, off_frames
        self.value = False
        self._count = 0

    def update(self, raw: bool) -> bool:
        if raw != self.value:
            self._count += 1
            if self._count >= (self.on_frames if raw else self.off_frames):
                self.value = raw
                self._count = 0
        else:
            self._count = 0
        return self.value

    def reset(self) -> None:
        self.value = False
        self._count = 0


class GestureRecognizer:
    """One instance per tracked hand role (control hand, action hand)."""

    def __init__(self, gcfg: GestureConfig, fcfg: FingerConfig) -> None:
        self.gcfg = gcfg
        self.fcfg = fcfg
        self._lp: dict[str, LowPass1D] = {}
        self._hyst: dict[str, Hysteresis] = {}
        self._prev: dict[str, tuple[float, float]] = {}      # contact -> (t, filtered d)
        self._vel: dict[str, float] = {}                     # contact -> previous frame's velocity
        self._closing_since: dict[str, float | None] = {}
        self._closing_d0: dict[str, float] = {}              # distance when that convergence began
        self._slot = _Stable(2, 2)
        self._slot_name: str | None = None
        self._pending: str | None = None
        self._pending_count = 0
        self._pending_cross = 0.0
        self._mapping: dict[str, str] = {}
        self._scroll = _Stable(gcfg.pose_frames, gcfg.pose_frames)
        self._fist = _Stable(gcfg.pose_frames, 2)
        self._ily = _Stable(2, 3)
        self._dmin: dict[str, float] = {}                    # deepest distance of the current press
        self._peak: dict[str, float] = {}                    # widest distance since the last press (rearm)
        self._still: dict[str, int] = {}                     # frames a near contact has stopped moving
        self._near_t: dict[str, float] = {}                  # when it entered the near zone
        self._rel: dict[str, float] = {}                     # effective release threshold of the press
        self._prev_ratio: dict[str, float] = {}
        self._settled_slot: str | None = None
        self._evidence = ""
        self._need_open: set[str] = set()                   # contacts that must re-open before engaging
        self._fast_slot: str | None = None
        self._cross: float | None = None
        self._palm_track: deque[tuple[float, float, float, float]] = deque(maxlen=40)
        self._open_all = False                              # next frame: closed contacts must re-open
        self._last_swipe = -1e9
        self.aspect = 16 / 9   # inference frame width / height
        self.configure(gcfg, fcfg)

    def configure(self, gcfg: GestureConfig, fcfg: FingerConfig) -> None:
        self.gcfg, self.fcfg = gcfg, fcfg
        self._hyst.clear()
        press_frames, confirm_ms, lp_hz, release_frames = pinch_timing(gcfg)
        self._press_frames, self._confirm_s = max(1, press_frames), confirm_ms / 1000.0
        self._release_frames = max(1, release_frames)
        self._slot.on_frames, self._slot.off_frames = self._press_frames, self._release_frames
        self._lp_hz = lp_hz
        for lp in self._lp.values():
            lp.cutoff = lp_hz
        self._scroll.on_frames = self._scroll.off_frames = gcfg.pose_frames
        self._fist.on_frames = gcfg.pose_frames

    def reset(self) -> None:
        for lp in self._lp.values():
            lp.reset()
        self._prev.clear()
        self._vel.clear()
        self.cancel_press()
        self._scroll.reset()
        self._fist.reset()
        self._ily.reset()
        self._palm_track.clear()

    def cancel_press(self) -> None:
        """Drop any pressed or pending contact (click style / mode / pointer
        change): the next press needs fresh evidence under the new mapping.
        A contact that is still closed must open first - the cancelled
        pinch is never re-read as a new one."""
        for c, h in self._hyst.items():
            if h.engaged or (c in self._prev and self._prev[c][1] < h.release):
                self._need_open.add(c)
            h.engaged = False
        self._slot.reset()
        self._slot_name = None
        self._pending = None
        self._pending_count = 0
        self._closing_since.clear()
        self._dmin.clear()
        self._still.clear()
        self._near_t.clear()
        self._rel.clear()
        # The open reference survives: a contact still closed after a
        # cancel needs a real re-opening before a settled confirmation.
        for c, v in list(self._peak.items()):
            self._peak[c] = min(v, self._prev.get(c, (0.0, v))[1])

    def require_open(self) -> None:
        """Every contact that is closed on the next frame must re-open
        before it can engage (hand-over between the desktop pointer and the
        Air Keyboard, a fresh hand, a drop): a pinch already held at that
        moment is never read as a new click or key."""
        self.cancel_press()
        self._open_all = True

    # ---------------------------------------------------------------
    def _thresholds(self, contact: str) -> tuple[float, float]:
        if contact == "thumb_press":
            e, r = self.fcfg.thumb_press
        else:
            e, r = self.fcfg.pinch_thresholds.get(contact.split(":")[1], [0.26, 0.36])
        return float(e), float(r)

    def _distance(self, feats: HandFeatures, contact: str) -> float:
        if contact == "thumb_press":
            return feats.thumb_press
        return feats.pinch[Finger.from_name(contact.split(":")[1])]

    def _gated(self, feats: HandFeatures, contact: str) -> bool:
        """True when this contact must not *start*: the geometry says it is
        not a deliberate click."""
        if contact == "thumb_press":
            # A long finger extending while the thumb folds is a finger
            # switch (thumb -> other pointer), not a click.
            return any(feats.extension[f] >= EXTENDED_ON for f in NON_THUMB)
        # A fingertip folded into the palm often projects next to the
        # resting thumb (e.g. the old pointer folding away during a finger
        # switch). A deep deliberate pinch bends the finger as much but
        # keeps its tip away from its own knuckle, so it is not gated.
        f = Finger.from_name(contact.split(":")[1])
        return (feats.extension[f] < self.fcfg.pinch_min_extension
                and feats.reach.get(f, 1.0) < self.fcfg.pinch_min_reach)

    def update(self, feats: HandFeatures, t: float, pointer: Finger | None, *,
               style: str | None = None, dragging: bool = False,
               allow_fist: bool = True, slots: bool = True, block_new: bool = False,
               confidence: float = 1.0) -> GestureFrame:
        """Feed one frame. `block_new` (a finger switch is under way) and a
        low landmark `confidence` stop new contacts from engaging; contacts
        already engaged continue normally."""
        g = self.gcfg
        out = GestureFrame()
        contacts = slot_contacts(pointer, style or g.click_style) if slots else {}
        if contacts != self._mapping:
            if self._slot.value or self._pending:
                self.cancel_press()   # mapping changed mid-gesture: never re-label a press
            self._mapping = contacts

        # 🤟 keyboard pose. Its thumb is far from every fingertip, so it never
        # competes with a real pinch; while it is present no new contact may
        # start (the pose is being held, not a click).
        out.ily_raw = g.keyboard_gesture and self._slot_name is None and ily_pose(feats)
        out.ily = self._ily.update(out.ily_raw)
        block = block_new or confidence < 0.5 or out.ily_raw

        best: tuple[float, str] | None = None
        cross_t: dict[str, float] = {}
        motion: dict[str, tuple[float, float, float]] = {}   # slot -> (velocity, ratio, previous velocity)
        pratio: dict[str, float] = {}                        # slot -> previous frame's ratio
        settle_cands: list[tuple[float, str, str]] = []
        for slot, contact in contacts.items():
            raw = self._distance(feats, contact)
            lp = self._lp.get(contact)
            if lp is None:
                lp = self._lp[contact] = LowPass1D(self._lp_hz)
            d = lp(raw, t)
            out.contacts[slot] = d
            engage, release = self._thresholds(contact)
            h = self._hyst.get(contact)
            if h is None:
                h = self._hyst[contact] = Hysteresis(engage, release)
            prev = self._prev.get(contact)
            self._prev[contact] = (t, d)
            v = (d - prev[1]) / max(t - prev[0], 1e-3) if prev else 0.0
            motion[slot] = (v, d / engage, self._vel.get(contact, 0.0))
            self._vel[contact + "#prev"] = self._vel.get(contact, 0.0)
            self._vel[contact] = v
            if self._open_all and d < release:
                self._need_open.add(contact)
            if contact in self._need_open and d > release:
                self._need_open.discard(contact)
            gated = not h.engaged and (block or contact in self._need_open or self._gated(feats, contact))
            ratio = d / engage

            # Pinch onset (Smart Aim Lock): remember when this contact began
            # converging; forget it once it clearly re-opens.
            # Onset = start of the *current* continuous convergence (reset
            # whenever it pauses away from contact).
            if v < -_CLOSING_MIN_SPEED:
                if self._closing_since.get(contact) is None:
                    self._closing_since[contact] = prev[0] if prev else t
                    self._closing_d0[contact] = prev[1] if prev else d
            elif ratio > g.aim_near_ratio:
                self._closing_since[contact] = None
            if not gated:
                if ratio < out.approach:
                    out.approach = ratio
                    out.near_slot, out.near_d, out.near_engage = slot, d, engage
                    out.opening = v > _OPENING_SPEED and ratio > 1.0
                # Speed-armed onset needs *sustained* convergence (two frames
                # closing and a real drop): landmark noise on a relaxed, curled
                # finger near the thumb produced single-frame "closing" speeds
                # that froze an idle cursor up to half the time in v1.1.1.
                closing = ratio < g.aim_near_ratio or (
                    d < g.aim_arm_dist and v < -g.aim_close_speed
                    and self._vel.get(contact + "#prev", 0.0) < -_CLOSING_MIN_SPEED
                    and self._closing_d0.get(contact, d) - d >= g.aim_arm_drop)
                if closing and not out.closing:
                    out.closing = True
                    out.closing_onset = self._closing_since.get(contact) or t

            # Settled-contact evidence (v1.1.2): the widest opening since the
            # last press is the rearm reference; a near contact counts its
            # frames without movement.
            if h.engaged:
                self._peak[contact] = d
            else:
                self._peak[contact] = max(self._peak.get(contact, d), d)
            if ratio < g.contact_settle_ratio:
                self._near_t.setdefault(contact, prev[0] if prev else t)
                self._still[contact] = self._still.get(contact, 0) + 1 if abs(v) < g.contact_settle_speed else 0
            else:
                self._near_t.pop(contact, None)
                self._still[contact] = 0
            if not gated and not h.engaged and ratio < g.contact_settle_ratio:
                if (self._still[contact] >= g.contact_settle_frames
                        and self._peak[contact] - d >= g.contact_settle_drop):
                    settle_cands.append((ratio, slot, contact))
                elif out.note == "" and self._peak[contact] - d < g.contact_settle_drop and \
                        self._still[contact] >= g.contact_settle_frames:
                    out.note = "waiting for the fingers to open (rearm)"

            holding_this = dragging and self._slot_name == slot
            extra = g.drag_release_extra if holding_this else 0.0
            if h.engaged and not holding_this:
                # Adaptive release: a clear re-opening from the deepest point
                # of this press ends it, even if the hand never opens past the
                # fixed release threshold (a relaxed, partial opening). A press
                # that settled above its engage threshold keeps a small margin
                # above its deepest point, so landmark jitter cannot end it.
                dmin = min(self._dmin.get(contact, d), d)
                self._dmin[contact] = dmin
                rel = max(min(release, max(engage + _RELEASE_MIN_OVER, dmin + _RELEASE_GAP)),
                          dmin + _SETTLE_RELEASE_GAP)
                self._rel[contact] = rel
                extra = rel - release
            was = h.engaged
            if h.update(math.inf if gated else d, extra):
                if not was:
                    # Interpolated threshold crossing: confirmation time is
                    # measured from the physical event, not the frame grid.
                    if prev and prev[1] > engage > d:
                        cross_t[slot] = prev[0] + (t - prev[0]) * (prev[1] - engage) / (prev[1] - d)
                    else:
                        cross_t[slot] = t
                depth = d / engage
                if best is None or depth < best[0]:
                    best = (depth, slot)
            elif not (self._slot.value and self._slot_name == slot):
                self._dmin.pop(contact, None)          # (kept while its slot is still pressed)
                self._rel.pop(contact, None)
            pratio[slot] = self._prev_ratio.get(contact, math.inf)
            self._prev_ratio[contact] = ratio

        # Settled contact: confirmed when it is the only near contact (the
        # next one is clearly farther) - never a guess between two fingers.
        if best is None and settle_cands and not self._slot.value:
            settle_cands.sort()
            r0, slot, contact = settle_cands[0]
            others = [d_ / self._thresholds(c_)[0] for s_, c_ in contacts.items()
                      if s_ != slot and (d_ := out.contacts.get(s_)) is not None]
            if all(r >= r0 + g.contact_settle_margin for r in others):
                self._hyst[contact].engaged = True
                self._dmin[contact] = out.contacts[slot]
                cross_t[slot] = self._near_t.get(contact, t)
                best = (r0, slot)
                self._settled_slot = slot
            else:
                out.note = "ambiguous: two fingers equally close to the thumb"

        # Debounce: a slot must win for press_frames frames *and* for the
        # preset's evidence span after its threshold crossing; while a slot
        # is pressed it stays the slot until released for release_frames
        # (another contact cannot steal it mid-gesture).
        cand = best[1] if best else None
        passing = False
        if cand is not None and not self._slot.value:
            # Pass-through: the thumb sweeping past a bent fingertip on its way
            # to another one. The passed contact is already opening while the
            # other is still closing in - that frame is no evidence for it.
            vc = motion[cand][0]
            if vc > _OPENING_SPEED and any(
                    sl != cand and v < -_CLOSING_MIN_SPEED and r < g.aim_near_ratio
                    for sl, (v, r, _) in motion.items()):
                cand, passing = None, True
        if self._slot.value:
            # A contact that re-opened past its fixed release threshold is
            # unambiguous: one frame of evidence ends the click (button-up
            # sooner). Adaptive (partial) releases and drags keep the preset's
            # release frames.
            c = contacts.get(self._slot_name) if self._slot_name else None
            # (A one-frame tracking glitch jumps from a steady contact; a real
            # opening was already under way the frame before.)
            clear = (c is not None and not dragging
                     and out.contacts.get(self._slot_name, 0.0) > max(self._thresholds(c)[1],
                                                                       self._rel.get(c, 0.0))
                     and motion[self._slot_name][2] > _OPENING_SPEED * 1.5)
            self._slot.off_frames = 1 if clear else self._release_frames
            still = cand == self._slot_name or (self._slot_name in out.contacts and
                                                self._hyst[contacts[self._slot_name]].engaged)
            if not still and c is not None and self._slot_name in out.contacts and not dragging:
                # A one-frame landmark jump can release the hysteresis; if the
                # next frame is back below this press's release threshold (in
                # the engage/release gap) it is the same press - re-latch it
                # instead of ending it and starting a second click.
                rel = max(self._rel.get(c, 0.0), self._thresholds(c)[0] + _RELEASE_MIN_OVER)
                if out.contacts[self._slot_name] < rel and self._slot._count > 0:
                    self._hyst[c].engaged = True
                    still = True
            if not self._slot.update(still):
                self._slot_name = None
                if c is not None:
                    self._dmin.pop(c, None)
                    self._rel.pop(c, None)
        else:
            if cand is not None and cand == self._pending:
                self._pending_count += 1
            else:
                p = self._pending
                if (p is not None and cand is None and not passing and not block
                        and self._pending_count >= self._press_frames and p in contacts
                        and not self._hyst[contacts[p]].engaged
                        and not any(v < -_CLOSING_MIN_SPEED and r < g.aim_near_ratio
                                    for sl, (v, r, _) in motion.items() if sl != p)):
                    # Quick tap: the contact had the preset's frames of evidence
                    # but re-opened before its time span completed (a fast,
                    # deliberate pinch of ~2 frames). v1.1 discarded it; it is
                    # one complete click now.
                    out.tap, out.cross_t = p, self._pending_cross
                self._pending, self._pending_count = cand, (1 if cand else 0)
                self._pending_cross = cross_t.get(cand, t) if cand else 0.0
            instant = False
            if cand is not None and cand == self._settled_slot:
                # Settled contact: its still frames were the evidence.
                instant = True
            elif (cand is not None and best is not None and best[0] < g.contact_fast_ratio
                  and g.pinch_sensitivity != "stable"):
                # Deep contact reached through the near zone (not a one-frame
                # landmark jump), and no other mapped contact closing in.
                instant = (pratio.get(cand, math.inf) < g.aim_near_ratio
                           and not any(sl != cand and v < -_CLOSING_MIN_SPEED and r < g.aim_near_ratio
                                       for sl, (v, r, _) in motion.items()))
            if (cand is not None and (instant or (self._pending_count >= self._press_frames
                                                  and t - self._pending_cross >= self._confirm_s - 1e-6))):
                self._slot.value = True
                self._slot_name = cand
                self._cross = self._pending_cross
                out.settled = cand == self._settled_slot
                self._evidence = "settled" if out.settled else "deep" if instant else "threshold"
                self._pending, self._pending_count = None, 0
                self._settled_slot = None
        if self._settled_slot is not None and self._settled_slot != cand:
            self._settled_slot = None
        self._open_all = False
        out.slot = self._slot_name if self._slot.value else None
        out.cross_t = self._cross if out.slot else out.cross_t      # (a tap sets its own)
        out.evidence = self._evidence if out.slot else ("tap" if out.tap else "")
        if best and out.slot == best[1]:
            out.slot_depth = best[0]

        out.scroll_pose = self._scroll.update(feats.two_finger and out.slot is None)
        out.fist = self._fist.update(allow_fist and feats.fist and out.slot is None)
        out.open_palm = feats.open_palm
        out.swipe = self._swipe(feats, t)
        return out

    def _swipe(self, feats: HandFeatures, t: float) -> str | None:
        g = self.gcfg
        if not feats.open_palm:
            self._palm_track.clear()
            return None
        x, y = feats.palm_center
        self._palm_track.append((t, x, y, feats.hand_scale))
        while self._palm_track and t - self._palm_track[0][0] > g.swipe_max_time_s:
            self._palm_track.popleft()
        if len(self._palm_track) < 3 or t - self._last_swipe < g.swipe_cooldown_s:
            return None
        t0, x0, y0, _ = self._palm_track[0]
        scale = max(feats.hand_scale, 1e-3)
        # hand_scale is aspect-corrected; express dx in the same units.
        dx = (x - x0) * self.aspect / scale
        dy = (y - y0) / scale
        if abs(dx) >= g.swipe_min_dist and abs(dx) > 2.5 * abs(dy):
            self._last_swipe = t
            self._palm_track.clear()
            return "right" if dx > 0 else "left"
        return None


def hand_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
