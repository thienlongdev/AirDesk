"""Pointer engine: from a hand feature point to a desktop cursor position.

Design (the core of AirDesk's "no-jump" behaviour)
-------------------------------------------------
The cursor is computed from a *source point* in normalized image space -
normally the active fingertip blended slightly toward its knuckle. Many
events change which physical point is the source, or would make the
cursor jump if mapped naively:

* switching the active finger (thumb tip and little tip are centimetres apart),
* entering a drag (the pointer then follows the rigid knuckle centroid so
  pinch flexion cannot move the cursor),
* releasing a hold/clutch/press freeze,
* the physical mouse moving the cursor.

All of these are handled by two continuity offsets instead of resetting
anything:

1. `source offset` (image space, *before* the filter): when the source
   point changes, the offset absorbs the difference between the old and
   new raw points measured in the *same frame*, so the filter input is
   continuous and the filter never sees a step.
2. `screen offset` (normalized screen space, *after* mapping): when a hold
   ends, the offset is set so the cursor resumes exactly where it was held.

In DIRECT mode both offsets then fade with *hand motion* (not with time),
so the absolute hand-to-screen relationship is gradually restored while
the user moves - imperceptibly - and never while the hand is still.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from ..config import PointerConfig
from .filters import PointFilter, make_filter
from .screen import ScreenLayout
from .types import PointerMode


@dataclass(slots=True)
class PointerOutput:
    x: float
    y: float
    moved: bool


class PointerEngine:
    def __init__(self, cfg: PointerConfig, screen: ScreenLayout) -> None:
        self.cfg = cfg
        self.screen = screen
        self.mode = PointerMode(cfg.mode)
        self.filter: PointFilter = self._make_filter()
        self._src_off = [0.0, 0.0]       # image space
        self._scr_off = [0.0, 0.0]       # normalized screen space (direct)
        self._last_filtered: tuple[float, float] | None = None
        self._last_raw: tuple[float, float] | None = None
        self._cursor: tuple[float, float] | None = None   # float px, last emitted
        self._held: tuple[float, float] | None = None
        self._rel: tuple[float, float] | None = None       # relative-mode virtual position
        self._history: deque[tuple[float, float, float]] = deque(maxlen=48)
        self._stationary = True
        self._release_pending = False
        self._motion = 0.0

    # ----------------------------------------------------------- config
    def _make_filter(self) -> PointFilter:
        c = self.cfg
        precise = self.mode is PointerMode.PRECISION
        return make_filter(c.filter,
                           min_cutoff=c.precision_min_cutoff if precise else c.min_cutoff,
                           beta=c.precision_beta if precise else c.beta,
                           prediction_ms=c.kalman_prediction_ms)

    def configure(self, cfg: PointerConfig, screen: ScreenLayout | None = None) -> None:
        mode_changed = PointerMode(cfg.mode) is not self.mode
        filt_changed = cfg.filter != self.cfg.filter
        self.cfg = cfg
        if screen is not None:
            self.screen = screen
        if mode_changed:
            self.set_mode(PointerMode(cfg.mode))
        elif filt_changed:
            self.filter = self._make_filter()
            self._last_filtered = None
        elif hasattr(self.filter, "configure"):
            self.filter.configure(cfg.min_cutoff, cfg.beta)  # type: ignore[attr-defined]

    def set_mode(self, mode: PointerMode) -> None:
        """Switch modes without moving the cursor."""
        self.mode = mode
        self.cfg.mode = mode.value
        self.filter = self._make_filter()
        self._last_filtered = None
        if self._cursor is not None:
            self._anchor_to(self._cursor)

    # ------------------------------------------------------------ events
    def reset(self) -> None:
        """Tracking lost: forget motion state but keep the cursor where it is."""
        self.filter.reset()
        self._last_filtered = None
        self._last_raw = None
        self._src_off = [0.0, 0.0]
        self._history.clear()
        if self._cursor is not None:
            # Next acquisition must not teleport: resume from here.
            self._held = self._cursor

    def change_source(self, old_raw: tuple[float, float], new_raw: tuple[float, float]) -> None:
        """The physical source point changed this frame (finger switch,
        drag start/end). Absorb the step so the filter input stays continuous."""
        dx, dy = new_raw[0] - old_raw[0], new_raw[1] - old_raw[1]
        if self._last_filtered is not None:
            # Re-base so the filter's next input equals its current output:
            # any lag still pending from the old source (e.g. a finger that
            # was curling away) is dropped instead of replayed as a jump.
            self._src_off = [new_raw[0] - self._last_filtered[0], new_raw[1] - self._last_filtered[1]]
        else:
            self._src_off[0] += dx
            self._src_off[1] += dy
        if self._last_raw is not None:
            # Keep the motion measure (used for offset fading) in the new
            # source's frame, otherwise the switch itself reads as motion.
            self._last_raw = (self._last_raw[0] + dx, self._last_raw[1] + dy)

    def hold(self, at: tuple[float, float] | None = None, *, firm: bool = False) -> None:
        """Freeze the cursor (press pending, clutch, keyboard dwell...).
        `firm` also drops a release requested earlier in this frame (a new
        hold that must not be undone by the pending release)."""
        if at is not None:
            self._cursor = at
        if self._cursor is not None:
            self._held = self._cursor
        if firm:
            self._release_pending = False

    def release_hold(self) -> None:
        """Resume tracking from the held position (re-anchored on next update)."""
        if self._held is not None:
            self._release_pending = True

    @property
    def holding(self) -> bool:
        return self._held is not None

    def sync_cursor(self, x: float, y: float) -> None:
        """The cursor was moved by something else (physical mouse):
        continue from there instead of snapping back to the hand."""
        self._cursor = (x, y)
        self._held = (x, y) if self._held is not None else None
        self._anchor_to((x, y))

    def position_at(self, t: float) -> tuple[float, float] | None:
        """Cursor position at time t (used to rewind a click to where the
        user was aiming before the pinch motion disturbed the hand)."""
        best = None
        for ht, x, y in reversed(self._history):
            best = (x, y)
            if ht <= t:
                break
        return best if best is not None else self._cursor

    @property
    def cursor(self) -> tuple[float, float] | None:
        return self._cursor

    # ------------------------------------------------------------ update
    def update(self, raw: tuple[float, float], t: float) -> PointerOutput | None:
        """Feed the current raw source point; returns the new cursor target."""
        sx, sy = raw[0] - self._src_off[0], raw[1] - self._src_off[1]
        fx, fy = self.filter(sx, sy, t)
        prev_f = self._last_filtered
        self._last_filtered = (fx, fy)
        prev_raw = self._last_raw
        self._last_raw = raw

        # Deliberate hand motion this frame, measured on the *filtered* point
        # with a noise floor: landmark jitter on a still hand must not count
        # as motion, or the continuity offsets would fade and the cursor drift.
        motion = 0.0
        if prev_f is not None:
            motion = max(0.0, math.hypot(fx - prev_f[0], fy - prev_f[1]) - self.cfg.offset_motion_floor)
        self._motion = motion

        # Motion-gated fading of the source offset (direct mode only).
        if self.mode is PointerMode.DIRECT and motion > 0 and (self._src_off[0] or self._src_off[1]):
            k = 1.0 - math.exp(-motion / max(self.cfg.offset_decay_dist, 1e-3))
            # Only fades while the hand moves, so a still hand never drifts.
            self._src_off[0] -= self._src_off[0] * k
            self._src_off[1] -= self._src_off[1] * k

        if self._held is not None:
            if self._release_pending:
                self._release_pending = False
                held = self._held
                self._held = None
                self._anchor_to(held)
                prev_f = (fx, fy)
            else:
                self._history.append((t, *self._held))
                return PointerOutput(self._held[0], self._held[1], False)

        if self.mode is PointerMode.DIRECT:
            target = self._direct(fx, fy, prev_raw, raw)
        else:
            target = self._relative(fx, fy, prev_f, t)
        if target is None:
            return None

        target = self.screen.clamp(*target)
        moved = True
        if self._cursor is not None:
            d = math.hypot(target[0] - self._cursor[0], target[1] - self._cursor[1])
            dz = self.cfg.deadzone_px
            if self._stationary and d < dz:
                target, moved = self._cursor, False
            else:
                self._stationary = d < dz * 0.5
        self._cursor = target
        self._history.append((t, *target))
        return PointerOutput(target[0], target[1], moved)

    # ----------------------------------------------------------- mapping
    def _box_norm(self, fx: float, fy: float) -> tuple[float, float]:
        c = self.cfg
        nx = 0.5 + ((fx - c.box_cx) / max(c.box_w, 1e-3)) * c.direct_gain
        ny = 0.5 + ((fy - c.box_cy) / max(c.box_h, 1e-3)) * c.direct_gain
        return nx, ny

    def _direct(self, fx: float, fy: float, prev_raw, raw) -> tuple[float, float]:
        nx, ny = self._box_norm(fx, fy)
        if (self._scr_off[0] or self._scr_off[1]) and self._motion > 0:
            k = math.exp(-self._motion / max(self.cfg.offset_decay_dist, 1e-3))
            self._scr_off[0] *= k
            self._scr_off[1] *= k
        nx += self._scr_off[0]
        ny += self._scr_off[1]
        r = self.screen.target_rect(self.cfg.monitor)
        nx = min(max(nx, 0.0), 1.0)
        ny = min(max(ny, 0.0), 1.0)
        return r.left + nx * (r.width - 1), r.top + ny * (r.height - 1)

    def _relative(self, fx: float, fy: float, prev_f, t: float) -> tuple[float, float] | None:
        if self._cursor is None:
            r = self.screen.target_rect("primary")
            self._cursor = (r.left + r.width / 2, r.top + r.height / 2)
        if self._rel is None:
            self._rel = self._cursor
        if prev_f is None:
            return self._rel
        dx, dy = fx - prev_f[0], fy - prev_f[1]
        vx, vy = self.filter.velocity()
        speed = math.hypot(vx, vy)
        c = self.cfg
        if self.mode is PointerMode.PRECISION:
            gain = c.precision_gain
        else:
            a = (speed - c.relative_speed_low) / max(c.relative_speed_high - c.relative_speed_low, 1e-6)
            a = min(max(a, 0.0), 1.0)
            a = a * a * (3 - 2 * a)  # smoothstep transfer curve
            gain = c.relative_gain_low + (c.relative_gain_high - c.relative_gain_low) * a
        ref = self.screen.target_rect("primary").height
        # Integrate in a sub-pixel virtual position so slow precision
        # motion accumulates instead of being eaten by the dead-zone.
        self._rel = self.screen.clamp(self._rel[0] + dx * gain * ref, self._rel[1] + dy * gain * ref)
        return self._rel

    def _anchor_to(self, pos: tuple[float, float]) -> None:
        """Make the *current* filtered point map exactly onto `pos`."""
        self._cursor = pos
        self._rel = pos
        if self.mode is not PointerMode.DIRECT or self._last_filtered is None:
            return
        r = self.screen.target_rect(self.cfg.monitor)
        want_nx = (pos[0] - r.left) / max(r.width - 1, 1)
        want_ny = (pos[1] - r.top) / max(r.height - 1, 1)
        nx, ny = self._box_norm(*self._last_filtered)
        self._scr_off = [want_nx - nx, want_ny - ny]
