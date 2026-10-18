"""Temporal filters for 2-D pointer signals.

All filters share one interface (`PointFilter`): feed a timestamped 2-D
measurement, get the filtered point back. Timestamps come from the camera
frame, never from the time the filter happens to run, so a stalled
inference frame does not distort velocity estimates.

Implemented:
  * `OneEuroFilter2D` - Casiez, Roussel & Vogel, "1 Euro Filter: A Simple
    Speed-based Low-pass Filter for Noisy Input in Interactive Systems",
    CHI 2012 (algorithm re-implemented from the paper; reference code is
    BSD-3 licensed, https://github.com/casiez/OneEuroFilter).
  * `AdaptiveKalman2D` - constant-velocity Kalman filter whose process
    noise scales with hand speed. Adapted from AirTouch (MIT,
    https://github.com/Jhnbismark/airtouch @ 8bd72f3), re-written without
    the dense 4x4 matrix algebra (the x and y axes are independent, so two
    2-state filters are exact and ~10x cheaper in Python).
  * `EmaFilter2D` - exponential moving average (baseline for benchmarks).
  * `PassThrough2D` - no filtering (baseline for benchmarks).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


class PointFilter(ABC):
    name = "base"

    @abstractmethod
    def __call__(self, x: float, y: float, t: float) -> tuple[float, float]:
        """Filter one measurement taken at time `t` (seconds)."""

    @abstractmethod
    def reset(self) -> None:
        """Forget all history (tracking loss, mode change)."""

    def velocity(self) -> tuple[float, float]:
        """Best current velocity estimate (units/second)."""
        return 0.0, 0.0


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class _OneEuro1D:
    __slots__ = ("min_cutoff", "beta", "d_cutoff", "x", "dx")

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x: float | None = None
        self.dx = 0.0

    def step(self, value: float, dt: float) -> float:
        if self.x is None:
            self.x = value
            self.dx = 0.0
            return value
        raw_dx = (value - self.x) / dt
        self.dx += _alpha(self.d_cutoff, dt) * (raw_dx - self.dx)
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        self.x += _alpha(cutoff, dt) * (value - self.x)
        return self.x


class OneEuroFilter2D(PointFilter):
    """Speed-adaptive low-pass: heavy smoothing when slow, little lag when fast.

    `min_cutoff` (Hz) sets jitter suppression at rest; `beta` sets how fast
    the cutoff opens with speed. Units of `beta` depend on the signal
    units; AirDesk feeds normalized image coordinates (0..1).
    """

    name = "one_euro"

    def __init__(self, min_cutoff: float = 1.2, beta: float = 8.0,
                 d_cutoff: float = 1.0) -> None:
        self._fx = _OneEuro1D(min_cutoff, beta, d_cutoff)
        self._fy = _OneEuro1D(min_cutoff, beta, d_cutoff)
        self._t: float | None = None

    def configure(self, min_cutoff: float, beta: float) -> None:
        for f in (self._fx, self._fy):
            f.min_cutoff = min_cutoff
            f.beta = beta

    def __call__(self, x: float, y: float, t: float) -> tuple[float, float]:
        dt = 1 / 30 if self._t is None else max(t - self._t, 1e-3)
        self._t = t
        return self._fx.step(x, dt), self._fy.step(y, dt)

    def velocity(self) -> tuple[float, float]:
        return self._fx.dx, self._fy.dx

    def reset(self) -> None:
        self._fx.x = self._fy.x = None
        self._t = None


class _Kalman1D:
    """Constant-velocity Kalman filter for one axis: state [p, v]."""

    __slots__ = ("p", "v", "P00", "P01", "P11")

    def __init__(self) -> None:
        self.p: float | None = None
        self.v = 0.0
        self.P00 = self.P01 = self.P11 = 0.0

    def step(self, z: float, dt: float, q: float, r: float) -> None:
        if self.p is None:
            self.p, self.v = z, 0.0
            self.P00, self.P01, self.P11 = r, 0.0, 1.0
            return
        # Predict with the discrete white-noise-acceleration model.
        p = self.p + self.v * dt
        dt2 = dt * dt
        P00 = self.P00 + dt * (2 * self.P01 + dt * self.P11) + q * dt2 * dt / 3
        P01 = self.P01 + dt * self.P11 + q * dt2 / 2
        P11 = self.P11 + q * dt
        # Update.
        s = P00 + r
        k0, k1 = P00 / s, P01 / s
        innov = z - p
        self.p = p + k0 * innov
        self.v = self.v + k1 * innov
        self.P00 = (1 - k0) * P00
        self.P01 = (1 - k0) * P01
        self.P11 = P11 - k1 * P01


class AdaptiveKalman2D(PointFilter):
    """Kalman filter whose process noise follows the measured hand speed.

    Hovering (speed ~ 0) collapses Q so tremor is crushed; deliberate fast
    motion opens Q geometrically so the filter tracks without lag. An
    optional speed-gated prediction compensates part of the pipeline
    latency, clamped so direction reversals cannot overshoot noticeably.
    """

    name = "kalman"
    SPEED_LO = 0.05   # normalized units / s: hovering
    SPEED_HI = 0.60   # deliberate fast motion
    PRED_LO = 0.06
    PRED_HI = 0.25

    def __init__(self, q_min: float = 0.002, q_max: float = 400.0,
                 r: float = 2.0e-5, prediction_ms: float = 0.0,
                 max_prediction: float = 0.008) -> None:
        self.q_min, self.q_max, self.r = q_min, q_max, r
        self.prediction = prediction_ms / 1000.0
        self.max_prediction = max_prediction
        self._kx, self._ky = _Kalman1D(), _Kalman1D()
        self._t: float | None = None
        self._speed = 0.0
        self._last: tuple[float, float] | None = None

    def _q(self) -> float:
        a = (self._speed - self.SPEED_LO) / (self.SPEED_HI - self.SPEED_LO)
        a = min(max(a, 0.0), 1.0)
        return self.q_min * (self.q_max / self.q_min) ** a

    def __call__(self, x: float, y: float, t: float) -> tuple[float, float]:
        dt = 1 / 30 if self._t is None else max(t - self._t, 1e-3)
        self._t = t
        if self._last is not None:
            raw = math.hypot(x - self._last[0], y - self._last[1]) / dt
            self._speed += (dt / (dt + 0.15)) * (raw - self._speed)
        self._last = (x, y)
        q = self._q()
        self._kx.step(x, dt, q, self.r)
        self._ky.step(y, dt, q, self.r)
        px, py = float(self._kx.p), float(self._ky.p)
        if self.prediction > 0:
            vx, vy = self._kx.v, self._ky.v
            speed = math.hypot(vx, vy)
            ramp = min(max((speed - self.PRED_LO) / (self.PRED_HI - self.PRED_LO), 0.0), 1.0)
            if ramp > 0:
                ox, oy = vx * self.prediction * ramp, vy * self.prediction * ramp
                mag = math.hypot(ox, oy)
                if mag > self.max_prediction:
                    ox, oy = ox * self.max_prediction / mag, oy * self.max_prediction / mag
                px, py = px + ox, py + oy
        return px, py

    def velocity(self) -> tuple[float, float]:
        return self._kx.v, self._ky.v

    def reset(self) -> None:
        self._kx, self._ky = _Kalman1D(), _Kalman1D()
        self._t = None
        self._speed = 0.0
        self._last = None


class EmaFilter2D(PointFilter):
    name = "ema"

    def __init__(self, alpha: float = 0.35) -> None:
        self.alpha = alpha
        self._p: tuple[float, float] | None = None
        self._v = (0.0, 0.0)
        self._t: float | None = None

    def __call__(self, x: float, y: float, t: float) -> tuple[float, float]:
        if self._p is None:
            self._p, self._t = (x, y), t
            return x, y
        dt = max(t - (self._t or t), 1e-3)
        a = self.alpha
        nx = self._p[0] + a * (x - self._p[0])
        ny = self._p[1] + a * (y - self._p[1])
        self._v = ((nx - self._p[0]) / dt, (ny - self._p[1]) / dt)
        self._p, self._t = (nx, ny), t
        return nx, ny

    def velocity(self) -> tuple[float, float]:
        return self._v

    def reset(self) -> None:
        self._p, self._t, self._v = None, None, (0.0, 0.0)


class PassThrough2D(PointFilter):
    name = "none"

    def __init__(self) -> None:
        self._last: tuple[float, float, float] | None = None
        self._v = (0.0, 0.0)

    def __call__(self, x: float, y: float, t: float) -> tuple[float, float]:
        if self._last is not None:
            dt = max(t - self._last[2], 1e-3)
            self._v = ((x - self._last[0]) / dt, (y - self._last[1]) / dt)
        self._last = (x, y, t)
        return x, y

    def velocity(self) -> tuple[float, float]:
        return self._v

    def reset(self) -> None:
        self._last, self._v = None, (0.0, 0.0)


class LowPass1D:
    """First-order low-pass with a cutoff in Hz (used for pinch distances:
    stable thresholds without smearing a quick 200 ms tap-pinch)."""

    def __init__(self, cutoff_hz: float = 10.0) -> None:
        self.cutoff = cutoff_hz
        self.y: float | None = None
        self._t: float | None = None

    def __call__(self, value: float, t: float) -> float:
        if self.y is None or self._t is None:
            self.y, self._t = value, t
            return value
        dt = max(t - self._t, 1e-3)
        self._t = t
        self.y += _alpha(self.cutoff, dt) * (value - self.y)
        return self.y

    def reset(self) -> None:
        self.y = None
        self._t = None


FILTER_NAMES = ("one_euro", "kalman", "ema", "none")


def make_filter(name: str, *, min_cutoff: float = 1.2, beta: float = 8.0,
                kalman_q_min: float = 0.002, kalman_q_max: float = 400.0,
                kalman_r: float = 2.0e-5, prediction_ms: float = 0.0,
                ema_alpha: float = 0.35) -> PointFilter:
    if name == "one_euro":
        return OneEuroFilter2D(min_cutoff, beta)
    if name == "kalman":
        return AdaptiveKalman2D(kalman_q_min, kalman_q_max, kalman_r, prediction_ms)
    if name == "ema":
        return EmaFilter2D(ema_alpha)
    if name == "none":
        return PassThrough2D()
    raise ValueError(f"unknown filter {name!r}")
