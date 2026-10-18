"""Synthetic (algorithm) benchmarks - reproducible, seeded, no camera.

These measure the *software* on generated landmark streams with
MediaPipe-like noise. They are clearly labelled "synthetic" everywhere
and are never presented as human usability results.

Noise model: per-landmark Gaussian noise of sigma = 0.0012 normalized
image units by default (~0.8 px on a 640 px inference frame), a typical
order of magnitude for MediaPipe hand landmarks on a still hand.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import asdict, dataclass

import numpy as np

from ..config import PointerConfig, Settings
from ..core.engine import InteractionEngine
from ..core.pointer import PointerEngine
from ..core.screen import ScreenLayout
from ..core.types import Finger
from ..sim.synthetic import Pose, frame, observation

SCREEN = ScreenLayout.single(1920, 1080)
FPS = 30.0


@dataclass
class FilterResult:
    filter: str
    jitter_px: float          # cursor std while the hand is still
    lag_ms: float             # time behind the true target at mid-reach
    overshoot_px: float       # max travel beyond the reach end point
    settle_ms: float          # time to come within 3 px of the end point


class _NullMouse:
    def __init__(self):
        self.events = []
        self.held = set()
        self.pos = (960.0, 540.0)

    def move(self, x, y):
        self.pos = (x, y)
        self.events.append(("move", x, y))

    def button(self, b, down):
        (self.held.add if down else self.held.discard)(b)
        self.events.append(("button", b, down))

    def click(self, b, n=1):
        self.events.append(("click", b, n))

    def scroll(self, dx, dy):
        self.events.append(("scroll", dx, dy))

    def xbutton(self, w):
        self.events.append(("xbutton", w))

    def zoom(self, n):
        self.events.append(("zoom", n))

    def release_all(self):
        self.held.clear()

    def held_buttons(self):
        return tuple(self.held)

    def cursor_pos(self):
        return self.pos


def _min_jerk(t: float, T: float) -> float:
    s = min(max(t / T, 0.0), 1.0)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def filter_comparison(noise: float = 0.0012, seed: int = 7) -> list[FilterResult]:
    """Still hand (2 s) then a 0.45 s minimum-jerk reach of 0.25 image widths."""
    configs = [("one_euro", {}), ("one_euro (precision)", {"precision": True}),
               ("kalman", {}), ("kalman + 30 ms prediction", {"pred": 30.0}),
               ("ema", {}), ("none", {})]
    out: list[FilterResult] = []
    for label, opt in configs:
        rng = np.random.default_rng(seed)
        cfg = PointerConfig()
        cfg.filter = label.split()[0]
        cfg.kalman_prediction_ms = opt.get("pred", 0.0)
        cfg.deadzone_px = 0.0           # measure the filter, not the dead-zone
        if opt.get("precision"):
            cfg.min_cutoff, cfg.beta = cfg.precision_min_cutoff, cfg.precision_beta
        pe = PointerEngine(cfg, SCREEN)
        dt = 1 / FPS
        still_xs, still_ys = [], []
        t = 0.0
        x0, y0 = 0.40, 0.45
        for _ in range(int(2.0 * FPS)):
            t += dt
            o = pe.update((x0 + rng.normal(0, noise), y0 + rng.normal(0, noise)), t)
            if t > 0.7 and o:
                still_xs.append(o.x)
                still_ys.append(o.y)
        jitter = math.sqrt(statistics.pvariance(still_xs) + statistics.pvariance(still_ys))
        T, dist = 0.45, 0.25
        t_start = t
        truth, got = [], []
        for _ in range(int(1.5 * FPS)):
            t += dt
            xt = x0 + dist * _min_jerk(t - t_start, T)
            o = pe.update((xt + rng.normal(0, noise), y0 + rng.normal(0, noise)), t)
            truth.append((t - t_start, pe._box_norm(xt, y0)[0] * 1919))
            got.append((t - t_start, o.x if o else float("nan")))
        end_px = truth[-1][1]
        mid_px = pe._box_norm(x0 + dist / 2, y0)[0] * 1919
        t_true_mid = T / 2
        t_got_mid = float("nan")
        for (ta, xa), (tb, xb) in zip(got, got[1:]):
            if xa < mid_px <= xb:           # interpolate the crossing between frames
                t_got_mid = ta + (tb - ta) * (mid_px - xa) / max(xb - xa, 1e-9)
                break
        overshoot = max(0.0, max(x for _, x in got) - end_px)
        settle = next((tt for tt, x in got if tt > T and all(abs(x2 - end_px) < 3 for t2, x2 in got if t2 >= tt)),
                      float("nan"))
        out.append(FilterResult(label, round(jitter, 3), round((t_got_mid - t_true_mid) * 1000, 1),
                                round(overshoot, 2), round((settle - T) * 1000, 1) if settle == settle else -1))
    return out


def _engine(settings: Settings | None = None):
    s = settings or Settings()
    m = _NullMouse()
    e = InteractionEngine(s, SCREEN, m)
    e.set_armed(True, 0.0)
    return e, m


def false_clicks(noise: float, seconds: float = 120.0, seed: int = 3) -> dict:
    """A pointing hand wandering for `seconds` must never click."""
    rng = np.random.default_rng(seed)
    e, m = _engine()
    t, x, y = 0.0, 0.5, 0.7
    fingers = [Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.LITTLE, Finger.THUMB]
    f = Finger.INDEX
    for i in range(int(seconds * FPS)):
        t += 1 / FPS
        if i % 90 == 0:
            f = fingers[rng.integers(0, 5)]
        x = min(max(x + rng.normal(0, 0.004), 0.25), 0.75)
        y = min(max(y + rng.normal(0, 0.004), 0.5), 0.85)
        e.process(frame(t, observation(Pose(extended={f}), x, y, noise=noise, rng=rng)))
    clicks = sum(1 for ev in m.events if ev[0] in ("click", "button"))
    return {"noise": noise, "seconds": seconds, "false_clicks": clicks,
            "per_minute": round(clicks / (seconds / 60), 2)}


def pinch_detection(noise: float, trials: int = 60, seed: int = 5) -> dict:
    """Scripted quick pinches (click = one down/up, no motion) and 1 s pinches
    with motion (drag = one down, motion, one up)."""
    rng = np.random.default_rng(seed)
    e, m = _engine()
    t = 0.0
    ok_clicks = ok_drags = 0
    for i in range(trials):
        drag = i % 2 == 1
        n0 = len(m.events)
        for _ in range(12):
            t += 1 / FPS
            e.process(frame(t, observation(Pose.pointing(Finger.INDEX), 0.5, 0.7, noise=noise, rng=rng)))
        hold = 33 if drag else 4          # drag needs the 1.0 s hold (v1.1)
        x = 0.5
        for k in range(hold + (15 if drag else 0)):
            t += 1 / FPS
            if drag and k >= hold:
                x += 0.004
            e.process(frame(t, observation(Pose(extended={Finger.INDEX}, pinch=Finger.INDEX), x, 0.7,
                                           noise=noise, rng=rng)))
        for _ in range(10):
            t += 1 / FPS
            e.process(frame(t, observation(Pose.pointing(Finger.INDEX), x, 0.7, noise=noise, rng=rng)))
        ev = m.events[n0:]
        clicks = [x for x in ev if x[0] == "click"]
        btn = [x for x in ev if x[0] == "button"]
        one_press = btn == [("button", "left", True), ("button", "left", False)] and not clicks
        if one_press:
            i0, i1 = (i for i, x in enumerate(ev) if x[0] == "button")
            moved = len({(x[1], x[2]) for x in ev[i0:i1] if x[0] == "move"}) > 1
        if drag:
            ok_drags += int(one_press and moved)
        else:
            ok_clicks += int(one_press and not moved)   # v1.1: button down at confirm, up at release
    half = trials // 2
    return {"noise": noise, "click_success": round(ok_clicks / half, 3),
            "drag_success": round(ok_drags / (trials - half), 3), "trials": trials,
            "stuck_buttons": len(m.held)}


def finger_switch_continuity(noise: float, switches: int = 40, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    e, m = _engine()
    t = 0.0
    order = [Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.LITTLE, Finger.THUMB]
    for _ in range(20):
        t += 1 / FPS
        e.process(frame(t, observation(Pose.pointing(Finger.INDEX), 0.5, 0.7, noise=noise, rng=rng)))
    steps, drift, delays, wrong = [], [], [], 0
    cur = Finger.INDEX
    for i in range(switches):
        nxt = order[(order.index(cur) + 1 + rng.integers(0, 4)) % 5]
        start = m.pos
        n0 = len(m.events)
        t_sw = t
        switched_at = None
        for _ in range(30):
            t += 1 / FPS
            st = e.process(frame(t, observation(Pose.pointing(nxt), 0.5, 0.7, noise=noise, rng=rng)))
            if switched_at is None and st.finger == nxt.label:
                switched_at = t
            if st.finger not in (cur.label, nxt.label):
                wrong += 1
        mv = [(x[1], x[2]) for x in m.events[n0:] if x[0] == "move"]
        pts = [start] + mv
        steps += [math.dist(a, b) for a, b in zip(pts, pts[1:])]
        drift.append(math.dist(start, m.pos))
        delays.append(((switched_at or t) - t_sw) * 1000)
        cur = nxt
    return {"noise": noise, "switches": switches, "max_step_px": round(max(steps, default=0), 2),
            "median_drift_px": round(statistics.median(drift), 2), "max_drift_px": round(max(drift), 2),
            "median_switch_delay_ms": round(statistics.median(delays), 1), "unwanted_switch_frames": wrong}


def engine_cost(frames: int = 3000) -> dict:
    e, m = _engine()
    rng = np.random.default_rng(1)
    obs = [observation(Pose.pointing(Finger.INDEX), 0.5 + 0.1 * math.sin(i / 20), 0.7, noise=0.001, rng=rng)
           for i in range(200)]
    t0 = time.perf_counter()
    for i in range(frames):
        e.process(frame(i / FPS, obs[i % 200]))
    us = (time.perf_counter() - t0) / frames * 1e6
    return {"frames": frames, "engine_us_per_frame": round(us, 1)}


def telex_accuracy() -> dict:
    from ..core.telex import telex
    cases = [("Vieejt Nam", "Việt Nam"), ("tieengs Vieetj", "tiếng Việt"), ("dduowngf", "đường"),
             ("nguowif", "người"), ("quyeenf", "quyền"), ("giuwax", "giữa"), ("thuowr", "thuở"),
             ("khuyur", "khuỷu"), ("dduwowcj", "được"), ("Truwowngf", "Trường"), ("hoafn", "hoàn"),
             ("chuyeenj", "chuyện"), ("nuwowcs", "nước"), ("phaanf", "phần"), ("quoocs", "quốc")]
    ok = sum(telex(a) == b for a, b in cases)
    t0 = time.perf_counter()
    n = 0
    for _ in range(300):
        for a, _b in cases:
            telex(a)
            n += len(a)
    kps = n / (time.perf_counter() - t0)
    return {"words": len(cases), "correct": ok, "keys_per_second": int(kps)}


def run_all(progress=None) -> dict:
    res: dict = {"kind": "synthetic", "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    steps = [
        ("filters", lambda: [asdict(r) for r in filter_comparison()]),
        ("false_clicks", lambda: [false_clicks(n) for n in (0.0008, 0.0015, 0.003)]),
        ("pinch_detection", lambda: [pinch_detection(n) for n in (0.0008, 0.0015, 0.003)]),
        ("finger_switching", lambda: [finger_switch_continuity(n) for n in (0.0008, 0.0015)]),
        ("engine_cost", engine_cost),
        ("telex", telex_accuracy),
    ]
    for i, (name, fn) in enumerate(steps):
        if progress:
            progress(name, i, len(steps))
        res[name] = fn()
    return res
