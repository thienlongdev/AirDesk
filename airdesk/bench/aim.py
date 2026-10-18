"""Synthetic v1.1 benchmarks: pinch recognition latency, Smart Aim Lock
displacement, small-target clicking and false-lock rates.

Everything here runs the production InteractionEngine on generated
landmark streams (seeded, MediaPipe-like noise, random camera phase). It
is labelled *synthetic* everywhere: it measures the algorithm, not a
human hand in front of a real webcam.

Pinch model: the thumb travels to the fingertip along a minimum-jerk
profile while the pinched finger flexes by `curl` (the pointer finger
itself in Normal style), and the knuckles drift slightly as real hands
do while pinching. The *aim point* is the cursor just before the closing
movement starts; displacement = distance from it to where Windows
received the button-down (and, separately, the farthest the visible
cursor strayed while the pinch formed).
"""

from __future__ import annotations

import math
import statistics

import numpy as np

from ..config import Settings
from ..core.engine import InteractionEngine
from ..core.hand_model import analyze
from ..core.screen import ScreenLayout
from ..core.types import Finger
from ..sim.synthetic import Pose, frame, observation
from .synthetic import _min_jerk, _NullMouse

SCREEN = ScreenLayout.single(1920, 1080)
FPS = 30.0


def _pct(v: list[float], q: float) -> float:
    if not v:
        return float("nan")
    s = sorted(v)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def _summary(v: list[float]) -> dict:
    if not v:
        return {"n": 0}
    return {"n": len(v), "median": round(statistics.median(v), 1), "p95": round(_pct(v, 0.95), 1),
            "max": round(max(v), 1)}


class _Trial:
    """One scripted pinch on a fresh engine."""

    def __init__(self, settings: Settings, rng: np.random.Generator, noise: float) -> None:
        self.s = settings
        self.m = _NullMouse()
        self.e = InteractionEngine(settings, SCREEN, self.m)
        self.rng, self.noise = rng, noise
        self.t = float(rng.uniform(0, 1 / FPS))      # random camera phase
        self.x, self.y = 0.5, 0.72
        self.e.set_armed(True, self.t)

    def step(self, pose: Pose, dx: float = 0.0, dy: float = 0.0) -> None:
        self.t += 1 / FPS
        self.x += dx
        self.y += dy
        self.e.process(frame(self.t, observation(pose, self.x, self.y, noise=self.noise, rng=self.rng)))


def _crossing_time(finger: Finger, curl: float, t0: float, T: float, engage: float) -> float:
    """Exact (noise-free) time the thumb-finger distance crosses `engage`."""
    lo, hi = 0.0, 1.0
    for _ in range(40):          # distance decreases monotonically with approach
        mid = (lo + hi) / 2
        d = analyze(observation(Pose.closing(finger, mid, curl=curl))).pinch[finger]
        lo, hi = (mid, hi) if d > engage else (lo, mid)
    a = (lo + hi) / 2
    # invert the minimum-jerk profile
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _min_jerk(mid, 1.0) < a else (lo, mid)
    return t0 + T * (lo + hi) / 2


def pinch_trial(settings: Settings, rng: np.random.Generator, *, noise: float, finger: Finger,
                curl: float, close_s: float, hold_s: float, drift: float = 0.004,
                aim_motion: float = 0.0) -> dict:
    tr = _Trial(settings, rng, noise)
    for i in range(30):          # aim (optionally moving), then settle on the target
        tr.step(Pose.pointing(Finger.INDEX), dx=aim_motion if i < 20 else 0.0)
    aim = tr.e.pointer.cursor
    n0 = len(tr.m.events)
    t_start = tr.t
    engage = settings.finger.pinch_thresholds[finger.name.lower()][0]
    t_cross = _crossing_time(finger, curl, t_start, close_s, engage)
    vis = []
    confirm_t = None
    total = close_s + hold_s
    while tr.t - t_start < total:
        a = _min_jerk(tr.t + 1 / FPS - t_start, close_s)
        tr.step(Pose.closing(finger, a, curl=curl), dy=drift / max(total * FPS, 1))
        if tr.e.pointer.cursor is not None:
            vis.append(math.dist(tr.e.pointer.cursor, aim))
        if confirm_t is None and tr.e.state.value in ("PRESS_PENDING", "DRAGGING"):
            confirm_t = tr.t
    for k in range(3):
        tr.step(Pose.closing(finger, 1 - (k + 1) / 3, curl=curl))
    for _ in range(10):
        tr.step(Pose.pointing(Finger.INDEX))
    ev = tr.m.events[n0:]
    downs = [i for i, e in enumerate(ev) if (e[0] == "button" and e[2]) or e[0] == "click"]
    pos = None
    if downs:
        mv = [e for e in ev[:downs[0]] if e[0] == "move"]
        pos = (mv[-1][1], mv[-1][2]) if mv else aim
    return {"detected": bool(downs), "buttons": [e for e in ev if e[0] in ("button", "click")],
            "latency_ms": (confirm_t - t_cross) * 1000 if confirm_t is not None else None,
            "click_disp": math.dist(pos, aim) if pos is not None else None,
            "visual_disp": max(vis) if vis else 0.0, "aim": aim, "click": pos}


def recognition_latency(trials: int = 120, noise: float = 0.0012, seed: int = 21) -> list[dict]:
    """Threshold crossing -> confirmed press, per sensitivity preset."""
    out = []
    for preset in ("fast", "balanced", "stable"):
        rng = np.random.default_rng(seed)
        lat, missed, extra = [], 0, 0
        for i in range(trials):
            s = Settings()
            s.gesture.pinch_sensitivity = preset
            close_s = float(rng.uniform(0.10, 0.25))
            r = pinch_trial(s, rng, noise=noise, finger=Finger.INDEX, curl=float(rng.uniform(0.45, 0.85)),
                            close_s=close_s, hold_s=0.2)
            if r["latency_ms"] is None:
                missed += 1
            else:
                lat.append(r["latency_ms"])
            extra += int(len([b for b in r["buttons"] if (b[0] == "button" and b[2]) or b[0] == "click"]) > 1)
        out.append({"preset": preset, "noise": noise, "trials": trials, "missed": missed,
                    "duplicate_presses": extra, "latency_ms": _summary(lat),
                    "mean_ms": round(statistics.fmean(lat), 1) if lat else None})
    return out


def aim_displacement(trials: int = 80, noise: float = 0.0012, seed: int = 33) -> list[dict]:
    """Click and visual displacement with Smart Aim Lock on vs off."""
    out = []
    for style, finger in (("pinch_pointer", Finger.INDEX), ("pinch_free", Finger.MIDDLE)):
        for lock in (True, False):
            rng = np.random.default_rng(seed)
            cd, vd, missed = [], [], 0
            for _ in range(trials):
                s = Settings()
                s.gesture.click_style = style
                s.gesture.aim_lock = lock
                r = pinch_trial(s, rng, noise=noise, finger=finger, curl=float(rng.uniform(0.45, 0.85)),
                                close_s=float(rng.uniform(0.12, 0.35)), hold_s=0.15)
                if r["click_disp"] is None:
                    missed += 1
                    continue
                cd.append(r["click_disp"])
                vd.append(r["visual_disp"])
            out.append({"style": style, "aim_lock": lock, "noise": noise, "trials": trials, "missed": missed,
                        "click_px": _summary(cd), "visual_px": _summary(vd)})
    return out


def small_targets(trials: int = 60, noise: float = 0.0012, seed: int = 44,
                  styles: tuple[str, ...] = ("pinch_pointer", "pinch_free"),
                  locks: tuple[bool, ...] = (True, False)) -> list[dict]:
    """Click success on 12/16/20 px targets in a row with 4 px gaps.

    The target under the aim point is the intended one; a click that lands
    on a neighbour is a wrong-target click, anywhere else a miss.
    """
    out = []
    for style, finger in (("pinch_pointer", Finger.INDEX), ("pinch_free", Finger.MIDDLE)):
        if style not in styles:
            continue
        for lock in locks:
            for size in (12, 16, 20):
                rng = np.random.default_rng(seed + size)
                ok = wrong = miss = 0
                for _ in range(trials):
                    s = Settings()
                    s.gesture.click_style = style
                    s.gesture.aim_lock = lock
                    r = pinch_trial(s, rng, noise=noise, finger=finger, curl=float(rng.uniform(0.45, 0.85)),
                                    close_s=float(rng.uniform(0.12, 0.35)), hold_s=0.15)
                    if r["click"] is None:
                        miss += 1
                        continue
                    # target grid centred on the aim point, rows and columns
                    pitch = size + 4
                    dx, dy = r["click"][0] - r["aim"][0], r["click"][1] - r["aim"][1]
                    half = size / 2
                    if abs(dx) <= half and abs(dy) <= half:
                        ok += 1
                    elif abs(dx) <= pitch + half and abs(dy) <= pitch + half and (
                            (abs(dx) % pitch <= half or abs(dx) % pitch >= pitch - half)
                            and (abs(dy) % pitch <= half or abs(dy) % pitch >= pitch - half)):
                        wrong += 1
                    else:
                        miss += 1
                out.append({"style": style, "aim_lock": lock, "size_px": size, "trials": trials,
                            "hit": ok, "wrong_target": wrong, "miss": miss,
                            "hit_rate": round(ok / trials, 3)})
    return out


def false_locks(seconds: float = 90.0, noise: float = 0.0015, seed: int = 55) -> dict:
    """Natural pointing (smooth wandering, no pinches): how often the cursor
    is held by the aim lock, and whether anything is clicked."""
    rng = np.random.default_rng(seed)
    s = Settings()
    tr = _Trial(s, rng, noise)
    frames = int(seconds * FPS)
    locked = 0
    for i in range(frames):
        tr.step(Pose.pointing(Finger.INDEX), dx=0.004 * math.sin(i / 23) + 0.002 * math.sin(i / 7),
                dy=0.003 * math.cos(i / 31) + 0.0015 * math.sin(i / 5))
        locked += int(tr.e._aim is not None)
    presses = [e for e in tr.m.events if (e[0] == "button" and e[2]) or e[0] == "click"]
    return {"seconds": seconds, "noise": noise, "locked_fraction": round(locked / frames, 4),
            "presses": len(presses)}


def relaxed_idle(seconds: float = 40.0, noise: float = 0.003, curl: float = 0.7, seed: int = 7,
                 style: str = "pinch_pointer") -> dict:
    """v1.1.2: a relaxed pointing hand (middle/ring/little half-curled, not
    a fist) wandering with landmark noise. v1.1.1 froze this cursor in Aim
    Lock up to half of the time; nothing may ever be clicked."""
    rng = np.random.default_rng(seed)
    s = Settings()
    s.gesture.click_style = style
    s.finger.mode = "manual"
    tr = _Trial(s, rng, noise)
    curls = {Finger.MIDDLE: curl, Finger.RING: curl, Finger.LITTLE: curl}
    frames = int(seconds * FPS)
    locked = 0
    for i in range(frames):
        tr.step(Pose(extended={Finger.INDEX}, curls=curls), dx=0.001 * math.sin(i / 23),
                dy=0.0007 * math.cos(i / 31))
        locked += int(tr.e._aim is not None)
    presses = [e for e in tr.m.events if (e[0] == "button" and e[2]) or e[0] == "click"]
    return {"seconds": seconds, "noise": noise, "curl": curl, "style": style,
            "locked_fraction": round(locked / frames, 4), "presses": len(presses)}


def aim_recovery(trials: int = 30, noise: float = 0.0012, seed: int = 81, stop_at: float = 0.5) -> dict:
    """v1.1.2: a pinch that starts but stops short of contact (thumb halts
    at ~1.8x the threshold) - Aim Lock arms, no click may happen, and the
    cursor must follow the hand again quickly. Reports the time from the
    moment the fingers stop until the lock is released."""
    rng = np.random.default_rng(seed)
    rec, armed, presses = [], 0, 0
    for _ in range(trials):
        tr = _Trial(Settings(), rng, noise)
        for _ in range(30):
            tr.step(Pose.pointing(Finger.INDEX))
        n0 = len(tr.m.events)
        for k in range(5):
            tr.step(Pose.closing(Finger.INDEX, stop_at * _min_jerk((k + 1) / 5, 1.0), curl=0.6))
        t_stop = tr.t
        was = False
        for _ in range(30):
            tr.step(Pose.closing(Finger.INDEX, stop_at, curl=0.6))
            if tr.e._aim is not None:
                was = True
            elif was or tr.t - t_stop > 0.05:
                break
        armed += int(was)
        if was:
            rec.append((tr.t - t_stop) * 1000)
        presses += len([e for e in tr.m.events[n0:] if (e[0] == "button" and e[2]) or e[0] == "click"])
    return {"trials": trials, "aim_lock_armed": armed, "presses": presses,
            "release_after_stop_ms": _summary(rec)}


def rapid_aim_then_pinch(trials: int = 40, noise: float = 0.0012, seed: int = 66) -> dict:
    """Fast aiming motion that ends in a pinch: the click must land near
    where the motion *ended*, not where an over-long rewind would put it."""
    rng = np.random.default_rng(seed)
    disp = []
    for _ in range(trials):
        s = Settings()
        tr = _Trial(s, rng, noise)
        for _ in range(20):
            tr.step(Pose.pointing(Finger.INDEX))
        for _ in range(8):                      # fast move ...
            tr.step(Pose.pointing(Finger.INDEX), dx=0.012)
        for _ in range(3):                      # ... brief stop on the target
            tr.step(Pose.pointing(Finger.INDEX))
        aim = tr.e.pointer.cursor
        n0 = len(tr.m.events)
        for k in range(5):
            tr.step(Pose.closing(Finger.INDEX, (k + 1) / 5, curl=0.7))
        for _ in range(3):
            tr.step(Pose.closing(Finger.INDEX, 1.0, curl=0.7))
        for _ in range(8):
            tr.step(Pose.pointing(Finger.INDEX))
        ev = tr.m.events[n0:]
        downs = [i for i, e in enumerate(ev) if (e[0] == "button" and e[2]) or e[0] == "click"]
        if downs:
            mv = [e for e in ev[:downs[0]] if e[0] == "move"]
            pos = (mv[-1][1], mv[-1][2]) if mv else aim
            disp.append(math.dist(pos, aim))
    return {"trials": trials, "detected": len(disp), "click_px": _summary(disp)}


def run_all() -> dict:
    return {
        "label": "SYNTHETIC - generated landmarks, not a human hand",
        "recognition_latency": recognition_latency(),
        "aim_displacement": aim_displacement(),
        "small_targets": small_targets(),
        "false_locks": false_locks(),
        "rapid_aim_then_pinch": rapid_aim_then_pinch(),
        "relaxed_idle": [relaxed_idle(noise=n, curl=c) for n in (0.0015, 0.003) for c in (0.55, 0.7, 0.85)],
        "aim_recovery": aim_recovery(),
    }


if __name__ == "__main__":   # pragma: no cover
    import json
    print(json.dumps(run_all(), indent=1))
