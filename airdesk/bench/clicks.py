"""Click-reliability benchmark (v1.1.1): one deliberate pinch must give
exactly one complete click, at the aimed pixel, promptly.

Runs the production InteractionEngine on *synthetic* landmark streams that
add the real-hand properties the v1.1 benchmarks left out (all labelled
synthetic - this measures the algorithm, not a human hand):

* relaxed hands - the non-pointing fingers half-curled instead of a fist;
* a contact gap - MediaPipe fingertip landmarks stay ~0.1-0.2 hand scales
  apart when the finger pads touch;
* incomplete releases - after a click the thumb opens only part-way before
  the next pinch;
* sympathetic flexion - the pointing index bends while the middle finger
  pinches (right click in Normal, left click in Pinch Free);
* quick taps and slow, deep pinches.

Every intended click is scored from the recorded MouseSink calls:
``ok`` (exactly one down + one up of the intended button, up within 0.25 s
of the physical release, at <= 5 px from the aim point - or snapped onto the
previous click inside the double-click time, counted as ``sticky``),
``missed``, ``stuck`` (down but no timely up), ``wrong_pos``, ``duplicate``.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

import numpy as np

from ..config import Settings
from ..core.engine import InteractionEngine
from ..core.screen import ScreenLayout
from ..core.types import Finger
from ..sim.synthetic import Pose, frame, observation
from .synthetic import _min_jerk

SCREEN = ScreenLayout.single(1920, 1080)
FPS = 30.0
I, M, R, L = Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.LITTLE


class TimedMouse:
    """MouseSink recording (engine time, event)."""

    def __init__(self) -> None:
        self.t = 0.0
        self.events: list[tuple] = []
        self.held: set[str] = set()
        self.pos = (960.0, 540.0)

    def move(self, x, y):
        self.pos = (x, y)

    def button(self, b, down):
        (self.held.add if down else self.held.discard)(b)
        self.events.append((self.t, "button", b, down, self.pos))

    def click(self, b, n=1):
        self.events.append((self.t, "click", b, n, self.pos))

    def scroll(self, dx, dy):
        self.events.append((self.t, "scroll", dx, dy, self.pos))

    def xbutton(self, w):
        pass

    def zoom(self, n):
        pass

    def release_all(self):
        for b in list(self.held):
            self.events.append((self.t, "button", b, False, self.pos))
        self.held.clear()

    def held_buttons(self):
        return tuple(sorted(self.held))

    def cursor_pos(self):
        return self.pos


@dataclass
class Scenario:
    name: str
    style: str = "pinch_pointer"
    finger: Finger = I                 # the pinched finger
    button: str = "left"
    rest: dict = field(default_factory=dict)   # curls of the non-extended fingers at rest
    pointer_flex: float = 0.0          # index flexion while another finger pinches
    gap: float = 0.0                   # contact landmark gap (square units, hand scale 0.18)
    release_to: float = 0.0            # approach the thumb opens back to between clicks
    close_s: tuple = (0.10, 0.25)
    hold_s: tuple = (0.06, 0.25)
    open_s: tuple = (0.08, 0.15)
    curl: tuple = (0.45, 0.85)         # pinched-finger flexion at contact
    finger_mode: str = "manual"
    squeeze: float = 0.0               # grip variation while held (approach amplitude)
    glitch: bool = False               # one-frame thumb-landmark jump in the middle of the hold
    noise: float = 0.0012
    depth_noise: float | None = None   # world landmarks (production path) with fingertip z noise (m)
    scale: float = 1.0                 # hand size in the image (camera distance)


RELAXED = {M: 0.55, R: 0.6, L: 0.6}

SCENARIOS = [
    Scenario("baseline (v1.1 bench conditions)"),
    Scenario("relaxed hand", rest=RELAXED),
    Scenario("contact gap 0.15", gap=0.027),
    Scenario("partial release", release_to=0.55, gap=0.02),
    Scenario("quick taps", close_s=(0.06, 0.10), hold_s=(0.03, 0.07), open_s=(0.05, 0.08), gap=0.02),
    Scenario("deep slow pinch", close_s=(0.30, 0.45), curl=(0.8, 0.95), gap=0.02),
    Scenario("right click, index flexes", finger=M, button="right", rest={R: 0.6, L: 0.6},
             pointer_flex=0.5, gap=0.02),
    Scenario("Pinch Free left, index flexes", style="pinch_free", finger=M, rest={R: 0.6, L: 0.6},
             pointer_flex=0.5, gap=0.02),
    Scenario("realistic mix", rest=RELAXED, gap=0.025, release_to=0.5),
    Scenario("realistic mix, right click", finger=M, button="right", rest={R: 0.55, L: 0.6},
             pointer_flex=0.45, gap=0.025, release_to=0.5),
    # re-opens to 0.31-0.36 hand scales: below v1.1's fixed 0.36 release
    Scenario("small release (re-opens to 0.31-0.36)", release_to=0.66, gap=0.02, curl=(0.6, 0.85)),
    Scenario("index flexes hard, right click", finger=M, button="right", rest={R: 0.6, L: 0.6},
             pointer_flex=0.7, gap=0.02),
    Scenario("long noisy holds, grip varies", gap=0.02, hold_s=(0.4, 1.2), squeeze=0.12, noise=0.003),
    Scenario("auto finger mode, relaxed", rest=RELAXED, gap=0.025, finger_mode="auto"),
    Scenario("1-frame tracking glitch in hold", gap=0.02, hold_s=(0.3, 0.6), glitch=True),
    # v1.1.2 - the owner's "stuck in Aim Lock" geometry: the contact stops a
    # little above the engage threshold (v1.1.1: 0 % clicked)
    Scenario("contact settles at 1.17x threshold", rest=RELAXED, gap=0.05, release_to=0.3),
    Scenario("contact settles at 1.27x threshold", rest=RELAXED, gap=0.055, release_to=0.3),
    Scenario("right click settles at 1.19x", finger=M, button="right", rest={R: 0.55, L: 0.6},
             pointer_flex=0.45, gap=0.05, release_to=0.3, curl=(0.6, 0.85)),
    Scenario("world landmarks, 8 mm depth noise", rest=RELAXED, gap=0.025, release_to=0.5, depth_noise=0.008),
    Scenario("hand far (0.6x size)", rest=RELAXED, gap=0.025, scale=0.6, noise=0.0015),
    Scenario("hand close (1.6x size)", rest=RELAXED, gap=0.025, scale=1.6),
]


def _pose(sc: Scenario, approach: float, curl: float) -> Pose:
    curls = dict(sc.rest)
    extended = {I}
    if sc.finger is not I and sc.pointer_flex:
        extended = set()
        curls[I] = sc.pointer_flex * min(approach / 0.6, 1.0)
    if approach <= 0.0:
        return Pose(extended=extended, curls=curls or None)
    return Pose(extended=extended, pinch=sc.finger, pinch_curl=curl, approach=approach,
                curls=curls or None, gap=sc.gap)


def run_scenario(sc: Scenario, trials: int = 30, clicks: int = 3, noise: float | None = None,
                 seed: int = 101, settings: Settings | None = None) -> dict:
    rng = np.random.default_rng(seed)
    noise = sc.noise if noise is None else noise
    res = {"ok": 0, "sticky": 0, "missed": 0, "stuck": 0, "wrong_pos": 0, "duplicate": 0,
           "wrong_button": 0}
    latency, up_lag, disp, wrong = [], [], [], []
    labels: dict[str, int] = {}
    for _ in range(trials):
        s = settings or Settings()
        s = Settings.from_dict(s.to_dict())
        s.gesture.click_style = sc.style
        s.finger.mode = sc.finger_mode
        s.finger.finger = "index"
        m = TimedMouse()
        e = InteractionEngine(s, SCREEN, m)
        t = float(rng.uniform(0, 1 / FPS))
        x, y = 0.5, 0.72
        e.set_armed(True, t)

        def step(pose: Pose):
            nonlocal t
            t += 1 / FPS
            m.t = t
            st = e.process(frame(t, observation(pose, x, y, noise=noise, rng=rng, scale=sc.scale,
                                                world=sc.depth_noise is not None,
                                                depth_noise=sc.depth_noise or 0.0)))
            labels[st.gesture.split(" (")[0]] = labels.get(st.gesture.split(" (")[0], 0) + 1

        for _ in range(30):
            step(_pose(sc, 0.0, 0.0))
        start_a = 0.0
        prev_down = None
        curl = float(rng.uniform(*sc.curl))     # one hand, one pinch shape per trial
        for k in range(clicks):
            aim = e.pointer.cursor
            close_s, hold_s, open_s = (float(rng.uniform(*r)) for r in (sc.close_s, sc.hold_s, sc.open_s))
            n0 = len(m.events)
            t0 = t
            while t - t0 < close_s:
                a = start_a + (1 - start_a) * _min_jerk(t + 1 / FPS - t0, close_s)
                step(_pose(sc, a, curl))
            t_contact = t
            glitched = False
            while t - t_contact < hold_s:
                wob = sc.squeeze * (0.5 - 0.5 * math.cos(2 * math.pi * (t - t_contact) / 0.45))
                if sc.glitch and not glitched and t - t_contact > hold_s / 2:
                    glitched, wob = True, 0.6
                step(_pose(sc, 1.0 - wob, curl))
            t_open = t
            end_a = sc.release_to if k < clicks - 1 else 0.0
            while t - t_open < open_s:
                a = 1 - (1 - end_a) * _min_jerk(t + 1 / FPS - t_open, open_s)
                step(_pose(sc, a, curl))
            for _ in range(8):                         # brief pause before the next pinch
                step(_pose(sc, end_a, curl))
            start_a = end_a
            ev = [x_ for x_ in m.events[n0:] if x_[1] in ("button", "click")]
            downs = [x_ for x_ in ev if (x_[1] == "button" and x_[3]) or x_[1] == "click"]
            ups = [x_ for x_ in ev if (x_[1] == "button" and not x_[3]) or x_[1] == "click"]
            if not downs:
                res["missed"] += 1
                continue
            if len(downs) > 1:
                res["duplicate"] += 1
                continue
            if downs[0][2] != sc.button:
                res["wrong_button"] += 1
                continue
            if not ups or ups[0][0] - t_open > 0.25:
                res["stuck"] += 1
                if m.held:
                    m.release_all()
                    e._safe_release()
                continue
            if aim is not None:
                disp.append(math.dist(downs[0][4], aim))
            if aim is not None and math.dist(downs[0][4], aim) > 5.0:
                sticky = (prev_down is not None and downs[0][0] - prev_down[0] < 0.5
                          and math.dist(downs[0][4], prev_down[1]) < 0.5)
                prev_down = (downs[0][0], downs[0][4])
                res["sticky" if sticky else "wrong_pos"] += 1
                if not sticky:
                    wrong.append((round(math.dist(downs[0][4], aim), 1), k))
                continue
            prev_down = (downs[0][0], downs[0][4])
            res["ok"] += 1
            latency.append((downs[0][0] - t_contact) * 1000)
            up_lag.append((ups[0][0] - t_open) * 1000)
    n = trials * clicks
    return {"scenario": sc.name, "clicks": n, **res, "success": round((res["ok"] + res["sticky"]) / n, 3),
            "down_after_contact_ms": round(statistics.median(latency), 1) if latency else None,
            "up_after_open_ms": round(statistics.median(up_lag), 1) if up_lag else None,
            "click_px_p95": round(sorted(disp)[int(0.95 * (len(disp) - 1))], 1) if disp else None,
            "wrong_pos_px": wrong[:6],
            "labels": dict(sorted(labels.items(), key=lambda kv: -kv[1])[:6])}


def run_all(trials: int = 30, settings: Settings | None = None) -> list[dict]:
    return [run_scenario(sc, trials=trials, settings=settings) for sc in SCENARIOS]


if __name__ == "__main__":   # pragma: no cover
    import json
    import sys
    rows = run_all(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
    for r in rows:
        print(json.dumps(r))
