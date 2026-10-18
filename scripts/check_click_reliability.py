"""Controlled Windows click-reliability check (v1.1.2: Instant Auto-Click, Drag Mode).

Synthetic hands drive the production InteractionEngine -> MouseDispatcher ->
real SendInput, with the on-screen feedback ring shown exactly as in the app.
A *separate process* (scripts/_click_target.py, real Tk widgets) records,
independently of AirDesk, every mouse press/release it receives and every
widget activation. For each interaction three things are checked separately:

  gesture   AirDesk accepted the press (interaction log)
  Windows   button-down and button-up were both submitted (SendInput result)
  target    the intended widget activated, nothing else did

Phases (default Instant Auto-Click unless named): 30 primary clicks (15
Normal, then 15 after switching to Pinch Free), 30 secondary clicks, rapid
repeated clicks, a double click, 10 small-checkbox clicks, pinches held 1.0 /
2.0 s (one click each, at the pinch, no drag), a contact that settles at
~1.2x the threshold (the owner's "stuck in Aim Lock" geometry), Aim Lock
alone (pinch stops short: nothing may reach the target), Drag Mode (1.25 s
hold + move = one drag, an early release = nothing), tracking loss and the
emergency stop during a Drag Mode drag, a restart (new engine from the saved
settings), 🤟 keyboard toggles (no mouse events), and the feedback states
per click (✓ only for completed clicks).

Moves the real cursor for about two minutes; do not touch the mouse (that
makes AirDesk yield, by design). Writes bench_results/click_integration.json.

  .venv\\Scripts\\python scripts\\check_click_reliability.py
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication  # noqa: E402

from airdesk.bench.clicks import Scenario, _pose  # noqa: E402
from airdesk.config import Settings  # noqa: E402
from airdesk.core.engine import InteractionEngine  # noqa: E402
from airdesk.core.types import Finger  # noqa: E402
from airdesk.platform import win32  # noqa: E402
from airdesk.platform.dispatcher import MouseDispatcher  # noqa: E402
from airdesk.sim.synthetic import Pose, frame, observation  # noqa: E402
from airdesk.ui.feedback import FeedbackMarker, feedback_state  # noqa: E402

I, M, R, L = Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.LITTLE


class Check:
    def __init__(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="airdesk-click-"))
        self.layout_p, self.events_p = self.tmp / "layout.json", self.tmp / "events.jsonl"
        self.target = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "_click_target.py"),
                                        str(self.layout_p), str(self.events_p)])
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.marker = FeedbackMarker()
        self.rng = random.Random(7)
        self.s = Settings()
        self.s.finger.mode = "manual"
        self.disp = MouseDispatcher(120, 10.0)
        self.commands: list[str] = []
        self.eng = InteractionEngine(self.s, win32.screen_layout(), self.disp, on_command=self.commands.append)
        self.x, self.y = 0.5, 0.6
        self.marker_kinds: list[str] = []
        # Frames are processed on a worker thread, like the app's vision
        # thread, while this (GUI) thread owns the feedback ring and pumps
        # Qt. (SendInput from the ring's own thread measured ~4 ms slower.)
        self.vision = ThreadPoolExecutor(1, thread_name_prefix="vision")
        self._ev_pos = 0
        self.events: list[dict] = []

    # ---------------------------------------------------------------- io
    def layout(self) -> dict:
        for _ in range(100):
            if self.layout_p.exists():
                try:
                    return json.loads(self.layout_p.read_text(encoding="utf-8"))
                except ValueError:
                    pass
            time.sleep(0.1)
        raise RuntimeError("target window did not start")

    def read_events(self) -> list[dict]:
        if self.events_p.exists():
            with self.events_p.open("rb") as f:
                f.seek(self._ev_pos)
                data = f.read()
            cut = data.rfind(b"\n") + 1
            self._ev_pos += cut
            self.events += [json.loads(line) for line in data[:cut].decode("utf-8").splitlines()
                            if line.strip()]
        return self.events

    def pump(self, ms: float) -> None:
        end = time.perf_counter() + ms / 1000
        while time.perf_counter() < end:
            self.app.processEvents()
            time.sleep(0.004)

    # ------------------------------------------------------------- driving
    def feed(self, pose: Pose | None, n: int = 1, target=None, dx: float = 0.0) -> None:
        for _ in range(n):
            if target is not None and self.eng.pointer.cursor is not None:
                cx, cy = self.eng.pointer.cursor
                vs = self.eng.screen.target_rect(self.s.pointer.monitor)
                self.x += (target[0] - cx) / vs.width * self.s.pointer.box_w * 0.6
                self.y += (target[1] - cy) / vs.height * self.s.pointer.box_h * 0.6
            self.x += dx
            hands = [observation(pose, self.x, self.y)] if pose is not None else []
            fut = self.vision.submit(self.eng.process, frame(time.perf_counter(), *hands))
            while not fut.done():
                self.app.processEvents()
                time.sleep(0.0005)
            st = fut.result()
            self.disp.enabled = self.eng.armed and not self.eng.paused
            self.marker.update_from(st, True, self.eng.clicks.latest())
            kind = feedback_state(st, self.eng.clicks.latest(), time.perf_counter())[0]
            if not self.marker_kinds or self.marker_kinds[-1] != kind:
                self.marker_kinds.append(kind)
            self.pump(33)

    def aim(self, pt, rest=None) -> tuple[float, float]:
        sc = Scenario("aim", rest=rest or {})
        ok = 0
        for _ in range(60):
            self.feed(_pose(sc, 0.0, 0.0), target=pt)
            c = self.eng.pointer.cursor
            ok = ok + 1 if c and math.dist(c, pt) < 1.0 else 0
            if ok >= 3:
                break
        self.feed(_pose(sc, 0.0, 0.0), 3)
        return self.eng.pointer.cursor

    def pinch(self, sc: Scenario, close=3, hold=3, opening=3, curl=0.65, hold_dx=0.0) -> None:
        for k in range(close):
            self.feed(_pose(sc, (k + 1) / close, curl))
        self.feed(_pose(sc, 1.0, curl), hold, dx=hold_dx)
        self.t_open = time.perf_counter()
        for k in range(opening):
            self.feed(_pose(sc, 1 - (k + 1) / opening, curl))
        self.feed(_pose(sc, 0.0, curl), 6)

    def click(self, pt, sc: Scenario, **kw) -> dict:
        aim = self.aim(pt, sc.rest)
        t0 = time.perf_counter()
        last = self.eng.clicks.latest()
        self.marker_kinds = []
        self.pinch(sc, **kw)
        self.pump(120)
        it = self.eng.clicks.latest()
        it = it if it is not None and (last is None or it.id != last.id) else None
        ev = [e for e in self.read_events() if e["t"] >= t0]
        return {"aim": aim, "it": it, "ev": ev, "marker": list(self.marker_kinds), "t_open": self.t_open}


def center(r):
    return (r[0] + r[2] / 2, r[1] + r[3] / 2)


def summarise(rows: list[dict], name: str, button: int) -> dict:
    acc = sum(1 for r in rows if r["it"] is not None)
    comp = sum(1 for r in rows if r["it"] is not None and r["it"].phase == "completed")
    act = sum(1 for r in rows if [e for e in r["ev"] if e["kind"] == "activate" and e["name"] == name])
    other = sum(len([e for e in r["ev"] if e["kind"] == "activate" and e["name"] != name]) for r in rows)
    pairs = sum(1 for r in rows if [(e["kind"], e["button"]) for e in r["ev"] if e["kind"] in ("press", "release")]
                == [("press", button), ("release", button)])
    err = [math.dist((p["x"], p["y"]), r["aim"]) for r in rows
           for p in [e for e in r["ev"] if e["kind"] == "press"][:1]]
    dl = [r["it"].confirm_to_down_ms() for r in rows if r["it"] is not None and r["it"].wall_down]
    rx = [p["t"] - r["it"].wall_down for r in rows if r["it"] is not None and r["it"].wall_down
          for p in [e for e in r["ev"] if e["kind"] == "press"][:1]]
    rec = [r["it"].recognition_ms() for r in rows if r["it"] is not None and r["it"].recognition_ms() is not None]
    before_open = sum(1 for r in rows if r["it"] is not None and r["it"].wall_up is not None
                      and r.get("t_open") is not None and r["it"].wall_up <= r["t_open"])
    ring = sum(1 for r in rows if "done" in r["marker"])
    false_ring = sum(1 for r in rows if "done" in r["marker"] and (r["it"] is None or r["it"].phase != "completed"))
    return {"trials": len(rows), "gesture_accepted": acc, "windows_completed": comp, "target_activated": act,
            "other_activations": other, "exact_press_release_pairs": pairs,
            "press_error_px_max": round(max(err), 1) if err else None,
            "accept_to_sendinput_ms_median": round(statistics.median(dl), 3) if dl else None,
            "accept_to_sendinput_ms_max": round(max(dl), 3) if dl else None,
            "sendinput_to_target_ms_median": round(statistics.median(rx) * 1000, 2) if rx else None,
            "sendinput_to_target_ms_max": round(max(rx) * 1000, 2) if rx else None,
            "contact_to_recognition_ms_median": round(statistics.median(rec), 1) if rec else None,
            "completed_before_fingers_opened": before_open,
            "done_marker_shown": ring, "done_ring_without_completed_click": false_ring}


def main() -> int:
    c = Check()
    lay = c.layout()
    saved = win32.cursor_pos()
    c.disp.start()
    c.eng.set_armed(True, time.perf_counter())
    c.disp.enabled = True
    res: dict = {}
    try:
        c.feed(Pose.pointing(I), 15)
        # 1) 30 primary clicks: 15 Normal (thumb+index), then switch to Pinch Free (thumb+middle)
        rows = []
        for k in range(30):
            if k == 15:
                c.s.gesture.click_style = "pinch_free"
                c.eng.configure(c.s)
            free = k >= 15
            sc = Scenario("p", finger=M if free else I, rest=({R: 0.6, L: 0.6} if free else
                                                             ({M: 0.55, R: 0.6, L: 0.6} if k % 2 else {})),
                          pointer_flex=0.4 if free else 0.0, gap=0.02)
            r0 = lay["primary"]
            pt = (r0[0] + r0[2] * (0.3 + 0.4 * c.rng.random()), r0[1] + r0[3] * (0.3 + 0.4 * c.rng.random()))
            quick = k % 5 == 0
            rows.append(c.click(pt, sc, close=2 if quick else c.rng.choice((3, 4)), hold=1 if quick else
                                c.rng.choice((2, 3, 5, 8)), opening=2 if quick else 3, curl=c.rng.uniform(0.5, 0.8)))
        res["primary_clicks"] = summarise(rows, "primary", 1)
        res["quick_pinch_subset"] = summarise(rows[::5], "primary", 1)
        c.s.gesture.click_style = "pinch_pointer"
        c.eng.configure(c.s)
        # 2) 30 secondary clicks (Normal: thumb+middle while the index bends a little)
        rows = []
        r0 = lay["secondary"]
        for k in range(30):
            pt = (r0[0] + r0[2] * (0.3 + 0.4 * c.rng.random()), r0[1] + r0[3] * (0.3 + 0.4 * c.rng.random()))
            sc = Scenario("s", finger=M, button="right", rest={R: 0.6, L: 0.6}, pointer_flex=0.45, gap=0.02)
            rows.append(c.click(pt, sc, close=c.rng.choice((2, 3, 4)), hold=c.rng.choice((1, 2, 4, 6)),
                                curl=c.rng.uniform(0.5, 0.8)))
        res["secondary_clicks"] = summarise(rows, "secondary", 3)
        # 3) rapid repeated deliberate clicks on one target (no re-aim between them)
        pt = center(lay["primary"])
        c.aim(pt)
        t0 = time.perf_counter()
        n_ok0 = c.eng.clicks.snapshot_counters()["completed"]
        for _ in range(10):
            c.pinch(Scenario("r", gap=0.02), close=2, hold=2, opening=2)
        c.pump(150)
        ev = [e for e in c.read_events() if e["t"] >= t0]
        res["rapid_repeated"] = {"pinches": 10,
                                 "windows_completed": c.eng.clicks.snapshot_counters()["completed"] - n_ok0,
                                 "target_activations": sum(1 for e in ev if e["kind"] == "activate"
                                                           and e["name"] == "primary"),
                                 "presses": sum(1 for e in ev if e["kind"] == "press"),
                                 "releases": sum(1 for e in ev if e["kind"] == "release")}
        # 4) double click
        pt = center(lay["double"])
        c.aim(pt)
        t0 = time.perf_counter()
        for _ in range(2):
            for k in range(2):
                c.feed(_pose(Scenario("d", gap=0.02), (k + 1) / 2, 0.65))
            c.feed(_pose(Scenario("d", gap=0.02), 1.0, 0.65), 1)
            for k in range(2):
                c.feed(_pose(Scenario("d", gap=0.02), 1 - (k + 1) / 2, 0.65))
        c.feed(Pose.pointing(I), 8)
        c.pump(150)
        ev = [e for e in c.read_events() if e["t"] >= t0]
        res["double_click"] = {"double_activations": sum(1 for e in ev if e["kind"] == "activate"
                                                         and e["name"] == "double"),
                               "presses": sum(1 for e in ev if e["kind"] == "press")}
        # 5) small checkboxes (13-15 px, 20 px pitch): click a given one, neighbours must not toggle
        rows, wrong = [], 0
        for k in range(10):
            i = 1 + k % 5
            pt = center(lay["checks"][i])
            r = c.click(pt, Scenario("c", rest={M: 0.55, R: 0.6, L: 0.6} if k % 2 else {}, gap=0.02),
                        close=3, hold=c.rng.choice((2, 3, 4)), curl=c.rng.uniform(0.5, 0.8))
            wrong += sum(1 for e in r["ev"] if e["kind"] == "activate" and e["name"].startswith("check")
                         and e["name"] != f"check{i}")
            r["ev"] = [dict(e, name="check") if e.get("name") == f"check{i}" else e for e in r["ev"]]
            rows.append(r)
        res["small_checkboxes"] = summarise(rows, "check", 1)
        res["small_checkboxes"]["neighbour_toggles"] = wrong
        # 6) instant: pinches held 1.0 / 2.0 s click once, at the pinch, never drag
        for hold_s in (1.0, 2.0):
            r = c.click(center(lay["primary"]), Scenario("h", gap=0.02), hold=int(hold_s * 30) - 3)
            pr = [e for e in r["ev"] if e["kind"] == "press"]
            res[f"hold_{hold_s}s"] = {"windows": r["it"].phase if r["it"] else None,
                                      "kind": r["it"].kind if r["it"] else None,
                                      "presses": len(pr),
                                      "press_before_fingers_opened": bool(pr) and pr[0]["t"] < r["t_open"],
                                      "target_activated": sum(1 for e in r["ev"] if e["kind"] == "activate"
                                                              and e["name"] == "primary")}
        # 7) the owner's failure: the contact settles at ~1.2x the threshold
        rows = [c.click(center(lay["primary"]), Scenario("settle", gap=0.05, rest={M: 0.55, R: 0.6, L: 0.6}),
                        close=4, hold=8, curl=0.7) for _ in range(5)]
        res["settled_contact_1.2x"] = summarise(rows, "primary", 1)
        # 8) Aim Lock alone (the pinch stops short at ~1.8x): nothing may reach the target
        c.aim(center(lay["primary"]))
        t0 = time.perf_counter()
        for _ in range(3):
            for k in range(4):
                c.feed(Pose.closing(I, 0.5 * (k + 1) / 4, curl=0.6))
            c.feed(Pose.closing(I, 0.5, curl=0.6), 20)
            c.feed(Pose.pointing(I), 8)
        ev = [e for e in c.read_events() if e["t"] >= t0]
        res["aim_lock_alone"] = {"mouse_events": sum(1 for e in ev if e["kind"] in ("press", "release")),
                                 "aim_released_logged": c.eng.clicks.snapshot_counters()["aim_released"]}
        # 9) Drag Mode: release before 1.2 s = nothing; 1.25 s hold + move = one drag (one-shot)
        d = lay["drag"]
        sc = Scenario("g", gap=0.02)
        c.eng.set_click_mode("drag")
        c.aim((d[0] + 120, d[1] + d[3] / 2))
        t0 = time.perf_counter()
        c.pinch(sc, close=3, hold=20)                      # ~0.8 s: released before the threshold
        ev = [e for e in c.read_events() if e["t"] >= t0]
        early = {"mouse_events": sum(1 for e in ev if e["kind"] in ("press", "release")),
                 "logged": c.eng.clicks.latest().reason if c.eng.clicks.latest() else None}
        c.aim((d[0] + 120, d[1] + d[3] / 2))
        t0 = time.perf_counter()
        for k in range(3):
            c.feed(_pose(sc, (k + 1) / 3, 0.65))
        t_conf = None
        held_at = None
        for _ in range(45):
            c.feed(_pose(sc, 1.0, 0.65))
            if t_conf is None and c.eng.state.value == "PRESS_PENDING":
                t_conf = time.perf_counter()
            if held_at is None and c.disp.held_buttons():
                held_at = time.perf_counter()
                break
        pre = [e for e in c.read_events() if e["t"] >= t0 and e["kind"] == "press"]
        c.feed(_pose(sc, 1.0, 0.65), 25, dx=0.006)
        c.feed(Pose.pointing(I), 8)
        c.pump(150)
        ev = [e for e in c.read_events() if e["t"] >= t0]
        dr = [e for e in ev if e["kind"] == "activate" and e["name"] == "drag"]
        res["drag_mode"] = {"early_release": early,
                            "hold_to_button_down_s": round(held_at - t_conf, 3) if held_at and t_conf else None,
                            "press_before_threshold": bool(pre and held_at and pre[0]["t"] < held_at - 0.05),
                            "drags": len(dr),
                            "distance_px": (dr[0]["end"][0] - dr[0]["start"][0]) if dr else None,
                            "moves_while_held": dr[0]["moves"] if dr else 0,
                            "presses": sum(1 for e in ev if e["kind"] == "press"),
                            "releases": sum(1 for e in ev if e["kind"] == "release"),
                            "mode_after_drop": c.eng.click_mode}
        # 10) tracking loss and emergency stop during a Drag Mode drag -> button released
        for how in ("tracking_loss", "emergency_stop"):
            c.eng.set_click_mode("drag")
            c.aim((d[0] + 300, d[1] + 60))
            t0 = time.perf_counter()
            for k in range(3):
                c.feed(_pose(sc, (k + 1) / 3, 0.65))
            c.feed(_pose(sc, 1.0, 0.65), 40)
            held = c.disp.held_buttons()
            if how == "tracking_loss":
                c.feed(None, 14)
            else:
                c.eng.emergency_stop()
                c.disp.enabled = False
                c.disp.release_all("emergency stop")
                c.feed(_pose(sc, 1.0, 0.65), 5)
            c.pump(120)
            ev = [e for e in c.read_events() if e["t"] >= t0]
            it = c.eng.clicks.latest()
            res[how] = {"held_before": held, "held_after": c.disp.held_buttons(),
                        "system_left_down_after": win32.key_is_down(0x01),
                        "target_press_release": [e["kind"] for e in ev if e["kind"] in ("press", "release")],
                        "logged": f"{it.phase}: {it.reason}" if it else None}
            if how == "emergency_stop":
                c.eng.set_armed(True, time.perf_counter())
                c.disp.enabled = True
            c.eng.set_click_mode("instant")
            c.feed(Pose.pointing(I), 10)
        # 11) restart: a new engine from the saved settings keeps Instant + 1.2 s and clicks
        path = c.tmp / "settings.json"
        c.s.save(path)
        s2 = Settings.load(path)
        c.eng = InteractionEngine(s2, win32.screen_layout(), c.disp, on_command=c.commands.append)
        c.eng.set_armed(True, time.perf_counter())
        c.feed(Pose.pointing(I), 15)
        r = c.click(center(lay["primary"]), Scenario("restart", gap=0.02))
        res["after_restart"] = {"click_mode": s2.gesture.click_mode, "hold_to_drag_s": s2.gesture.hold_to_drag_s,
                                "keyboard_gesture_s": s2.gesture.keyboard_gesture_s,
                                "target_activated": sum(1 for e in r["ev"] if e["kind"] == "activate"
                                                        and e["name"] == "primary")}
        # 12) 🤟 toggles the keyboard command twice (open, close) and never clicks
        c.aim(center(lay["primary"]))
        t0 = time.perf_counter()
        n0 = len(c.commands)
        for _ in range(2):
            c.feed(Pose(extended={I}, curls={L: 0.5}), 2)
            c.feed(Pose.ily(), 30)
            c.feed(Pose.pointing(I), 10)
        ev = [e for e in c.read_events() if e["t"] >= t0]
        res["ily_keyboard"] = {"toggles": c.commands[n0:].count("toggle_keyboard"),
                               "mouse_events": sum(1 for e in ev if e["kind"] in ("press", "release"))}
        # 13) disarmed: nothing reaches the target
        c.eng.set_armed(False, time.perf_counter())
        c.disp.enabled = False
        t0 = time.perf_counter()
        c.pinch(Scenario("x", gap=0.02))
        c.pump(120)
        res["disarmed_events"] = sum(1 for e in c.read_events() if e["t"] >= t0)
        res["dispatcher"] = dict(c.disp.stats)
        res["interaction_log"] = c.eng.clicks.snapshot_counters()
    finally:
        c.disp.stop()
        c.marker.hide()
        c.target.terminate()
        win32.mouse_move_abs(*saved)

    p, s2 = res["primary_clicks"], res["secondary_clicks"]
    ok = (p["trials"] == p["gesture_accepted"] == p["windows_completed"] == p["target_activated"] == 30
          and p["other_activations"] == 0 and p["exact_press_release_pairs"] == 30
          and s2["gesture_accepted"] == s2["windows_completed"] == s2["target_activated"] == 30
          and s2["exact_press_release_pairs"] == 30 and s2["other_activations"] == 0
          and p["done_ring_without_completed_click"] == 0 and s2["done_ring_without_completed_click"] == 0
          and p["completed_before_fingers_opened"] == 30 and s2["completed_before_fingers_opened"] == 30
          and res["rapid_repeated"]["target_activations"] == 10 == res["rapid_repeated"]["presses"]
          and res["double_click"]["double_activations"] == 1
          and res["small_checkboxes"]["target_activated"] == 10 and res["small_checkboxes"]["neighbour_toggles"] == 0
          and all(res[f"hold_{h}s"]["target_activated"] == 1 and res[f"hold_{h}s"]["presses"] == 1
                  and res[f"hold_{h}s"]["press_before_fingers_opened"] for h in (1.0, 2.0))
          and res["settled_contact_1.2x"]["target_activated"] == 5
          and res["settled_contact_1.2x"]["other_activations"] == 0
          and res["aim_lock_alone"]["mouse_events"] == 0
          and res["drag_mode"]["early_release"]["mouse_events"] == 0
          and res["drag_mode"]["drags"] == 1 and (res["drag_mode"]["distance_px"] or 0) > 100
          and res["drag_mode"]["presses"] == res["drag_mode"]["releases"] == 1
          and not res["drag_mode"]["press_before_threshold"]
          and 1.15 <= (res["drag_mode"]["hold_to_button_down_s"] or 0) <= 1.3
          and res["drag_mode"]["mode_after_drop"] == "instant"
          and all(res[k]["held_before"] == ("left",) and res[k]["held_after"] == ()
                  and not res[k]["system_left_down_after"] for k in ("tracking_loss", "emergency_stop"))
          and res["after_restart"]["click_mode"] == "instant" and res["after_restart"]["target_activated"] == 1
          and res["ily_keyboard"] == {"toggles": 2, "mouse_events": 0}
          and res["disarmed_events"] == 0)
    res["result"] = "PASS" if ok else "FAIL"
    res["date"] = time.strftime("%Y-%m-%d %H:%M")
    res["label"] = "real SendInput into a separate-process Tk window; hands are synthetic (no camera)"
    out = ROOT / "bench_results" / "click_integration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    for k, v in res.items():
        print(f"{k:22} {v}")
    print(res["result"])
    return 0 if ok else 1


if __name__ == "__main__":
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    raise SystemExit(main())
