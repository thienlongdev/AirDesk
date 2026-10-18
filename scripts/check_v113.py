"""Controlled Windows check for v1.1.3: Pinky Drag and the Dual-Hand Air Keyboard.

Synthetic hands drive the production InteractionEngine; everything after it
is real: MouseDispatcher -> SendInput for the mouse, KeyboardOverlay ->
KeyboardOutput -> SendInput for keys. Two *separate processes* record what
Windows actually delivered, independently of AirDesk:

Part A - scripts/_click_target.py (real Tk widgets):
  * 10 Instant Auto-Clicks (Normal style) - the v1.1.2 path, unchanged
  * 5 Pinky Drags on the drag canvas: exactly one press at the aimed point,
    motion, one release, no click first; acquisition time (0.2 s target)
  * 3 pinky poses released before 0.2 s: nothing reaches the target
  * a drag ended by tracking loss and one by the emergency stop: the button
    is released by AirDesk itself
  * an Instant click right after a drop (hand-over back to clicking)

Part B - scripts/_keyboard_target.py (a Tk editor that logs key events):
  * left hand only, right hand only, both hands alternating, both hands in
    the same frame (two different keys)
  * Ctrl+A / Ctrl+C / Ctrl+V / Ctrl+Z with the modifier on one hand and the
    letter on the other (clipboard content verified), Shift+A, Ctrl+Shift+S
    (delivered as a chord with both modifier bits), Telex "việt"
  * an armed Ctrl is cleared when both hands leave the camera
  * the editor stays the foreground window, the overlay never takes focus,
    the mouse cursor never moves and no mouse button reaches anything
  * 🤟 closes the keyboard

Moves the real mouse in part A (~40 s) and types into its own editor in
part B (~60 s): do not touch the mouse or keyboard meanwhile. The user's
clipboard text is saved and restored. Writes bench_results/v113_integration.json.

  .venv\\Scripts\\python scripts\\check_v113.py
"""

from __future__ import annotations

import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from check_click_reliability import Check, center, summarise  # noqa: E402

from airdesk.bench.clicks import Scenario  # noqa: E402
from airdesk.config import KeyboardConfig, Settings  # noqa: E402
from airdesk.core.dual_keyboard import EDGE  # noqa: E402
from airdesk.core.engine import InteractionEngine  # noqa: E402
from airdesk.core.hand_model import analyze  # noqa: E402
from airdesk.core.types import Finger  # noqa: E402
from airdesk.platform import win32  # noqa: E402
from airdesk.platform.dispatcher import MouseDispatcher  # noqa: E402
from airdesk.sim.synthetic import Pose, frame, observation  # noqa: E402
from airdesk.ui.keyboard_overlay import KeyboardOverlay  # noqa: E402

I, L = Finger.INDEX, Finger.LITTLE


def transition(c: float) -> Pose:
    return Pose(extended=set(), curls={I: c, L: 1.0 - c})


PINKY = transition(1.0)


# ===================================================================== part A
def part_a(c: Check, lay: dict) -> dict:
    res: dict = {}
    c.feed(Pose.pointing(I), 15)
    rows = []
    r0 = lay["primary"]
    for k in range(10):
        pt = (r0[0] + r0[2] * (0.3 + 0.4 * c.rng.random()), r0[1] + r0[3] * (0.3 + 0.4 * c.rng.random()))
        rows.append(c.click(pt, Scenario("p", gap=0.02), close=c.rng.choice((2, 3, 4)),
                            hold=c.rng.choice((1, 2, 3, 5)), curl=c.rng.uniform(0.5, 0.8)))
    res["instant_clicks"] = summarise(rows, "primary", 1)

    d = lay["drag"]
    drags = []
    for k in range(5):
        start = (d[0] + d[2] * (0.1 + 0.15 * k), d[1] + d[3] * (0.3 + 0.1 * k))
        aim = c.aim(start)
        t0 = time.perf_counter()
        last = c.eng.clicks.latest()
        for j in range(4):
            c.feed(transition((j + 1) / 4))
        c.feed(PINKY, 8)
        c.feed(PINKY, 15, dx=0.0025)               # ~ +190 px on a 2560 px screen
        for j in range(4):
            c.feed(transition(1 - (j + 1) / 4))
        c.feed(Pose.pointing(I), 6)
        c.pump(150)
        ev = [e for e in c.read_events() if e["t"] >= t0]
        it = c.eng.clicks.latest()
        it = it if it is not None and (last is None or it.id != last.id) else None
        presses = [e for e in ev if e["kind"] == "press"]
        act = [e for e in ev if e["kind"] == "activate" and e["name"] == "drag"]
        drags.append({
            "presses": len(presses), "releases": sum(1 for e in ev if e["kind"] == "release"),
            "press_button": presses[0]["button"] if presses else None,
            "press_error_px": round(math.dist((presses[0]["x"], presses[0]["y"]), aim), 1) if presses else None,
            "drag_activations": len(act),
            "moved_px": round(act[0]["end"][0] - act[0]["start"][0], 1) if act else None,
            "motion_events": act[0]["moves"] if act else 0,
            "other_activations": sum(1 for e in ev if e["kind"] == "activate" and e["name"] != "drag"),
            "acquisition_ms": round((it.wall_down - it.wall_confirm) * 1000, 1)
            if it is not None and it.wall_down else None,
            "phase": it.phase if it else None})
    res["pinky_drags"] = {
        "trials": len(drags),
        "exact_one_press_one_release": sum(1 for x in drags if x["presses"] == 1 and x["releases"] == 1),
        "drag_activations": sum(x["drag_activations"] for x in drags),
        "other_activations": sum(x["other_activations"] for x in drags),
        "press_error_px_max": max((x["press_error_px"] for x in drags if x["press_error_px"] is not None),
                                  default=None),
        "moved_px_min": min((x["moved_px"] for x in drags if x["moved_px"] is not None), default=None),
        "acquisition_ms_median": statistics.median([x["acquisition_ms"] for x in drags if x["acquisition_ms"]])
        if any(x["acquisition_ms"] for x in drags) else None,
        "completed": sum(1 for x in drags if x["phase"] == "completed"), "rows": drags}

    # Pose released before 0.2 s: nothing at all.
    t0 = time.perf_counter()
    for _ in range(3):
        c.aim(center(lay["drag"]))
        for j in range(4):
            c.feed(transition((j + 1) / 4))
        c.feed(PINKY, 2)
        for j in range(4):
            c.feed(transition(1 - (j + 1) / 4))
        c.feed(Pose.pointing(I), 6)
    c.pump(150)
    ev = [e for e in c.read_events() if e["t"] >= t0]
    res["early_release"] = {"trials": 3, "presses": sum(1 for e in ev if e["kind"] == "press"),
                            "activations": sum(1 for e in ev if e["kind"] == "activate")}

    # Tracking loss and emergency stop during a drag.
    for how in ("tracking_loss", "emergency_stop"):
        c.aim(center(lay["drag"]))
        t0 = time.perf_counter()
        for j in range(4):
            c.feed(transition((j + 1) / 4))
        c.feed(PINKY, 10, dx=0.002)
        held = c.disp.held_buttons()
        if how == "tracking_loss":
            c.feed(None, 15)
        else:
            c.eng.emergency_stop()
            c.disp.enabled = False
            c.pump(100)
        c.pump(150)
        ev = [e for e in c.read_events() if e["t"] >= t0]
        res[how] = {"held_during_drag": list(held), "held_after": list(c.disp.held_buttons()),
                    "presses": sum(1 for e in ev if e["kind"] == "press"),
                    "releases": sum(1 for e in ev if e["kind"] == "release")}
        if how == "emergency_stop":
            c.eng.set_armed(True, time.perf_counter())
            c.disp.enabled = True
        c.feed(Pose.pointing(I), 15)

    # Hand-over: a normal Instant click right after a Pinky Drag drop.
    c.aim(center(lay["drag"]))
    for j in range(4):
        c.feed(transition((j + 1) / 4))
    c.feed(PINKY, 9, dx=0.001)
    for j in range(4):
        c.feed(transition(1 - (j + 1) / 4))
    c.feed(Pose.pointing(I), 8)
    r = c.click(center(lay["primary"]), Scenario("p", gap=0.02), close=3, hold=2)
    res["click_after_drop"] = summarise([r], "primary", 1)
    return res


# ===================================================================== part B
class Hand:
    def __init__(self, side: str, x: float, s: Settings, kb: KeyboardOverlay):
        self.side, self.x, self.y, self.s, self.kb = side, x, 0.72, s, kb

    def obs(self, pose=None) -> object:
        return observation(pose or Pose.pointing(I), self.x, self.y,
                           handedness="Left" if self.side == "left" else "Right", mirror=self.side == "left",
                           noise=0.0008)

    def aim(self, key: str) -> None:
        r = next(r for r in self.kb.rects if r.key == key)
        fx, fy = r.x + r.w / 2, r.y + r.h / 2
        nx, ny = (fx + EDGE) / (1 + 2 * EDGE), (fy + EDGE) / (1 + 2 * EDGE)
        p = self.s.pointer
        ix = p.box_cx + (nx - 0.5) * p.box_w / p.direct_gain
        iy = p.box_cy + (ny - 0.5) * p.box_h / p.direct_gain
        f = analyze(observation(Pose.pointing(I), self.x, self.y, mirror=self.side == "left"), 16 / 9)
        a = p.anchor_blend
        self.x += ix - (f.tips[I][0] * (1 - a) + f.mcps[I][0] * a)
        self.y += iy - (f.tips[I][1] * (1 - a) + f.mcps[I][1] * a)


class Keys:
    def __init__(self, app) -> None:
        self.app = app
        self.tmp = Path(tempfile.mkdtemp(prefix="airdesk-v113-"))
        self.text_p, self.events_p = self.tmp / "text.txt", self.tmp / "events.jsonl"
        self.target = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "_keyboard_target.py"),
                                        str(self.text_p), str(self.events_p)])
        self.s = Settings()
        self.s.keyboard.suggestions = False
        self.kb = KeyboardOverlay(KeyboardConfig(layout="full", suggestions=False))
        self.kb.cfg.one_shot_modifiers = True
        self.kb.out.mods.one_shot = True
        self.disp = MouseDispatcher(120, 10.0)
        self.commands: list[str] = []
        self.eng = InteractionEngine(self.s, win32.screen_layout(), self.disp, self.kb,
                                     on_command=self.commands.append)
        self.vision = ThreadPoolExecutor(1, thread_name_prefix="vision")
        self.L = Hand("left", 0.36, self.s, self.kb)
        self.R = Hand("right", 0.64, self.s, self.kb)
        self.t = time.perf_counter()
        self.fg_changes: list[str] = []
        self.hwnd = 0
        self.unsafe = False

    def pump(self, ms: float) -> None:
        end = time.perf_counter() + ms / 1000
        while time.perf_counter() < end:
            self.app.processEvents()
            time.sleep(0.004)

    def watch(self) -> None:
        fg = win32.foreground_window()
        if fg != self.hwnd:
            self.fg_changes.append(win32.window_title(fg))
            self.unsafe = True

    def step(self, left=None, right=None, n=1, use_left=True, use_right=True) -> object:
        st = None
        for _ in range(n):
            if self.unsafe:
                raise RuntimeError("the editor lost the foreground - stopped typing (safety)")
            hs = ([self.L.obs(left)] if use_left else []) + ([self.R.obs(right)] if use_right else [])
            fut = self.vision.submit(self.eng.process, frame(time.perf_counter(), *hs))
            while not fut.done():
                self.app.processEvents()
                time.sleep(0.0005)
            st = fut.result()
            while "toggle_keyboard" in self.commands:
                self.commands.remove("toggle_keyboard")
                self.kb.toggle()
            self.pump(30)
            self.watch()
        return st

    def press(self, side: str, key: str, closing=(0.35, 0.7, 1.0), hold=2) -> None:
        (self.L if side == "left" else self.R).aim(key)
        self.step(n=7)
        for a in closing:
            self.step(**{side: Pose.closing(I, a)})
        self.step(**{side: Pose.closing(I, 1.0)}, n=hold)
        for a in (0.6, 0.2, 0.0):
            self.step(**{side: Pose.closing(I, a)})
        self.step(n=2)

    def both(self, lkey: str, rkey: str, offset: int = 0) -> None:
        """Both hands pinch together (offset = frames the right hand lags)."""
        self.L.aim(lkey)
        self.R.aim(rkey)
        self.step(n=7)
        seq = [0.35, 0.7, 1.0, 1.0, 1.0, 0.6, 0.2, 0.0]
        for k in range(len(seq) + offset):
            la = seq[k] if k < len(seq) else 0.0
            ra = seq[k - offset] if 0 <= k - offset < len(seq) else 0.0
            self.step(left=Pose.closing(I, la), right=Pose.closing(I, ra))
        self.step(n=2)

    def text(self) -> str:
        self.pump(250)
        return self.text_p.read_text(encoding="utf-8") if self.text_p.exists() else ""

    def events(self) -> list[dict]:
        if not self.events_p.exists():
            return []
        return [json.loads(x) for x in self.events_p.read_text(encoding="utf-8").splitlines() if x.strip()]


def part_b(app) -> dict:
    k = Keys(app)
    res: dict = {}
    clip = app.clipboard()
    saved_clip = clip.text()
    for _ in range(100):
        k.pump(50)
        fg = win32.foreground_window()
        if "keyboard target" in win32.window_title(fg):
            k.hwnd = fg
            break
    if not k.hwnd:
        k.target.terminate()
        return {"error": "editor target did not become the foreground window"}
    cursor0 = win32.cursor_pos()
    k.disp.start()
    k.eng.set_armed(True, time.perf_counter())
    k.disp.enabled = True
    k.kb.set_shown(True)
    k.pump(400)
    res["overlay_noactivate"] = win32.has_noactivate(int(k.kb.winId()))
    k.watch()
    try:
        k.step(n=12)
        res["dual_mode"] = k.eng.state.value
        for ch in "sad":
            k.press("left", ch)
        for ch in [" ", "o", "k"]:
            k.press("right", "space" if ch == " " else ch)
        res["after_single_hands"] = k.text()
        k.press("right", "space")
        for _ in range(3):
            k.both("a", "l", offset=1)                 # overlapping pinches, left first
        res["after_alternating"] = k.text()
        k.press("right", "space")
        k.both("x", "y", offset=0)                     # the very same frames
        res["after_simultaneous"] = k.text()
        before_copy = k.text()
        # Ctrl+A (left Ctrl, right A), Ctrl+C (right Ctrl, left C)
        k.press("left", "ctrl")
        k.press("right", "a")
        k.press("right", "ctrl")
        k.press("left", "c")
        k.pump(200)
        res["clipboard_matches_editor"] = clip.text() == before_copy
        k.press("right", "end")                        # deselect, caret at the end
        k.press("left", "ctrl")
        k.press("right", "v")
        res["after_paste"] = k.text()
        k.press("right", "ctrl")
        k.press("left", "z")
        res["after_undo"] = k.text()
        k.press("left", "shift")
        k.press("right", "a")
        res["after_shift_a"] = k.text()
        k.press("left", "ctrl")
        k.press("left", "shift")
        k.press("right", "s")
        res["after_ctrl_shift_s"] = k.text()
        # Telex: EN -> VI, type v-i-e-e-j-t across both hands, VI -> IME -> EN
        k.press("right", "lang")
        k.press("right", "space")
        for side, ch in (("left", "v"), ("right", "i"), ("left", "e"), ("left", "e"), ("right", "j"),
                         ("left", "t")):
            k.press(side, ch)
        k.press("right", "lang")
        k.press("right", "lang")
        res["after_telex"] = k.text()
        # Armed Ctrl + both hands leaving the camera: cleared.
        k.press("left", "ctrl")
        armed = list(k.kb.out.mods.armed())
        k.step(n=40, use_left=False, use_right=False)
        k.pump(100)
        res["modifier_cleared_on_tracking_loss"] = {"armed_before": armed,
                                                    "armed_after": list(k.kb.out.mods.armed())}
        k.press("right", "q")
        res["after_lost_hands"] = k.text()
        res["cursor_moved_px"] = round(math.dist(cursor0, win32.cursor_pos()), 1)
        res["dispatcher"] = {kk: v for kk, v in k.disp.stats.items() if kk in ("moves", "buttons", "wheel")}
        # 🤟 (right hand) closes the keyboard (the desktop pointer resumes after it).
        k.R.aim("h")
        k.step(n=6)
        k.step(right=Pose.ily(), n=18)
        k.pump(300)
        res["closed_by_ily"] = not k.kb.isVisible()
        k.step(n=10)
        res["state_after_close"] = k.eng.state.value
    except RuntimeError as e:
        res["error"] = str(e)
    finally:
        k.kb.set_shown(False)
        k.eng.set_armed(False, time.perf_counter())
        k.disp.enabled = False
        k.disp.stop()
        k.pump(200)
    ev = k.events()
    keys = [e for e in ev if e["kind"] == "key"]
    css = [e for e in keys if e["keysym"].lower() == "s" and e["state"] & 0x4 and e["state"] & 0x1]
    res["ctrl_shift_s_chord_received"] = len(css)
    res["ctrl_c_received"] = sum(1 for e in keys if e["keysym"].lower() == "c" and e["state"] & 0x4)
    res["ctrl_v_received"] = sum(1 for e in keys if e["keysym"].lower() == "v" and e["state"] & 0x4)
    res["mouse_presses_at_editor"] = sum(1 for e in ev if e["kind"] == "press")
    res["foreground_changes"] = k.fg_changes
    res["dispatch_log"] = [(r["hand"], r["combo"] or r["key"], r["ok"]) for r in k.kb.results]
    res["dual_counters"] = dict(k.eng.dual.counters)
    clip.setText(saved_clip)
    k.target.terminate()
    return res


def main() -> int:
    c = Check()
    lay = c.layout()
    saved = win32.cursor_pos()
    c.disp.start()
    c.eng.set_armed(True, time.perf_counter())
    c.disp.enabled = True
    out: dict = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": "synthetic hands -> real SendInput"}
    try:
        out["part_a"] = part_a(c, lay)
    finally:
        c.eng.set_armed(False, time.perf_counter())
        c.disp.enabled = False
        c.disp.release_all()
        c.disp.stop()
        c.target.terminate()
        win32.mouse_move_abs(*saved)
    out["part_b"] = part_b(c.app)
    win32.mouse_move_abs(*saved)

    a, b = out["part_a"], out["part_b"]
    # Two keys pinched in the very same frames: either order is correct, both must arrive once.
    sim = b.get("after_simultaneous", "")
    base = sim if sim in ("sad ok alalal xy", "sad ok alalal yx") else "sad ok alalal xy"
    expect = {
        "after_single_hands": "sad ok",
        "after_alternating": "sad ok alalal",
        "after_simultaneous": base,
    }
    expect.update({"after_paste": base + base, "after_undo": base, "after_shift_a": base + "A",
                   "after_ctrl_shift_s": base + "A", "after_telex": base + "A việt",
                   "after_lost_hands": base + "A việtq"})
    checks = {
        "instant clicks 10/10 activated": a["instant_clicks"]["target_activated"] == 10
        and a["instant_clicks"]["other_activations"] == 0,
        "pinky drags: 1 press + 1 release each": a["pinky_drags"]["exact_one_press_one_release"] == 5,
        "pinky drags: 5 drag activations, nothing else": a["pinky_drags"]["drag_activations"] == 5
        and a["pinky_drags"]["other_activations"] == 0,
        "pinky drag starts at the aimed point (<= 3 px)": (a["pinky_drags"]["press_error_px_max"] or 99) <= 3,
        "pinky drag moved the target": (a["pinky_drags"]["moved_px_min"] or 0) > 100,
        "early pinky release sends nothing": a["early_release"]["presses"] == 0,
        "tracking loss releases the button": a["tracking_loss"]["held_after"] == []
        and a["tracking_loss"]["releases"] == a["tracking_loss"]["presses"] == 1,
        "emergency stop releases the button": a["emergency_stop"]["held_after"] == []
        and a["emergency_stop"]["releases"] == a["emergency_stop"]["presses"] == 1,
        "click right after a drop": a["click_after_drop"]["target_activated"] == 1,
        "keyboard: no error": "error" not in b,
        "keyboard: dual-hand mode": b.get("dual_mode") == "KEYBOARD",
        "keyboard: clipboard = copied text": b.get("clipboard_matches_editor") is True,
        "keyboard: Ctrl+Shift+S chord delivered": b.get("ctrl_shift_s_chord_received", 0) >= 1,
        "keyboard: armed Ctrl cleared when hands left": b.get("modifier_cleared_on_tracking_loss", {})
        .get("armed_after") == [],
        "keyboard: 🤟 closed it": b.get("closed_by_ily") is True,
        "keyboard: no mouse input at all": b.get("mouse_presses_at_editor") == 0 and b.get("cursor_moved_px") == 0
        and not any(b.get("dispatcher", {}).values()),
        "keyboard: editor kept the focus": b.get("foreground_changes") == [] and b.get("overlay_noactivate"),
    }
    for kk, v in expect.items():
        checks[f"text {kk} == {v!r}"] = b.get(kk) == v
    out["checks"] = checks
    out["passed"] = all(checks.values())
    dest = ROOT / "bench_results" / "v113_integration.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for kk, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'}  {kk}")
    print(json.dumps({"pinky_drags": {k2: v2 for k2, v2 in a["pinky_drags"].items() if k2 != "rows"},
                      "instant": {k2: a["instant_clicks"][k2] for k2 in
                                  ("target_activated", "accept_to_sendinput_ms_median",
                                   "contact_to_recognition_ms_median", "press_error_px_max")}},
                     indent=1, ensure_ascii=False))
    print("PASS" if out["passed"] else "FAIL", "->", dest)
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
