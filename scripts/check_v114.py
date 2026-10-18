"""Controlled Windows check for v1.1.4: dual-hand tracking and independent
two-hand typing.

Synthetic hands drive the production InteractionEngine (Dual-Hand Air
Keyboard); everything after it is real: KeyboardOverlay -> its ordered
dispatch queue -> KeyboardOutput -> SendInput into a *separate-process*
Tk editor (scripts/_keyboard_target.py) that logs what Windows delivered.

  * right hand only; the right hand leaves (1.5 s) and comes back ALONE with
    MediaPipe's label wrong for its first 3 frames, then types again
  * the same for the left hand
  * both hands sequentially; a held left pinch while the right types; a held
    right pinch while the left types; two keys in the very same frames
  * Ctrl (right) + A (left) in the SAME frame -> Ctrl+A; Ctrl+C across hands
    (clipboard verified), Ctrl+V across hands
  * the left hand vanishes during a right-hand key
  * one-shot Ctrl armed, then the emergency stop: modifiers cleared,
    keyboard mode left, nothing typed afterwards
  * the editor keeps the foreground, the mouse cursor never moves, no mouse
    button reaches anything, no Ctrl / Shift / Alt / Win is left down
  * dispatch timing: engine emission -> GUI-thread send (queue wait), send
    duration; queue depth after every step

Types into its own editor for ~40 s: do not touch the mouse or keyboard
meanwhile. The clipboard text is saved and restored. Writes
bench_results/v114_integration.json.

  .venv\\Scripts\\python scripts\\check_v114.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from check_v113 import I, Keys  # noqa: E402

from airdesk.platform import win32  # noqa: E402
from airdesk.platform.keyboard_output import VK  # noqa: E402
from airdesk.sim.synthetic import Pose, frame  # noqa: E402

CYCLE = (0.35, 0.7, 1.0, 1.0, 1.0, 0.6, 0.2, 0.0)


class Keys114(Keys):
    def __init__(self, app) -> None:
        super().__init__(app)
        self.labels: dict[str, str | None] = {"left": None, "right": None}
        self.max_depth = 0

    def step(self, left=None, right=None, n=1, use_left=True, use_right=True):
        st = None
        for _ in range(n):
            if self.unsafe:
                raise RuntimeError("the editor lost the foreground - stopped typing (safety)")
            hs = []
            for side, pose, use in (("left", left, use_left), ("right", right, use_right)):
                if use:
                    o = (self.L if side == "left" else self.R).obs(pose)
                    if self.labels[side] is not None:
                        o.handedness = self.labels[side]
                    hs.append(o)
            fut = self.vision.submit(self.eng.process, frame(time.perf_counter(), *hs))
            while not fut.done():
                self.app.processEvents()
                time.sleep(0.0005)
            st = fut.result()
            self.max_depth = max(self.max_depth, self.kb.queue_depth)
            while "toggle_keyboard" in self.commands:
                self.commands.remove("toggle_keyboard")
                self.kb.toggle()
            self.pump(30)
            self.watch()
        return st

    def type(self, side: str, key: str, only: bool = False) -> None:
        kw = {"use_left": side == "left", "use_right": side == "right"} if only else {}
        (self.L if side == "left" else self.R).aim(key)
        self.step(n=7, **kw)
        for a in CYCLE:
            self.step(**{side: Pose.closing(I, a)}, **kw)
        self.step(n=2, **kw)

    def leave_and_return(self, side: str) -> dict:
        """The hand leaves for 1.5 s and returns alone, MediaPipe's label
        wrong for its first 3 frames (re-entry at the image edge). Frames
        until any cursor, and until its own controller has it."""
        only = {"use_left": side == "left", "use_right": side == "right"}
        self.step(n=45, use_left=False, use_right=False)
        self.labels[side] = "Right" if side == "left" else "Left"
        shown = own = None
        for f in range(15):
            if f == 3:
                self.labels[side] = None
            st = self.step(**only)
            v = next(v for v in st.vhands if v["side"] == side)
            if shown is None and any(x["tracked"] for x in st.vhands):
                shown = f + 1
            own = (own or f + 1) if v["tracked"] else None
        return {"cursor_shown_frame": shown, "own_controller_frame": own, "life": v["life"],
                "reacqs": v["reacqs"], "reacq_ms": v["reacq_ms"]}

    def hold_and_type(self, holder: str, hold_key: str, word: str) -> None:
        typer = "right" if holder == "left" else "left"
        (self.L if holder == "left" else self.R).aim(hold_key)
        self.step(n=7)
        for a in (0.35, 0.7, 1.0):
            self.step(**{holder: Pose.closing(I, a)})
        for ch in word:
            (self.R if typer == "right" else self.L).aim(ch)
            self.step(**{holder: Pose.closing(I, 1.0)}, n=6)
            for a in CYCLE:
                self.step(**{holder: Pose.closing(I, 1.0), typer: Pose.closing(I, a)})
        for a in (0.6, 0.2, 0.0):
            self.step(**{holder: Pose.closing(I, a)})
        self.step(n=2)


def mods_down() -> list[str]:
    return [m for m, vk in (("ctrl", VK["ctrl"]), ("shift", VK["shift"]), ("alt", VK["alt"]), ("win", 0x5B))
            if win32.key_is_down(vk)]


def run(app) -> dict:
    k = Keys114(app)
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
        k.step(n=6, use_left=False)
        res["dual_mode"] = k.eng.state.value
        # Right hand only, leaves, returns alone (label wrong at first), types again.
        for ch in "hello":
            k.type("right", ch, only=True)
        res["right_return"] = k.leave_and_return("right")
        for ch in ["space"] + list("world"):
            k.type("right", ch, only=True)
        res["after_right"] = k.text()
        # Left hand only, same.
        k.step(n=45, use_left=False, use_right=False)
        for ch in ["space"] + list("sweat"):
            k.type("left", ch, only=True)
        res["left_return"] = k.leave_and_return("left")
        for ch in ["space"] + list("grade"):
            k.type("left", ch, only=True)
        res["after_left"] = k.text()
        # Both hands: sequential, held pinch + typing (both ways), same frames.
        k.type("right", "space")
        k.type("left", "a")
        k.type("right", "l")
        k.type("right", "space")
        k.hold_and_type("left", "x", "ok")
        k.type("right", "space")
        k.hold_and_type("right", "p", "as")
        k.type("right", "space")
        k.both("f", "j", offset=0)
        res["after_two_hands"] = k.text()
        before_copy = k.text()
        # Ctrl (right) and A (left) in the very same frames -> Ctrl+A.
        k.L.aim("a")
        k.R.aim("ctrl")
        k.step(n=7)
        for a in CYCLE:
            k.step(left=Pose.closing(I, a), right=Pose.closing(I, a))
        k.step(n=2)
        k.type("left", "ctrl")
        k.type("right", "c")
        k.pump(200)
        res["clipboard_matches_editor"] = clip.text() == before_copy
        k.type("right", "end")
        k.type("right", "ctrl")
        k.type("left", "v")
        res["after_paste"] = k.text()
        # The left hand vanishes while the right types.
        k.type("right", "space")
        k.R.aim("q")
        k.step(n=7)
        for i, a in enumerate(CYCLE):
            k.step(right=Pose.closing(I, a), use_left=i < 2)
        k.step(n=2, use_left=False)
        res["after_vanish"] = k.text()
        res["mods_down_while_typing_done"] = mods_down()
        # Emergency stop with a one-shot Ctrl armed.
        k.type("right", "ctrl", only=True)
        res["armed_before_stop"] = list(k.kb.out.mods.armed())
        k.eng.emergency_stop()
        k.pump(200)
        res["armed_after_stop"] = list(k.kb.out.mods.armed())
        res["state_after_stop"] = k.eng.state.value
        k.type("right", "z", only=True)                 # disarmed: must type nothing
        res["after_stop"] = k.text()
        res["cursor_moved_px"] = round(math.dist(cursor0, win32.cursor_pos()), 1)
        res["dispatcher"] = {kk: v for kk, v in k.disp.stats.items() if kk in ("moves", "buttons", "wheel")}
    except RuntimeError as e:
        res["error"] = str(e)
    finally:
        k.kb.set_shown(False)
        k.eng.set_armed(False, time.perf_counter())
        k.disp.enabled = False
        k.disp.stop()
        k.pump(200)
    res["mods_down_at_end"] = mods_down()
    ev = k.events()
    keys = [e for e in ev if e["kind"] == "key"]
    res["ctrl_a_received"] = sum(1 for e in keys if e["keysym"].lower() == "a" and e["state"] & 0x4)
    res["ctrl_c_received"] = sum(1 for e in keys if e["keysym"].lower() == "c" and e["state"] & 0x4)
    res["ctrl_v_received"] = sum(1 for e in keys if e["keysym"].lower() == "v" and e["state"] & 0x4)
    res["mouse_presses_at_editor"] = sum(1 for e in ev if e["kind"] == "press")
    res["foreground_changes"] = k.fg_changes
    rs = list(k.kb.results)
    res["dispatch_hands"] = [(r["hand"], r["combo"] or r["key"]) for r in rs]
    q = [r["queue_ms"] for r in rs if "queue_ms" in r]
    sd = [r["send_ms"] for r in rs if "send_ms" in r]
    res["dispatch_timing_ms"] = {"queue_wait_median": round(statistics.median(q), 2) if q else None,
                                 "queue_wait_max": round(k.kb.max_queue_ms, 2),
                                 "send_median": round(statistics.median(sd), 2) if sd else None,
                                 "send_max": round(max(sd), 2) if sd else None,
                                 "emitted": k.kb.emitted, "processed": k.kb.processed,
                                 "max_queue_depth_seen_between_frames": k.max_depth}
    res["dual_counters"] = dict(k.eng.dual.counters)
    clip.setText(saved_clip)
    k.target.terminate()
    return res


def main() -> int:
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    out = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": "synthetic hands -> real SendInput",
           "result": run(app)}
    b = out["result"]
    two = "hello world sweat grade al xok pas fj"
    checks = {
        "no error": "error" not in b,
        "dual-hand mode": b.get("dual_mode") == "KEYBOARD",
        "right hand back alone (label wrong 3 frames): cursor at once, own controller <= 10 frames":
            b.get("right_return", {}).get("cursor_shown_frame") == 1
            and (b.get("right_return", {}).get("own_controller_frame") or 99) <= 10,
        "left hand back alone (label wrong 3 frames): cursor at once, own controller <= 10 frames":
            b.get("left_return", {}).get("cursor_shown_frame") == 1
            and (b.get("left_return", {}).get("own_controller_frame") or 99) <= 10,
        "text after right hand only": b.get("after_right") == "hello world",
        "text after left hand only": b.get("after_left") == "hello world sweat grade",
        "text after two-hand / held / same-frame typing": b.get("after_two_hands") == two,
        # (The target's own Ctrl+A binding consumes the event before its logger;
        # the clipboard check below proves the selection.)
        "same-frame Ctrl (right) + A (left) = Ctrl+A chord": ("left", "Ctrl+A") in b.get("dispatch_hands", []),
        "Ctrl+C across hands -> clipboard = editor text": b.get("clipboard_matches_editor") is True,
        "Ctrl+V across hands": b.get("after_paste") == two + two and b.get("ctrl_v_received", 0) >= 1,
        "left hand vanishing during a right key": b.get("after_vanish") == two + two + " q",
        "emergency stop clears the armed Ctrl": b.get("armed_before_stop") == ["ctrl"]
        and b.get("armed_after_stop") == [],
        "emergency stop leaves keyboard mode, nothing typed after": b.get("state_after_stop") == "DISARMED"
        and b.get("after_stop") == two + two + " q",
        "no modifier left down": b.get("mods_down_while_typing_done") == [] and b.get("mods_down_at_end") == [],
        "no mouse input at all": b.get("mouse_presses_at_editor") == 0 and b.get("cursor_moved_px") == 0
        and not any(b.get("dispatcher", {}).values()),
        "editor kept the focus": b.get("foreground_changes") == [] and b.get("overlay_noactivate") is True,
        "every emitted key was sent": b.get("dispatch_timing_ms", {}).get("emitted")
        == b.get("dispatch_timing_ms", {}).get("processed"),
    }
    out["checks"] = checks
    out["passed"] = all(checks.values())
    dest = ROOT / "bench_results" / "v114_integration.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for kk, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'}  {kk}")
    print(json.dumps({kk: b.get(kk) for kk in ("right_return", "left_return", "dispatch_timing_ms",
                                               "dual_counters", "after_two_hands", "after_vanish")},
                     indent=1, ensure_ascii=False))
    print("PASS" if out["passed"] else "FAIL", "->", dest)
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
