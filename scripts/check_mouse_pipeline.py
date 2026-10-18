"""Integration check: gesture engine -> 120 Hz dispatcher -> real SendInput.

A synthetic hand (no camera) drives the production InteractionEngine and
MouseDispatcher. A Qt test window records the *real* Windows mouse events
it receives and the check verifies:

  1. a quick, *natural* pinch (the index bends toward the thumb) produces
     exactly one left click on the aimed pixel (Smart Aim Lock), repeated on
     several small-target positions with the click error measured;
  2. Drag Mode (v1.1.2): a 1.2 s pinch hold + hand motion produces press ->
     moves -> release (drag); in the default Instant Auto-Click a 0.6 s hold
     with hand motion is one plain click (completed at the pinch);
  3. the V scroll pose produces vertical and horizontal wheel events;
  4. Pinch Free style: thumb + middle produces a left click;
  5. a tracking loss during a press releases the button (no stuck button);
  6. after disarming, nothing is injected.

It moves the real cursor for ~10 s, then restores it. Do not touch the
mouse while it runs (doing so makes AirDesk yield, by design).

  .venv\\Scripts\\python scripts\\check_mouse_pipeline.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from airdesk.config import Settings  # noqa: E402
from airdesk.core.engine import InteractionEngine  # noqa: E402
from airdesk.core.types import Finger  # noqa: E402
from airdesk.platform import win32  # noqa: E402
from airdesk.platform.dispatcher import MouseDispatcher  # noqa: E402
from airdesk.sim.synthetic import Pose, frame, observation  # noqa: E402


class Recorder(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AirDesk mouse pipeline check")
        self.events: list[tuple] = []
        self.setMouseTracking(True)

    def mousePressEvent(self, e):  # noqa: N802
        self.events.append(("press", e.button().name, e.globalPosition().toPoint().toTuple()))

    def mouseReleaseEvent(self, e):  # noqa: N802
        self.events.append(("release", e.button().name, e.globalPosition().toPoint().toTuple()))

    def mouseMoveEvent(self, e):  # noqa: N802
        if e.buttons():
            self.events.append(("dragmove", e.globalPosition().toPoint().toTuple()))

    def wheelEvent(self, e):  # noqa: N802
        self.events.append(("wheel", e.angleDelta().x(), e.angleDelta().y()))

    def of(self, kind):
        return [e for e in self.events if e[0] == kind]


def main() -> int:
    app = QApplication(sys.argv)
    win = Recorder()
    scr = app.primaryScreen().geometry()
    win.setGeometry(scr.x() + 200, scr.y() + 150, 900, 600)
    win.show()
    win.raise_()
    win.activateWindow()
    saved = win32.cursor_pos()
    s = Settings()
    disp = MouseDispatcher(120, 10.0)
    eng = InteractionEngine(s, win32.screen_layout(), disp)
    disp.start()
    state = {"x": 0.5, "y": 0.6, "t": time.perf_counter()}
    results: dict[str, object] = {}

    def rect_center():
        r = win32.window_rect(int(win.winId()))
        return (r.left + r.right) // 2, (r.top + r.bottom) // 2

    def feed(pose, n=1, target=None, hand=True, dx=0.0, dy=0.0):
        for _ in range(n):
            state["t"] = time.perf_counter()
            if target is not None and eng.pointer.cursor is not None:
                cx, cy = eng.pointer.cursor
                vs = eng.screen.target_rect(s.pointer.monitor)
                state["x"] += (target[0] - cx) / vs.width * s.pointer.box_w * 0.5
                state["y"] += (target[1] - cy) / vs.height * s.pointer.box_h * 0.5
            state["x"] += dx
            state["y"] += dy
            hands = [observation(pose, state["x"], state["y"])] if hand else []
            eng.process(frame(state["t"], *hands))
            disp.enabled = eng.armed and not eng.paused
            for _ in range(3):
                app.processEvents()
                time.sleep(0.011)

    def pinch(finger=Finger.INDEX, hold=4, close=4, curl=0.7, dx=0.0):
        for k in range(close):
            feed(Pose.closing(finger, (k + 1) / close, curl=curl))
        feed(Pose.closing(finger, 1.0, curl=curl), hold, dx=dx)
        for k in range(3):
            feed(Pose.closing(finger, 1 - (k + 1) / 3, curl=curl))

    def run():
        from _windows_target import activate_own_window
        if not activate_own_window(int(win.winId())):
            results["focus_error"] = "test target could not become foreground; no input sent"
            app.quit()
            return
        point = Pose.pointing(Finger.INDEX)
        eng.set_armed(True, time.perf_counter())
        cx, cy = rect_center()
        # 1) aim, then a natural quick pinch -> one click on the aimed pixel
        feed(point, 45, target=(cx - 150, cy))
        aim0 = eng.pointer.cursor
        n0 = len(win.events)
        pinch()
        feed(point, 12)
        ev = win.events[n0:]
        results["click"] = [e[:2] for e in ev if e[0] in ("press", "release")]
        results["click_pos_error_px"] = round(math.dist(ev[0][2], aim0), 1) if ev else None
        # 1b) small targets: five aimed pixels, the click must land within 6 px (a 12 px control)
        errs = []
        for i, (ox, oy) in enumerate(((-60, -40), (40, -40), (120, 30), (-120, 60), (10, 90))):
            feed(point, 30, target=(cx + ox, cy + oy))
            feed(point, 8)
            aim = eng.pointer.cursor
            n0 = len(win.events)
            pinch(curl=0.55 + 0.1 * (i % 3))
            feed(point, 10)
            pr = [e for e in win.events[n0:] if e[0] == "press"]
            errs.append(round(math.dist(pr[0][2], aim), 1) if len(pr) == 1 else None)
        results["small_target_err_px"] = errs
        # 2) drag: Drag Mode, hold the pinch 1.2 s, move the hand right, release
        eng.set_click_mode("drag")
        feed(point, 20, target=(cx - 250, cy))
        n0 = len(win.events)
        for k in range(4):
            feed(Pose.closing(Finger.INDEX, (k + 1) / 4, curl=0.7))
        feed(Pose.closing(Finger.INDEX, 1.0, curl=0.7), 42)          # > 1.2 s after acceptance
        feed(Pose.closing(Finger.INDEX, 1.0, curl=0.7), 25, dx=0.004)
        for k in range(3):
            feed(Pose.closing(Finger.INDEX, 1 - (k + 1) / 3, curl=0.7))
        feed(point, 10)
        ev = win.events[n0:]
        pr = [e for e in ev if e[0] in ("press", "release")]
        results["drag"] = [e[:2] for e in pr]
        results["drag_distance_px"] = (pr[-1][2][0] - pr[0][2][0]) if len(pr) == 2 else None
        results["drag_moves"] = sum(1 for e in ev if e[0] == "dragmove")
        results["mode_after_drag"] = eng.click_mode                  # one-shot -> instant
        # 2b) 0.6 s hold while the hand drifts: still one click, no drag motion
        feed(point, 20, target=(cx + 100, cy))
        n0 = len(win.events)
        pinch(hold=14, dx=0.003)
        feed(point, 10)
        ev = win.events[n0:]
        results["hold_0.6s"] = [e[:2] for e in ev if e[0] in ("press", "release")]
        results["hold_0.6s_dragmoves"] = sum(1 for e in ev if e[0] == "dragmove")
        # 3) V scroll pose: hand up -> wheel up; then sideways -> horizontal wheel
        feed(point, 15, target=(cx, cy))
        n0 = len(win.events)
        feed(Pose.two_finger(), 8)
        feed(Pose.two_finger(), 30, dy=-0.006)
        feed(point, 10)
        wh = [e for e in win.events[n0:] if e[0] == "wheel"]
        results["wheel_events"] = len(wh)
        results["wheel_all_up"] = bool(wh) and all(e[2] > 0 and e[1] == 0 for e in wh)
        feed(point, 15, target=(cx, cy))
        n0 = len(win.events)
        feed(Pose.two_finger(), 8)
        feed(Pose.two_finger(), 30, dx=0.008)
        feed(point, 10)
        wh = [e for e in win.events[n0:] if e[0] == "wheel"]
        results["hwheel_gesture"] = len(wh)
        results["hwheel_gesture_only_x"] = bool(wh) and all(e[1] != 0 and e[2] == 0 for e in wh)
        # 4) Pinch Free: thumb + middle = left click
        s.gesture.click_style = "pinch_free"
        eng.configure(s)
        feed(point, 20, target=(cx - 100, cy + 40))
        n0 = len(win.events)
        pinch(Finger.MIDDLE, curl=0.6)
        feed(point, 10)
        results["pinch_free_middle"] = [e[:2] for e in win.events[n0:] if e[0] in ("press", "release")]
        s.gesture.click_style = "pinch_pointer"
        eng.configure(s)
        # 5) tracking loss during a press must release the button (classic
        # mode holds the button while pinched; Drag Mode is covered by
        # check_click_reliability.py)
        eng.set_click_mode("classic")
        feed(point, 15, target=(cx, cy))
        n0 = len(win.events)
        for k in range(4):
            feed(Pose.closing(Finger.INDEX, (k + 1) / 4, curl=0.7))
        feed(Pose.closing(Finger.INDEX, 1.0, curl=0.7), 12)
        held_before = disp.held_buttons()
        feed(point, 12, hand=False)
        ev = win.events[n0:]
        results["loss_held_before"] = held_before
        eng.set_click_mode("instant")
        results["loss_released"] = [e[:2] for e in ev if e[0] in ("press", "release")]
        results["loss_held_after"] = disp.held_buttons()
        # other buttons through the dispatcher's real SendInput path
        feed(point, 10, target=(cx, cy))
        n0 = len(win.events)
        for cmd in (lambda: disp.click("right"), lambda: disp.click("middle"), lambda: disp.click("left", 2),
                    lambda: disp.xbutton(1), lambda: disp.xbutton(2), lambda: disp.scroll(-2.0, 0.0)):
            cmd()
            feed(point, 4)
        ev = win.events[n0:]
        results["buttons"] = [e[1] for e in ev if e[0] == "press"]
        results["hwheel"] = [e[1] for e in ev if e[0] == "wheel" and e[1] != 0][:3]
        # 6) disarmed: no injection
        eng.set_armed(False, time.perf_counter())
        n0 = len(win.events)
        moves0 = disp.stats["moves"]
        pinch()
        feed(point, 20, dx=0.005)
        results["disarmed_events"] = len(win.events) - n0
        results["disarmed_moves"] = disp.stats["moves"] - moves0
        app.quit()

    QTimer.singleShot(500, run)
    app.exec()
    disp.stop()
    win32.mouse_move_abs(*saved)

    lr = [("press", "LeftButton"), ("release", "LeftButton")]
    errs = results.get("small_target_err_px") or [None]
    ok = (results.get("click") == lr and (results.get("click_pos_error_px") or 99) <= 6
          and all(e is not None and e <= 6 for e in errs)
          and results.get("drag") == lr and results.get("mode_after_drag") == "instant"
          and results.get("hold_0.6s") == lr and results.get("hold_0.6s_dragmoves") == 0
          and results.get("hwheel_gesture_only_x") and results.get("pinch_free_middle") == lr
          and (results.get("drag_distance_px") or 0) > 100
          and results.get("wheel_all_up")
          and results.get("loss_held_before") == ("left",) and results.get("loss_held_after") == ()
          and results.get("loss_released") == [("press", "LeftButton"), ("release", "LeftButton")]
          and results.get("disarmed_events") == 0 and results.get("disarmed_moves") == 0
          and results.get("buttons") == ["RightButton", "MiddleButton", "LeftButton", "LeftButton",
                                         "BackButton", "ForwardButton"]
          and bool(results.get("hwheel")))
    for k, v in results.items():
        print(f"{k:22} {v}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
