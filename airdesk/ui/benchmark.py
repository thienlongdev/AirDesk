"""Benchmark Laboratory: synthetic algorithm benchmarks and human usability tests.

Every saved result records `kind` ("synthetic" or "human") and, for human
tests, the actual `input_source`: if AirDesk was not armed while the test
ran, the clicks came from the physical mouse and the result is labelled as
a physical-input baseline, never as a gesture result.

Results: %APPDATA%\\AirDesk\\bench_results\\*.json (+ summary.csv). Nothing
typed and no camera images are stored - only timings, counts and distances.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QCursor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QSpinBox,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..config import app_dir
from . import theme
from .widgets import group, label, page

if TYPE_CHECKING:
    from .main_window import AppContext

PHRASES = {
    "en": ["the quick brown fox", "hello world", "open the file menu", "air typing works", "see you tomorrow"],
    "vi": ["xin chào", "cảm ơn bạn", "tiếng việt", "hôm nay trời đẹp", "người dùng"],
}


def results_dir() -> Path:
    p = app_dir() / "bench_results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_result(res: dict) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = results_dir() / f"{stamp}_{res['test']}.json"
    path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    summ = results_dir() / "summary.csv"
    new = not summ.exists()
    with summ.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "test", "kind", "input_source", "metric", "value"])
        for k, v in res.get("summary", {}).items():
            w.writerow([stamp, res["test"], res["kind"], res.get("input_source", ""), k, v])
    return path


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# ============================================================ test windows
class TestWindow(QWidget):
    finished = Signal(dict)

    def __init__(self, ctx: "AppContext", name: str) -> None:
        super().__init__(None, Qt.Window | Qt.FramelessWindowHint)
        self.ctx, self.name = ctx, name
        self.setMouseTracking(True)
        self.samples: list[tuple[float, float, float]] = []
        self._sampler = QTimer(self, interval=16, timeout=self._sample)
        self.t0 = time.perf_counter()
        self.trials: list[dict] = []
        self.armed_frames = self.frames = 0

    def start(self) -> None:
        self.showFullScreen()
        self.activateWindow()
        self._sampler.start()

    def _sample(self) -> None:
        p = QCursor.pos()
        self.samples.append((time.perf_counter(), p.x(), p.y()))
        self.frames += 1
        self.armed_frames += int(self.ctx.runtime.engine.armed and not self.ctx.runtime.engine.paused)
        self.update()

    def keyPressEvent(self, e) -> None:  # noqa: N802
        if e.key() == Qt.Key_Escape:
            self.finish(aborted=True)

    def summary(self) -> dict:
        return {}

    def finish(self, aborted: bool = False) -> None:
        self._sampler.stop()
        s = self.ctx.s
        res = {
            "test": self.name, "kind": "human",
            "input_source": ("AirDesk gestures" if self.armed_frames > 0.8 * max(self.frames, 1)
                             else "physical input (AirDesk not armed) - baseline"),
            "aborted": aborted, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {"pointer_mode": s.pointer.mode, "filter": s.pointer.filter, "min_cutoff": s.pointer.min_cutoff,
                       "beta": s.pointer.beta, "finger_mode": s.finger.mode, "finger": s.finger.finger,
                       "camera": f"{s.camera.width}x{s.camera.height}", "profile": s.profile},
            "trials": self.trials, "summary": self.summary() if self.trials else {},
        }
        self.close()
        self.finished.emit(res)

    def draw_header(self, p: QPainter, text: str) -> None:
        p.setPen(theme.qcolor("text"))
        f = QFont(self.font())
        f.setPointSize(16)
        p.setFont(f)
        p.drawText(QRectF(0, 20, self.width(), 40), Qt.AlignHCenter, text)
        f.setPointSize(10)
        p.setFont(f)
        p.setPen(theme.qcolor("muted"))
        p.drawText(QRectF(0, 60, self.width(), 24), Qt.AlignHCenter, "Esc aborts · Ctrl+Alt+F12 disarms")


class TargetTest(TestWindow):
    """Fitts-style target acquisition: time, misses, path efficiency,
    overshoot (target re-entries), click drift and hover jitter."""

    SIZES = (28, 56, 96)

    def __init__(self, ctx, n: int = 15) -> None:
        super().__init__(ctx, "target_acquisition")
        self.n = n
        self.rng = random.Random()
        self.target: tuple[QPointF, int] | None = None
        self._t_start = 0.0
        self._misses = 0
        self._entries = 0
        self._inside = False
        self._start_pos = QPointF()

    def start(self) -> None:
        super().start()
        QTimer.singleShot(400, self._next)

    def _next(self) -> None:
        if len(self.trials) >= self.n:
            self.finish()
            return
        size = self.SIZES[len(self.trials) % 3]
        m = 120
        c = QPointF(self.rng.uniform(m, self.width() - m), self.rng.uniform(m + 60, self.height() - m))
        self.target = (c, size)
        self._t_start = time.perf_counter()
        self._misses = self._entries = 0
        self._inside = False
        self._start_pos = QPointF(self.mapFromGlobal(QCursor.pos()))
        self._sample_i0 = len(self.samples)
        self.update()

    def _sample(self) -> None:
        super()._sample()
        if self.target:
            c, size = self.target
            p = self.mapFromGlobal(QCursor.pos())
            inside = math.dist((p.x(), p.y()), (c.x(), c.y())) <= size / 2
            if inside and not self._inside:
                self._entries += 1
            self._inside = inside

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if not self.target:
            return
        c, size = self.target
        p = e.position()
        if math.dist((p.x(), p.y()), (c.x(), c.y())) > size / 2:
            self._misses += 1
            return
        now = time.perf_counter()
        pts = [(x, y) for t, x, y in self.samples[self._sample_i0:]]
        path = sum(math.dist(a, b) for a, b in zip(pts, pts[1:])) or 1.0
        gp = self.mapToGlobal(c.toPoint())
        dist = math.dist((self._start_pos.x(), self._start_pos.y()), (c.x(), c.y()))
        last = [(x, y) for t, x, y in self.samples if now - t <= 0.3]
        jitter = math.sqrt(statistics.pvariance([x for x, _ in last]) + statistics.pvariance([y for _, y in last])) \
            if len(last) > 2 else 0.0
        before = [(x, y) for t, x, y in self.samples if 0.08 <= now - t <= 0.15]
        drift = math.dist(before[-1], (QCursor.pos().x(), QCursor.pos().y())) if before else 0.0
        mt = now - self._t_start
        self.trials.append({"size_px": size, "distance_px": round(dist, 1), "time_ms": round(mt * 1000, 1),
                            "missed_clicks": self._misses, "target_entries": self._entries,
                            "path_efficiency": round(min(dist / path, 1.0), 3),
                            "hover_jitter_px": round(jitter, 2), "click_drift_px": round(drift, 2),
                            "error_px": round(math.dist((p.x(), p.y()), (c.x(), c.y())), 1),
                            "fitts_id": round(math.log2(dist / size + 1), 3), "target_global": [gp.x(), gp.y()]})
        self.target = None
        QTimer.singleShot(250, self._next)

    def summary(self) -> dict:
        t = self.trials
        clicks = len(t) + sum(x["missed_clicks"] for x in t)
        tp = [x["fitts_id"] / (x["time_ms"] / 1000) for x in t if x["time_ms"] > 0]
        out = {"trials": len(t), "click_success_rate": round(len(t) / max(clicks, 1), 3),
               "median_time_ms": statistics.median(x["time_ms"] for x in t),
               "median_path_efficiency": statistics.median(x["path_efficiency"] for x in t),
               "mean_overshoot_reentries": round(statistics.mean(max(x["target_entries"] - 1, 0) for x in t), 2),
               "median_hover_jitter_px": statistics.median(x["hover_jitter_px"] for x in t),
               "median_click_drift_px": statistics.median(x["click_drift_px"] for x in t),
               "throughput_bits_s": round(statistics.mean(tp), 2) if tp else 0}
        for s in self.SIZES:
            v = [x["time_ms"] for x in t if x["size_px"] == s]
            if v:
                out[f"median_time_ms_{s}px"] = statistics.median(v)
        return out

    def paintEvent(self, e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), theme.qcolor("bg"))
        self.draw_header(p, f"Click the target · {len(self.trials)}/{self.n}")
        if self.target:
            c, size = self.target
            p.setPen(Qt.NoPen)
            p.setBrush(theme.qcolor("accent"))
            p.drawEllipse(c, size / 2, size / 2)
            p.setBrush(theme.qcolor("bg"))
            p.drawEllipse(c, 2, 2)
        p.end()


class PrecisionTest(TestWindow):
    """Small, tightly spaced controls (v1.1 acceptance): rows of 12/16/20 px
    checkboxes, close icons and toolbar buttons 4 px apart. One control is
    the target; clicking a neighbour is a *wrong-target* click, anything else
    a miss. Also records time to activate, accidental double clicks and how
    far the cursor moved between 150 ms before the press and the press
    (pinch-induced displacement). Nothing here does anything destructive."""

    SIZES = (12, 16, 20)
    KINDS = ("checkbox", "close", "toolbar")
    GAP = 4
    COUNT = 7

    def __init__(self, ctx, n: int = 18) -> None:
        super().__init__(ctx, "precision_targets")
        self.n = n
        self.rng = random.Random()
        self.row: list[QRectF] = []
        self.target_i = 0
        self.kind = "checkbox"
        self.size = 12
        self._t_start = 0.0
        self._wrong = self._miss = 0
        self._last_press: tuple[float, tuple[float, float]] | None = None
        self._doubles = 0

    def start(self) -> None:
        super().start()
        QTimer.singleShot(400, self._next)

    def _next(self) -> None:
        if len(self.trials) >= self.n:
            self.finish()
            return
        i = len(self.trials)
        self.size = self.SIZES[i % 3]
        self.kind = self.KINDS[(i // 3) % 3]
        pitch = self.size + self.GAP
        w = pitch * self.COUNT
        x0 = self.rng.uniform(120, self.width() - 120 - w)
        y0 = self.rng.uniform(160, self.height() - 120)
        self.row = [QRectF(x0 + k * pitch, y0, self.size, self.size) for k in range(self.COUNT)]
        self.target_i = self.rng.randrange(1, self.COUNT - 1)     # always has neighbours on both sides
        self._t_start = time.perf_counter()
        self._wrong = self._miss = 0
        self.update()

    def mousePressEvent(self, e) -> None:  # noqa: N802
        p = e.position()
        now = time.perf_counter()
        last, self._last_press = self._last_press, (now, (p.x(), p.y()))
        if last and now - last[0] < 0.4 and math.dist(last[1], (p.x(), p.y())) <= 4:
            self._doubles += 1                       # a second press on the same spot within 0.4 s
            return
        if not self.row:
            return
        idx = next((k for k, r in enumerate(self.row) if r.contains(p)), None)
        if idx is None:
            self._miss += 1
            return
        if idx != self.target_i:
            self._wrong += 1
            return
        before = [(x, y) for t, x, y in self.samples if 0.12 <= now - t <= 0.2]
        gp = QCursor.pos()
        disp = math.dist(before[-1], (gp.x(), gp.y())) if before else 0.0
        c = self.row[idx].center()
        self.trials.append({"kind": self.kind, "size_px": self.size, "time_ms": round((now - self._t_start) * 1000, 1),
                            "wrong_target_clicks": self._wrong, "missed_clicks": self._miss,
                            "error_px": round(math.dist((p.x(), p.y()), (c.x(), c.y())), 1),
                            "press_displacement_px": round(disp, 1)})
        self.row = []
        self.update()
        QTimer.singleShot(300, self._next)

    def summary(self) -> dict:
        t = self.trials
        out = {"trials": len(t), "wrong_target_clicks": sum(x["wrong_target_clicks"] for x in t),
               "missed_clicks": sum(x["missed_clicks"] for x in t), "accidental_double_clicks": self._doubles,
               "median_time_ms": statistics.median(x["time_ms"] for x in t),
               "median_press_displacement_px": statistics.median(x["press_displacement_px"] for x in t),
               "max_press_displacement_px": max(x["press_displacement_px"] for x in t)}
        for s in self.SIZES:
            v = [x for x in t if x["size_px"] == s]
            if v:
                tries = len(v) + sum(x["wrong_target_clicks"] + x["missed_clicks"] for x in v)
                out[f"first_try_rate_{s}px"] = round(len(v) / max(tries, 1), 3)
                out[f"median_time_ms_{s}px"] = statistics.median(x["time_ms"] for x in v)
        return out

    def paintEvent(self, e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), theme.qcolor("bg"))
        self.draw_header(p, f"Click the highlighted {self.kind} ({self.size} px) · {len(self.trials)}/{self.n}")
        for k, r in enumerate(self.row):
            target = k == self.target_i
            p.setPen(QPen(theme.qcolor("accent") if target else theme.qcolor("muted"), 1.4))
            p.setBrush(theme.qcolor("accent") if target else Qt.NoBrush)
            if self.kind == "close":
                p.drawEllipse(r)
                p.setPen(QPen(theme.qcolor("bg") if target else theme.qcolor("muted"), 1.4))
                m = r.width() * 0.3
                p.drawLine(QPointF(r.left() + m, r.top() + m), QPointF(r.right() - m, r.bottom() - m))
                p.drawLine(QPointF(r.right() - m, r.top() + m), QPointF(r.left() + m, r.bottom() - m))
            elif self.kind == "toolbar":
                p.drawRoundedRect(r, 2, 2)
            else:
                p.drawRect(r)
        p.end()


class DragTest(TestWindow):
    def __init__(self, ctx, n: int = 8) -> None:
        super().__init__(ctx, "drag_and_drop")
        self.n = n
        self.rng = random.Random()
        self.ball = QPointF()
        self.slot = QRectF()
        self._drag = False
        self._t0 = 0.0
        self._accidental = 0

    def start(self) -> None:
        super().start()
        self._mode0 = self.ctx.runtime.engine.click_mode
        QTimer.singleShot(400, self._next)

    def _next(self) -> None:
        if len(self.trials) >= self.n:
            self.finish()
            return
        # (v1.1.3: Pinky Drag needs no mode switch; v1.1.2 switched to Drag Mode here)
        self.ball = QPointF(self.rng.uniform(150, self.width() * 0.35), self.rng.uniform(200, self.height() - 150))
        sx = self.rng.uniform(self.width() * 0.55, self.width() - 220)
        self.slot = QRectF(sx, self.rng.uniform(200, self.height() - 250), 120, 120)
        self._t0 = time.perf_counter()
        self._accidental = 0
        self.update()

    def mousePressEvent(self, e) -> None:  # noqa: N802
        p = e.position()
        if math.dist((p.x(), p.y()), (self.ball.x(), self.ball.y())) < 34:
            self._drag = True

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if self._drag:
            self.ball = e.position()
            self.update()

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if not self._drag:
            return
        self._drag = False
        if self.slot.contains(self.ball):
            c = self.slot.center()
            self.trials.append({"success": True, "time_ms": round((time.perf_counter() - self._t0) * 1000, 1),
                                "accidental_releases": self._accidental,
                                "endpoint_error_px": round(math.dist((c.x(), c.y()), (self.ball.x(), self.ball.y())), 1)})
            QTimer.singleShot(250, self._next)
        else:
            self._accidental += 1
            if self._accidental >= 3:
                self.trials.append({"success": False, "time_ms": round((time.perf_counter() - self._t0) * 1000, 1),
                                    "accidental_releases": self._accidental, "endpoint_error_px": None})
                QTimer.singleShot(250, self._next)
        self.update()

    def finish(self, aborted: bool = False) -> None:
        if getattr(self, "_mode0", None):
            self.ctx.runtime.set_click_mode(self._mode0)
            self._mode0 = None
        super().finish(aborted)

    def summary(self) -> dict:
        t = self.trials
        ok = [x for x in t if x["success"]]
        return {"trials": len(t), "success_rate": round(len(ok) / max(len(t), 1), 3),
                "first_try_success_rate": round(sum(1 for x in ok if x["accidental_releases"] == 0) / max(len(t), 1), 3),
                "accidental_releases": sum(x["accidental_releases"] for x in t),
                "median_time_ms": statistics.median(x["time_ms"] for x in ok) if ok else None,
                "median_endpoint_error_px": statistics.median(x["endpoint_error_px"] for x in ok) if ok else None}

    def paintEvent(self, e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), theme.qcolor("bg"))
        self.draw_header(p, "Pinky Drag: aim at the ball, only the little finger up 0.2 s, drop it in the box "
                            f"(curl the little finger) · {len(self.trials)}/{self.n}")
        p.setPen(QPen(theme.qcolor("ok"), 3, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(self.slot, 14, 14)
        p.setPen(Qt.NoPen)
        p.setBrush(theme.qcolor("warn"))
        p.drawEllipse(self.ball, 30, 30)
        p.end()


class ClickAcceptanceTest(TestWindow):
    """Owner click-reliability acceptance (v1.1.2): 20 left clicks, 10 right
    clicks, 10 small-checkbox clicks, 10 rapid repeated clicks, 5 Drag Mode
    drags (hold 1.2 s), 3 safe releases after tracking loss and 5 🤟
    keyboard open/close cycles (0.4 s hold).

    Separate numbers per stage, never merged:
      * gesture side   - presses AirDesk accepted, from the interaction log,
                         and pinch candidates released without a click;
      * Windows side   - interactions whose button-down and -up were both
                         submitted (completed) / cancelled / rejected, with
                         the reasons;
      * target side    - real Qt mouse events on this window: activations
                         (press and release inside the target, right button),
                         wrong-button, wrong-target, outside presses and
                         suspected duplicates (a second activation < 150 ms
                         after the previous one - faster than a new pinch).
    Attempts = activations + wrong + outside + the misses the owner reports.
    The owner presses M on the physical keyboard whenever a deliberate pinch
    did nothing (missed clicks the camera never saw are only countable that
    way). N skips a stage, Esc aborts."""

    STAGES = (("left", 20, "Left-click the blue button (one pinch - it clicks at once)"),
              ("right", 10, "Right-click the orange button"),
              ("checkbox", 10, "Click the highlighted 14 px checkbox"),
              ("rapid", 10, "Click the button 10 times in a row - open your fingers between pinches"),
              ("drag", 5, "Drag Mode is on: pinch the ball, hold 1.2 s until the ring fills, drop it in the box"),
              ("trackloss", 3, "Drag Mode: pinch and hold the ball until it drags, then pull your hand "
                               "out of the camera view - the drag must drop by itself"),
              ("keyboard", 10, "🤟 hold 0.4 s: open the Air Keyboard, relax, 🤟 again to close"))

    def __init__(self, ctx) -> None:
        super().__init__(ctx, "click_acceptance")
        self.rng = random.Random()
        self.stage = 0
        self.done_in_stage = 0
        self.target = QRectF()
        self.row: list[QRectF] = []
        self.target_i = 0
        self.ball = QPointF()
        self.slot = QRectF()
        self._press: tuple | None = None
        self._dragging = False
        self._toggles = 0
        self._stage_counts: dict = {}
        self._log0: dict = {}
        self._listener = self._on_runtime_event

    def start(self) -> None:
        super().start()
        self.ctx.runtime.listeners.append(self._listener)
        QTimer.singleShot(400, self._begin_stage)

    def _on_runtime_event(self, name: str) -> None:   # vision / hotkey thread: count only
        if name == "toggle_keyboard":
            self._toggles += 1

    def _kind(self) -> str:
        return self.STAGES[self.stage][0]

    def _begin_stage(self) -> None:
        if self.stage >= len(self.STAGES):
            self.finish()
            return
        self.done_in_stage = 0
        self._stage_counts = {"activations": 0, "wrong_button": 0, "wrong_target": 0, "outside": 0,
                              "released_outside": 0, "user_reported_missed": 0, "keyboard_toggles": 0,
                              "suspected_duplicates": 0, "safe_releases": 0}
        self._log0 = self.ctx.runtime.engine.clicks.snapshot_counters()
        last = self.ctx.runtime.engine.clicks.latest()
        self._id0 = last.id if last else 0
        self._seen_loss: set[int] = set()
        self._last_act = -1.0
        if self._kind() in ("drag", "trackloss"):
            self._mode0 = getattr(self, "_mode0", None) or self.ctx.runtime.engine.click_mode
            self.ctx.runtime.set_click_mode("drag")
        elif getattr(self, "_mode0", None):
            self.ctx.runtime.set_click_mode(self._mode0)
            self._mode0 = None
        self._toggles = 0
        self._t_stage = time.perf_counter()
        self._place()

    def _place(self) -> None:
        k = self._kind()
        w, h = self.width(), self.height()
        if k in ("left", "right"):
            self.target = QRectF(self.rng.uniform(120, w - 260), self.rng.uniform(180, h - 160), 120, 56)
        elif k == "checkbox":
            pitch, size = 18, 14
            x0, y0 = self.rng.uniform(150, w - 300), self.rng.uniform(200, h - 150)
            self.row = [QRectF(x0 + i * pitch, y0, size, size) for i in range(7)]
            self.target_i = self.rng.randrange(1, 6)
        elif k == "rapid":
            if self.target.isNull():
                self.target = QRectF(self.rng.uniform(120, w - 260), self.rng.uniform(180, h - 160), 120, 56)
        elif k in ("drag", "trackloss"):
            self.ctx.runtime.set_click_mode("drag")          # one-shot Drag Mode ends after each drop
            self.ball = QPointF(self.rng.uniform(150, w * 0.35), self.rng.uniform(220, h - 160))
            self.slot = QRectF(self.rng.uniform(w * 0.55, w - 240), self.rng.uniform(220, h - 260), 130, 130)
        self.update()

    def _advance(self) -> None:
        self.done_in_stage += 1
        if self.done_in_stage >= self.STAGES[self.stage][1]:
            self._end_stage()
        else:
            self._place()

    def _end_stage(self, skipped: bool = False) -> None:
        log1 = self.ctx.runtime.engine.clicks.snapshot_counters()
        delta = {k: log1[k] - self._log0.get(k, 0) for k in log1}
        kind, goal, _ = self.STAGES[self.stage]
        if kind == "keyboard":
            self._stage_counts["keyboard_toggles"] = self._toggles
        c = self._stage_counts
        reasons: dict[str, int] = {}
        for it in self.ctx.runtime.engine.clicks.recent(64):
            if it.id > self._id0 and it.phase in ("cancelled", "rejected"):
                key = f"{it.phase}: {it.reason or '-'}"
                reasons[key] = reasons.get(key, 0) + 1
        self.trials.append({"stage": kind, "goal": goal, "skipped": skipped,
                            "time_s": round(time.perf_counter() - self._t_stage, 1),
                            "target": dict(c),
                            "attempts": c["activations"] + c["wrong_button"] + c["wrong_target"] + c["outside"]
                            + c["released_outside"] + c["user_reported_missed"],
                            "airdesk": {"presses_accepted": delta["accepted"],
                                        "button_downs_submitted": delta["down_submitted"],
                                        "button_ups_submitted": delta["up_submitted"],
                                        "completed": delta["completed"], "cancelled": delta["cancelled"],
                                        "rejected": delta["rejected"], "drags": delta["drags"],
                                        "aim_released_without_click": delta.get("aim_released", 0),
                                        "not_completed_reasons": reasons}})
        self.stage += 1
        self.target, self.row = QRectF(), []
        QTimer.singleShot(500, self._begin_stage)

    def _sample(self) -> None:
        super()._sample()
        if self.stage < len(self.STAGES) and self._kind() == "keyboard" and self._toggles > self.done_in_stage:
            self.done_in_stage = self._toggles - 1
            self._advance()
        elif self.stage < len(self.STAGES) and self._kind() == "trackloss":
            # A drag ended by tracking loss: AirDesk cancels it and releases
            # the button by itself. Success = nothing left held.
            for it in self.ctx.runtime.engine.clicks.recent(16):
                if (it.id > self._id0 and it.id not in self._seen_loss and it.done
                        and "tracking lost" in it.reason and not self.ctx.runtime.dispatcher.held_buttons()):
                    self._seen_loss.add(it.id)
                    self._stage_counts["safe_releases"] += 1
                    self._dragging = False
                    self._advance()
                    break

    def keyPressEvent(self, e) -> None:  # noqa: N802
        if e.key() == Qt.Key_M and self.stage < len(self.STAGES):
            self._stage_counts["user_reported_missed"] += 1
            self.update()
        elif e.key() == Qt.Key_N and self.stage < len(self.STAGES):
            self._end_stage(skipped=True)
        else:
            super().keyPressEvent(e)

    def _hit(self, p: QPointF):
        k = self._kind()
        if k in ("left", "right", "rapid"):
            return "target" if self.target.contains(p) else None
        if k == "checkbox":
            idx = next((i for i, r in enumerate(self.row) if r.adjusted(-1, -1, 1, 1).contains(p)), None)
            return None if idx is None else ("target" if idx == self.target_i else "neighbour")
        return None

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if self.stage >= len(self.STAGES):
            return
        p = e.position()
        k = self._kind()
        if k in ("drag", "trackloss"):
            if math.dist((p.x(), p.y()), (self.ball.x(), self.ball.y())) < 36:
                self._dragging = True
            return
        self._press = (e.button(), self._hit(p))

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if self._dragging:
            self.ball = e.position()
            self.update()

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if self.stage >= len(self.STAGES):
            return
        c = self._stage_counts
        if self._kind() == "trackloss":
            self._dragging = False             # counted by _sample from the interaction log
            return
        if self._kind() == "drag":
            if self._dragging:
                self._dragging = False
                if self.slot.contains(self.ball):
                    c["activations"] += 1
                    self._advance()
                else:
                    c["released_outside"] += 1
            return
        press, self._press = self._press, None
        if press is None:
            return
        want = Qt.RightButton if self._kind() == "right" else Qt.LeftButton
        hit = self._hit(e.position())
        if press[1] is None:
            c["outside"] += 1
        elif press[1] == "neighbour":
            c["wrong_target"] += 1
        elif press[0] != want:
            c["wrong_button"] += 1
        elif hit != "target":
            c["released_outside"] += 1
        else:
            now = time.perf_counter()
            if now - self._last_act < 0.15:
                c["suspected_duplicates"] += 1
            self._last_act = now
            c["activations"] += 1
            self._advance()
        self.update()

    def contextMenuEvent(self, e) -> None:  # noqa: N802
        e.accept()                       # right clicks only count, never open a menu

    def summary(self) -> dict:
        out = {}
        for t in self.trials:
            s, a = t["target"], t["airdesk"]
            pre = t["stage"]
            out[f"{pre}_activations"] = s["activations"] if pre != "keyboard" else s["keyboard_toggles"]
            out[f"{pre}_goal"] = t["goal"]
            out[f"{pre}_user_reported_missed"] = s["user_reported_missed"]
            out[f"{pre}_wrong_target_or_button"] = s["wrong_target"] + s["wrong_button"]
            out[f"{pre}_accepted"] = a["presses_accepted"]
            out[f"{pre}_completed"] = a["completed"]
            out[f"{pre}_not_completed"] = a["cancelled"] + a["rejected"]
            out[f"{pre}_attempts"] = t.get("attempts", 0)
            out[f"{pre}_duplicates"] = s.get("suspected_duplicates", 0)
            if pre == "trackloss":
                out["trackloss_safe_releases"] = s.get("safe_releases", 0)
        return out

    def paintEvent(self, e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), theme.qcolor("bg"))
        if self.stage >= len(self.STAGES):
            self.draw_header(p, "Done")
            p.end()
            return
        kind, goal, text = self.STAGES[self.stage]
        self.draw_header(p, f"Stage {self.stage + 1}/{len(self.STAGES)}: {text} · {self.done_in_stage}/{goal}")
        f = QFont(self.font())
        f.setPointSize(10)
        p.setFont(f)
        p.setPen(theme.qcolor("muted"))
        c = self._stage_counts
        p.drawText(QRectF(0, 86, self.width(), 24), Qt.AlignHCenter,
                   f"Press M on the keyboard each time a deliberate pinch did nothing (so far {c.get('user_reported_missed', 0)}) "
                   f"· N skips this stage")
        if kind in ("left", "right", "rapid") and not self.target.isNull():
            col = theme.qcolor("warn") if kind == "right" else theme.qcolor("accent")
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            p.drawRoundedRect(self.target, 10, 10)
            p.setPen(theme.qcolor("bg"))
            p.drawText(self.target, Qt.AlignCenter, {"left": "Left click", "right": "Right click"}.get(
                kind, f"Click {self.done_in_stage + 1}/{goal}"))
        elif kind == "checkbox":
            for i, r in enumerate(self.row):
                tgt = i == self.target_i
                p.setPen(QPen(theme.qcolor("accent") if tgt else theme.qcolor("muted"), 1.4))
                p.setBrush(theme.qcolor("accent") if tgt else Qt.NoBrush)
                p.drawRect(r)
        elif kind in ("drag", "trackloss"):
            p.setPen(QPen(theme.qcolor("ok"), 3, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(self.slot, 14, 14)
            p.setPen(Qt.NoPen)
            p.setBrush(theme.qcolor("warn"))
            p.drawEllipse(self.ball, 30, 30)
        elif kind == "keyboard":
            f.setPointSize(48)
            p.setFont(f)
            p.setPen(theme.qcolor("text"))
            p.drawText(QRectF(0, 0, self.width(), self.height()), Qt.AlignCenter, "🤟")
        p.end()

    def finish(self, aborted: bool = False) -> None:
        try:
            self.ctx.runtime.listeners.remove(self._listener)
        except ValueError:
            pass
        if getattr(self, "_mode0", None):
            self.ctx.runtime.set_click_mode(self._mode0)     # leave the owner's click mode as it was
            self._mode0 = None
        super().finish(aborted)


class OwnerAcceptanceV113(ClickAcceptanceTest):
    """Owner physical acceptance for v1.1.3 - one short guided run:

      1. 10 ordinary clicks (v1.1.2 Instant Auto-Click must feel unchanged)
      2. 5 Pinky Drags (ball into the box; no mode switch)
      3. 5 drag-releases (start a Pinky Drag, curl the little finger: it must drop at once)
      4. left-hand typing, 5. right-hand typing, 6. two-handed typing
      7. Ctrl+A / Ctrl+C / Ctrl+V with the modifier on one hand and the letter on the other
      8. 5 🤟 show / hide cycles

    Typing stages type into a text field in this window through the real Air
    Keyboard; which hand typed each key comes from the keyboard's dispatch
    log (hand attribution only, never the text). Results are stored with the
    other human tests; nothing here is a claim until the owner has run it."""

    WORDS = {"type_left": "sad", "type_right": "lion", "type_both": "airdesk"}
    STAGES = (("left", 10, "Ten ordinary clicks: left-click the blue button (one pinch each)"),
              ("pinkydrag", 5, "Pinky Drag: point at the ball, then only the little finger up (thumb tucked) "
                               "0.2 s, move it into the box, curl the little finger to drop"),
              ("release", 5, "Drag-release: Pinky Drag the ball a little, then curl the little finger - "
                             "it must drop at once"),
              ("type_left", 1, "Open the Air Keyboard (🤟) and type  sad  with your LEFT hand only (blue cursor)"),
              ("type_right", 1, "Type  lion  with your RIGHT hand only (mint cursor)"),
              ("type_both", 1, "Type  airdesk  using BOTH hands"),
              ("shortcut", 1, "Ctrl+A, then Ctrl+C (Ctrl on one hand, the letter on the other), → , then Ctrl+V"),
              ("keyboard", 10, "🤟 hold 0.4 s: show the Air Keyboard, relax, 🤟 again to hide (5 cycles)"))

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.name = "owner_acceptance_v113"
        self.field = QLineEdit(self)
        self.field.setVisible(False)
        f = QFont(self.font())
        f.setPointSize(20)
        self.field.setFont(f)
        self._res0 = 0

    def _begin_stage(self) -> None:
        super()._begin_stage()
        if self.stage >= len(self.STAGES):
            return
        k = self._kind()
        typing = k.startswith("type_") or k == "shortcut"
        self.field.setVisible(typing)
        if typing:
            w = self.width()
            self.field.setGeometry(int(w * 0.25), 160, int(w * 0.5), 56)
            self.field.setText("copy me" if k == "shortcut" else "")
            self.field.setFocus()
            self.activateWindow()
        self._res0 = len(getattr(self.ctx.overlay, "results", []))
        self._hands: dict[str, int] = {}

    def _place(self) -> None:
        k = self._kind()
        if k in ("pinkydrag", "release"):
            w, h = self.width(), self.height()
            self.ball = QPointF(self.rng.uniform(150, w * 0.35), self.rng.uniform(220, h - 160))
            self.slot = QRectF(self.rng.uniform(w * 0.55, w - 240), self.rng.uniform(220, h - 260), 130, 130)
            self._moved = False
            self.update()
            return
        super()._place()

    def _sample(self) -> None:
        super()._sample()
        if self.stage >= len(self.STAGES):
            return
        k = self._kind()
        res = list(getattr(self.ctx.overlay, "results", []))[self._res0:]
        self._hands = {}
        for r in res:
            self._hands[r["hand"]] = self._hands.get(r["hand"], 0) + 1
        text = self.field.text()
        done = False
        if k in self.WORDS:
            done = text == self.WORDS[k]
        elif k == "shortcut":
            done = text == "copy mecopy me"
        if done:
            self._stage_counts["activations"] += 1
            self._stage_counts["hands"] = dict(self._hands)
            self._stage_counts["keys"] = [(r["hand"], r["combo"] or r["key"]) for r in res
                                          if r["key"] not in self.WORDS.get(k, "")] if k == "shortcut" else []
            self._advance()

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if self.stage < len(self.STAGES) and self._kind() in ("pinkydrag", "release"):
            p = e.position()
            if math.dist((p.x(), p.y()), (self.ball.x(), self.ball.y())) < 36:
                self._dragging = True
                self._moved = False
            return
        if self.stage < len(self.STAGES) and (self._kind().startswith("type_") or self._kind() == "shortcut"):
            self._stage_counts["outside"] += 1          # a click while typing: should never happen
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if self._dragging:
            self._moved = True
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if self.stage < len(self.STAGES) and self._kind() in ("pinkydrag", "release"):
            c = self._stage_counts
            if self._dragging:
                self._dragging = False
                if self._kind() == "release" and self._moved:
                    c["activations"] += 1
                    self._advance()
                elif self._kind() == "pinkydrag" and self.slot.contains(self.ball):
                    c["activations"] += 1
                    self._advance()
                else:
                    c["released_outside"] += 1
            self.update()
            return
        if self.stage < len(self.STAGES) and (self._kind().startswith("type_") or self._kind() == "shortcut"):
            return
        super().mouseReleaseEvent(e)

    def paintEvent(self, e) -> None:  # noqa: N802
        if self.stage < len(self.STAGES) and self._kind() in ("pinkydrag", "release"):
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            p.fillRect(self.rect(), theme.qcolor("bg"))
            kind, goal, text = self.STAGES[self.stage]
            self.draw_header(p, f"Stage {self.stage + 1}/{len(self.STAGES)}: {text} · {self.done_in_stage}/{goal}")
            if kind == "pinkydrag":
                p.setPen(QPen(theme.qcolor("ok"), 3, Qt.DashLine))
                p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(self.slot, 14, 14)
            p.setPen(Qt.NoPen)
            p.setBrush(theme.qcolor("warn"))
            p.drawEllipse(self.ball, 30, 30)
            p.end()
            return
        if self.stage < len(self.STAGES) and (self._kind().startswith("type_") or self._kind() == "shortcut"):
            p = QPainter(self)
            p.fillRect(self.rect(), theme.qcolor("bg"))
            kind, goal, text = self.STAGES[self.stage]
            self.draw_header(p, f"Stage {self.stage + 1}/{len(self.STAGES)}: {text}")
            f = QFont(self.font())
            f.setPointSize(11)
            p.setFont(f)
            p.setPen(theme.qcolor("muted"))
            hands = ", ".join(f"{k}: {v} keys" for k, v in self._hands.items()) or "no keys yet"
            p.drawText(QRectF(0, 230, self.width(), 24), Qt.AlignHCenter,
                       f"Keys so far by hand - {hands} · Backspace on the Air Keyboard fixes typos · N skips")
            p.end()
            return
        super().paintEvent(e)

    def summary(self) -> dict:
        out = super().summary()
        for t in self.trials:
            if t["stage"].startswith("type_") or t["stage"] == "shortcut":
                out[f"{t['stage']}_keys_by_hand"] = t["target"].get("hands", {})
            if t["stage"] == "shortcut":
                out["shortcut_sequence"] = t["target"].get("keys", [])
        return out



class OwnerAcceptanceV114(OwnerAcceptanceV113):
    """Owner physical acceptance for v1.1.4 - dual-hand tracking and
    independent two-hand typing (the owner's 10-step checklist):

      1. right hand enters, types five keys     2. right hand leaves, returns alone, five more
      3./4. the same with the left hand          5. both hands type together
      6./7. one hand holds a pinch while the other types
      8. two keys pinched at nearly the same moment   9. Ctrl on one hand, C / V on the other
      10. cross the hands, separate, type        11./12. keyboard closed: clicks and Pinky Drags

    Each typing stage stores which hand typed (dispatch log, never the text)
    and a snapshot of the keyboard's hand diagnostics (reacquisitions and
    their latency, identity corrections, ambiguous frames, rejections), so a
    failure shows whether MediaPipe lost the hand or AirDesk did not
    associate it. Nothing here is a claim until the owner has run it."""

    WORDS = {"type_right": "hello", "type_right_return": "world", "type_left": "sweat",
             "type_left_return": "grade", "type_both": "airdesk", "type_hold_left": "xok",
             "type_hold_right": "pas", "type_cross": "ok"}
    STAGES = (("type_right", 1, "Open the Air Keyboard (🤟). RIGHT hand only (mint cursor): type  hello"),
              ("type_right_return", 1, "RIGHT hand: leave the camera view, wait 2 s, come back (keep the left "
                                       "hand out) and type  world"),
              ("type_left", 1, "LEFT hand only (blue cursor): type  sweat"),
              ("type_left_return", 1, "LEFT hand: leave the view, wait 2 s, come back (right hand out), "
                                      "type  grade"),
              ("type_both", 1, "Both hands together: type  airdesk"),
              ("type_hold_left", 1, "LEFT hand: pinch  x  and KEEP holding it; meanwhile type  ok  with the "
                                    "right hand, then release"),
              ("type_hold_right", 1, "RIGHT hand: pinch  p  and KEEP holding it; meanwhile type  as  with the "
                                     "left hand, then release"),
              ("type_simul", 1, "Pinch  f  (left) and  j  (right) at the same moment - three times"),
              ("shortcut", 1, "Ctrl+A, then Ctrl+C (Ctrl on one hand, the letter on the other), → , then Ctrl+V"),
              ("type_cross", 1, "Cross your hands, separate them again, then type  ok  (each cursor must "
                                "stay with its hand)"),
              ("left", 5, "Close the Air Keyboard (🤟). Five ordinary clicks on the blue button"),
              ("pinkydrag", 3, "Three Pinky Drags: ball into the box"))

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.name = "owner_acceptance_v114"

    def _sample(self) -> None:
        if self.stage < len(self.STAGES) and self._kind() == "type_simul":
            super(OwnerAcceptanceV113, self)._sample()
            res = list(getattr(self.ctx.overlay, "results", []))[self._res0:]
            text = self.field.text()
            if len(text) >= 6 and text.count("f") == 3 and text.count("j") == 3:
                self._stage_counts["activations"] += 1
                self._stage_counts["hands"] = {h: sum(1 for r in res if r["hand"] == h) for h in ("left", "right")}
                self._finish_typing_stage()
                self._advance()
            return
        k = self._kind() if self.stage < len(self.STAGES) else ""
        if k in self.WORDS and self.field.text() == self.WORDS[k]:
            self._finish_typing_stage()
        super()._sample()

    def _finish_typing_stage(self) -> None:
        eng = self.ctx.runtime.engine if getattr(self.ctx, "runtime", None) is not None else None
        if eng is None or not eng._dual_on:
            return
        d = eng.dual.diag()
        self._stage_counts["dual"] = {
            "counters": d["counters"], "polarity": d["polarity"], "label_agree": d["label_agree"],
            "hands": [{k: v[k] for k in ("side", "life", "reacqs", "reacq_ms", "reacq_frames", "rec_ms", "keys")}
                      for v in eng.dual.status()]}

class ScrollTest(TestWindow):
    PROMPTS = ["down", "up", "right", "left", "rest", "down", "up", "rest", "right", "left"]

    def __init__(self, ctx) -> None:
        super().__init__(ctx, "scroll_direction")
        self.i = -1
        self._t0 = 0.0
        self._events: list[tuple[float, int, int]] = []

    def start(self) -> None:
        super().start()
        QTimer.singleShot(400, self._next)

    def _next(self) -> None:
        if self.i >= 0:
            self._score()
        self.i += 1
        if self.i >= len(self.PROMPTS):
            self.finish()
            return
        self._events = []
        self._t0 = time.perf_counter()
        QTimer.singleShot(3000 if self.PROMPTS[self.i] == "rest" else 4000, self._next)
        self.update()

    def wheelEvent(self, e) -> None:  # noqa: N802
        d = e.angleDelta()
        self._events.append((time.perf_counter() - self._t0, d.x(), d.y()))

    def _score(self) -> None:
        want = self.PROMPTS[self.i]
        evs = self._events
        if want == "rest":
            self.trials.append({"prompt": "rest", "undesired_scroll_events": len(evs)})
            return

        def dir_of(dx, dy):
            if abs(dy) >= abs(dx):
                return "up" if dy > 0 else "down"
            return "left" if dx > 0 else "right"     # Qt: positive x = scroll left
        dirs = [dir_of(dx, dy) for _, dx, dy in evs if dx or dy]
        first = dirs[0] if dirs else None
        self.trials.append({"prompt": want, "first_direction": first, "correct": first == want,
                            "time_to_first_ms": round(evs[0][0] * 1000, 1) if evs else None,
                            "events": len(dirs), "wrong_direction_events": sum(d != want for d in dirs)})

    def summary(self) -> dict:
        moves = [t for t in self.trials if t["prompt"] != "rest"]
        rests = [t for t in self.trials if t["prompt"] == "rest"]
        times = [t["time_to_first_ms"] for t in moves if t["time_to_first_ms"] is not None]
        tot = sum(t["events"] for t in moves)
        return {"direction_accuracy": round(sum(t["correct"] for t in moves) / max(len(moves), 1), 3),
                "wrong_direction_event_share": round(sum(t["wrong_direction_events"] for t in moves) / max(tot, 1), 3),
                "undesired_scroll_events_at_rest": sum(t["undesired_scroll_events"] for t in rests),
                "median_time_to_scroll_ms": statistics.median(times) if times else None}

    def paintEvent(self, e) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), theme.qcolor("bg"))
        if 0 <= self.i < len(self.PROMPTS):
            want = self.PROMPTS[self.i]
            self.draw_header(p, "Hold still - do NOT scroll" if want == "rest" else f"Scroll {want.upper()}")
            arrow = {"up": "↑", "down": "↓", "left": "←", "right": "→", "rest": "■"}[want]
            f = QFont(self.font())
            f.setPointSize(120)
            p.setFont(f)
            p.setPen(theme.qcolor("accent"))
            p.drawText(self.rect(), Qt.AlignCenter, arrow)
        p.end()


class TypingTest(TestWindow):
    def __init__(self, ctx, lang: str) -> None:
        super().__init__(ctx, f"typing_{lang}")
        self.lang = lang
        self.phrases = list(PHRASES[lang])
        self.i = 0
        lay = QVBoxLayout(self)
        lay.setContentsMargins(200, 160, 200, 200)
        self.prompt = QLabel()
        self.prompt.setAlignment(Qt.AlignCenter)
        self.prompt.setStyleSheet("font-size: 28pt; font-weight: 600;")
        self.edit = QLineEdit()
        self.edit.setStyleSheet("font-size: 24pt; padding: 10px;")
        self.edit.returnPressed.connect(self._submit)
        self.edit.textEdited.connect(self._edited)
        self.info = QLabel()
        self.info.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.prompt)
        lay.addWidget(self.edit)
        lay.addWidget(self.info)
        lay.addStretch(1)
        self._t_first = None
        self._bs = 0
        self._keys = 0
        self.edit.installEventFilter(self)
        ctx.overlay.key_typed.connect(self._key)
        self._load()

    def start(self) -> None:
        super().start()
        self.edit.setFocus()
        if not self.ctx.overlay.isVisible():
            self.ctx.overlay.set_shown(True)

    def _key(self, key: str) -> None:
        if self.isVisible():
            self._keys += 1

    def _load(self) -> None:
        self.prompt.setText(self.phrases[self.i])
        self.edit.clear()
        self._t_first, self._bs, self._keys = None, 0, 0
        hint = "Air Keyboard language: VI (Telex), e.g. 'xin chafo'" if self.lang == "vi" else "Air Keyboard: EN"
        self.info.setText(f"Phrase {self.i + 1}/{len(self.phrases)} · type it, then press Enter · {hint}")

    def _edited(self, txt: str) -> None:
        if self._t_first is None and txt:
            self._t_first = time.perf_counter()

    def eventFilter(self, obj, e):  # noqa: N802
        from PySide6.QtCore import QEvent
        if obj is self.edit and e.type() == QEvent.KeyPress and e.key() == Qt.Key_Backspace:
            self._bs += 1
        return False

    def _submit(self) -> None:
        typed = self.edit.text()
        target = self.phrases[self.i]
        secs = (time.perf_counter() - self._t_first) if self._t_first else 0.0
        dist = levenshtein(typed, target)
        self.trials.append({"chars": len(target), "seconds": round(secs, 2),
                            "wpm": round((len(typed) / 5) / (secs / 60), 2) if secs > 0 else 0,
                            "char_accuracy": round(max(0.0, 1 - dist / max(len(target), 1)), 3),
                            "uncorrected_errors": dist, "air_keyboard_keypresses": self._keys,
                            "backspaces": self._bs})
        self.i += 1
        if self.i >= len(self.phrases):
            self.ctx.overlay.key_typed.disconnect(self._key)
            self.finish()
        else:
            self._load()

    def summary(self) -> dict:
        t = self.trials
        return {"phrases": len(t), "median_wpm": statistics.median(x["wpm"] for x in t),
                "mean_char_accuracy": round(statistics.mean(x["char_accuracy"] for x in t), 3),
                "total_uncorrected_errors": sum(x["uncorrected_errors"] for x in t),
                "keystrokes_per_char": round(sum(x["air_keyboard_keypresses"] for x in t)
                                             / max(sum(x["chars"] for x in t), 1), 2)}

    def paintEvent(self, e) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), theme.qcolor("bg"))
        self.draw_header(p, f"Typing test ({'Vietnamese' if self.lang == 'vi' else 'English'})")
        p.end()


class FingerSwitchTest(TestWindow):
    def __init__(self, ctx, n: int = 10) -> None:
        super().__init__(ctx, "finger_switching")
        self.n = n
        self.want: str | None = None
        self._t0 = 0.0
        self._pos0 = (0, 0)
        self._max_dev = 0.0
        self._wrong: set[str] = set()
        self._stable_since: float | None = None
        self._prev = None
        self._cur = None

    def start(self) -> None:
        super().start()
        QTimer.singleShot(600, self._next)

    def _next(self) -> None:
        if len(self.trials) >= self.n:
            self.finish()
            return
        st = self.ctx.runtime.status
        cur = st.finger if st.tracking else "Index"
        choices = [f for f in ("Thumb", "Index", "Middle", "Ring", "Little") if f != cur]
        self.want = random.choice(choices)
        self._prev = cur
        self._t0 = time.perf_counter()
        p = QCursor.pos()
        self._pos0 = (p.x(), p.y())
        self._max_dev = 0.0
        self._wrong = set()
        self._stable_since = None

    def _sample(self) -> None:
        super()._sample()
        if not self.want:
            return
        st = self.ctx.runtime.status
        p = QCursor.pos()
        now = time.perf_counter()
        if st.tracking and st.finger not in (self.want, self._prev):
            self._wrong.add(st.finger)
        if st.tracking and st.finger == self.want:
            self._stable_since = self._stable_since or now
            if now - self._stable_since >= 0.3:
                self.trials.append({"from": self._prev, "to": self.want,
                                    "switch_time_ms": round((self._stable_since - self._t0) * 1000, 1),
                                    "cursor_max_deviation_px": round(self._max_dev, 1),
                                    "unwanted_fingers": sorted(self._wrong)})
                self.want = None
                QTimer.singleShot(700, self._next)
                return
        else:
            self._stable_since = None
            self._max_dev = max(self._max_dev, math.dist(self._pos0, (p.x(), p.y())))
        if now - self._t0 > 8:
            self.trials.append({"from": self._prev, "to": self.want, "switch_time_ms": None,
                                "cursor_max_deviation_px": round(self._max_dev, 1), "unwanted_fingers": sorted(self._wrong),
                                "timeout": True})
            self.want = None
            QTimer.singleShot(700, self._next)

    def summary(self) -> dict:
        t = self.trials
        ok = [x for x in t if x["switch_time_ms"] is not None]
        return {"trials": len(t), "recognised_rate": round(len(ok) / max(len(t), 1), 3),
                "median_switch_time_ms": statistics.median(x["switch_time_ms"] for x in ok) if ok else None,
                "median_cursor_deviation_px": statistics.median(x["cursor_max_deviation_px"] for x in t),
                "trials_with_unwanted_finger": sum(1 for x in t if x["unwanted_fingers"])}

    def paintEvent(self, e) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), theme.qcolor("bg"))
        self.draw_header(p, f"Keep your hand still and point with the shown finger · {len(self.trials)}/{self.n}")
        f = QFont(self.font())
        f.setPointSize(64)
        p.setFont(f)
        p.setPen(theme.qcolor("accent"))
        p.drawText(self.rect(), Qt.AlignCenter, self.want or "✔")
        st = self.ctx.runtime.status
        f.setPointSize(14)
        p.setFont(f)
        p.setPen(theme.qcolor("muted"))
        p.drawText(QRectF(0, self.height() - 120, self.width(), 30), Qt.AlignHCenter,
                   f"AirDesk sees: {st.finger if st.tracking else 'no hand'}")
        p.end()


# ==================================================================== page
class BenchmarkPage:
    def __init__(self, ctx: "AppContext") -> None:
        self.ctx = ctx
        self.widget, lay = page("Benchmark Laboratory",
                                "Synthetic benchmarks test the algorithms on generated landmark streams. Human "
                                "tests measure you using AirDesk with the real camera. They are stored and "
                                "labelled separately, and a human test run without arming AirDesk is recorded "
                                "as a physical-mouse baseline.")
        g, f = group("Synthetic algorithm benchmarks (reproducible, no camera)", lay)
        row = QHBoxLayout()
        self.run_syn = QPushButton("Run synthetic suite")
        self.run_syn.setObjectName("Primary")
        self.run_syn.clicked.connect(self._run_synthetic)
        row.addWidget(self.run_syn)
        self.syn_status = label("", "Muted")
        row.addWidget(self.syn_status, 1)
        f.addRow(row)
        self.chart = QLabel()
        self.chart.setAlignment(Qt.AlignCenter)
        f.addRow(self.chart)

        g, f = group("Human usability tests (real camera and gestures - arm AirDesk first)", lay)
        grid = QHBoxLayout()
        self.n_targets = QSpinBox()
        self.n_targets.setRange(6, 60)
        self.n_targets.setValue(15)
        acc4 = QPushButton("Owner acceptance (v1.1.4): two-hand tracking and independent typing")
        acc4.setObjectName("Primary")
        acc4.clicked.connect(lambda: self._open(OwnerAcceptanceV114(ctx)))
        f.addRow(acc4)
        f.addRow(label("Right hand alone · right hand leaves and returns alone · the same with the left · both "
                       "hands · hold a pinch on one hand while the other types (both ways) · two keys at the "
                       "same moment · Ctrl on one hand, C / V on the other · crossed hands · then ordinary "
                       "clicks and Pinky Drags with the keyboard closed. Diagnostics → Keyboard LEFT / RIGHT "
                       "hand rows show whether MediaPipe saw the hand and which controller got it.",
                       "Muted", True))
        acc3 = QPushButton("Owner acceptance (v1.1.3): clicks, Pinky Drag, two-hand keyboard")
        acc3.clicked.connect(lambda: self._open(OwnerAcceptanceV113(ctx)))
        f.addRow(acc3)
        f.addRow(label("10 ordinary clicks · 5 Pinky Drags · 5 drag-releases · type with the left hand, the "
                       "right hand, both hands · Ctrl+A / C / V with the modifier and the letter on different "
                       "hands · 5 🤟 show/hide cycles. Press M whenever a deliberate pinch did nothing; N skips "
                       "a stage. Diagnostics shows the recognized hand, pose, cursors, dispatch and release "
                       "state live.", "Muted", True))
        acc = QPushButton("Owner click acceptance (v1.1.2, legacy Drag Mode)")
        acc.clicked.connect(lambda: self._open(ClickAcceptanceTest(ctx)))
        f.addRow(acc)
        f.addRow(label("20 left clicks · 10 right clicks · 10 small checkboxes · 10 rapid repeats · 5 legacy "
                       "Drag Mode drags (hold 1.2 s) · 3 drops by tracking loss · 5 🤟 keyboard cycles (0.4 s).",
                       "Muted", True))
        for text, fn in (("Target acquisition", lambda: self._open(TargetTest(ctx, self.n_targets.value()))),
                         ("Small targets (12-20 px)", lambda: self._open(PrecisionTest(ctx))),
                         ("Drag and drop", lambda: self._open(DragTest(ctx))),
                         ("Scroll direction", lambda: self._open(ScrollTest(ctx))),
                         ("Finger switching", lambda: self._open(FingerSwitchTest(ctx))),
                         ("Typing (English)", lambda: self._open(TypingTest(ctx, "en"))),
                         ("Typing (Vietnamese)", lambda: self._open(TypingTest(ctx, "vi")))):
            b = QPushButton(text)
            b.clicked.connect(fn)
            grid.addWidget(b)
        f.addRow(grid)
        f.addRow("Targets per run", self.n_targets)
        f.addRow(label("Multi-monitor mapping: the target test runs on the monitor that contains the AirDesk "
                       "window; move the window to another monitor to test it there.", "Muted", True))

        g, f = group("Performance", lay)
        self.perf_btn = QPushButton("Measure capture modes and inference (pauses tracking ~25 s)")
        self.perf_btn.clicked.connect(self._run_perf)
        f.addRow(self.perf_btn)
        self.perf_lbl = label("", "Muted", wrap=True)
        f.addRow(self.perf_lbl)

        g, f = group("Results", lay)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["When", "Test", "Kind / input", "Summary"])
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(260)
        f.addRow(self.table)
        r = QHBoxLayout()
        op = QPushButton("Open results folder")
        op.clicked.connect(lambda: os.startfile(str(results_dir())))
        r.addWidget(op)
        r.addStretch(1)
        f.addRow(r)
        lay.addStretch(1)
        self._win: TestWindow | None = None
        self.refresh()

    def on_tick(self, st, m) -> None:
        pass

    def refresh(self) -> None:
        files = sorted(results_dir().glob("*.json"), reverse=True)[:60]
        self.table.setRowCount(0)
        for p in files:
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            r = self.table.rowCount()
            self.table.insertRow(r)
            summ = d.get("summary", {})
            txt = ", ".join(f"{k}={v}" for k, v in list(summ.items())[:6]) if summ else "(see file)"
            for c, v in enumerate((p.stem[:15], d.get("test", "?"),
                                   d.get("kind", "?") + (f" / {d['input_source']}" if d.get("input_source") else ""),
                                   txt)):
                self.table.setItem(r, c, QTableWidgetItem(str(v)))
        self.table.resizeColumnsToContents()

    def _open(self, w: TestWindow) -> None:
        self._win = w
        w.finished.connect(self._done)
        w.start()

    def _done(self, res: dict) -> None:
        if res["trials"]:
            save_result(res)
        self.refresh()
        self.ctx.window.show_normal()

    def _run_synthetic(self) -> None:
        self.run_syn.setEnabled(False)

        def work() -> None:
            from ..bench.synthetic import run_all
            res = run_all(lambda n, i, k: setattr(self, "_syn_msg", f"Running {n} ({i + 1}/{k})…"))
            res["test"] = "synthetic_suite"
            res["summary"] = {"one_euro_jitter_px": res["filters"][0]["jitter_px"],
                              "one_euro_lag_ms": res["filters"][0]["lag_ms"],
                              "false_clicks_per_min_noise_0.0015": res["false_clicks"][1]["per_minute"],
                              "click_success_noise_0.0015": res["pinch_detection"][1]["click_success"],
                              "drag_success_noise_0.0015": res["pinch_detection"][1]["drag_success"],
                              "switch_max_step_px": res["finger_switching"][1]["max_step_px"],
                              "engine_us_per_frame": res["engine_cost"]["engine_us_per_frame"]}
            path = save_result(res)
            self._chart_path = render_filter_chart(res, path.with_suffix(".png"))
            self._syn_msg = f"Done - saved {path.name}"

        self._syn_msg = "Starting…"
        th = threading.Thread(target=work, daemon=True)
        th.start()
        timer = QTimer(self.widget, interval=200)

        def poll() -> None:
            self.syn_status.setText(self._syn_msg)
            if not th.is_alive():
                timer.stop()
                self.run_syn.setEnabled(True)
                if getattr(self, "_chart_path", None):
                    self.chart.setPixmap(QPixmap(str(self._chart_path)).scaledToWidth(760, Qt.SmoothTransformation))
                self.refresh()
        timer.timeout.connect(poll)
        timer.start()

    def _run_perf(self) -> None:
        self.perf_btn.setEnabled(False)
        rt = self.ctx.runtime
        rt.stop_camera()                  # never two streams on one device

        def work() -> None:
            from ..bench.performance import run_performance
            try:
                self._perf = run_performance(self.ctx.s.camera.index)
            except Exception as e:  # report, never crash the UI
                self._perf = {"error": str(e)}

        th = threading.Thread(target=work, daemon=True)
        th.start()
        timer = QTimer(self.widget, interval=300)

        def poll() -> None:
            if th.is_alive():
                self.perf_lbl.setText("Measuring… (camera paused)")
                return
            timer.stop()
            rt.start_camera()
            self.perf_btn.setEnabled(True)
            res = self._perf
            if "error" in res:
                self.perf_lbl.setText(f"Failed: {res['error']}")
                return
            res.update({"test": "performance", "kind": "measurement"})
            save_result(res)
            lines = [f"{c['backend'].upper()} {c['got']}: {c['fps']} FPS" for c in res["capture"]]
            lines += [f"Inference {i['source']} → {i['inference_width']}px on {i['delegate_used'].upper()}: "
                      f"{i['total_ms_median']} ms median" for i in res["inference"]]
            self.perf_lbl.setText("\n".join(lines))
            self.refresh()
        timer.timeout.connect(poll)
        timer.start()


def render_filter_chart(res: dict, path: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    rows = res["filters"]
    fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=110)
    for r in rows:
        ax.scatter(r["lag_ms"], r["jitter_px"], s=60)
        ax.annotate(r["filter"], (r["lag_ms"], r["jitter_px"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Mid-reach lag (ms)  — lower is more responsive")
    ax.set_ylabel("Rest jitter (px)  — lower is steadier")
    ax.set_title("Pointer filters on a synthetic reach (synthetic benchmark)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
