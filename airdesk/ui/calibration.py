"""First-run calibration wizard, quick recalibration and profile management.

Measurements use the live landmark stream (a runtime frame hook) and never
store images: only the derived numbers (control box, noise, pinch
distances) end up in the settings/profile JSON.
"""

from __future__ import annotations

import math
import statistics
import time
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import (QButtonGroup, QDialog, QHBoxLayout, QInputDialog, QLabel, QListWidget,
                               QMessageBox, QProgressBar, QPushButton, QStackedWidget, QVBoxLayout, QWidget)

from ..config import Settings, list_profiles, load_profile, profiles_dir, save_profile
from . import theme
from .widgets import Binder, PreviewWidget, group, label, page

if TYPE_CHECKING:
    from .main_window import AppContext

FINGERS = ["Thumb", "Index", "Middle", "Ring", "Little"]
TIP = {"Thumb": 4, "Index": 8, "Middle": 12, "Ring": 16, "Little": 20}
MCP = {"Thumb": 2, "Index": 5, "Middle": 9, "Ring": 13, "Little": 17}


class Recorder:
    """Collects the active pointer source point from the vision thread."""

    def __init__(self, ctx: "AppContext") -> None:
        self.ctx = ctx
        self.points: list[tuple[float, float, float]] = []
        self.pinch: list[float] = []
        self.active = False

    def __enter__(self):
        self.ctx.runtime.frame_hooks.append(self._hook)
        return self

    def __exit__(self, *a):
        if self._hook in self.ctx.runtime.frame_hooks:
            self.ctx.runtime.frame_hooks.remove(self._hook)

    def start(self) -> None:
        self.points.clear()
        self.pinch.clear()
        self.active = True

    def _hook(self, fi, st) -> None:
        if not self.active or not st.tracking or st.finger not in TIP:
            return
        wrist = self.ctx.runtime.engine.hands._last_wrist
        hand = next((h for h in fi.hands if wrist and abs(h.image[0, 0] - wrist[0]) < 1e-9), None)
        if hand is None:
            return
        a = self.ctx.s.pointer.anchor_blend
        tip, mcp = hand.image[TIP[st.finger]], hand.image[MCP[st.finger]]
        self.points.append((fi.t, tip[0] * (1 - a) + mcp[0] * a, tip[1] * (1 - a) + mcp[1] * a))
        if "primary" in st.pinch:
            self.pinch.append(st.pinch["primary"])


class ClickArena(QWidget):
    """Targets to click and a ball to drag into a slot (real mouse events)."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(300)
        self.targets: list[QPointF] = []
        self.hits = self.misses = 0
        self.drag_ok = self.drag_fail = 0
        self._ball = QPointF(60, 250)
        self._dragging = False
        self.reset()

    def reset(self) -> None:
        rng = np.random.default_rng(int(time.time()))
        self.targets = [QPointF(rng.uniform(80, max(self.width(), 500) - 80), rng.uniform(40, 200))
                        for _ in range(5)]
        self.hits = self.misses = self.drag_ok = self.drag_fail = 0
        self._ball = QPointF(60, 260)
        self.update()

    def _slot(self) -> QRectF:
        return QRectF(self.width() - 140, 225, 90, 70)

    def mousePressEvent(self, e) -> None:  # noqa: N802
        p = e.position()
        if math.dist((p.x(), p.y()), (self._ball.x(), self._ball.y())) < 26:
            self._dragging = True
            return
        for t in self.targets:
            if math.dist((p.x(), p.y()), (t.x(), t.y())) < 22:
                self.targets.remove(t)
                self.hits += 1
                self.update()
                return
        self.misses += 1
        self.update()

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if self._dragging:
            self._ball = e.position()
            self.update()

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if self._dragging:
            self._dragging = False
            if self._slot().contains(self._ball):
                self.drag_ok += 1
            else:
                self.drag_fail += 1
            self._ball = QPointF(60, 260)
            self.update()

    def paintEvent(self, e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), theme.qcolor("panel"))
        p.setPen(Qt.NoPen)
        p.setBrush(theme.qcolor("accent"))
        for t in self.targets:
            p.drawEllipse(t, 20, 20)
        p.setPen(QPen(theme.qcolor("ok"), 2, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(self._slot(), 10, 10)
        p.setPen(Qt.NoPen)
        p.setBrush(theme.qcolor("warn"))
        p.drawEllipse(self._ball, 22, 22)
        p.setPen(theme.qcolor("text"))
        p.drawText(QRectF(10, self.height() - 26, self.width() - 20, 22), Qt.AlignLeft,
                   f"Targets hit {self.hits}/5 · missed clicks {self.misses} · "
                   f"drags into the slot {self.drag_ok} · dropped outside {self.drag_fail}")
        p.end()


class Wizard(QDialog):
    STEPS = ["Welcome", "Camera", "Hand and finger", "Five-finger test", "Comfortable zone",
             "Resting noise", "Pinch distances", "Sensitivity", "Click and drag test", "Save profile"]

    def __init__(self, ctx: "AppContext", only_zone: bool = False) -> None:
        super().__init__(ctx.window)
        self.ctx = ctx
        self.only_zone = only_zone
        self.setWindowTitle("AirDesk calibration")
        self.resize(980, 760)
        self.binder = Binder(ctx.changed)
        self.rec = Recorder(ctx).__enter__()
        lay = QVBoxLayout(self)
        self.title = label("", "H1")
        self.stepinfo = label("", "Muted")
        lay.addWidget(self.title)
        lay.addWidget(self.stepinfo)
        body = QHBoxLayout()
        self.stack = QStackedWidget()
        body.addWidget(self.stack, 3)
        self.preview = PreviewWidget()
        self.preview.setMinimumSize(380, 240)
        body.addWidget(self.preview, 2)
        lay.addLayout(body, 1)
        nav = QHBoxLayout()
        self.back = QPushButton("Back")
        self.back.clicked.connect(lambda: self._go(self.stack.currentIndex() - 1))
        self.skip = QPushButton("Skip step")
        self.skip.clicked.connect(lambda: self._go(self.stack.currentIndex() + 1))
        self.next = QPushButton("Next")
        self.next.setObjectName("Primary")
        self.next.clicked.connect(self._next)
        nav.addWidget(self.back)
        nav.addStretch(1)
        nav.addWidget(self.skip)
        nav.addWidget(self.next)
        lay.addLayout(nav)
        self._build()
        self.timer = QTimer(self, interval=33, timeout=self._tick)
        self.timer.start()
        self._phase_t = 0.0
        self._phase = ""
        self._seen: dict[str, float] = {}
        self._go(4 if only_zone else 0)

    # ------------------------------------------------------------- pages
    def _text_page(self, text: str) -> tuple[QWidget, QVBoxLayout]:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(label(text, wrap=True))
        return w, v

    def _build(self) -> None:
        s = self.ctx.s
        b = self.binder
        w, v = self._text_page(
            "This wizard tunes AirDesk to your camera position, your hand and your preferred finger. "
            "It takes about two minutes. Gesture input stays DISARMED until the click-and-drag test, "
            "where you arm it yourself. Only numbers are saved - never images.\n\n"
            "Sit where you normally work and make sure your hand is visible in the preview.")
        v.addStretch(1)
        self.stack.addWidget(w)

        w, v = self._text_page("Choose the camera and mirroring. With mirroring on, moving your hand to the "
                               "right moves the preview hand to the right.")
        g, f = group("Camera")
        b.check(f, "Mirror camera (selfie view)", s.camera, "mirror", "camera")
        from PySide6.QtWidgets import QComboBox
        res = QComboBox()
        for wh in ((1280, 720), (1920, 1080), (3840, 2160)):
            res.addItem(f"{wh[0]} × {wh[1]}", wh)
        res.setCurrentIndex(max(res.findData((s.camera.width, s.camera.height)), 0))

        def set_res(_i) -> None:
            s.camera.width, s.camera.height = res.currentData()
            self.ctx.changed("camera")
        res.currentIndexChanged.connect(set_res)
        f.addRow("Capture resolution", res)
        v.addWidget(g)
        v.addStretch(1)
        self.stack.addWidget(w)

        w, v = self._text_page("Which hand will you point with, and with which finger? Auto mode lets you "
                               "switch fingers simply by pointing with a different one.")
        g, f = group("Hand and finger")
        b.combo(f, "Hand", s.finger, "hand", [("auto", "Automatic"), ("right", "Right"), ("left", "Left")], "finger")
        b.combo(f, "Finger mode", s.finger, "mode", [("auto", "Auto"), ("manual", "Manual"), ("lock", "Lock")],
                "finger")
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        self.fgroup = QButtonGroup(self)
        for i, n in enumerate(FINGERS):
            btn = QPushButton(n)
            btn.setCheckable(True)
            btn.setChecked(n.lower() == s.finger.finger)
            self.fgroup.addButton(btn, i)
            h.addWidget(btn)
        self.fgroup.idClicked.connect(lambda i: (self.ctx.runtime.set_finger(FINGERS[i]), self.ctx.changed("finger")))
        f.addRow("Preferred finger", row)
        v.addWidget(g)
        v.addStretch(1)
        self.stack.addWidget(w)

        w, v = self._text_page("Point with each finger in turn: extend only that finger and curl the others "
                               "(for the thumb, make a fist with the thumb out). Each one ticks when AirDesk "
                               "has recognised it for half a second.")
        self.finger_lbls = {}
        for n in FINGERS:
            lb = QLabel(f"○  {n}")
            lb.setObjectName("H2")
            self.finger_lbls[n] = lb
            v.addWidget(lb)
        v.addStretch(1)
        self.stack.addWidget(w)

        w, v = self._text_page("Comfortable zone: press Start, then for 6 seconds move your pointing finger "
                               "around the whole area you can reach comfortably - left, right, top and bottom. "
                               "That area becomes the full screen in Direct mode.")
        self.zone_btn = QPushButton("Start recording")
        self.zone_btn.clicked.connect(lambda: self._start("zone", 6.0))
        self.zone_bar = QProgressBar()
        self.zone_res = label("", "Muted", wrap=True)
        for x in (self.zone_btn, self.zone_bar, self.zone_res):
            v.addWidget(x)
        v.addStretch(1)
        self.stack.addWidget(w)

        w, v = self._text_page("Resting noise: press Start and hold your pointing hand as still as you "
                               "comfortably can for 3 seconds. AirDesk measures tracking jitter and picks "
                               "smoothing that removes it with the least lag.")
        self.noise_btn = QPushButton("Start measuring")
        self.noise_btn.clicked.connect(lambda: self._start("noise", 3.0))
        self.noise_bar = QProgressBar()
        self.noise_res = label("", "Muted", wrap=True)
        for x in (self.noise_btn, self.noise_bar, self.noise_res):
            v.addWidget(x)
        v.addStretch(1)
        self.stack.addWidget(w)

        w, v = self._text_page("Pinch distances for your primary click: 1) keep pointing normally and press "
                               "'Open' (2 s), 2) touch the thumb to the pointing finger, keep it touching and "
                               "press 'Pinched' (2 s).")
        row = QHBoxLayout()
        self.open_btn = QPushButton("Open (2 s)")
        self.open_btn.clicked.connect(lambda: self._start("open", 2.0))
        self.pinch_btn = QPushButton("Pinched (2 s)")
        self.pinch_btn.clicked.connect(lambda: self._start("pinched", 2.0))
        row.addWidget(self.open_btn)
        row.addWidget(self.pinch_btn)
        v.addLayout(row)
        self.pinch_bar = QProgressBar()
        self.pinch_res = label("", "Muted", wrap=True)
        v.addWidget(self.pinch_bar)
        v.addWidget(self.pinch_res)
        v.addStretch(1)
        self.stack.addWidget(w)
        self._open_d = self._pinch_d = None

        w, v = self._text_page("Sensitivity. Direct mode maps your zone to the screen; Relative mode works "
                               "like a trackpad (make a fist to clutch and re-centre your hand).")
        g, f = group("Pointer")
        b.combo(f, "Pointer mode", s.pointer, "mode", [("direct", "Direct"), ("relative", "Relative trackpad"),
                                                       ("precision", "Precision")], "pointer")
        b.number(f, "Direct gain", s.pointer, "direct_gain", 0.5, 2.5, 0.05, 2, section="pointer")
        b.number(f, "Relative gain (fast)", s.pointer, "relative_gain_high", 0.5, 8, 0.1, 1, section="pointer")
        b.number(f, "Smoothing (One Euro min cutoff)", s.pointer, "min_cutoff", 0.1, 5, 0.05, 2, "Hz", "pointer")
        v.addWidget(g)
        v.addStretch(1)
        self.stack.addWidget(w)

        w, v = self._text_page("Click and drag test with real input. Press 'Arm', then click the five blue "
                               "targets - one pinch clicks at once (green ring with ✓ = the click was sent). "
                               "To drag the orange ball into the dashed slot, aim at it, then leave only the "
                               "little finger up (thumb tucked) for 0.2 s - Pinky Drag, no mode switch - move, "
                               "and curl the little finger to drop. "
                               "Show/hide the Air Keyboard with 🤟 (thumb, index, little out) held 0.4 s. "
                               "Ctrl+Alt+F12 disarms at any time.")
        row = QHBoxLayout()
        arm = QPushButton("Arm gesture input")
        arm.setObjectName("Primary")
        arm.clicked.connect(lambda: self.ctx.runtime.arm(True))
        dis = QPushButton("Disarm")
        dis.clicked.connect(lambda: self.ctx.runtime.arm(False))
        again = QPushButton("Reset test")
        row.addWidget(arm)
        row.addWidget(dis)
        row.addWidget(again)
        row.addStretch(1)
        v.addLayout(row)
        self.arena = ClickArena()
        again.clicked.connect(self.arena.reset)
        v.addWidget(self.arena, 1)
        self.stack.addWidget(w)

        w, v = self._text_page("Save these settings as a profile (for example one per desk, camera position or "
                               "person). Profiles can be switched from the Calibration page.")
        self.prof_lbl = label("", "Muted", wrap=True)
        save = QPushButton("Save as profile…")
        save.clicked.connect(self._save_profile)
        v.addWidget(save)
        v.addWidget(self.prof_lbl)
        v.addStretch(1)
        self.stack.addWidget(w)

    # ------------------------------------------------------------ flow
    def _go(self, i: int) -> None:
        i = max(0, min(i, self.stack.count() - 1))
        if self.stack.currentIndex() == 8 and i != 8:
            self.ctx.runtime.arm(False)            # never leave the test armed
        self.stack.setCurrentIndex(i)
        self.title.setText(self.STEPS[i])
        self.stepinfo.setText(f"Step {i + 1} of {len(self.STEPS)}")
        self.back.setEnabled(i > 0)
        self.next.setText("Finish" if i == self.stack.count() - 1 or self.only_zone else "Next")
        self.binder.refresh()
        if i == 8:
            self.arena.reset()

    def _next(self) -> None:
        if self.only_zone or self.stack.currentIndex() == self.stack.count() - 1:
            self.ctx.s.ui.first_run_done = True
            self.ctx.changed("ui")
            self.accept()
            return
        self._go(self.stack.currentIndex() + 1)

    def done(self, r: int) -> None:
        self.timer.stop()
        self.rec.active = False
        self.rec.__exit__()
        self.ctx.runtime.arm(False)
        if not self.ctx.s.ui.first_run_done:
            self.ctx.s.ui.first_run_done = True     # never force the wizard on every start
            self.ctx.changed("ui")
        super().done(r)

    def _start(self, phase: str, seconds: float) -> None:
        self._phase, self._phase_t, self._phase_len = phase, time.monotonic(), seconds
        self.rec.start()

    # ------------------------------------------------------------- tick
    def _tick(self) -> None:
        rt = self.ctx.runtime
        st, m = rt.snapshot()
        pc = self.ctx.s.pointer
        self.preview.set_frame(rt.preview, rt.engine.hands._last_wrist, st.finger if st.tracking else None,
                               (pc.box_cx, pc.box_cy, pc.box_w, pc.box_h), pc.anchor_blend, st.gesture)
        if self.stack.currentIndex() == 3 and st.tracking:
            conf = st.finger_confidence.get(st.finger, 0)
            for n in FINGERS:
                if conf > 0.8 and n == st.finger:
                    self._seen.setdefault(n, time.monotonic())
            for n, lb in self.finger_lbls.items():
                ok = n in self._seen and time.monotonic() - self._seen[n] > 0.5
                lb.setText(f"{'✔' if ok else '○'}  {n}" + ("   ← now" if n == st.finger else ""))
                lb.setStyleSheet(f"color: {theme.current['ok']};" if ok else "")
        if not self._phase:
            return
        frac = (time.monotonic() - self._phase_t) / self._phase_len
        bar = {"zone": self.zone_bar, "noise": self.noise_bar}.get(self._phase, self.pinch_bar)
        bar.setValue(int(min(frac, 1) * 100))
        if frac >= 1:
            phase, self._phase = self._phase, ""
            self.rec.active = False
            getattr(self, f"_finish_{phase}")()

    def _finish_zone(self) -> None:
        pts = np.array([(x, y) for _, x, y in self.rec.points])
        if len(pts) < 30:
            self.zone_res.setText("Not enough tracking data - make sure your hand is visible and try again.")
            return
        lo, hi = np.percentile(pts, 4, axis=0), np.percentile(pts, 96, axis=0)
        w, h = (hi - lo) * 0.92                    # inner 92%: edges reachable without straining
        pc = self.ctx.s.pointer
        pc.box_cx, pc.box_cy = float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)
        pc.box_w, pc.box_h = float(max(w, 0.12)), float(max(h, 0.12))
        self.ctx.changed("pointer")
        self.zone_res.setText(f"Zone set: centre ({pc.box_cx:.2f}, {pc.box_cy:.2f}), "
                              f"size {pc.box_w:.2f} × {pc.box_h:.2f} of the camera image "
                              f"({len(pts)} samples). The dashed box in the preview shows it.")

    def _finish_noise(self) -> None:
        pts = np.array([(x, y) for _, x, y in self.rec.points])
        if len(pts) < 20:
            self.noise_res.setText("Not enough tracking data - try again.")
            return
        pc = self.ctx.s.pointer
        # Jitter = high-frequency part: subtract a 9-sample moving average so
        # slow drift of the "still" hand is not counted as tracking noise.
        k = np.ones(9) / 9
        resid = np.stack([pts[:, i] - np.convolve(pts[:, i], k, "same") for i in (0, 1)], axis=1)[5:-5]
        std = float(np.sqrt(resid[:, 0].var() + resid[:, 1].var()))
        scr = self.ctx.runtime.screen.target_rect(pc.monitor) if self.ctx.runtime.screen else None
        px_per_unit = (scr.height / max(pc.box_h, 1e-3)) if scr else 2000
        jitter_px = std * px_per_unit
        if jitter_px < 2:
            cutoff = 0.9
        elif jitter_px < 5:
            cutoff = 0.6
        elif jitter_px < 10:
            cutoff = 0.45
        else:
            cutoff = 0.3
        pc.min_cutoff = cutoff
        pc.deadzone_px = float(min(max(round(jitter_px * 0.15, 1), 1.0), 4.0))
        self.ctx.changed("pointer")
        self.noise_res.setText(f"Measured raw hand/tracking motion ≈ {jitter_px:.1f} px on screen "
                               f"(std, {len(pts)} samples). Set One Euro min cutoff {cutoff} Hz and "
                               f"dead-zone {pc.deadzone_px} px.")

    def _finish_open(self) -> None:
        self._open_d = statistics.median(self.rec.pinch) if len(self.rec.pinch) > 10 else None
        self._apply_pinch()

    def _finish_pinched(self) -> None:
        self._pinch_d = statistics.median(self.rec.pinch) if len(self.rec.pinch) > 10 else None
        self._apply_pinch()

    def _apply_pinch(self) -> None:
        o, p = self._open_d, self._pinch_d
        txt = f"Open: {o:.2f}  " if o else "Open: not measured  "
        txt += f"Pinched: {p:.2f}" if p else "Pinched: not measured"
        if o and p and o > p + 0.1:
            engage = round(p + 0.35 * (o - p), 2)
            release = round(p + 0.55 * (o - p), 2)
            st = self.ctx.runtime.status
            fc = self.ctx.s.finger
            name = st.finger.lower() if st.finger in TIP else fc.finger
            if name == "thumb":
                fc.thumb_press = [engage, release]
            else:
                fc.pinch_thresholds[name] = [engage, release]
            self.ctx.changed("finger")
            txt += f"  →  {name}: engage below {engage}, release above {release}"
        elif o and p:
            txt += "  →  the two poses were too similar; try a firmer pinch."
        self.pinch_res.setText(txt)

    def _save_profile(self) -> None:
        name, ok = QInputDialog.getText(self, "Save profile", "Profile name:", text=self.ctx.s.profile)
        if ok and name.strip():
            save_profile(name.strip(), self.ctx.s)
            self.ctx.s.profile = name.strip()
            self.ctx.changed("profile")
            self.prof_lbl.setText(f"Saved to {profiles_dir()}")


class CalibrationPage:
    def __init__(self, ctx: "AppContext") -> None:
        self.ctx = ctx
        self.widget, lay = page("Calibration", "Personalise AirDesk for your camera position, hand and finger. "
                                "Run the full wizard once; use quick recalibration after moving the camera.")
        row = QHBoxLayout()
        full = QPushButton("Start calibration wizard")
        full.setObjectName("Primary")
        full.clicked.connect(lambda: Wizard(ctx).exec())
        quick = QPushButton("Quick recalibration (zone only)")
        quick.clicked.connect(lambda: Wizard(ctx, only_zone=True).exec())
        row.addWidget(full)
        row.addWidget(quick)
        row.addStretch(1)
        lay.addLayout(row)
        g, f = group("Profiles", lay)
        self.list = QListWidget()
        self.list.setMinimumHeight(160)
        f.addRow(self.list)
        r = QHBoxLayout()
        for text, fn in (("Load", self._load), ("Save current as…", self._save), ("Delete", self._delete),
                         ("Reset calibration to defaults", self._reset)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            r.addWidget(b)
        r.addStretch(1)
        f.addRow(r)
        self.cur = label("", "Muted")
        f.addRow(self.cur)
        g, f = group("Current calibration", lay)
        self.summary = label("", wrap=True)
        f.addRow(self.summary)
        lay.addStretch(1)
        self.refresh()

    def refresh(self) -> None:
        self.list.clear()
        self.list.addItems(list_profiles())
        s = self.ctx.s
        self.cur.setText(f"Active profile: {s.profile}")
        pc, fc = s.pointer, s.finger
        self.summary.setText(
            f"Direct-mode zone: centre ({pc.box_cx:.2f}, {pc.box_cy:.2f}), size {pc.box_w:.2f} × {pc.box_h:.2f}\n"
            f"Smoothing: {pc.filter}, min cutoff {pc.min_cutoff} Hz, beta {pc.beta}; dead-zone {pc.deadzone_px} px\n"
            f"Pinch thresholds: " + ", ".join(f"{k} {v[0]:.2f}/{v[1]:.2f}" for k, v in fc.pinch_thresholds.items())
            + f", thumb press {fc.thumb_press[0]:.2f}/{fc.thumb_press[1]:.2f}\n"
            f"Hand {fc.hand}, finger mode {fc.mode}, preferred finger {fc.finger}")

    def on_tick(self, st, m) -> None:
        pass

    def _sel(self) -> str | None:
        it = self.list.currentItem()
        return it.text() if it else None

    def _load(self) -> None:
        n = self._sel()
        if n and load_profile(n, self.ctx.s):
            self.ctx.runtime.camera.cfg = self.ctx.s.camera
            self.ctx.changed("camera")
            self.ctx.window.refresh_pages()

    def _save(self) -> None:
        name, ok = QInputDialog.getText(self.widget, "Save profile", "Profile name:", text=self.ctx.s.profile)
        if ok and name.strip():
            save_profile(name.strip(), self.ctx.s)
            self.ctx.s.profile = name.strip()
            self.ctx.changed("profile")
            self.refresh()

    def _delete(self) -> None:
        n = self._sel()
        if n and QMessageBox.question(self.widget, "Delete profile", f"Delete profile '{n}'?") == QMessageBox.Yes:
            (profiles_dir() / f"{n}.json").unlink(missing_ok=True)
            self.refresh()

    def _reset(self) -> None:
        if QMessageBox.question(self.widget, "Reset calibration",
                                "Reset camera, pointer, finger and gesture calibration to defaults?") != QMessageBox.Yes:
            return
        fresh = Settings()
        self.ctx.s.apply_profile(fresh.export_profile())
        self.ctx.s.profile = "Default"
        self.ctx.changed("camera")
        self.ctx.window.refresh_pages()
