"""Reusable widgets: stat cards, settings binder, live preview, bars, sparkline."""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Iterable

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QGroupBox,
                               QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QSlider, QSpinBox,
                               QVBoxLayout, QWidget)

from . import theme

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
                    (17, 18), (18, 19), (19, 20)]
FINGER_TIPS = {"Thumb": 4, "Index": 8, "Middle": 12, "Ring": 16, "Little": 20}
FINGER_MCP = {"Thumb": 2, "Index": 5, "Middle": 9, "Ring": 13, "Little": 17}


def label(text: str, obj: str | None = None, wrap: bool = False) -> QLabel:
    lb = QLabel(text)
    if obj:
        lb.setObjectName(obj)
    lb.setWordWrap(wrap)
    return lb


def page(title: str, subtitle: str = "") -> tuple[QScrollArea, QVBoxLayout]:
    """Scrollable page with a heading; returns (widget, content layout)."""
    inner = QWidget()
    inner.setObjectName("Page")
    lay = QVBoxLayout(inner)
    lay.setContentsMargins(28, 22, 28, 22)
    lay.setSpacing(12)
    lay.addWidget(label(title, "H1"))
    if subtitle:
        lay.addWidget(label(subtitle, "Muted", wrap=True))
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setWidget(inner)
    sa.setFrameShape(QFrame.NoFrame)
    return sa, lay


class Card(QFrame):
    def __init__(self, title: str, value: str = "-") -> None:
        super().__init__()
        self.setObjectName("Card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)
        self.title = label(title, "CardTitle")
        self.value = label(value, "CardValue")
        lay.addWidget(self.title)
        lay.addWidget(self.value)
        self.setMinimumWidth(150)

    def set(self, text: str, color: str | None = None) -> None:
        if self.value.text() != text:
            self.value.setText(text)
        style = f"color: {theme.current[color]};" if color else ""
        if self.value.styleSheet() != style:
            self.value.setStyleSheet(style)


class Binder:
    """Builds form rows bound to dataclass fields; calls `changed(section)`
    after every edit so settings are saved and pushed to the runtime."""

    def __init__(self, changed: Callable[[str], None]) -> None:
        self.changed = changed
        self._refreshers: list[Callable[[], None]] = []

    def refresh(self) -> None:
        for r in self._refreshers:
            r()

    def _set(self, obj: Any, name: str, value: Any, section: str) -> None:
        if getattr(obj, name) != value:
            setattr(obj, name, value)
            self.changed(section)

    def combo(self, form: QFormLayout, text: str, obj: Any, name: str,
              options: Iterable[tuple[Any, str]], section: str = "", tip: str = "") -> QComboBox:
        cb = QComboBox()
        opts = list(options)
        for v, lab in opts:
            cb.addItem(lab, v)

        def load() -> None:
            i = cb.findData(getattr(obj, name))
            cb.blockSignals(True)
            cb.setCurrentIndex(max(i, 0))
            cb.blockSignals(False)
        load()
        self._refreshers.append(load)
        cb.currentIndexChanged.connect(lambda _: self._set(obj, name, cb.currentData(), section))
        cb.setToolTip(tip)
        form.addRow(text, cb)
        return cb

    def check(self, form: QFormLayout, text: str, obj: Any, name: str, section: str = "",
              tip: str = "") -> QCheckBox:
        c = QCheckBox(text)

        def load() -> None:
            c.blockSignals(True)
            c.setChecked(bool(getattr(obj, name)))
            c.blockSignals(False)
        load()
        self._refreshers.append(load)
        c.toggled.connect(lambda v: self._set(obj, name, bool(v), section))
        c.setToolTip(tip)
        form.addRow("", c)
        return c

    def number(self, form: QFormLayout, text: str, obj: Any, name: str, lo: float, hi: float,
               step: float = 1, decimals: int = 0, suffix: str = "", section: str = "",
               tip: str = "", slider: bool = True) -> QWidget:
        is_int = isinstance(getattr(obj, name), int) and decimals == 0
        spin = QSpinBox() if is_int else QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        if not is_int:
            spin.setDecimals(decimals)
        if suffix:
            spin.setSuffix(" " + suffix)
        spin.setFixedWidth(110)
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        sl = None
        scale = 10 ** decimals
        if slider:
            sl = QSlider(Qt.Horizontal)
            sl.setRange(int(lo * scale), int(hi * scale))
            h.addWidget(sl, 1)
        h.addWidget(spin)

        def load() -> None:
            v = getattr(obj, name)
            spin.blockSignals(True)
            spin.setValue(v)
            spin.blockSignals(False)
            if sl:
                sl.blockSignals(True)
                sl.setValue(int(round(v * scale)))
                sl.blockSignals(False)
        load()
        self._refreshers.append(load)

        def from_spin(v) -> None:
            if sl:
                sl.blockSignals(True)
                sl.setValue(int(round(v * scale)))
                sl.blockSignals(False)
            self._set(obj, name, int(v) if is_int else float(v), section)
        spin.valueChanged.connect(from_spin)
        if sl:
            sl.valueChanged.connect(lambda v: spin.setValue(v / scale if not is_int else v))
        row.setToolTip(tip)
        form.addRow(text, row)
        return row


def group(title: str, parent_layout: QVBoxLayout | None = None) -> tuple[QGroupBox, QFormLayout]:
    g = QGroupBox(title)
    f = QFormLayout(g)
    f.setLabelAlignment(Qt.AlignLeft)
    f.setHorizontalSpacing(18)
    f.setVerticalSpacing(8)
    if parent_layout is not None:
        parent_layout.addWidget(g)
    return g, f


class Bars(QWidget):
    """Horizontal labelled bars (finger extension / confidence)."""

    def __init__(self, names: list[str]) -> None:
        super().__init__()
        self.names = names
        self.values: dict[str, float] = {}
        self.highlight: str | None = None
        self.setMinimumHeight(24 * len(names) + 6)

    def set(self, values: dict[str, float], highlight: str | None = None) -> None:
        self.values, self.highlight = values, highlight
        self.update()

    def paintEvent(self, e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        for i, n in enumerate(self.names):
            y = 4 + i * 24
            v = max(0.0, min(1.0, float(self.values.get(n, 0.0))))
            hi = n == self.highlight
            p.setPen(theme.qcolor("text") if hi else theme.qcolor("muted"))
            f = QFont(self.font())
            f.setBold(hi)
            p.setFont(f)
            p.drawText(QRectF(0, y, 70, 18), Qt.AlignVCenter, n)
            track = QRectF(76, y + 6, w - 120, 7)
            p.setPen(Qt.NoPen)
            p.setBrush(theme.qcolor("border"))
            p.drawRoundedRect(track, 3, 3)
            p.setBrush(theme.qcolor("accent") if hi else theme.qcolor("muted"))
            p.drawRoundedRect(QRectF(track.x(), track.y(), track.width() * v, track.height()), 3, 3)
            p.setPen(theme.qcolor("muted"))
            p.drawText(QRectF(w - 40, y, 40, 18), Qt.AlignVCenter | Qt.AlignRight, f"{v:.2f}")
        p.end()


class Sparkline(QWidget):
    def __init__(self, title: str, unit: str, maxlen: int = 240, y_max: float | None = None) -> None:
        super().__init__()
        self.title, self.unit, self.y_max = title, unit, y_max
        self.data: deque[float] = deque(maxlen=maxlen)
        self.setMinimumHeight(90)

    def push(self, v: float) -> None:
        self.data.append(v)
        self.update()

    def paintEvent(self, e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        p.setPen(QPen(theme.qcolor("border"), 1))
        p.setBrush(theme.qcolor("panel"))
        p.drawRoundedRect(r, 8, 8)
        p.setPen(theme.qcolor("muted"))
        last = self.data[-1] if self.data else 0.0
        p.drawText(r.adjusted(10, 6, -10, 0), Qt.AlignLeft | Qt.AlignTop, self.title)
        p.setPen(theme.qcolor("text"))
        p.drawText(r.adjusted(10, 6, -10, 0), Qt.AlignRight | Qt.AlignTop, f"{last:.1f} {self.unit}")
        if len(self.data) > 1:
            top = self.y_max or max(max(self.data) * 1.2, 1e-6)
            plot = r.adjusted(10, 26, -10, -8)
            path = QPainterPath()
            n = self.data.maxlen or len(self.data)
            for i, v in enumerate(self.data):
                x = plot.left() + plot.width() * i / max(n - 1, 1)
                y = plot.bottom() - plot.height() * min(v / top, 1.0)
                path.moveTo(x, y) if i == 0 else path.lineTo(x, y)
            p.setPen(QPen(theme.qcolor("accent"), 1.6))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
        p.end()


class PreviewWidget(QWidget):
    """Webcam preview with landmarks, active hand, active fingertip, pointer
    anchor and the direct-mode control box."""

    clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._img: QImage | None = None
        self._hands = []
        self._active_wrist = None
        self._finger = None
        self._box = None
        self._blend = 0.25
        self._gesture = ""
        self._buf: np.ndarray | None = None

    def set_frame(self, preview, active_wrist, finger: str | None, box, blend: float, gesture: str) -> None:
        if preview is None:
            return
        img = preview.image
        self._buf = np.ascontiguousarray(img)
        h, w = self._buf.shape[:2]
        self._img = QImage(self._buf.data, w, h, 3 * w, QImage.Format_RGB888)
        self._hands = preview.hands
        self._active_wrist, self._finger, self._box = active_wrist, finger, box
        self._blend, self._gesture = blend, gesture
        self.update()

    def paintEvent(self, e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(12, 13, 16))
        if self._img is None:
            p.setPen(theme.qcolor("muted"))
            p.drawText(self.rect(), Qt.AlignCenter, "Waiting for camera…")
            p.end()
            return
        iw, ih = self._img.width(), self._img.height()
        s = min(self.width() / iw, self.height() / ih)
        dw, dh = iw * s, ih * s
        ox, oy = (self.width() - dw) / 2, (self.height() - dh) / 2
        p.drawImage(QRectF(ox, oy, dw, dh), self._img)

        def pt(x: float, y: float) -> QPointF:
            return QPointF(ox + x * dw, oy + y * dh)

        if self._box is not None:
            cx, cy, bw, bh = self._box
            p.setPen(QPen(QColor(255, 255, 255, 90), 1, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(pt(cx - bw / 2, cy - bh / 2), pt(cx + bw / 2, cy + bh / 2)))
        for hand in self._hands:
            lm = hand.image
            active = (self._active_wrist is not None
                      and abs(lm[0, 0] - self._active_wrist[0]) < 1e-6
                      and abs(lm[0, 1] - self._active_wrist[1]) < 1e-6)
            col = QColor(76, 154, 255) if active else QColor(170, 175, 185)
            p.setPen(QPen(col, 2.0 if active else 1.2))
            for a, b in HAND_CONNECTIONS:
                p.drawLine(pt(lm[a, 0], lm[a, 1]), pt(lm[b, 0], lm[b, 1]))
            p.setBrush(col)
            p.setPen(Qt.NoPen)
            for i in range(21):
                p.drawEllipse(pt(lm[i, 0], lm[i, 1]), 2.4, 2.4)
            if active and self._finger in FINGER_TIPS:
                tip, mcp = FINGER_TIPS[self._finger], FINGER_MCP[self._finger]
                p.setBrush(QColor(60, 220, 140))
                p.drawEllipse(pt(lm[tip, 0], lm[tip, 1]), 7, 7)
                a = self._blend
                ax = lm[tip, 0] * (1 - a) + lm[mcp, 0] * a
                ay = lm[tip, 1] * (1 - a) + lm[mcp, 1] * a
                p.setPen(QPen(QColor(255, 210, 80), 2))
                p.setBrush(Qt.NoBrush)
                c = pt(ax, ay)
                p.drawLine(c + QPointF(-8, 0), c + QPointF(8, 0))
                p.drawLine(c + QPointF(0, -8), c + QPointF(0, 8))
                p.setPen(QColor(255, 255, 255))
                p.drawText(pt(lm[0, 0], lm[0, 1]) + QPointF(-30, 22), f"{hand.handedness} · {self._finger}")
        if self._gesture and self._gesture != "-":
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 150))
            p.drawRoundedRect(QRectF(ox + 10, oy + 10, 190, 26), 6, 6)
            p.setPen(QColor(255, 255, 255))
            p.drawText(QRectF(ox + 18, oy + 10, 180, 26), Qt.AlignVCenter, self._gesture)
        p.end()
