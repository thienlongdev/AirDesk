"""Interaction log: one record per accepted press, correlated end to end.

Every mouse-button interaction the engine starts gets an id. The record
follows it through the pipeline - threshold crossing, confirmation,
button-down submission (SendInput result), button-up submission - and
ends in exactly one outcome:

    completed   down and up both submitted (a click, or a finished drag)
    rejected    Windows refused the input, or it was blocked (reason)
    cancelled   no complete click happened (reason: tracking lost, ...)

The on-screen feedback and the Diagnostics page read this log, so what
they show is what was actually submitted, not what the recognizer meant.
Only timings, buttons and screen positions are stored - never typed text.

Thread-safe: written by the vision thread (engine), read by the GUI.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Interaction:
    id: int
    button: str
    kind: str                          # click | drag | dwell | long_press
    pos: tuple[int, int]
    slot: str = ""
    t_cross: float | None = None       # pinch threshold crossing (frame clock)
    t_confirm: float = 0.0             # press accepted (frame clock)
    wall_confirm: float = 0.0          # perf_counter when the engine accepted it
    wall_down: float | None = None     # perf_counter after SendInput(down) returned
    wall_up: float | None = None
    down_ok: bool | None = None
    up_ok: bool | None = None
    phase: str = "accepted"            # accepted | pressed | dragging | completed | rejected | cancelled
    reason: str = ""
    wall_end: float = 0.0
    notes: list[str] = field(default_factory=list)
    evidence: str = ""                 # threshold | settled | deep | tap (how the pinch was confirmed)
    mode: str = ""                     # click interaction mode: instant | drag | classic
    clock: str = ""                    # local time of acceptance (HH:MM:SS)

    @property
    def done(self) -> bool:
        return self.phase in ("completed", "rejected", "cancelled")

    def confirm_to_down_ms(self) -> float | None:
        return None if self.wall_down is None else (self.wall_down - self.wall_confirm) * 1000

    def recognition_ms(self) -> float | None:
        return None if self.t_cross is None else (self.t_confirm - self.t_cross) * 1000


class ClickLog:
    def __init__(self, maxlen: int = 64) -> None:
        self._lock = threading.Lock()
        self._items: deque[Interaction] = deque(maxlen=maxlen)
        self._next = 1
        self.counters = {"accepted": 0, "down_submitted": 0, "up_submitted": 0, "completed": 0,
                         "rejected": 0, "cancelled": 0, "drags": 0, "aim_released": 0}
        # Pinch candidates that ended without a press (Aim Lock released,
        # ambiguous contact, ...): small dicts, metadata only.
        self._events: deque[dict] = deque(maxlen=maxlen)
        self._ev_next = 1

    def begin(self, button: str, kind: str, pos, t: float, slot: str = "",
              t_cross: float | None = None) -> Interaction:
        with self._lock:
            it = Interaction(self._next, button, kind, (int(round(pos[0])), int(round(pos[1]))), slot,
                             t_cross, t, time.perf_counter())
            it.clock = time.strftime("%H:%M:%S")
            self._next += 1
            self._items.append(it)
            self.counters["accepted"] += 1
            return it

    def submitted(self, it: Interaction | None, down: bool, ok: bool, reason: str = "",
                  t_submit: float | None = None) -> None:
        """`t_submit`: perf_counter right before SendInput (the moment the
        input was handed to Windows); defaults to now."""
        if it is None:
            return
        with self._lock:
            now = t_submit or time.perf_counter()
            if down:
                it.wall_down, it.down_ok = now, ok
                if ok:
                    self.counters["down_submitted"] += 1
                    if "blocked by windows" in reason.lower():
                        self._finish(it, "rejected", reason)    # submitted, but dropped by UIPI
                    elif it.phase == "accepted":
                        it.phase = "pressed"
                else:
                    self._finish(it, "rejected", reason or "button-down not accepted by Windows")
            else:
                it.wall_up, it.up_ok = now, ok
                if ok:
                    self.counters["up_submitted"] += 1
                    if it.down_ok and not it.done:
                        self._finish(it, "completed", reason)
                else:
                    self._finish(it, "cancelled" if it.down_ok else "rejected",
                                 reason or "button-up not accepted")
            if reason and reason not in it.notes:
                it.notes.append(reason)

    def dragging(self, it: Interaction | None) -> None:
        if it is None:
            return
        with self._lock:
            it.kind, it.phase = "drag", "dragging"
            self.counters["drags"] += 1

    def end(self, it: Interaction | None, phase: str, reason: str = "") -> None:
        """Close an interaction that did not finish through `submitted`."""
        if it is None or it.done:
            return
        with self._lock:
            self._finish(it, phase, reason)

    def _finish(self, it: Interaction, phase: str, reason: str) -> None:
        if it.done:
            return
        it.phase, it.reason, it.wall_end = phase, reason, time.perf_counter()
        self.counters[phase] += 1
        if phase == "completed":
            log.debug("interaction #%d %s %s completed at %s (confirm->down %.1f ms)", it.id, it.kind,
                      it.button, it.pos, it.confirm_to_down_ms() or -1)
        else:
            log.info("interaction #%d %s %s %s: %s", it.id, it.kind, it.button, phase, reason or "-")

    def note(self, kind: str, text: str, **data) -> None:
        """A gesture event that is not a press (e.g. Aim Lock released
        without a click, and why). Numbers and labels only."""
        with self._lock:
            ev = {"id": f"c{self._ev_next}", "wall": time.perf_counter(), "time": time.strftime("%H:%M:%S"),
                  "kind": kind, "text": text, **data}
            self._ev_next += 1
            self._events.append(ev)
            if kind == "aim_released":
                self.counters["aim_released"] += 1
        log.info("gesture %s %s: %s", ev["id"], kind, text)

    def events(self, n: int = 10) -> list[dict]:
        with self._lock:
            return [dict(e) for e in list(self._events)[-n:]]

    # ----------------------------------------------------------- readers
    def latest(self) -> Interaction | None:
        with self._lock:
            return replace(self._items[-1], notes=list(self._items[-1].notes)) if self._items else None

    def recent(self, n: int = 10) -> list[Interaction]:
        with self._lock:
            return [replace(i, notes=list(i.notes)) for i in list(self._items)[-n:]]

    def snapshot_counters(self) -> dict:
        with self._lock:
            return dict(self.counters)
