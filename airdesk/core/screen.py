"""Multi-monitor desktop geometry in physical pixels (pure, testable)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Rect:
    left: int
    top: int
    right: int    # exclusive
    bottom: int   # exclusive

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x < self.right and self.top <= y < self.bottom

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        return (min(max(x, self.left), self.right - 1),
                min(max(y, self.top), self.bottom - 1))

    def dist2(self, x: float, y: float) -> float:
        cx, cy = self.clamp(x, y)
        return (cx - x) ** 2 + (cy - y) ** 2


@dataclass(frozen=True, slots=True)
class ScreenLayout:
    monitors: tuple[Rect, ...]
    primary: int = 0

    @property
    def virtual(self) -> Rect:
        return Rect(min(m.left for m in self.monitors), min(m.top for m in self.monitors),
                    max(m.right for m in self.monitors), max(m.bottom for m in self.monitors))

    def target_rect(self, which: str) -> Rect:
        """Rectangle the direct-mode control box maps onto."""
        if which == "primary":
            return self.monitors[self.primary]
        if which.isdigit() and int(which) < len(self.monitors):
            return self.monitors[int(which)]
        return self.virtual

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        """Clamp to the nearest real monitor so the cursor never targets
        the dead space of a non-rectangular multi-monitor desktop."""
        for m in self.monitors:
            if m.contains(x, y):
                return x, y
        best = min(self.monitors, key=lambda m: m.dist2(x, y))
        return best.clamp(x, y)

    @classmethod
    def single(cls, width: int = 1920, height: int = 1080) -> "ScreenLayout":
        return cls((Rect(0, 0, width, height),))
