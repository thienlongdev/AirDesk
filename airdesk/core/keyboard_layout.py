"""Air Keyboard layouts and hit testing (pure data, no Qt).

Keys are laid out in abstract units; `layout_rects` converts a layout to
fractional rectangles (0..1) of the keyboard surface. The overlay renders
those rectangles at any size, and the engine hit-tests the *same*
rectangles against the physical cursor position, so what you see is
exactly what you select at every DPI and size.
"""

from __future__ import annotations

from dataclasses import dataclass

SHIFTED = dict(zip("`1234567890-=[]\\;',./", "~!@#$%^&*()_+{}|:\"<>?"))

LABELS = {
    "backspace": "⌫", "tab": "Tab", "enter": "Enter", "shift": "Shift", "caps": "Caps",
    "ctrl": "Ctrl", "alt": "Alt", "win": "Win", "space": "", "left": "←", "right": "→",
    "up": "↑", "down": "↓", "home": "Home", "end": "End", "pgup": "PgUp", "pgdn": "PgDn",
    "del": "Del", "esc": "Esc", "lang": "EN", "fn": "Fn", "sym": "?123",
    "abc": "ABC", "copy": "Copy", "paste": "Paste", "cut": "Cut", "undo": "Undo",
    "selall": "Sel all", "zoomin": "Zoom+", "zoomout": "Zoom−", "move": "⠿ Air Keyboard",
    "size+": "+", "size-": "−", "layout": "Layout", "width-": "W−", "width+": "W+",
    "reset": "Reset", "hide": "×",
}
MODIFIERS = {"shift", "ctrl", "alt", "win"}
# Title bar: move handle (drag/pinch) · suggestions · language · layout ·
# − size% + · narrower/wider · reset · hide. "sizepct" is a read-out only.
TOOLBAR = ["move", "sug0", "sug1", "sug2", "lang", "layout", "size-", "sizepct", "size+",
           "width-", "width+", "reset", "hide"]
TOOLBAR_UNITS = {"move": 2.2, "sug0": 2.0, "sug1": 2.0, "sug2": 2.0}

# (key, width-in-units). Each row should sum to the same width.
FULL = [
    TOOLBAR,
    [("esc", 1), ("f1", 1), ("f2", 1), ("f3", 1), ("f4", 1), ("f5", 1), ("f6", 1), ("f7", 1),
     ("f8", 1), ("f9", 1), ("f10", 1), ("f11", 1), ("f12", 1), ("del", 2)],
    [("`", 1)] + [(c, 1) for c in "1234567890-="] + [("backspace", 2)],
    [("tab", 1.5)] + [(c, 1) for c in "qwertyuiop[]"] + [("\\", 1.5)],
    [("caps", 1.75)] + [(c, 1) for c in "asdfghjkl;'"] + [("enter", 2.25)],
    [("shift", 2.25)] + [(c, 1) for c in "zxcvbnm,./"] + [("up", 1), ("shift", 1.75)],
    [("ctrl", 1.5), ("win", 1.25), ("alt", 1.25), ("space", 5.0), ("home", 1), ("end", 1),
     ("undo", 1), ("copy", 1), ("paste", 1), ("left", 1), ("down", 1), ("right", 1)],
]
COMPACT = [
    TOOLBAR,
    [(c, 1) for c in "qwertyuiop"] + [("backspace", 1.5)],
    [("caps", 1)] + [(c, 1) for c in "asdfghjkl"] + [("enter", 1.5)],
    [("shift", 1.5)] + [(c, 1) for c in "zxcvbnm"] + [(",", 1), (".", 1), ("shift", 1)],
    # v1.1.3: Esc, Tab, Ctrl and Alt on the compact layout too (shortcuts).
    [("sym", 1.25), ("esc", 1), ("tab", 1), ("ctrl", 1), ("alt", 1), ("space", 3.25), ("left", 1),
     ("right", 1), ("del", 1)],
]
SYMBOLS = [
    TOOLBAR,
    [(c, 1) for c in "1234567890"] + [("backspace", 1.5)],
    [(c, 1) for c in "@#$%&*-+=/"] + [("enter", 1.5)],
    [(c, 1) for c in "()[]{}<>!?"] + [("tab", 1.5)],
    [("abc", 1.5), (";", 1), (":", 1), ("'", 1), ("\"", 1), ("space", 3.0), ("_", 1), ("\\", 1),
     ("up", 1)],
]
LAYOUTS = {"full": FULL, "compact": COMPACT, "symbols": SYMBOLS}


@dataclass(frozen=True, slots=True)
class KeyRect:
    key: str
    x: float
    y: float
    w: float
    h: float
    index: int        # disambiguates duplicate keys (two Shift keys)

    @property
    def id(self) -> str:
        return f"{self.key}#{self.index}"

    def contains(self, fx: float, fy: float) -> bool:
        return self.x <= fx < self.x + self.w and self.y <= fy < self.y + self.h


def layout_rects(name: str, gap: float = 0.006) -> list[KeyRect]:
    rows = LAYOUTS[name]
    out: list[KeyRect] = []
    toolbar_h = 0.7  # toolbar row is shorter than key rows
    heights = [toolbar_h] + [1.0] * (len(rows) - 1)
    total_h = sum(heights)
    y = 0.0
    idx = 0
    for r, row in enumerate(rows):
        h = heights[r] / total_h
        items = [(k, 1.0) for k in row] if r == 0 else row
        if r == 0:
            items = [(k, TOOLBAR_UNITS.get(k, 1.0)) for k in row]
        units = sum(w for _, w in items)
        x = 0.0
        for key, w in items:
            fw = w / units
            out.append(KeyRect(key, x + gap / 2, y + gap / 2, fw - gap, h - gap, idx))
            idx += 1
            x += fw
        y += h
    return out


# ------------------------------------------------------------------ sizing
SIZE_RANGE = (40, 130)      # % (overall size, both dimensions)
WIDTH_RANGE = (60, 150)     # % (extra width factor)
SIZE_DEFAULT, WIDTH_DEFAULT = 70, 100
# At SIZE_DEFAULT the full layout is 65 % of the work-area width and 27 % of
# its height (v1.0: 56 % x 39 %); compact/symbols are narrower and have fewer rows.
_BASE = {"full": (0.65, 0.27), "compact": (0.47, 0.20), "symbols": (0.47, 0.20)}
MIN_KEY_W, MIN_KEY_H = 30, 26        # logical px: keys stay readable and pinchable


def _row_units(name: str) -> tuple[float, float]:
    """(widest key row in units, total height in row units) of a layout."""
    rows = LAYOUTS[name]
    return max(sum(w for _, w in r) for r in rows[1:]), 0.7 + (len(rows) - 1)


def keyboard_size(work_w: int, work_h: int, layout: str, size_pct: int = SIZE_DEFAULT,
                  width_pct: int = WIDTH_DEFAULT) -> tuple[int, int]:
    """Window size (logical px) for a work area. Size scales both dimensions;
    width scales only the width (keys get wider or narrower, the layout
    redistributes because every key rectangle is fractional)."""
    size = min(max(size_pct, SIZE_RANGE[0]), SIZE_RANGE[1]) / SIZE_DEFAULT
    wide = min(max(width_pct, WIDTH_RANGE[0]), WIDTH_RANGE[1]) / 100.0
    bw, bh = _BASE.get(layout, _BASE["full"])
    units_w, units_h = _row_units(layout)
    w = work_w * bw * size * wide
    h = work_h * bh * size
    w = min(max(w, units_w * MIN_KEY_W), work_w)
    h = min(max(h, units_h * MIN_KEY_H), work_h * 0.6)
    return int(round(w)), int(round(h))


def clamp_rect(x: int, y: int, w: int, h: int, area: tuple[int, int, int, int]) -> tuple[int, int]:
    """Top-left that keeps a w x h window fully inside area (x, y, w, h)."""
    ax, ay, aw, ah = area
    return min(max(x, ax), ax + max(aw - w, 0)), min(max(y, ay), ay + max(ah - h, 0))


def hit(rects: list[KeyRect], fx: float, fy: float) -> KeyRect | None:
    for r in rects:
        if r.contains(fx, fy):
            return r
    return None


def hit_near(rects: list[KeyRect], fx: float, fy: float, tol: float = 0.006) -> KeyRect | None:
    """`hit`, plus the nearest key when the point falls in the thin gap
    between keys (virtual hand cursors: a pinch on a key edge is that key)."""
    k = hit(rects, fx, fy)
    if k is not None or not (-tol <= fx <= 1 + tol and -tol <= fy <= 1 + tol):
        return k
    best, bd = None, tol
    for r in rects:
        dx = max(r.x - fx, 0.0, fx - (r.x + r.w))
        dy = max(r.y - fy, 0.0, fy - (r.y + r.h))
        d = max(dx, dy * 0.5)
        if d <= bd:
            best, bd = r, d
    return best


def key_label(key: str, shift: bool, caps: bool, lang: str = "en") -> str:
    if key == "lang":
        return {"en": "EN", "vi": "VI", "raw": "IME"}[lang]
    if key in LABELS:
        return LABELS[key]
    if key.startswith("f") and key[1:].isdigit():
        return key.upper()
    if len(key) == 1:
        if key.isalpha():
            return key.upper() if (shift ^ caps) else key
        return SHIFTED.get(key, key) if shift else key
    return key


def char_for(key: str, shift: bool, caps: bool) -> str | None:
    if len(key) != 1:
        return None
    if key.isalpha():
        return key.upper() if (shift ^ caps) else key
    return SHIFTED.get(key, key) if shift else key
