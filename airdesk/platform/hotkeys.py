"""System-wide hotkeys via RegisterHotKey on a dedicated message-loop thread.

These work no matter which window has focus and do not depend on hand
tracking, so the emergency stop is always reachable with the physical
keyboard. No keyboard hook is installed and no input is suppressed.
"""

from __future__ import annotations

import ctypes
import logging
import threading
from ctypes import wintypes
from typing import Callable

from . import win32

log = logging.getLogger(__name__)

MODS = {"alt": 0x1, "ctrl": 0x2, "control": 0x2, "shift": 0x4, "win": 0x8}
MOD_NOREPEAT = 0x4000
WM_HOTKEY, WM_QUIT = 0x0312, 0x0012
NAMED_VK = {"space": 0x20, "enter": 0x0D, "esc": 0x1B, "escape": 0x1B, "tab": 0x09,
            "pause": 0x13, "end": 0x23, "home": 0x24, "insert": 0x2D, "delete": 0x2E,
            "backspace": 0x08, "scrolllock": 0x91}


def parse_hotkey(spec: str) -> tuple[int, int]:
    """'ctrl+alt+f12' -> (modifiers, vk). Raises ValueError if malformed."""
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    if not parts:
        raise ValueError("empty hotkey")
    mods = 0
    for p in parts[:-1]:
        if p not in MODS:
            raise ValueError(f"unknown modifier {p!r}")
        mods |= MODS[p]
    key = parts[-1]
    if key.startswith("f") and key[1:].isdigit() and 1 <= int(key[1:]) <= 24:
        vk = 0x6F + int(key[1:])
    elif key in NAMED_VK:
        vk = NAMED_VK[key]
    elif len(key) == 1 and key.isalnum():
        vk = ord(key.upper())
    else:
        raise ValueError(f"unknown key {key!r}")
    return mods, vk


class HotkeyService:
    def __init__(self) -> None:
        self._bindings: dict[int, tuple[str, Callable[[], None]]] = {}
        self._specs: list[tuple[int, str, Callable[[], None]]] = []
        self._thread: threading.Thread | None = None
        self._tid = 0
        self.failed: list[str] = []
        self._ready = threading.Event()

    def add(self, spec: str, callback: Callable[[], None]) -> None:
        self._specs.append((len(self._specs) + 1, spec, callback))

    def start(self) -> None:
        if not win32.IS_WINDOWS:
            return
        self._thread = threading.Thread(target=self._run, name="airdesk-hotkeys", daemon=True)
        self._thread.start()
        self._ready.wait(2.0)

    def stop(self) -> None:
        if self._tid:
            ctypes.windll.user32.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)
        if self._thread:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        user32 = ctypes.windll.user32
        self._tid = ctypes.windll.kernel32.GetCurrentThreadId()
        for hid, spec, cb in self._specs:
            try:
                mods, vk = parse_hotkey(spec)
            except ValueError as e:
                self.failed.append(f"{spec}: {e}")
                continue
            if user32.RegisterHotKey(None, hid, mods | MOD_NOREPEAT, vk):
                self._bindings[hid] = (spec, cb)
            else:
                self.failed.append(f"{spec}: already in use by another program")
                log.warning("hotkey %s could not be registered", spec)
        self._ready.set()
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY and msg.wParam in self._bindings:
                spec, cb = self._bindings[msg.wParam]
                try:
                    cb()
                except Exception:
                    log.exception("hotkey %s handler failed", spec)
        for hid in self._bindings:
            user32.UnregisterHotKey(None, hid)
