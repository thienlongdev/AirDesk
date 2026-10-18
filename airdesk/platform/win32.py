"""Thin, typed ctypes bindings for the documented Win32 APIs AirDesk uses.

* SendInput for real mouse and keyboard events (no pynput/pyautogui).
* Monitor enumeration in physical pixels (process is per-monitor DPI aware).
* Foreground window, process elevation and password-field checks.
* WS_EX_NOACTIVATE overlay helpers.

Lessons taken from Alpha-OSK (MIT, owenpkent/alpha-osk @ eb12523): the
INPUT union must include MOUSEINPUT or sizeof(INPUT) is wrong and
SendInput silently rejects every call; KEYEVENTF_UNICODE is layout
independent; WS_EX_NOACTIVATE must be (re)applied after Qt creates the HWND.

Every event AirDesk injects carries `AIRDESK_SIGNATURE` in dwExtraInfo.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes

from ..core.screen import Rect, ScreenLayout

IS_WINDOWS = sys.platform == "win32"
AIRDESK_SIGNATURE = 0xA1DE5C

if IS_WINDOWS:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
else:  # pragma: no cover - lets the pure modules import on other OSes
    user32 = kernel32 = advapi32 = None

ULONG_PTR = ctypes.c_size_t

INPUT_MOUSE, INPUT_KEYBOARD = 0, 1
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP = 0x0002, 0x0004
MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP = 0x0008, 0x0010
MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP = 0x0020, 0x0040
MOUSEEVENTF_XDOWN, MOUSEEVENTF_XUP = 0x0080, 0x0100
MOUSEEVENTF_WHEEL, MOUSEEVENTF_HWHEEL = 0x0800, 0x1000
MOUSEEVENTF_VIRTUALDESK = 0x4000
MOUSEEVENTF_ABSOLUTE = 0x8000
KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP = 0x0001, 0x0002
KEYEVENTF_UNICODE, KEYEVENTF_SCANCODE = 0x0004, 0x0008
WHEEL_DELTA = 120

BUTTON_FLAGS = {
    "left": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
    "right": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
    "middle": (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
}

# Virtual-key codes that need KEYEVENTF_EXTENDEDKEY.
EXTENDED_VK = {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E,
               0xA3, 0xA5, 0x5B, 0x5C, 0x5D, 0x6F, 0x90, 0x2C}


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


if IS_WINDOWS:
    user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
    user32.SendInput.restype = wintypes.UINT
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetWindowLongW.restype = ctypes.c_long
    user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
    user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
    user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
    user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, wintypes.UINT]
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.WindowFromPoint.argtypes = [wintypes.POINT]
    user32.WindowFromPoint.restype = wintypes.HWND
    user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    user32.GetAncestor.restype = wintypes.HWND
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
    user32.VkKeyScanW.argtypes = [wintypes.WCHAR]
    user32.VkKeyScanW.restype = ctypes.c_short
    user32.GetKeyState.argtypes = [ctypes.c_int]
    user32.GetKeyState.restype = ctypes.c_short
    user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
    user32.GetAsyncKeyState.restype = ctypes.c_short
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                     wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    advapi32.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                             wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]


def _send(inputs: list[INPUT]) -> int:
    if not inputs:
        return 0
    arr = (INPUT * len(inputs))(*inputs)
    return int(user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT)))


# ------------------------------------------------------------------ mouse
def _mouse(flags: int, dx: int = 0, dy: int = 0, data: int = 0) -> INPUT:
    inp = INPUT(type=INPUT_MOUSE)
    inp.mi = MOUSEINPUT(dx, dy, data & 0xFFFFFFFF, flags, 0, AIRDESK_SIGNATURE)
    return inp


def virtual_screen() -> Rect:
    x, y = user32.GetSystemMetrics(76), user32.GetSystemMetrics(77)
    w, h = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79)
    return Rect(x, y, x + w, y + h)


def buttons_swapped() -> bool:
    return bool(user32.GetSystemMetrics(23))  # SM_SWAPBUTTON


def mouse_move_abs(x: float, y: float, vs: Rect | None = None) -> bool:
    """Move the real cursor to physical pixel (x, y) on the virtual desktop."""
    vs = vs or virtual_screen()
    nx = int(round((x - vs.left) * 65535 / max(vs.width - 1, 1)))
    ny = int(round((y - vs.top) * 65535 / max(vs.height - 1, 1)))
    flags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
    return _send([_mouse(flags, nx, ny)]) == 1


def _physical_button(button: str, swapped: bool | None = None) -> str:
    """SendInput talks in physical buttons; honour the user's swap setting
    so a logical 'left' click is still the primary action. Pass `swapped`
    (cached by the dispatcher) to avoid an extra API call - and an extra
    GIL release - right before a button event."""
    if button in ("left", "right") and (buttons_swapped() if swapped is None else swapped):
        return "right" if button == "left" else "left"
    return button


def mouse_button(button: str, down: bool, swapped: bool | None = None) -> bool:
    if button.startswith("x"):
        which = int(button[1:])
        return _send([_mouse(MOUSEEVENTF_XDOWN if down else MOUSEEVENTF_XUP, data=which)]) == 1
    f_down, f_up = BUTTON_FLAGS[_physical_button(button, swapped)]
    return _send([_mouse(f_down if down else f_up)]) == 1


def mouse_click(button: str, count: int = 1, swapped: bool | None = None) -> bool:
    f_down, f_up = BUTTON_FLAGS[_physical_button(button, swapped)]
    seq: list[INPUT] = []
    for _ in range(count):
        seq += [_mouse(f_down), _mouse(f_up)]
    return _send(seq) == len(seq)


def mouse_wheel(delta: int, horizontal: bool = False) -> bool:
    return _send([_mouse(MOUSEEVENTF_HWHEEL if horizontal else MOUSEEVENTF_WHEEL, data=delta)]) == 1


def cursor_pos() -> tuple[int, int]:
    pt = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def window_at(x: int, y: int) -> int:
    """Top-level window under a screen point (0 if none)."""
    h = user32.WindowFromPoint(wintypes.POINT(int(x), int(y)))
    if not h:
        return 0
    return int(user32.GetAncestor(h, 2) or h)   # GA_ROOT


def double_click_time_s() -> float:
    return user32.GetDoubleClickTime() / 1000.0


# --------------------------------------------------------------- keyboard
def _key(vk: int, up: bool, scan: int | None = None) -> INPUT:
    inp = INPUT(type=INPUT_KEYBOARD)
    flags = KEYEVENTF_KEYUP if up else 0
    if vk in EXTENDED_VK:
        flags |= KEYEVENTF_EXTENDEDKEY
    sc = user32.MapVirtualKeyW(vk, 0) if scan is None else scan
    inp.ki = KEYBDINPUT(vk, sc, flags, 0, AIRDESK_SIGNATURE)
    return inp


def key_event(vk: int, down: bool) -> bool:
    return _send([_key(vk, not down)]) == 1


def key_tap(vk: int, modifiers: tuple[int, ...] = ()) -> bool:
    """One key with modifiers as a single ordered SendInput batch:
    modifiers down, key down, key up, modifiers up (Ctrl+C = Ctrl down,
    C down, C up, Ctrl up). SendInput inserts a batch atomically - no other
    input can land in the middle of the chord.

    A modifier the user is physically holding is left alone (its up would
    otherwise cancel the real key). If Windows inserts only part of the
    batch, every key this call pressed is released again, so nothing is
    ever left down."""
    mods = tuple(m for m in modifiers if not key_is_down(m))
    seq = [_key(m, False) for m in mods] + [_key(vk, False), _key(vk, True)]
    seq += [_key(m, True) for m in reversed(mods)]
    n = _send(seq)
    if n != len(seq):
        if n > 0:
            _send([_key(vk, True)] + [_key(m, True) for m in reversed(mods)])
        return False
    return True


def type_unicode(text: str) -> bool:
    """Insert text with KEYEVENTF_UNICODE (layout independent, bypasses IMEs)."""
    seq: list[INPUT] = []
    for ch in text:
        data = ch.encode("utf-16-le")
        for i in range(0, len(data), 2):
            unit = int.from_bytes(data[i:i + 2], "little")
            for up in (False, True):
                inp = INPUT(type=INPUT_KEYBOARD)
                inp.ki = KEYBDINPUT(0, unit, KEYEVENTF_UNICODE | (KEYEVENTF_KEYUP if up else 0),
                                    0, AIRDESK_SIGNATURE)
                seq.append(inp)
    return _send(seq) == len(seq)


def vk_for_char(ch: str) -> tuple[int, bool] | None:
    """(vk, needs_shift) for a character on the active layout, or None."""
    r = user32.VkKeyScanW(ch)
    if r == -1:
        return None
    vk, shift_state = r & 0xFF, (r >> 8) & 0xFF
    if shift_state & ~1:  # needs Ctrl/Alt (AltGr) -> use Unicode instead
        return None
    return vk, bool(shift_state & 1)


def key_is_down(vk: int) -> bool:
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


def caps_lock_on() -> bool:
    return bool(user32.GetKeyState(0x14) & 1)


# --------------------------------------------------------------- monitors
class MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]


def ensure_dpi_awareness() -> None:
    """Per-monitor-v2 DPI awareness for non-Qt entry points (Qt sets it itself)."""
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        pass


def screen_layout() -> ScreenLayout:
    mons: list[tuple[Rect, bool]] = []
    proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC,
                              ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)

    def cb(hmon, hdc, lprect, lparam):
        mi = MONITORINFO()
        mi.cbSize = ctypes.sizeof(MONITORINFO)
        user32.GetMonitorInfoW(hmon, ctypes.byref(mi))
        r = mi.rcMonitor
        mons.append((Rect(r.left, r.top, r.right, r.bottom), bool(mi.dwFlags & 1)))
        return True

    user32.EnumDisplayMonitors(None, None, proc(cb), 0)
    if not mons:
        return ScreenLayout.single()
    primary = next((i for i, (_, p) in enumerate(mons) if p), 0)
    return ScreenLayout(tuple(r for r, _ in mons), primary)


# ------------------------------------------------------------- foreground
def foreground_window() -> int:
    return int(user32.GetForegroundWindow() or 0)


def window_pid(hwnd: int) -> int:
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(pid.value)


def window_class(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def window_title(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(hwnd, buf, 512)
    return buf.value


def process_name(pid: int) -> str:
    h = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(1024)
        if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return os.path.basename(buf.value)
        return ""
    finally:
        kernel32.CloseHandle(h)


def process_elevated(pid: int) -> bool | None:
    """True if the process runs elevated (UIPI will block our input),
    None if it cannot be queried (treated as protected by callers)."""
    h = kernel32.OpenProcess(0x1000, False, pid)
    if not h:
        return None
    tok = wintypes.HANDLE()
    try:
        if not advapi32.OpenProcessToken(h, 0x0008, ctypes.byref(tok)):  # TOKEN_QUERY
            return None
        elev = wintypes.DWORD()
        ret = wintypes.DWORD()
        ok = advapi32.GetTokenInformation(tok, 20, ctypes.byref(elev), 4, ctypes.byref(ret))
        return bool(elev.value) if ok else None
    finally:
        if tok:
            kernel32.CloseHandle(tok)
        kernel32.CloseHandle(h)


def self_elevated() -> bool:
    return bool(process_elevated(os.getpid()))


class GUITHREADINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("flags", wintypes.DWORD),
                ("hwndActive", wintypes.HWND), ("hwndFocus", wintypes.HWND),
                ("hwndCapture", wintypes.HWND), ("hwndMenuOwner", wintypes.HWND),
                ("hwndMoveSize", wintypes.HWND), ("hwndCaret", wintypes.HWND),
                ("rcCaret", wintypes.RECT)]


def focused_control(hwnd: int) -> int:
    tid = user32.GetWindowThreadProcessId(hwnd, None)
    info = GUITHREADINFO()
    info.cbSize = ctypes.sizeof(GUITHREADINFO)
    if user32.GetGUIThreadInfo(tid, ctypes.byref(info)):
        return int(info.hwndFocus or 0)
    return 0


def is_password_field(hwnd: int) -> bool:
    """Classic Win32 edit controls with ES_PASSWORD. (Browser and UWP password
    boxes are not detectable this way; the privacy mode covers those.)"""
    focus = focused_control(hwnd)
    if not focus:
        return False
    if "edit" not in window_class(focus).lower():
        return False
    return bool(user32.GetWindowLongW(focus, -16) & 0x0020)


# ------------------------------------------------------------- overlay
GWL_EXSTYLE = -20
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
HWND_TOPMOST = -1
SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE, SWP_FRAMECHANGED = 0x1, 0x2, 0x10, 0x20


def make_noactivate(hwnd: int) -> bool:
    """Apply WS_EX_NOACTIVATE|WS_EX_TOOLWINDOW and topmost without activating."""
    style = user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE)
    want = style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
    if want != style:
        user32.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, want)
    return bool(user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                                    SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE | SWP_FRAMECHANGED))


def has_noactivate(hwnd: int) -> bool:
    return bool(user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE) & WS_EX_NOACTIVATE)


def window_rect(hwnd: int) -> Rect:
    r = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    return Rect(r.left, r.top, r.right, r.bottom)


def move_window_noactivate(hwnd: int, x: int, y: int, w: int | None = None, h: int | None = None) -> None:
    flags = SWP_NOACTIVATE | (SWP_NOSIZE if w is None else 0)
    user32.SetWindowPos(hwnd, HWND_TOPMOST, x, y, w or 0, h or 0, flags)
