r"""Optional "start with Windows" via HKCU\...\Run (per user, no elevation).

Only ever changed from the Settings page after an explicit confirmation.
The registered command starts AirDesk minimized and DISARMED.
"""

from __future__ import annotations

import sys

try:
    import winreg
except ImportError:  # pragma: no cover
    winreg = None

KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
NAME = "AirDesk"


def command() -> str | None:
    if not getattr(sys, "frozen", False):
        return None          # only the packaged executable is registered
    return f'"{sys.executable}" --minimized'


def is_enabled() -> bool:
    if winreg is None:
        return False
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, KEY) as k:
            winreg.QueryValueEx(k, NAME)
            return True
    except OSError:
        return False


def enable() -> bool:
    cmd = command()
    if winreg is None or cmd is None:
        return False
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, KEY, 0, winreg.KEY_SET_VALUE) as k:
        winreg.SetValueEx(k, NAME, 0, winreg.REG_SZ, cmd)
    return True


def disable() -> None:
    if winreg is None:
        return
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, KEY, 0, winreg.KEY_SET_VALUE) as k:
            winreg.DeleteValue(k, NAME)
    except OSError:
        pass
