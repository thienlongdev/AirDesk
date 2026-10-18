"""Activate only this test process's window, including from a background shell.

Temporary input-queue attachment handles Windows foreground-lock policy;
it never changes AirDesk's production focus behavior or sends any input.
"""
import ctypes
from ctypes import wintypes
import os


def activate_own_window(hwnd: int = 0) -> bool:
    user = ctypes.WinDLL("user32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    user.GetForegroundWindow.restype = wintypes.HWND
    user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user.GetWindowThreadProcessId.restype = wintypes.DWORD
    user.IsWindowVisible.argtypes = [wintypes.HWND]
    user.SetForegroundWindow.argtypes = [wintypes.HWND]
    user.BringWindowToTop.argtypes = [wintypes.HWND]
    user.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]

    def owns(handle):
        pid = wintypes.DWORD()
        user.GetWindowThreadProcessId(handle, ctypes.byref(pid))
        return pid.value == os.getpid() and user.IsWindowVisible(handle)

    if not hwnd:
        handles = []

        @callback_type
        def collect(handle, _):
            if owns(handle):
                handles.append(handle)
            return True

        user.EnumWindows(collect, 0)
        if not handles:
            return False
        hwnd = handles[0]
    if not owns(hwnd):
        return False
    current = kernel.GetCurrentThreadId()
    foreground = user.GetForegroundWindow()
    threads = {user.GetWindowThreadProcessId(foreground, None),
               user.GetWindowThreadProcessId(hwnd, None)} - {0, current}
    attached = []
    try:
        for thread in threads:
            if user.AttachThreadInput(current, thread, True):
                attached.append(thread)
        user.ShowWindow(hwnd, 9)
        user.BringWindowToTop(hwnd)
        user.SetForegroundWindow(hwnd)
    finally:
        for thread in reversed(attached):
            user.AttachThreadInput(current, thread, False)
    return user.GetForegroundWindow() == hwnd
