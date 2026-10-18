"""Integration check: Air Keyboard types into another app without stealing focus.

Uses REAL Windows input: it moves the mouse cursor and clicks the overlay
keys with SendInput, exactly like gesture clicks would, then verifies

  1. the external editor stays the foreground window throughout;
  2. the overlay never becomes foreground;
  3. the text that arrives in the editor is exactly what was typed
     (English via Unicode, Vietnamese via AirDesk Telex);
  4. the gesture path (press_key from a non-GUI thread) types as well.

Run:  .venv\\Scripts\\python scripts\\check_keyboard_focus.py
The cursor position is restored afterwards. Takes ~5 seconds.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from airdesk.config import KeyboardConfig  # noqa: E402
from airdesk.platform import win32  # noqa: E402
from airdesk.ui.keyboard_overlay import KeyboardOverlay  # noqa: E402


def main() -> int:
    out = Path(tempfile.mkdtemp()) / "editor.txt"
    editor = subprocess.Popen([sys.executable, str(Path(__file__).with_name("_editor_target.py")), str(out)])
    app = QApplication(sys.argv)
    cfg = KeyboardConfig(layout="full", suggestions=False)
    kb = KeyboardOverlay(cfg)
    results: dict[str, object] = {"fg_changes": []}
    saved_cursor = win32.cursor_pos()

    def editor_hwnd() -> int:
        fg = win32.foreground_window()
        return fg if "focus test editor" in win32.window_title(fg) else 0

    def key_center(key: str) -> tuple[int, int]:
        r = win32.window_rect(int(kb.winId()))
        k = next(k for k in kb.rects if k.key == key)
        return int(r.left + (k.x + k.w / 2) * r.width), int(r.top + (k.y + k.h / 2) * r.height)

    def click(key: str) -> None:
        x, y = key_center(key)
        win32.mouse_move_abs(x, y)
        time.sleep(0.03)
        win32.mouse_click("left")
        time.sleep(0.06)

    def pump(seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            time.sleep(0.01)

    def run() -> None:
        target = 0
        for _ in range(100):
            pump(0.05)
            target = editor_hwnd()
            if target:
                break
        results["target_found"] = bool(target)
        if not target:
            app.quit()
            return
        kb.set_shown(True)
        pump(0.4)
        results["noactivate_style"] = win32.has_noactivate(int(kb.winId()))
        results["fg_after_show"] = win32.foreground_window() == target

        def watch() -> None:
            fg = win32.foreground_window()
            if fg != target:
                results["fg_changes"].append(win32.window_title(fg))

        # English through physical-style mouse clicks on the overlay.
        for key in ["shift", "h", "i", "space"]:
            click(key)
            pump(0.03)
            watch()
        # Vietnamese Telex: switch language with the toolbar key, then type.
        click("lang")
        pump(0.05)
        for key in "vieejt":
            click(key)
            pump(0.03)
            watch()
        click("space")
        pump(0.05)
        # Gesture path: press_key() from a worker thread (no mouse events).
        t = threading.Thread(target=lambda: [kb.press_key(k) for k in "dduowngf"])
        t.start()
        t.join()
        pump(0.4)
        watch()
        # v1.1 title bar: resize with real clicks, then keep typing into the
        # editor at the new size (painted keys == hit-tested keys).
        w0 = kb.width()
        click("size-")
        pump(0.1)
        click("width+")
        pump(0.1)
        results["resized"] = (w0, kb.width(), kb.cfg.size_pct, kb.cfg.width_pct)
        watch()
        click("lang")          # vi -> raw IME; plain ASCII from here
        pump(0.05)
        click("lang")          # raw -> en
        pump(0.05)
        for key in ["space", "o", "k"]:
            click(key)
            pump(0.03)
            watch()
        click("reset")
        pump(0.1)
        results["reset"] = (kb.width() == w0, kb.cfg.size_pct, kb.cfg.width_pct)
        watch()
        results["overlay_never_foreground"] = int(kb.winId()) not in results["fg_changes"]
        pump(0.3)
        results["text"] = out.read_text(encoding="utf-8") if out.exists() else None
        app.quit()

    QTimer.singleShot(300, run)
    app.exec()
    editor.terminate()
    win32.mouse_move_abs(*saved_cursor)

    expected = "Hi việt đường ok"
    ok = (results.get("target_found") and results.get("fg_after_show") and not results["fg_changes"]
          and results.get("text") == expected and results.get("noactivate_style")
          and results.get("resized", (0, 0))[1] != results.get("resized", (0, 0))[0]
          and results.get("reset", (False,))[0])
    for k, v in results.items():
        print(f"{k:26} {v!r}")
    print(f"{'expected':26} {expected!r}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
