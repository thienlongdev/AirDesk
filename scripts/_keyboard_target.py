"""Harmless editor target for scripts/check_v113.py (separate process, own focus).

Records, independently of AirDesk, every key event Windows delivers to it
(keysym + modifier state bits) and every mouse button press, and dumps its
text. Ctrl+A selects all (bound explicitly - Tk's default differs per
platform); Ctrl+C / Ctrl+V / Ctrl+Z are Tk's own Text bindings.

  python _keyboard_target.py <text.txt> <events.jsonl>
"""
import json
import sys
import time
import tkinter as tk

text_path, events_path = sys.argv[1], sys.argv[2]
out = open(events_path, "a", encoding="utf-8", buffering=1)

root = tk.Tk()
root.title("AirDesk v1.1.3 keyboard target")
root.geometry("640x170+60+60")
text = tk.Text(root, font=("Segoe UI", 14), undo=True)
text.pack(fill="both", expand=True)


def log(kind, **kw):
    out.write(json.dumps({"t": time.perf_counter(), "kind": kind, **kw}) + "\n")


def select_all(_e):
    text.tag_add("sel", "1.0", "end-1c")
    return "break"


text.bind("<Control-a>", select_all)
text.bind("<Control-A>", select_all)
root.bind_all("<KeyPress>", lambda e: log("key", keysym=e.keysym, state=e.state, char=e.char != ""), add="+")
root.bind_all("<ButtonPress>", lambda e: log("press", button=e.num), add="+")


def dump():
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text.get("1.0", "end-1c"))
    root.after(80, dump)


root.after(80, dump)
def focus_target():
    from _windows_target import activate_own_window
    activate_own_window()
    text.focus_set()


root.after(150, focus_target)
root.mainloop()
