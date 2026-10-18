"""Harmless click target for scripts/check_click_reliability.py.

A separate process (own message queue, own focus) with real Tk widgets. It
records, independently of AirDesk, every mouse press/release it receives and
every widget *activation* (button command, checkbox toggle, right-click
handler, double-click, drag), timestamped with time.perf_counter (QPC, so
comparable with the AirDesk process on the same machine). Nothing typed,
nothing persisted beyond the two files passed on the command line.

  python _click_target.py <layout.json> <events.jsonl>
"""
import ctypes
import json
import sys
import time
import tkinter as tk

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)   # physical pixels, like AirDesk
except Exception:
    pass

layout_path, events_path = sys.argv[1], sys.argv[2]
out = open(events_path, "a", encoding="utf-8", buffering=1)


def log(kind, **kw):
    out.write(json.dumps({"t": time.perf_counter(), "kind": kind, **kw}) + "\n")


root = tk.Tk()
root.title("AirDesk click reliability target")
root.geometry("900x560+260+180")
root.attributes("-topmost", True)

big = tk.Button(root, text="Primary target", width=18, height=3,
                command=lambda: log("activate", name="primary"))
big.place(x=40, y=40)
rc = tk.Label(root, text="Right-click target", width=18, height=3, bg="#ffcc88", relief="raised")
rc.place(x=280, y=40)
rc.bind("<ButtonRelease-3>", lambda e: log("activate", name="secondary", x=e.x_root, y=e.y_root))
dbl = tk.Label(root, text="Double-click target", width=18, height=3, bg="#cce0ff", relief="raised")
dbl.place(x=520, y=40)
dbl.bind("<Double-Button-1>", lambda e: log("activate", name="double", x=e.x_root, y=e.y_root))

boxes = []
for i in range(7):
    v = tk.IntVar(value=0)
    cb = tk.Checkbutton(root, variable=v, padx=0, pady=0, bd=0, highlightthickness=0,
                        command=lambda i=i, v=v: log("activate", name=f"check{i}", value=v.get()))
    cb.place(x=60 + i * 20, y=180)
    boxes.append(cb)

drag = tk.Canvas(root, width=820, height=260, bg="white", highlightthickness=1)
drag.place(x=40, y=240)
_drag = {"down": None, "moves": 0}


def d_press(e):
    _drag.update(down=(e.x_root, e.y_root), moves=0)


def d_move(e):
    _drag["moves"] += 1


def d_release(e):
    if _drag["down"]:
        log("activate", name="drag", start=_drag["down"], end=(e.x_root, e.y_root), moves=_drag["moves"])
    _drag["down"] = None


drag.bind("<ButtonPress-1>", d_press)
drag.bind("<B1-Motion>", d_move)
drag.bind("<ButtonRelease-1>", d_release)
root.bind_all("<ButtonPress>", lambda e: log("press", button=e.num, x=e.x_root, y=e.y_root,
                                             widget=str(e.widget)))
root.bind_all("<ButtonRelease>", lambda e: log("release", button=e.num, x=e.x_root, y=e.y_root,
                                               widget=str(e.widget)))


def publish():
    root.update_idletasks()
    from _windows_target import activate_own_window
    activate_own_window()

    def rect(w):
        return [w.winfo_rootx(), w.winfo_rooty(), w.winfo_width(), w.winfo_height()]
    data = {"primary": rect(big), "secondary": rect(rc), "double": rect(dbl), "drag": rect(drag),
            "checks": [rect(b) for b in boxes]}
    with open(layout_path, "w", encoding="utf-8") as f:
        json.dump(data, f)


root.after(300, publish)
root.mainloop()
