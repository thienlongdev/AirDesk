"""Tiny external editor used by integration checks (separate process, own focus)."""
import sys
import tkinter as tk

out_path = sys.argv[1]
root = tk.Tk()
root.title("AirDesk focus test editor")
root.geometry("520x160+80+80")
text = tk.Text(root, font=("Segoe UI", 14))
text.pack(fill="both", expand=True)


def dump():
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text.get("1.0", "end-1c"))
    root.after(100, dump)


root.after(100, dump)
def focus_target():
    from _windows_target import activate_own_window
    activate_own_window()
    text.focus_set()


root.after(150, focus_target)
root.mainloop()
