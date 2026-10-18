"""Hardware performance measurement: capture FPS per backend/mode and
MediaPipe inference cost per inference width / delegate. No frames saved."""

from __future__ import annotations

import os
import platform
import statistics
import time

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2  # noqa: E402

from ..config import CameraConfig  # noqa: E402
from ..vision.tracker import HandTracker  # noqa: E402

BACK = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW}


def measure_capture(index: int, backend: str, w: int, h: int, seconds: float):
    t_open = time.perf_counter()
    cap = cv2.VideoCapture(index, BACK[backend])
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 30)
    ok, f = cap.read()
    open_s = time.perf_counter() - t_open
    if not ok:
        cap.release()
        return None, []
    for _ in range(5):
        cap.read()
    n, t0, frames = 0, time.perf_counter(), []
    while time.perf_counter() - t0 < seconds:
        ok, f = cap.read()
        if ok:
            n += 1
            if len(frames) < 40:
                frames.append(f)
    fps = n / (time.perf_counter() - t0)
    cap.release()
    return {"backend": backend, "request": f"{w}x{h}", "got": f"{f.shape[1]}x{f.shape[0]}",
            "fps": round(fps, 1), "open_s": round(open_s, 2)}, frames


def measure_inference(frames, width: int, delegate: str, n: int = 60) -> dict:
    tr = HandTracker(CameraConfig(inference_width=width, delegate=delegate))
    ms, hands = [], 0
    t = time.perf_counter()
    for i in range(n):
        t += 1 / 30
        t0 = time.perf_counter()
        fi, _ = tr.process(frames[i % len(frames)], t)
        ms.append((time.perf_counter() - t0) * 1000)
        hands += len(fi.hands)
    tr.close()
    ms = sorted(ms[5:])
    return {"inference_width": width, "delegate_requested": delegate, "delegate_used": tr.delegate,
            "note": tr.delegate_note, "total_ms_median": round(statistics.median(ms), 2),
            "total_ms_p95": round(ms[int(len(ms) * 0.95) - 1], 2), "frames_with_hands": hands}


def run_performance(index: int = 0, seconds: float = 2.5, backends=("msmf", "dshow"),
                    widths=(480, 640, 960), delegates=("cpu", "gpu")) -> dict:
    res = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "machine": platform.processor(),
           "python": platform.python_version(), "opencv": cv2.__version__, "capture": [], "inference": []}
    keep: dict[str, list] = {}
    for backend in backends:
        for w, h in ((3840, 2160), (1920, 1080), (1280, 720)):
            r, frames = measure_capture(index, backend, w, h, seconds)
            if r is None:
                res["capture"].append({"backend": backend, "request": f"{w}x{h}", "got": "failed", "fps": 0})
                continue
            res["capture"].append(r)
            if backend == "msmf" and w in (3840, 1280):
                keep["4K" if w == 3840 else "720p"] = frames
    for label, frames in keep.items():
        for width in widths:
            for d in delegates:
                r = measure_inference(frames, width, d)
                r["source"] = label
                res["inference"].append(r)
    return res
