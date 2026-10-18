"""Camera manager: one capture thread, latest-frame-wins, automatic recovery.

Backend choice (historical measurement on one webcam, not a hardware guarantee):
Media Foundation (MSMF) sustained ~30 FPS at 3840x2160, 1920x1080 and
1280x720, while DirectShow fell to 5-15 FPS after mode changes, so MSMF is
the default and DirectShow remains selectable.

Only the newest frame is kept. The vision thread waits on a condition
variable for a frame newer than the one it processed last, so there is
never a queue of stale frames and capture never blocks on inference.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass

# Opening MSMF devices is several seconds faster without hardware transforms.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ..config import CameraConfig  # noqa: E402

log = logging.getLogger(__name__)

BACKENDS = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW, "any": cv2.CAP_ANY}
MODES = [(3840, 2160), (2560, 1440), (1920, 1080), (1280, 720), (960, 540), (640, 480)]


@dataclass
class CameraInfo:
    index: int
    name: str
    width: int = 0
    height: int = 0


def camera_names() -> list[str]:
    """Friendly names of connected cameras (Windows PnP), best effort.

    The order usually, but not always, matches OpenCV indices; the UI shows
    the probed resolution next to each name so the user can tell.
    """
    try:
        cmd = ["powershell", "-NoProfile", "-Command",
               "Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -in 'Camera','Image' } "
               "| Select-Object -ExpandProperty Name"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=8,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    except Exception:
        return []


def enumerate_cameras(backend: str = "msmf", max_index: int = 5) -> list[CameraInfo]:
    """Probe indices 0..max_index. Do not call while the stream is running."""
    names = camera_names()
    found: list[CameraInfo] = []
    misses = 0
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, BACKENDS.get(backend, cv2.CAP_MSMF))
        ok = cap.isOpened() and cap.read()[0]
        if ok:
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            name = names[len(found)] if len(found) < len(names) else f"Camera {i}"
            found.append(CameraInfo(i, name, w, h))
            misses = 0
        else:
            misses += 1
        cap.release()
        if misses >= 2 and found:
            break
    return found


class CameraManager:
    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self.status = "stopped"        # stopped | opening | running | reconnecting | error
        self.error = ""
        self.negotiated = (0, 0, 0.0)  # width, height, driver-reported fps
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._t = 0.0
        self._seq = 0
        self._stop = threading.Event()
        self._restart = threading.Event()
        self._thread: threading.Thread | None = None
        self._times: deque[float] = deque(maxlen=90)
        self.read_ms = 0.0

    # ------------------------------------------------------------ control
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="airdesk-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._thread = None
        self.status = "stopped"

    def reconfigure(self, cfg: CameraConfig) -> None:
        """Apply a new camera/mode; the capture thread reopens the device."""
        self.cfg = cfg
        self._restart.set()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def capture_fps(self) -> float:
        ts = list(self._times)
        if len(ts) < 2 or time.perf_counter() - ts[-1] > 1.0:
            return 0.0
        return (len(ts) - 1) / (ts[-1] - ts[0])

    def wait_frame(self, after_seq: int, timeout: float) -> tuple[int, float, np.ndarray] | None:
        """Newest frame with seq > after_seq, or None on timeout."""
        with self._cond:
            if self._seq <= after_seq:
                self._cond.wait(timeout)
            if self._seq <= after_seq or self._frame is None:
                return None
            return self._seq, self._t, self._frame

    # ------------------------------------------------------------- thread
    def _open(self) -> cv2.VideoCapture | None:
        c = self.cfg
        self.status = "opening"
        cap = cv2.VideoCapture(c.index, BACKENDS.get(c.backend, cv2.CAP_MSMF))
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.height)
        cap.set(cv2.CAP_PROP_FPS, c.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            return None
        self.negotiated = (frame.shape[1], frame.shape[0], float(cap.get(cv2.CAP_PROP_FPS) or 0))
        log.info("camera %d opened via %s: %dx%d (driver fps %.1f)", c.index, c.backend,
                 frame.shape[1], frame.shape[0], self.negotiated[2])
        return cap

    def _run(self) -> None:
        cap = None
        fails = 0
        backoff = 0.5
        while not self._stop.is_set():
            if self._restart.is_set():
                self._restart.clear()
                if cap is not None:
                    cap.release()
                    cap = None
            if cap is None:
                cap = self._open()
                if cap is None:
                    self.status = "reconnecting" if self._seq else "error"
                    self.error = f"Camera {self.cfg.index} could not be opened ({self.cfg.backend})"
                    self._times.clear()
                    self._stop.wait(backoff)
                    backoff = min(backoff * 1.6, 3.0)
                    continue
                backoff, fails, self.error = 0.5, 0, ""
                self.status = "running"
            t0 = time.perf_counter()
            ok, frame = cap.read()
            t1 = time.perf_counter()
            if not ok or frame is None:
                fails += 1
                if fails >= 15:            # ~0.5 s of failures: device lost
                    log.warning("camera read failing; reopening")
                    cap.release()
                    cap = None
                    self.status = "reconnecting"
                    self.error = "Camera disconnected - retrying"
                    self._times.clear()
                else:
                    time.sleep(0.02)
                continue
            fails = 0
            self.read_ms = (t1 - t0) * 1000
            self._times.append(t1)
            with self._cond:
                self._frame, self._t = frame, t1
                self._seq += 1
                self._cond.notify_all()
        if cap is not None:
            cap.release()
        self.status = "stopped"
