"""MediaPipe Tasks HandLandmarker wrapper: frame -> `FrameInput`.

* Frames are downscaled to `inference_width` before inference (4K frames
  are never fed to the model; landmarks are normalized so the mapping back
  to any resolution is exact).
* Mirroring is applied to the small frame, so the preview, the landmarks
  and the cursor all share one "selfie" coordinate system.
* MediaPipe labels handedness assuming a mirrored (selfie) image, so the
  label is swapped when mirroring is off.
* The GPU delegate is attempted only when configured; MediaPipe's Python
  GPU delegate is not available on Windows builds, in which case the CPU
  delegate is used and reported as such (never assumed).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from ..config import CameraConfig
from ..core.types import FrameInput, HandObservation
from ..resources import model_path

log = logging.getLogger(__name__)


@dataclass
class Preview:
    image: np.ndarray                  # RGB, inference size, mirrored as configured
    hands: list[HandObservation]
    t: float


class HandTracker:
    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self.delegate = "cpu"
        self.delegate_note = ""
        self._last_ts = -1
        self._landmarker = self._create(cfg)
        self.infer_ms = 0.0

    def _create(self, cfg: CameraConfig):
        path = str(model_path())
        want_gpu = cfg.delegate == "gpu"
        for delegate in ([BaseOptions.Delegate.GPU] if want_gpu else []) + [BaseOptions.Delegate.CPU]:
            try:
                opts = vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=path, delegate=delegate),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=cfg.num_hands,
                    min_hand_detection_confidence=cfg.min_detection_confidence,
                    min_hand_presence_confidence=cfg.min_tracking_confidence,
                    min_tracking_confidence=cfg.min_tracking_confidence)
                lm = vision.HandLandmarker.create_from_options(opts)
                self.delegate = "gpu" if delegate == BaseOptions.Delegate.GPU else "cpu"
                return lm
            except Exception as e:  # GPU delegate unsupported on this platform/build
                if delegate == BaseOptions.Delegate.GPU:
                    self.delegate_note = f"GPU delegate unavailable ({type(e).__name__}); using CPU"
                    log.warning(self.delegate_note)
                else:
                    raise
        raise RuntimeError("no MediaPipe delegate available")

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass

    def process(self, frame_bgr: np.ndarray, t: float, want_preview: bool = False
                ) -> tuple[FrameInput, Preview | None]:
        c = self.cfg
        h, w = frame_bgr.shape[:2]
        iw = min(c.inference_width, w)
        ih = int(round(h * iw / w))
        small = cv2.resize(frame_bgr, (iw, ih), interpolation=cv2.INTER_AREA) if iw != w else frame_bgr
        if c.mirror:
            small = cv2.flip(small, 1)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        ts = int(t * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        t0 = time.perf_counter()
        res = self._landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        self.infer_ms = (time.perf_counter() - t0) * 1000
        hands: list[HandObservation] = []
        for i, lms in enumerate(res.hand_landmarks or []):
            img = np.array([(p.x, p.y, p.z) for p in lms], dtype=np.float64)
            world = None
            if res.hand_world_landmarks and i < len(res.hand_world_landmarks):
                world = np.array([(p.x, p.y, p.z) for p in res.hand_world_landmarks[i]], dtype=np.float64)
            label, score = "Right", 0.5
            if res.handedness and i < len(res.handedness) and res.handedness[i]:
                cat = res.handedness[i][0]
                label, score = cat.category_name, float(cat.score)
            if not c.mirror:
                label = "Left" if label == "Right" else "Right"
            hands.append(HandObservation(img, world, label, score))
        fi = FrameInput(t=t, hands=hands, frame_size=(iw, ih))
        return fi, (Preview(rgb, hands, t) if want_preview else None)
