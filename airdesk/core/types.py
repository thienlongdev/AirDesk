"""Shared value types used across the AirDesk engine.

Everything here is plain data: no OpenCV, MediaPipe, Qt or Win32 imports,
so the whole gesture/pointer engine can be unit-tested headlessly.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np


class Finger(enum.IntEnum):
    """The five pointer candidates, valued by MediaPipe fingertip index."""

    THUMB = 4
    INDEX = 8
    MIDDLE = 12
    RING = 16
    LITTLE = 20

    @property
    def label(self) -> str:
        return self.name.capitalize()

    @property
    def mcp(self) -> int:
        """Base joint (knuckle) landmark of this finger."""
        return {4: 2, 8: 5, 12: 9, 16: 13, 20: 17}[int(self)]

    @property
    def chain(self) -> tuple[int, int, int, int]:
        """The four landmarks of the finger, base to tip."""
        tip = int(self)
        return (tip - 3, tip - 2, tip - 1, tip)

    @classmethod
    def from_name(cls, name: str) -> "Finger":
        return cls[name.strip().upper()]


FINGERS: tuple[Finger, ...] = (Finger.THUMB, Finger.INDEX, Finger.MIDDLE,
                               Finger.RING, Finger.LITTLE)
NON_THUMB: tuple[Finger, ...] = FINGERS[1:]


class PointerMode(str, enum.Enum):
    DIRECT = "direct"        # calibrated absolute mapping
    RELATIVE = "relative"    # trackpad: displacement -> cursor delta
    PRECISION = "precision"  # relative with low gain and heavy smoothing


class FingerMode(str, enum.Enum):
    AUTO = "auto"      # pick the finger the user is pointing with
    MANUAL = "manual"  # always the configured finger
    LOCK = "lock"      # current finger frozen until explicitly switched


class HandPreference(str, enum.Enum):
    AUTO = "auto"
    LEFT = "left"
    RIGHT = "right"


@dataclass(slots=True)
class HandObservation:
    """One detected hand in one camera frame.

    `image` holds 21 landmarks in normalized image coordinates (x right,
    y down, already mirrored when the selfie view is enabled). `world`
    holds MediaPipe's metric hand-centred landmarks when available; the
    geometric classifiers prefer it because it is invariant to camera
    distance and much less sensitive to hand rotation.
    """

    image: np.ndarray                # (21, 3) float
    world: np.ndarray | None         # (21, 3) metres, or None
    handedness: str                  # "Left" | "Right" (user's anatomical hand)
    score: float                     # handedness / presence confidence

    @property
    def wrist(self) -> tuple[float, float]:
        return float(self.image[0, 0]), float(self.image[0, 1])


@dataclass(slots=True)
class FrameInput:
    """Everything the engine needs from one processed camera frame."""

    t: float                          # capture timestamp, seconds (monotonic)
    hands: list[HandObservation]
    frame_size: tuple[int, int] = (0, 0)  # inference image (w, h)


@dataclass(slots=True)
class HandFeatures:
    """Derived geometry for one hand (see `hand_model.analyze`)."""

    extension: dict[Finger, float]   # 0 = fully curled .. 1 = fully straight
    extended: dict[Finger, bool]     # hysteresis-free instantaneous decision
    pinch: dict[Finger, float]       # thumb-tip to fingertip / hand scale
    thumb_press: float               # thumb tip to index-side / hand scale
    hand_scale: float                # image-space hand size (normalized units)
    palm_center: tuple[float, float]
    knuckle_center: tuple[float, float]
    tips: dict[Finger, tuple[float, float]]
    mcps: dict[Finger, tuple[float, float]]
    fist: bool
    open_palm: bool
    two_finger: bool                 # index + middle extended, ring + little curled
    # fingertip-to-own-knuckle distance / hand scale (long fingers): a tip
    # folded into the palm is close to its knuckle, a pinching tip is not.
    reach: dict[Finger, float] = field(default_factory=dict)


@dataclass(slots=True)
class EngineStatus:
    """Snapshot published to the UI/diagnostics after each frame."""

    t: float = 0.0
    armed: bool = False
    paused: bool = False
    state: str = "DISARMED"
    hand: str = "-"
    finger: str = "-"
    finger_mode: str = "auto"
    pointer_mode: str = "direct"
    gesture: str = "-"
    buttons: tuple[str, ...] = ()
    keyboard_hover: str | None = None
    cursor: tuple[int, int] | None = None
    tracking: bool = False
    hands_seen: int = 0
    infer_ms: float = 0.0
    latency_ms: float = 0.0
    finger_confidence: dict[str, float] = field(default_factory=dict)
    extension: dict[str, float] = field(default_factory=dict)
    pinch: dict[str, float] = field(default_factory=dict)
    message: str = ""
    click_style: str = "pinch_pointer"
    aim_lock: tuple[int, int] | None = None   # Smart Aim Lock anchor (no button pressed yet)
    press_at: tuple[int, int] | None = None   # accepted press location (PRESS_PENDING)
    progress: float = 0.0                     # keyboard-gesture hold progress 0..1
    # v1.1.2
    click_mode: str = "instant"               # instant | drag | classic
    drag_progress: float = 0.0                # Drag Mode hold progress 0..1 (no button yet)
    hint: str = ""                            # camera-condition hint (too far / too close / ...)
    aim_reason: str = ""                      # why Aim Lock is holding the cursor
    contact: dict = field(default_factory=dict)  # nearest contact: pair, d, engage, ratio
    rearm: str = "ready"                      # ready | open fingers to click again
    candidate_ms: float = 0.0                 # age of the current pinch candidate / press
    # v1.1.3
    drag: str = ""                            # Pinky Drag: ready | dragging | released | cancelled
    pose: str = "-"                           # recognized pose of the controlling hand
    dual: bool = False                        # Dual-Hand Air Keyboard active (no desktop input)
    vhands: list = field(default_factory=list)  # per virtual hand: side, tracked, state, key ...
    # v1.1.4
    dual_diag: dict = field(default_factory=dict)  # identity / association diagnostics (no key ids)
