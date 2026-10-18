"""Synthetic MediaPipe-style hand landmarks for tests and synthetic benchmarks.

A planar kinematic hand: each finger is a chain of segments whose joints
bend by a per-pose angle. Geometry is built in square units (x already
multiplied by the image aspect) and converted to normalized image
coordinates, exactly the space `hand_model.analyze` reads.

This is *not* a substitute for real-hand testing; it exists so the
gesture engine, state machine and filters can be exercised
deterministically and so filter comparisons are reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.types import Finger, FrameInput, HandObservation

ASPECT = 16 / 9

_MCP = {Finger.INDEX: (-0.050, -0.170), Finger.MIDDLE: (-0.005, -0.180),
        Finger.RING: (0.040, -0.170), Finger.LITTLE: (0.080, -0.150)}
_SEG = {Finger.INDEX: (0.075, 0.045, 0.035), Finger.MIDDLE: (0.080, 0.050, 0.035),
        Finger.RING: (0.075, 0.045, 0.033), Finger.LITTLE: (0.060, 0.035, 0.030)}
_SPREAD = {Finger.INDEX: -8.0, Finger.MIDDLE: 0.0, Finger.RING: 7.0, Finger.LITTLE: 15.0}
_CURL = (70.0, 100.0, 60.0)   # MCP, PIP, DIP flexion of a fully curled finger


@dataclass
class Pose:
    """Which fingers are extended, plus optional contacts."""

    extended: set[Finger] = field(default_factory=lambda: {Finger.INDEX})
    pinch: Finger | None = None          # thumb tip touches this fingertip
    thumb_press: bool = False            # thumb tip pressed on the index side
    curl: float = 1.0                    # how curled the non-extended fingers are
    # Realistic pinch motion: the pinched finger flexes toward the thumb by
    # `pinch_curl` (even when it is the extended pointer finger) and the
    # thumb travels from rest to the fingertip; `approach` 0..1 interpolates
    # the whole closing movement (0 = still pointing, 1 = in contact).
    pinch_curl: float = 0.5
    approach: float = 1.0
    cluster: float = 0.0                 # 0..1: all five fingertips gather (keyboard toggle)
    # Real-hand realism (v1.1.1 reliability work): per-finger curl of the
    # non-extended fingers (a relaxed hand is half-curled, not a fist), and
    # the thumb-to-fingertip landmark gap that remains at physical contact
    # (MediaPipe tips sit at the nail end, so touching pads stay ~0.1-0.2
    # hand scales apart; square units, hand scale = 0.18).
    curls: dict | None = None
    gap: float = 0.0

    @classmethod
    def pointing(cls, finger: Finger) -> "Pose":
        return cls(extended={finger})

    @classmethod
    def open_palm(cls) -> "Pose":
        return cls(extended=set(Finger))

    @classmethod
    def fist(cls) -> "Pose":
        return cls(extended=set())

    @classmethod
    def two_finger(cls) -> "Pose":
        return cls(extended={Finger.INDEX, Finger.MIDDLE})

    @classmethod
    def closing(cls, finger: Finger, approach: float, pointer: Finger = Finger.INDEX,
                curl: float = 0.65) -> "Pose":
        """Part-way through a natural pinch of `finger` while `pointer` points."""
        return cls(extended={pointer}, pinch=finger, pinch_curl=curl, approach=approach)

    @classmethod
    def gather(cls, amount: float = 1.0) -> "Pose":
        """Five fingertips gathered into one cluster (v1.1's keyboard gesture,
        kept to prove it no longer toggles anything; 0 = open palm)."""
        return cls(extended=set(Finger), cluster=amount)

    @classmethod
    def ily(cls, little_curl: float = 0.0, thumb: bool = True) -> "Pose":
        """🤟: thumb, index and little finger extended; middle and ring curled."""
        ext = {Finger.INDEX, Finger.LITTLE} | ({Finger.THUMB} if thumb else set())
        return cls(extended=ext if not little_curl else ext - {Finger.LITTLE},
                   curls={Finger.LITTLE: little_curl} if little_curl else None)

    @classmethod
    def horns(cls) -> "Pose":
        """🤘: index and little up, thumb folded over the curled middle/ring."""
        return cls(extended={Finger.INDEX, Finger.LITTLE})


def _rodrigues(v: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    a = math.radians(deg)
    k = axis / np.linalg.norm(axis)
    return v * math.cos(a) + np.cross(k, v) * math.sin(a) + k * np.dot(k, v) * (1 - math.cos(a))


def geometry(pose: Pose) -> np.ndarray:
    """21 x 3 landmark positions in square units, wrist at the origin.

    The palm lies in the x-y plane facing the camera; fingers flex about
    an axis in that plane, so a curled finger folds toward the camera
    (-z) and its tip projects back onto the palm, as in real images.
    """
    g = np.zeros((21, 3))
    for f in (Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.LITTLE):
        s = math.radians(_SPREAD[f])
        mcp = np.array([*_MCP[f], 0.0])
        g[f.mcp] = mcp
        d = np.array([math.sin(s), -math.cos(s), 0.0])
        axis = np.array([math.cos(s), math.sin(s), 0.0])
        if f is pose.pinch:
            # A pinched finger bends toward the thumb, the pointer included.
            rest = 0.0 if f in pose.extended else (pose.curls or {}).get(f, pose.curl)
            curl = rest + (pose.pinch_curl - rest) * pose.approach
        elif pose.cluster > 0:
            curl = 0.42 * pose.cluster
        elif f in pose.extended:
            curl = 0.0
        elif pose.curls and f in pose.curls:
            curl = pose.curls[f]
        else:
            curl = pose.curl
        p = mcp
        for i, (length, bend) in enumerate(zip(_SEG[f], _CURL)):
            d = _rodrigues(d, axis, bend * curl)
            p = p + d * length
            g[f.mcp + 1 + i] = p

    g[1] = (-0.050, -0.040, 0.0)
    if Finger.THUMB in pose.extended and not (pose.pinch or pose.thumb_press):
        g[2:5] = [(-0.090, -0.080, 0.0), (-0.125, -0.110, 0.0), (-0.155, -0.135, 0.0)]
    else:
        # Relaxed thumb resting against the side of the curled index.
        g[2:5] = [(-0.085, -0.075, -0.010), (-0.100, -0.105, -0.025), (-0.095, -0.130, -0.040)]

    if pose.cluster > 0:
        # Fingertips (and their DIPs, less) converge on a point in front of
        # the knuckles; the thumb tip joins them.
        tips = [int(f) for f in (Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.LITTLE)]
        centre = g[tips].mean(axis=0) + np.array([-0.020, 0.010, 0.0])
        k = pose.cluster
        for f in (Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.LITTLE):
            g[int(f)] += (centre - g[int(f)]) * 0.85 * k
            g[int(f) - 1] += (centre - g[int(f) - 1]) * 0.45 * k
        rest = np.array([-0.155, -0.135, 0.0])
        g[4] = rest + (centre - rest) * k
        g[3] = g[2] + (g[4] - g[2]) * 0.6 + np.array([-0.012, 0.0, 0.0])
        g[2] = (-0.090, -0.080, 0.0)
        return g

    target = None
    if pose.pinch is not None:
        target = g[int(pose.pinch)] + np.array([-0.006, 0.004, 0.0])
        if pose.gap:
            away = g[4] - target
            target = target + away / max(float(np.linalg.norm(away)), 1e-9) * pose.gap
    elif pose.thumb_press:
        target = (g[6] + g[7]) / 2 + np.array([-0.004, 0.0, 0.0])
    if target is not None:
        if pose.pinch is not None and pose.approach < 1.0:
            rest = g[4].copy()
            target = rest + (target - rest) * pose.approach
        g[4] = target
        g[3] = g[2] + (target - g[2]) * 0.6 + np.array([-0.012, 0.0, 0.0])
    return g


def observation(pose: Pose, cx: float = 0.5, cy: float = 0.65, scale: float = 1.0,
                handedness: str = "Right", noise: float = 0.0,
                rng: np.random.Generator | None = None, roll_deg: float = 0.0,
                mirror: bool = False, world: bool = False, depth_noise: float = 0.0,
                score: float = 0.95) -> HandObservation:
    """Hand with its wrist at normalized image point (cx, cy). `roll_deg`
    rotates the hand in the image plane; `mirror` makes it the other hand.
    `world` adds MediaPipe-style world landmarks (metres, hand ~9 cm wrist to
    middle knuckle) - the production geometry path - with optional depth
    noise on the fingertips (`depth_noise`, metres: the estimated z of
    overlapping tips is the least reliable coordinate)."""
    g = geometry(pose) * scale
    if mirror:
        g[:, 0] = -g[:, 0]
    if roll_deg:
        a = math.radians(roll_deg)
        c, s_ = math.cos(a), math.sin(a)
        g[:, 0], g[:, 1] = g[:, 0] * c - g[:, 1] * s_, g[:, 0] * s_ + g[:, 1] * c
    img = np.zeros((21, 3))
    img[:, 0] = cx + g[:, 0] / ASPECT
    img[:, 1] = cy + g[:, 1]
    img[:, 2] = g[:, 2] / ASPECT
    if noise > 0:
        rng = rng or np.random.default_rng(0)
        img[:, :2] += rng.normal(0.0, noise, size=(21, 2))
    w = None
    if world:
        w = g * 0.5
        if noise > 0:
            w += (rng or np.random.default_rng(0)).normal(0.0, noise * 0.5, size=(21, 3))
        if depth_noise > 0:
            tips = [4, 8, 12, 16, 20]
            w[tips, 2] += (rng or np.random.default_rng(0)).normal(0.0, depth_noise, size=len(tips))
    return HandObservation(image=img, world=w, handedness=handedness, score=score)


def frame(t: float, *hands: HandObservation) -> FrameInput:
    return FrameInput(t=t, hands=list(hands), frame_size=(640, 360))
