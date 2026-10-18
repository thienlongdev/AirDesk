"""Landmark processing: turn 21 MediaPipe landmarks into hand geometry.

Finger extension is measured from *joint angles*, not fingertip height,
so it works when the hand is rotated, tilted, or pointing sideways. When
MediaPipe's metric world landmarks are available they are used for all
angles and pinch ratios (invariant to camera distance); otherwise the
aspect-corrected image landmarks are used.

Landmark indices (MediaPipe Hands):
    0 wrist
    1-4   thumb  (CMC, MCP, IP, TIP)
    5-8   index  (MCP, PIP, DIP, TIP)
    9-12  middle
    13-16 ring
    17-20 little
"""

from __future__ import annotations

import math

import numpy as np

from .types import FINGERS, NON_THUMB, Finger, HandFeatures, HandObservation

# Bend (degrees) mapped to extension 1.0 and 0.0 for the four long fingers.
_STRAIGHT_BEND = 35.0
_CURLED_BEND = 150.0
EXTENDED_ON = 0.62   # extension above this counts as "extended"
CURLED_BELOW = 0.42  # extension below this counts as "curled"
# 🤟 keyboard pose (see `ily_pose`)
ILY_LITTLE_ON = 0.55     # the little finger is shorter and rarely fully straight
ILY_THUMB_ON = 0.45      # thumb abducted from the palm
ILY_THUMB_CLEAR = 0.55   # thumb tip at least this far from the index tip (hand scales)
# Pinky Drag pose (v1.1.3, see `pinky_pose`)
PINKY_ON = 0.55          # little finger extended (entering the pose)
PINKY_THUMB_MAX = 0.45   # thumb tucked (curled / resting against the fingers)
PINKY_KEEP = 0.40        # while dragging, the little finger still counts as up above this


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    """Unsigned angle in degrees between two 3-D vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    c = float(np.dot(a, b) / (na * nb))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from point p to segment ab."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float(np.dot(p - a, ab)) / denom))
    return float(np.linalg.norm(p - (a + t * ab)))


def finger_bend(pts: np.ndarray, finger: Finger) -> float:
    """Weighted flexion (degrees) of one long finger: PIP + 0.6 DIP + 0.4 MCP."""
    mcp, pip, dip, tip = finger.chain
    wrist = pts[0]
    a_mcp = _angle(pts[mcp] - wrist, pts[pip] - pts[mcp])
    a_pip = _angle(pts[pip] - pts[mcp], pts[dip] - pts[pip])
    a_dip = _angle(pts[dip] - pts[pip], pts[tip] - pts[dip])
    return a_pip + 0.6 * a_dip + 0.4 * a_mcp


def thumb_extension(pts: np.ndarray, scale: float) -> float:
    """Thumb extension combines straightness with abduction from the palm.

    A straight thumb tucked against the index finger is not "pointing", so
    the distance from the thumb tip to the index knuckle (normalized by
    hand scale) carries half the weight.
    """
    a_mcp = _angle(pts[2] - pts[1], pts[3] - pts[2])
    a_ip = _angle(pts[3] - pts[2], pts[4] - pts[3])
    straight = 1.0 - min(max((a_mcp + a_ip - 20.0) / 90.0, 0.0), 1.0)
    reach = float(np.linalg.norm(pts[4] - pts[5])) / scale
    abduct = min(max((reach - 0.40) / 0.40, 0.0), 1.0)
    # Thumb tip should also be farther from the little-finger knuckle than
    # the thumb IP joint is (otherwise it is folded across the palm).
    across = float(np.linalg.norm(pts[4] - pts[17]) >= np.linalg.norm(pts[3] - pts[17]))
    return (0.45 * straight + 0.55 * abduct) * (0.5 + 0.5 * across)


def _geometry_points(obs: HandObservation, aspect: float) -> np.ndarray:
    if obs.world is not None:
        return np.asarray(obs.world, dtype=np.float64)
    pts = np.asarray(obs.image, dtype=np.float64).copy()
    pts[:, 0] *= aspect  # make x and y share units
    pts[:, 2] *= aspect
    return pts


def analyze(obs: HandObservation, aspect: float = 16 / 9, depth_weight: float = 1.0) -> HandFeatures:
    """Compute extension, pinch and pose features for one hand.

    `depth_weight` scales the z axis of the thumb-to-fingertip contact
    distances (world landmarks only; GestureConfig.contact_depth_weight)."""
    g = _geometry_points(obs, aspect)
    g_scale = float(np.linalg.norm(g[9] - g[0])) or 1e-6

    extension: dict[Finger, float] = {}
    for f in NON_THUMB:
        bend = finger_bend(g, f)
        e = 1.0 - (bend - _STRAIGHT_BEND) / (_CURLED_BEND - _STRAIGHT_BEND)
        extension[f] = min(max(e, 0.0), 1.0)
    extension[Finger.THUMB] = thumb_extension(g, g_scale)
    extended = {f: extension[f] >= EXTENDED_ON for f in FINGERS}

    # Contact distances weight the depth axis (world landmarks only):
    # MediaPipe estimates z, and touching fingertips that overlap in the
    # image get the least reliable depth of all (v1.1.2).
    gc = g
    if obs.world is not None and depth_weight != 1.0:
        gc = g.copy()
        gc[:, 2] *= depth_weight
    pinch = {f: float(np.linalg.norm(gc[4] - gc[int(f)])) / g_scale for f in NON_THUMB}
    reach = {f: float(np.linalg.norm(g[int(f)] - g[f.mcp])) / g_scale for f in NON_THUMB}
    thumb_press = min(_seg_dist(g[4], g[5], g[6]), _seg_dist(g[4], g[6], g[7])) / g_scale

    img = np.asarray(obs.image, dtype=np.float64)
    xy = img[:, :2]
    corrected = xy.copy()
    corrected[:, 0] *= aspect
    hand_scale = float(np.linalg.norm(corrected[9] - corrected[0]))
    palm = xy[[0, 5, 9, 13, 17]].mean(axis=0)
    knuckles = xy[[5, 9, 13, 17]].mean(axis=0)

    curled4 = all(extension[f] < CURLED_BELOW for f in NON_THUMB)
    fist = curled4 and extension[Finger.THUMB] < EXTENDED_ON
    open_palm = all(extension[f] >= EXTENDED_ON for f in NON_THUMB) and extension[Finger.THUMB] >= 0.45
    two_finger = (extension[Finger.INDEX] >= EXTENDED_ON and extension[Finger.MIDDLE] >= EXTENDED_ON
                  and extension[Finger.RING] < CURLED_BELOW and extension[Finger.LITTLE] < CURLED_BELOW)

    return HandFeatures(
        extension=extension,
        extended=extended,
        pinch=pinch,
        thumb_press=thumb_press,
        hand_scale=hand_scale,
        palm_center=(float(palm[0]), float(palm[1])),
        knuckle_center=(float(knuckles[0]), float(knuckles[1])),
        tips={f: (float(xy[int(f), 0]), float(xy[int(f), 1])) for f in FINGERS},
        mcps={f: (float(xy[f.mcp, 0]), float(xy[f.mcp, 1])) for f in FINGERS},
        fist=fist,
        open_palm=open_palm,
        two_finger=two_finger,
        reach=reach,
    )


def ily_pose(feats: HandFeatures) -> bool:
    """The 🤟 keyboard pose: thumb, index and little finger extended, middle
    and ring curled, the thumb clearly away from every fingertip.

    Joint-angle extension and hand-scale distances only, so it holds for
    either hand and any palm orientation. Mutually exclusive by geometry
    with the V scroll pose (middle up), a fist (index and little up), the
    🤘 horns (thumb holds the curled fingers) and every thumb pinch (the
    thumb tip is far from the index and middle tips).
    """
    e, d = feats.extension, feats.pinch
    return (e[Finger.INDEX] >= EXTENDED_ON and e[Finger.LITTLE] >= ILY_LITTLE_ON
            and e[Finger.MIDDLE] < CURLED_BELOW and e[Finger.RING] < CURLED_BELOW
            and e[Finger.THUMB] >= ILY_THUMB_ON
            and d[Finger.INDEX] >= ILY_THUMB_CLEAR and d[Finger.MIDDLE] >= ILY_THUMB_CLEAR * 0.6)


def pinky_pose(feats: HandFeatures) -> bool:
    """The Pinky Drag pose (v1.1.3): only the little finger extended - thumb,
    index, middle and ring curled. Exclusive by geometry with 🤟 (index up),
    V scroll (index + middle up), the fist (little curled) and every thumb
    pinch (the thumb is tucked, not reaching a fingertip). Little-finger
    *pointing* with the thumb out stays a pointer pose (`pointer_candidate`)."""
    e = feats.extension
    return (e[Finger.LITTLE] >= PINKY_ON and e[Finger.THUMB] < PINKY_THUMB_MAX
            and e[Finger.INDEX] < CURLED_BELOW and e[Finger.MIDDLE] < CURLED_BELOW
            and e[Finger.RING] < CURLED_BELOW)


def pinky_forming(feats: HandFeatures) -> bool:
    """Looser pinky-only test used while the 0.2 s acquisition runs (one
    noisy frame must not cancel a deliberate pose)."""
    e = feats.extension
    return (e[Finger.LITTLE] >= 0.45 and e[Finger.THUMB] < 0.55
            and e[Finger.INDEX] < 0.5 and e[Finger.MIDDLE] < 0.5 and e[Finger.RING] < 0.5)


def pinky_held(feats: HandFeatures) -> bool:
    """Drag continues while the little finger stays up and no other long
    finger is extended (the thumb may move freely while dragging)."""
    e = feats.extension
    return (e[Finger.LITTLE] >= PINKY_KEEP and e[Finger.INDEX] < EXTENDED_ON
            and e[Finger.MIDDLE] < EXTENDED_ON and e[Finger.RING] < EXTENDED_ON)


def pointer_candidate(feats: HandFeatures) -> Finger | None:
    """The finger this pose unambiguously points with, or None.

    * exactly one long finger extended, the other three curled -> that finger
      (the thumb is ignored: people point with the thumb relaxed or out);
    * all four long fingers curled and the thumb extended -> thumb;
    * anything else (open palm, V-sign, fist, half-curled) -> ambiguous.
    """
    ext = feats.extension
    up = [f for f in NON_THUMB if ext[f] >= EXTENDED_ON]
    down = [f for f in NON_THUMB if ext[f] < CURLED_BELOW]
    if len(up) == 1 and len(down) == 3:
        return up[0]
    if len(down) == 4 and ext[Finger.THUMB] >= EXTENDED_ON:
        return Finger.THUMB
    return None


def action_fingers(pointer: Finger | None) -> list[Finger]:
    """Long fingers available for thumb pinches, nearest the thumb first.

    With the index as pointer this is [middle, ring, little]; with the
    middle finger as pointer it is [index, ring, little], and so on. When
    the thumb is the pointer, pinches are not used (see thumb press).
    """
    return [f for f in NON_THUMB if f != pointer]
