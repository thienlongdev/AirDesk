"""User settings and calibration profiles (JSON, stored per Windows user).

Location: %APPDATA%\\AirDesk\\settings.json and %APPDATA%\\AirDesk\\profiles\\*.json
Nothing typed, no webcam imagery and no biometric data is ever stored here.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def app_dir() -> Path:
    base = os.environ.get("AIRDESK_HOME") or os.path.join(
        os.environ.get("APPDATA", str(Path.home())), "AirDesk")
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class CameraConfig:
    index: int = 0
    backend: str = "msmf"            # msmf | dshow | any
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True
    inference_width: int = 640       # frames are downscaled for MediaPipe
    num_hands: int = 2
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5
    delegate: str = "cpu"            # cpu | gpu (gpu falls back to cpu)


@dataclass
class PointerConfig:
    mode: str = "direct"             # direct | relative | precision
    filter: str = "one_euro"         # one_euro | kalman | ema | none
    # One Euro defaults retained from the v1.1.4 synthetic tuning.
    min_cutoff: float = 0.6          # One Euro: jitter suppression at rest (Hz)
    beta: float = 15.0               # One Euro: cutoff growth with speed
    precision_min_cutoff: float = 0.5
    precision_beta: float = 3.0
    kalman_prediction_ms: float = 0.0
    anchor_blend: float = 0.25       # 0 = pure fingertip, 1 = pure knuckle
    # Direct-mode control box in normalized (mirrored) image coordinates.
    box_cx: float = 0.5
    box_cy: float = 0.45
    box_w: float = 0.50
    box_h: float = 0.50
    direct_gain: float = 1.0
    relative_gain_low: float = 1.0   # screen heights per image unit, slow motion
    relative_gain_high: float = 3.2  # fast motion
    relative_speed_low: float = 0.05 # image units / s
    relative_speed_high: float = 0.8
    precision_gain: float = 0.35
    deadzone_px: float = 1.5         # stationary cursor stabilization
    offset_decay_dist: float = 0.25  # direct mode: re-anchoring offset fades after this much motion
    offset_motion_floor: float = 0.0015  # per-frame filtered motion below this is noise, not motion
    monitor: str = "all"             # all | primary | <monitor index>
    dispatch_hz: int = 120           # cursor interpolation rate
    smoothing_tau_ms: float = 10.0   # dispatcher sub-frame interpolation
    click_rewind_ms: float = 70.0    # fallback rewind when Smart Aim Lock did not engage
    release_settle_ms: float = 110.0
    yield_to_physical_s: float = 1.2 # physical mouse wins for this long


@dataclass
class FingerConfig:
    mode: str = "auto"               # auto | manual | lock
    finger: str = "index"            # manual/default finger
    hand: str = "auto"               # auto | left | right
    switch_delay_ms: float = 250.0
    switch_confidence: float = 0.6
    two_hand_mode: str = "off"       # off | action_hand
    # Per-finger pinch thresholds (normalized by hand scale): [engage, release]
    pinch_thresholds: dict[str, list[float]] = field(default_factory=lambda: {
        "index": [0.26, 0.36], "middle": [0.26, 0.36],
        "ring": [0.28, 0.38], "little": [0.30, 0.40]})
    thumb_press: list[float] = field(default_factory=lambda: [0.28, 0.40])
    # A pinch on a non-pointer finger only engages when that finger is at
    # least this extended: a fingertip curled into the palm often projects
    # next to the resting thumb, and must never fire a click by itself.
    pinch_min_extension: float = 0.2
    # ... unless the fingertip is still this far from its own knuckle (hand
    # scales): a deep, deliberate pinch bends the finger a lot but keeps the
    # tip out of the palm, unlike a finger folding away during a switch.
    pinch_min_reach: float = 0.45
    curl_freeze: float = 0.45        # active finger below this extension freezes the cursor


@dataclass
class GestureConfig:
    # Click style is an explicit user choice, never inferred from the pose:
    #   pinch_pointer ("Normal")     thumb+pointer = left, next fingers = right, middle
    #   pinch_free    ("Pinch Free") pointer stays free; thumb+next finger = left ...
    click_style: str = "pinch_pointer"
    # Click interaction mode (v1.1.2) - separate from the click style, which
    # only decides which contact means which button:
    #   instant  Instant Auto-Click (default): a confirmed pinch completes a
    #            click at once (down + up); holding never drags or repeats
    #   drag     Drag Mode: pinch and hold hold_to_drag_s -> button down and
    #            drag; release sooner = nothing; drag_one_shot returns to
    #            instant after one drag
    #   classic  v1.1.1: button down at the pinch, up at the release, held
    #            hold_to_drag_s = drag (the click completes on release)
    click_mode: str = "instant"
    drag_one_shot: bool = True
    # Instant click: while the pinch is still held, the cursor stays on the
    # clicked pixel; it follows the hand again after this long or once the
    # whole hand moves (aim_cancel_move). A new click still needs a release.
    instant_hold_follow_s: float = 0.8
    # Pinch recognition preset (see PINCH_PRESETS): fast | balanced | stable | custom
    pinch_sensitivity: str = "balanced"
    bindings: dict[str, str] = field(default_factory=lambda: {
        "pinch_primary": "left_click_drag",
        "pinch_secondary": "right_click",
        "pinch_tertiary": "middle_click",
        "two_finger_pose": "scroll",
        "fist": "clutch",
        "swipe_left": "forward",
        "swipe_right": "back",
        "two_hand_pinch": "zoom",
        "open_palm_hold": "none",
    })
    # Smart Aim Lock: as soon as a mapped thumb contact starts *closing*,
    # the cursor is anchored where it was when the closing began, so the
    # finger flexion of the pinch cannot drag the click off a small target.
    aim_lock: bool = True
    aim_close_speed: float = 1.5     # hand scales / s of thumb-finger convergence that arms the lock
    aim_arm_dist: float = 0.85       # ... only when the contact is already this close (hand scales)
    aim_arm_drop: float = 0.12       # ... and it has closed by this much over >= 2 frames (v1.1.2)
    aim_near_ratio: float = 1.35     # or within this multiple of the engage threshold
    aim_max_rewind_ms: float = 250.0 # never anchor further back than this
    aim_hand_guard: float = 0.08     # never anchor before the hand (knuckles) was this far away
    aim_cancel_move: float = 0.35    # whole-hand motion (hand scales) that cancels the lock: re-aiming
    aim_max_hold_s: float = 0.5      # lock without a confirmed pinch is released after this (v1.1.1: 0.6)
    aim_stall_s: float = 0.15        # ... or once the contact stopped closing in for this long (v1.1.2)
    aim_release_ratio: float = 2.4   # lock released when the contact re-opens past this x engage
    # Settled contact (v1.1.2): a contact that closed in from an open hand
    # and stopped within this multiple of its engage threshold is a pinch
    # (real fingertip landmarks often stop just above the threshold).
    contact_settle_ratio: float = 1.35
    contact_settle_speed: float = 0.8   # hand scales / s: below this the contact has stopped
    contact_settle_frames: int = 2      # frames it must stay stopped
    contact_settle_drop: float = 0.15   # it must have closed (and re-open to rearm) by this much
    contact_settle_margin: float = 0.3  # the next mapped contact must be this much farther (x engage)
    contact_fast_ratio: float = 0.7     # deep contact reached via the near zone: confirmed at once
    # Weight of the estimated depth axis in contact distances (world
    # landmarks). Synthetic depth noise of 6-10 mm on the fingertips made
    # the full 3D distance miss pinches and double-click in synthetic checks.
    contact_depth_weight: float = 0.5
    instant_min_interval_s: float = 0.12 # Instant: the same contact cannot click twice faster than this (flicker)
    # "custom" preset values (the named presets override these four):
    press_frames: int = 2            # minimum frames of contact evidence
    pinch_confirm_ms: float = 45.0   # contact evidence span after the threshold crossing
    contact_lowpass_hz: float = 14.0
    release_frames: int = 2
    pose_frames: int = 4             # scroll / clutch pose stability
    click_max_hold_s: float = 0.35   # (v1.0, unused: any release before the drag threshold clicks)
    hold_to_drag_s: float = 1.2      # owner's choice (v1.1.2; 1.5 s in v1.1.1): Drag Mode / classic hold
    drag_move_threshold: float = 0.22  # (v1.0, unused: hand motion no longer starts a drag)
    drag_release_extra: float = 0.06   # wider release threshold while dragging
    click_debounce_s: float = 0.08
    double_click_s: float = 0.5      # sticky double click: Windows' default double-click time
    sticky_double_click_px: float = 14.0
    long_press_right_click_thumb: bool = True
    long_press_s: float = 0.8
    scroll_style: str = "joystick"   # joystick | direct
    scroll_dead_zone: float = 0.12   # hand-scale units
    scroll_full: float = 0.75
    scroll_speed: float = 18.0       # wheel notches / s at full deflection
    scroll_direct_gain: float = 14.0 # notches per hand-scale of motion
    scroll_axis_lock: bool = True
    scroll_accel: float = 1.6        # joystick response exponent (1 = linear)
    scroll_h_bias: float = 1.5       # horizontal needs this much more motion than vertical
    natural_scroll: bool = False
    smooth_wheel: bool = True        # sub-notch wheel deltas (False = whole 120 notches)
    swipe_min_dist: float = 1.6      # hand scales
    swipe_max_time_s: float = 0.45
    swipe_cooldown_s: float = 0.9
    zoom_step: float = 0.12          # relative hand distance change per zoom notch
    tracking_loss_grace_s: float = 0.25
    auto_disarm_s: float = 0.0       # 0 = never auto-disarm when hand absent
    dwell_click: bool = False        # dwell-to-click for the mouse (accessibility)
    dwell_click_s: float = 1.0
    # The 🤟 pose (thumb, index, little finger out; middle, ring curled) held
    # this long toggles the Air Keyboard. (v1.1's five-fingertip cluster was
    # removed in v1.1.1: unreliable, and its arbitration blocked real pinches.)
    keyboard_gesture: bool = True
    keyboard_gesture_s: float = 0.4  # owner's choice (v1.1.2; 0.7 s in v1.1.1)
    keyboard_gesture_cooldown_s: float = 0.8
    # Pinky Drag (v1.1.3): only the little finger extended (thumb, index,
    # middle, ring curled) held pinky_drag_s = left button down at the aimed
    # point; the hand (knuckles) drags; curling the little finger drops.
    # Works alongside Instant Auto-Click - no mode switch.
    pinky_drag: bool = True
    pinky_drag_s: float = 0.2
    pinky_release_frames: int = 2    # frames without the pose before the drop (noise guard)


@dataclass
class KeyboardConfig:
    layout: str = "full"             # full | compact
    language: str = "en"             # en | vi (AirDesk Telex) | raw (system IME)
    selection: str = "pinch"         # pinch | dwell | both
    dwell_ms: float = 650.0
    dwell_tolerance_px: float = 18.0
    repeat_backspace: bool = True
    two_hand_typing: bool = False    # (v1.0 second-hand typing; superseded by dual_hand)
    # Dual-Hand Air Keyboard (v1.1.3): while the keyboard is open, each hand
    # gets its own miniature hand cursor over the keyboard (left = blue,
    # right = mint) and types with its own pinch; no desktop mouse input.
    dual_hand: bool = True
    hand_cursors: bool = True
    # Pinch that types: "same" = the desktop click style (Normal / Pinch Free)
    kb_click_style: str = "same"     # same | pinch_pointer | pinch_free
    # One-shot modifiers: Ctrl / Shift / Alt / Win arm for the next key and
    # clear after it; tapping an armed modifier again cancels it.
    # False = v1.0 latch cycle (one-shot -> locked -> off).
    one_shot_modifiers: bool = True
    modifier_timeout_s: float = 10.0 # an armed modifier nobody used is cleared after this
    key_repeat_delay_s: float = 0.7  # held pinch on Backspace / Del / arrows repeats after this
    opacity: float = 0.94
    scale: float = 1.0               # (v1.0, unused)
    size_pct: int = 70               # overall size (both dimensions), shown on the title bar
    width_pct: int = 100             # extra width factor (narrower / wider keys)
    suggestions: bool = True
    privacy_mode: bool = False
    vi_tone_style: str = "classic"   # classic (hòa) | modern (hoà)
    x: int = -1                      # logical px, -1 = auto (bottom centre)
    y: int = -1
    width: int = 0                   # (v1.0, unused: size_pct / width_pct decide the size)
    height: int = 0


@dataclass
class SafetyConfig:
    emergency_hotkey: str = "ctrl+alt+f12"
    arm_hotkey: str = "ctrl+alt+f11"
    keyboard_hotkey: str = "ctrl+alt+f10"
    precision_hotkey: str = "ctrl+alt+f9"
    drag_hotkey: str = "ctrl+alt+f8"   # Drag Mode on/off (v1.1.2)
    autostart: bool = False          # input still always starts DISARMED


@dataclass
class UiConfig:
    theme: str = "dark"              # dark | light
    show_preview: bool = True
    preview_fps: int = 30
    first_run_done: bool = False
    minimize_to_tray: bool = True
    feedback_marker: bool = True     # aim / pressed / clicked ring, keyboard-gesture progress, scroll badge


@dataclass
class Settings:
    camera: CameraConfig = field(default_factory=CameraConfig)
    pointer: PointerConfig = field(default_factory=PointerConfig)
    finger: FingerConfig = field(default_factory=FingerConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    keyboard: KeyboardConfig = field(default_factory=KeyboardConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    profile: str = "Default"
    version: int = 5

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        old = int(data.get("version", 1)) if isinstance(data, dict) else 1
        s = _merge(cls(), data)
        if old < 2:
            # v1.0 -> v1.1: the keyboard is sized by size/width percentages.
            s.keyboard.width = s.keyboard.height = 0
        g = s.gesture
        if (old < 2 or (old == 2 and abs(g.hold_to_drag_s - 1.0) < 1e-6)
                or (old == 3 and abs(g.hold_to_drag_s - 1.5) < 1e-6)):
            # Inherited defaults (v1.0 0.4 s, v1.1 1.0 s, v1.1.1 1.5 s) -> the
            # owner's 1.2 s (v1.1.2). A value the user changed is kept.
            g.hold_to_drag_s = 1.2
        if old < 4 and abs(g.keyboard_gesture_s - 0.7) < 1e-6:
            g.keyboard_gesture_s = 0.4          # v1.1.1 default -> 0.4 s
        if old < 4 and abs(g.aim_max_hold_s - 0.6) < 1e-6:
            g.aim_max_hold_s = 0.5
        if old < 5 and g.click_mode == "drag":
            # v1.1.3: dragging no longer needs a mode (Pinky Drag); the
            # pinch-hold Drag Mode stays available as a legacy option.
            g.click_mode = "instant"
        s.version = 5
        return s

    def save(self, path: Path | None = None) -> Path:
        path = path or app_dir() / "settings.json"
        if path.exists():
            # Rollback: keep the last file written by an older AirDesk once.
            try:
                old = int(json.loads(path.read_text(encoding="utf-8")).get("version", 1))
                bak = path.with_name(f"settings.v{old}.bak.json")
                if old < self.version and not bak.exists():
                    bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                pass
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or app_dir() / "settings.json"
        if not path.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:  # corrupt file must never brick the app
            log.exception("settings unreadable, using defaults (backup kept)")
            path.replace(path.with_suffix(".corrupt.json"))
            return cls()

    # ------------------------------------------------------------ profiles
    PROFILE_SECTIONS = ("camera", "pointer", "finger", "gesture")

    def export_profile(self) -> dict[str, Any]:
        return {k: asdict(getattr(self, k)) for k in self.PROFILE_SECTIONS}

    def apply_profile(self, data: dict[str, Any]) -> None:
        keep_hold, keep_kb = self.gesture.hold_to_drag_s, self.gesture.keyboard_gesture_s
        for k in self.PROFILE_SECTIONS:
            if k in data:
                setattr(self, k, _merge(getattr(self, k), data[k]))
        g = data.get("gesture", {"pinch_sensitivity": 1, "double_click_s": 1, "click_mode": 1})
        hold = float(g.get("hold_to_drag_s", 1.0))
        if ("pinch_sensitivity" not in g
                or ("double_click_s" not in g and abs(hold - 1.0) < 1e-6)
                or ("click_mode" not in g and abs(hold - 1.5) < 1e-6)):
            # v1.0 profile, or a v1.1 / v1.1.1 profile carrying that version's
            # default: keep the current drag hold (1.2 s since v1.1.2).
            self.gesture.hold_to_drag_s = keep_hold
        if "click_mode" not in g and abs(float(g.get("keyboard_gesture_s", 0.7)) - 0.7) < 1e-6:
            self.gesture.keyboard_gesture_s = keep_kb


# Pinch recognition presets: (press_frames, confirm_ms, contact low-pass Hz,
# release_frames). The confirmation span is measured using frame timestamps.
PINCH_PRESETS: dict[str, tuple[int, float, float, int]] = {
    "fast": (1, 20.0, 20.0, 1),
    "balanced": (2, 45.0, 14.0, 2),
    "stable": (3, 70.0, 10.0, 2),
}


def pinch_timing(g: "GestureConfig") -> tuple[int, float, float, int]:
    return PINCH_PRESETS.get(g.pinch_sensitivity, (g.press_frames, g.pinch_confirm_ms,
                                                   g.contact_lowpass_hz, g.release_frames))


def _merge(obj: Any, data: Any) -> Any:
    """Recursively overlay known keys from `data` onto dataclass `obj`."""
    if not dataclasses.is_dataclass(obj) or not isinstance(data, dict):
        return obj
    for f in fields(obj):
        if f.name not in data:
            continue
        cur = getattr(obj, f.name)
        val = data[f.name]
        if dataclasses.is_dataclass(cur):
            setattr(obj, f.name, _merge(cur, val))
        elif isinstance(cur, dict) and isinstance(val, dict):
            merged = dict(cur)
            merged.update(val)
            setattr(obj, f.name, merged)
        elif cur is None or isinstance(val, type(cur)) or (
                isinstance(cur, float) and isinstance(val, int)):
            setattr(obj, f.name, type(cur)(val) if isinstance(cur, float) else val)
    return obj


def profiles_dir() -> Path:
    p = app_dir() / "profiles"
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_profiles() -> list[str]:
    return sorted(p.stem for p in profiles_dir().glob("*.json"))


def save_profile(name: str, settings: Settings) -> Path:
    safe = "".join(c for c in name if c.isalnum() or c in " -_").strip() or "Profile"
    path = profiles_dir() / f"{safe}.json"
    path.write_text(json.dumps(settings.export_profile(), indent=2), encoding="utf-8")
    return path


def load_profile(name: str, settings: Settings) -> bool:
    path = profiles_dir() / f"{name}.json"
    if not path.exists():
        return False
    settings.apply_profile(json.loads(path.read_text(encoding="utf-8")))
    settings.profile = name
    return True
