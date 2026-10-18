"""Action vocabulary and the output interfaces the engine drives.

The interaction engine never touches Win32 directly; it talks to a
`MouseSink` and a `KeyboardSink`. Production code plugs in the Windows
dispatcher and the overlay keyboard controller; tests plug in recorders.
"""

from __future__ import annotations

from typing import Protocol

# Gesture -> action vocabularies shown in the Gesture Mapping page.
SLOT_ACTIONS = ("left_click_drag", "left_click", "right_click", "right_click_drag",
                "middle_click", "double_click", "triple_click", "back", "forward",
                "toggle_keyboard", "toggle_precision", "toggle_magnifier", "none")
POSE_ACTIONS = {
    "two_finger_pose": ("scroll", "none"),
    "fist": ("clutch", "none"),
    "swipe_left": ("forward", "back", "toggle_keyboard", "none"),
    "swipe_right": ("back", "forward", "toggle_keyboard", "none"),
    "two_hand_pinch": ("zoom", "none"),
    "open_palm_hold": ("none", "pause", "toggle_keyboard", "toggle_magnifier"),
}
GESTURE_LABELS = {
    "pinch_primary": "Primary contact (pinch / thumb press)",
    "pinch_secondary": "Secondary pinch",
    "pinch_tertiary": "Tertiary pinch",
    "two_finger_pose": "V pose (index + middle up) = Scroll Mode",
    "fist": "Closed fist",
    "swipe_left": "Open-palm swipe left",
    "swipe_right": "Open-palm swipe right",
    "two_hand_pinch": "Two-hand pinch spread",
    "open_palm_hold": "Open palm held 1.2 s",
}
DRAG_ACTIONS = {"left_click_drag": "left", "right_click_drag": "right"}
# Actions that are a real mouse button: down when the pinch is confirmed,
# up when it is released (drag variants become a drag after the hold).
BUTTON_ACTIONS = {**DRAG_ACTIONS, "left_click": "left", "right_click": "right", "middle_click": "middle"}


class MouseSink(Protocol):
    """Button and click calls return False when the input could not be
    submitted (with the reason in an optional `last_error` attribute); any
    other return value counts as submitted."""

    def move(self, x: float, y: float) -> None: ...
    def button(self, button: str, down: bool) -> bool | None: ...
    def click(self, button: str, count: int = 1) -> bool | None: ...
    def scroll(self, dx_notches: float, dy_notches: float) -> None: ...
    def xbutton(self, which: int) -> None: ...
    def zoom(self, notches: int) -> None: ...
    def release_all(self) -> None: ...
    def held_buttons(self) -> tuple[str, ...]: ...
    def cursor_pos(self) -> tuple[float, float] | None: ...


class KeyboardSink(Protocol):
    def is_visible(self) -> bool: ...
    def hit_test(self, x: float, y: float) -> str | None: ...
    def hover(self, hand: int, key: str | None, progress: float = 0.0) -> None: ...
    def press_key(self, key: str, hand: int = 0) -> bool: ...
    def release_all(self) -> None: ...


class NullKeyboard:
    def is_visible(self) -> bool:
        return False

    def hit_test(self, x: float, y: float) -> str | None:
        return None

    def hover(self, hand: int, key: str | None, progress: float = 0.0) -> None:
        pass

    def press_key(self, key: str, hand: int = 0) -> bool:
        return False

    def release_all(self) -> None:
        pass
