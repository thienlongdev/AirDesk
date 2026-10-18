"""v1.1.3 one-shot modifiers, real chord dispatch, blocked system shortcuts,
safe cleanup, settings migration."""

from __future__ import annotations

import json

import pytest

from airdesk.config import Settings
from airdesk.platform import win32
from airdesk.platform.keyboard_output import VK, KeyboardOutput


class Rec:
    """Injector recording every chord / text call; can be told to fail."""

    def __init__(self):
        self.calls = []
        self.fail = False

    def tap(self, vk, mods=()):
        self.calls.append(("tap", vk, tuple(mods)))
        return not self.fail

    def text(self, s):
        self.calls.append(("text", s))
        return not self.fail

    def vk_for_char(self, ch):
        return (ord(ch.upper()), ch.isupper()) if ch.isalnum() else None


def out(**kw):
    r = Rec()
    return r, KeyboardOutput(r, one_shot=True, **kw)


def test_one_shot_ctrl_then_letter_is_one_real_chord_and_clears():
    r, o = out()
    o.press("ctrl")
    res = o.press("c")
    assert r.calls == [("tap", ord("C"), (VK["ctrl"],))] and res.combo == "Ctrl+C"
    o.press("v")
    assert r.calls[-1] == ("text", "v")              # Ctrl cleared after one key


@pytest.mark.parametrize("key", list("avxzysf"))
def test_common_ctrl_shortcuts(key):
    r, o = out()
    o.press("ctrl")
    o.press(key)
    assert r.calls == [("tap", ord(key.upper()), (VK["ctrl"],))]


def test_tapping_an_armed_modifier_again_cancels_it_without_typing():
    r, o = out()
    o.press("ctrl")
    o.press("ctrl")
    assert r.calls == [] and o.mods.armed() == []
    o.press("c")
    assert r.calls == [("text", "c")]


def test_multiple_modifiers_ctrl_shift_s_and_shift_enter():
    r, o = out()
    o.press("ctrl")
    o.press("shift")
    res = o.press("s")
    assert r.calls == [("tap", ord("S"), (VK["ctrl"], VK["shift"]))] and res.combo == "Ctrl+Shift+S"
    o.press("shift")
    res = o.press("enter")
    assert r.calls[-1] == ("tap", VK["enter"], (VK["shift"],)) and res.combo == "Shift+Enter"
    o.press("shift")
    o.press("a")
    assert r.calls[-1] == ("text", "A")               # Shift+letter = the capital, as text


@pytest.mark.parametrize("mods,key", [(["win"], "l"), (["ctrl"], "esc"), (["alt"], "tab"), (["alt"], "esc"),
                                      (["ctrl", "shift"], "esc")])
def test_os_wide_shortcuts_are_never_sent(mods, key):
    r, o = out()
    for m in mods:
        o.press(m)
    res = o.press(key)
    assert not res.ok and res.kind == "blocked" and r.calls == []
    assert o.mods.armed() == []                       # and nothing stays armed


def test_failed_dispatch_never_leaves_a_modifier_armed():
    r, o = out()
    r.fail = True
    o.press("ctrl")
    res = o.press("c")
    assert not res.ok and o.mods.armed() == []


def test_unused_modifier_expires():
    r, o = out()
    o.press("alt")
    t0 = o.mods.since["alt"]
    assert not o.mods.expire(t0 + 5, 10)
    assert o.mods.expire(t0 + 10.5, 10) and o.mods.armed() == []


def test_legacy_latch_cycle_still_available():
    r = Rec()
    o = KeyboardOutput(r)                             # one_shot=False: v1.0 cycle
    o.press("shift")
    o.press("shift")                                  # locked
    o.press("a")
    o.press("b")
    assert [c for c in r.calls if c[0] == "text"] == [("text", "A"), ("text", "B")]


def test_telex_and_unicode_unchanged_with_one_shot_modifiers():
    r, o = out(language="vi")
    for k in "vieejt":
        o.press(k)
    assert "".join(c[1] for c in r.calls if c[0] == "text").endswith("t")
    assert o.composer.output == "việt"


# --------------------------------------------------------------- Win32 chord
@pytest.mark.skipif(not win32.IS_WINDOWS, reason="Windows only")
def test_chord_order_and_partial_failure_cleanup(monkeypatch):
    sent = []

    def fake_send(seq):
        sent.append([(i.ki.wVk, bool(i.ki.dwFlags & win32.KEYEVENTF_KEYUP)) for i in seq])
        return 2 if len(sent) == 1 else len(seq)     # first batch only partly inserted

    monkeypatch.setattr(win32, "_send", fake_send)
    monkeypatch.setattr(win32, "key_is_down", lambda vk: False)
    assert win32.key_tap(ord("C"), (0x11,)) is False
    assert sent[0] == [(0x11, False), (ord("C"), False), (ord("C"), True), (0x11, True)]
    assert sent[1] == [(ord("C"), True), (0x11, True)]       # every pressed key released again


@pytest.mark.skipif(not win32.IS_WINDOWS, reason="Windows only")
def test_physically_held_modifier_is_left_alone(monkeypatch):
    sent = []
    monkeypatch.setattr(win32, "_send", lambda seq: sent.append([(i.ki.wVk, bool(i.ki.dwFlags & 2)) for i in seq])
                        or len(seq))
    monkeypatch.setattr(win32, "key_is_down", lambda vk: vk == 0x11)
    assert win32.key_tap(ord("V"), (0x11,))
    assert sent == [[(ord("V"), False), (ord("V"), True)]]   # the user's own Ctrl stays down


# ------------------------------------------------------------------ settings
def test_v4_settings_migrate_to_v5_with_backup(tmp_path):
    old = Settings().to_dict()
    old["version"] = 4
    old["gesture"]["click_mode"] = "drag"
    old["gesture"]["hold_to_drag_s"] = 1.3
    old["keyboard"]["size_pct"] = 90
    old["pointer"]["min_cutoff"] = 0.9
    for k in ("pinky_drag", "pinky_drag_s"):
        old["gesture"].pop(k)
    for k in ("dual_hand", "hand_cursors", "one_shot_modifiers", "kb_click_style"):
        old["keyboard"].pop(k)
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(old), encoding="utf-8")
    s = Settings.load(path)
    assert s.version == 5 and s.gesture.click_mode == "instant"
    assert s.gesture.hold_to_drag_s == 1.3 and s.keyboard.size_pct == 90 and s.pointer.min_cutoff == 0.9
    assert s.gesture.pinky_drag and s.gesture.pinky_drag_s == 0.2
    assert s.keyboard.dual_hand and s.keyboard.one_shot_modifiers and s.keyboard.kb_click_style == "same"
    s.save(path)
    assert json.loads((tmp_path / "settings.v4.bak.json").read_text(encoding="utf-8"))["version"] == 4
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 5


def test_v113_defaults():
    s = Settings()
    g, k = s.gesture, s.keyboard
    assert g.click_mode == "instant" and g.pinky_drag and g.pinky_drag_s == 0.2
    assert g.keyboard_gesture_s == 0.4 and g.hold_to_drag_s == 1.2
    assert k.dual_hand and k.hand_cursors and k.one_shot_modifiers and s.camera.num_hands == 2
