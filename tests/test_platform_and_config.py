import math

import numpy as np
import pytest

from airdesk.config import Settings, list_profiles, load_profile, save_profile
from airdesk.core.filters import AdaptiveKalman2D, OneEuroFilter2D, PassThrough2D
from airdesk.core.pointer import PointerEngine
from airdesk.core.screen import Rect, ScreenLayout
from airdesk.platform import dispatcher as dmod
from airdesk.platform.hotkeys import parse_hotkey


def test_settings_roundtrip_and_corrupt_file(tmp_path):
    s = Settings()
    s.pointer.mode = "relative"
    s.finger.pinch_thresholds["ring"] = [0.2, 0.3]
    p = tmp_path / "settings.json"
    s.save(p)
    s2 = Settings.load(p)
    assert s2.pointer.mode == "relative" and s2.finger.pinch_thresholds["ring"] == [0.2, 0.3]
    p.write_text("{not json", encoding="utf-8")
    s3 = Settings.load(p)                      # never bricks the app
    assert s3.pointer.mode == "direct"
    assert (tmp_path / "settings.corrupt.json").exists()


def test_unknown_and_mistyped_keys_are_ignored():
    s = Settings.from_dict({"pointer": {"mode": "precision", "bogus": 1, "beta": "fast"}, "nope": {}})
    assert s.pointer.mode == "precision" and s.pointer.beta == 15.0


def test_profiles():
    s = Settings()
    s.pointer.box_w = 0.33
    save_profile("Desk A", s)
    assert "Desk A" in list_profiles()
    t = Settings()
    assert load_profile("Desk A", t) and t.pointer.box_w == 0.33 and t.profile == "Desk A"


def test_multimonitor_clamp_avoids_dead_space():
    # 1920x1080 primary + 1280x1024 to the right, top-aligned: dead space below the 2nd.
    lay = ScreenLayout((Rect(0, 0, 1920, 1080), Rect(1920, 0, 3200, 1024)))
    assert lay.virtual == Rect(0, 0, 3200, 1080)
    assert lay.clamp(2500, 1060) == (2500, 1023)      # pulled onto the real monitor
    assert lay.clamp(100, 100) == (100, 100)


def test_direct_mapping_spans_selected_monitor():
    lay = ScreenLayout((Rect(-1280, 0, 0, 1024), Rect(0, 0, 1920, 1080)), primary=1)
    s = Settings()
    s.pointer.filter = "none"
    s.pointer.monitor = "primary"
    pe = PointerEngine(s.pointer, lay)
    c = s.pointer
    left = pe.update((c.box_cx - c.box_w / 2, c.box_cy), 0.0)
    pe2 = PointerEngine(s.pointer, lay)
    right = pe2.update((c.box_cx + c.box_w / 2, c.box_cy), 0.0)
    assert left.x == pytest.approx(0, abs=1) and right.x == pytest.approx(1919, abs=1)


@pytest.mark.parametrize("spec,expected", [("ctrl+alt+f12", (0x3, 0x7B)), ("shift+win+k", (0xC, ord("K"))),
                                           ("ctrl+space", (0x2, 0x20))])
def test_hotkey_parse(spec, expected):
    assert parse_hotkey(spec) == expected


@pytest.mark.parametrize("bad", ["", "ctrl+", "hyper+a", "ctrl+f99"])
def test_hotkey_parse_rejects(bad):
    with pytest.raises(ValueError):
        parse_hotkey(bad)


def test_filters_reduce_jitter_and_track_motion():
    rng = np.random.default_rng(0)
    res = {}
    for f in (OneEuroFilter2D(0.6, 15), AdaptiveKalman2D(), PassThrough2D()):
        xs = [f(0.5 + rng.normal(0, 0.002), 0.5, i / 30)[0] for i in range(120)]
        res[f.name] = np.std(xs[30:])
        for i in range(120, 150):       # fast ramp: output must follow
            out = f(0.5 + (i - 120) * 0.01, 0.5, i / 30)[0]
        assert out > 0.7, f.name
    assert res["one_euro"] < res["none"] * 0.5
    assert res["kalman"] < res["none"]


class FakeWin32:
    def __init__(self):
        self.calls = []
        self.WHEEL_DELTA = 120
        self.pos = (0, 0)
        self.fail_buttons = False

    def virtual_screen(self):
        return Rect(0, 0, 1920, 1080)

    def mouse_move_abs(self, x, y, vs=None):
        self.pos = (x, y)
        self.calls.append(("move", x, y))

    def mouse_button(self, b, down, swapped=None):
        self.calls.append(("button", b, down))
        return not self.fail_buttons

    def mouse_click(self, b, n=1, swapped=None):
        self.calls.append(("click", b, n))
        return not self.fail_buttons

    def mouse_wheel(self, d, horizontal=False):
        self.calls.append(("wheel", d, horizontal))

    def cursor_pos(self):
        return self.pos

    def key_is_down(self, vk):
        return False

    def key_event(self, vk, down):
        self.calls.append(("key", vk, down))


def test_dispatcher_button_bookkeeping(monkeypatch):
    fake = FakeWin32()
    monkeypatch.setattr(dmod, "win32", fake)
    d = dmod.MouseDispatcher(hz=120, tau_ms=0)
    d.enabled = True
    d.move(100, 100)
    # Synchronous: the button is submitted inside the call (no tick needed)
    # and the call reports the outcome.
    assert d.button("left", True) is True
    assert [c for c in fake.calls if c[0] == "button"] == [("button", "left", True)]
    assert fake.calls.index(("move", 100, 100)) < fake.calls.index(("button", "left", True))  # snap first
    assert d.button("left", True) is False and "already held" in d.last_error   # never a second down
    assert [c for c in fake.calls if c[0] == "button"] == [("button", "left", True)]
    d.release_all("test")
    assert fake.calls[-1] == ("button", "left", False) and d.held_buttons() == ()
    assert d.button("left", False) is False and "released earlier: test" in d.last_error
    # Disabled: refused immediately with a reason, nothing injected, nothing replayed.
    d.enabled = False
    assert d.click("right") is False and "disarmed" in d.last_error
    d.enabled = True
    d._tick(0.2, 1 / 120)
    assert ("click", "right", 1) not in fake.calls
    # SendInput failure: reported, and the button is not recorded as held.
    fake.fail_buttons = True
    assert d.button("right", True) is False and "SendInput rejected" in d.last_error
    assert d.held_buttons() == ()


def test_dispatcher_wheel_accumulates_fractional_notches(monkeypatch):
    fake = FakeWin32()
    monkeypatch.setattr(dmod, "win32", fake)
    d = dmod.MouseDispatcher(smooth_wheel=False)
    d.enabled = True
    d.move(5, 5)
    for i in range(10):
        d.scroll(0, 0.25)
        d._tick(i / 100, 0.01)
    wheels = [c for c in fake.calls if c[0] == "wheel"]
    assert sum(w[1] for w in wheels) == 240 and all(w[1] % 120 == 0 for w in wheels)
    assert math.isclose(d._wheel[1], 0.5)
