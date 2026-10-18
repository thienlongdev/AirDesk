"""v1.1 Air Keyboard: compact default size, size/width controls, clamping,
persistence and hit-test alignment after resizing."""

import pytest

from airdesk.config import KeyboardConfig, Settings
from airdesk.core.keyboard_layout import (SIZE_RANGE, TOOLBAR, WIDTH_RANGE, clamp_rect, hit, keyboard_size,
                                          layout_rects)


def test_compact_default_is_smaller_than_v10():
    w, h = keyboard_size(1920, 1032, "full")
    assert 0.63 <= w / 1920 <= 0.70                       # ~65 % of the work-area width
    assert h < 1920 * 0.56 / 2.55                         # v1.0: 56 % wide at aspect 2.55
    assert h / 1032 <= 0.30


def test_size_scales_both_and_width_only_width():
    w0, h0 = keyboard_size(1920, 1032, "full", 70, 100)
    w1, h1 = keyboard_size(1920, 1032, "full", 100, 100)
    assert w1 > w0 and h1 > h0
    wn, hn = keyboard_size(1920, 1032, "full", 70, 80)
    ww, hw = keyboard_size(1920, 1032, "full", 70, 130)
    assert wn < w0 < ww and hn == h0 == hw


@pytest.mark.parametrize("work", [(1280, 680), (1920, 1032), (2560, 1400), (3840, 2112)])
@pytest.mark.parametrize("layout", ["full", "compact", "symbols"])
def test_extremes_stay_usable_and_on_screen(work, layout):
    rects = layout_rects(layout)
    for size in SIZE_RANGE:
        for width in WIDTH_RANGE:
            w, h = keyboard_size(*work, layout, size, width)
            assert w <= work[0] and h <= work[1] * 0.6
            keys = [r for r in rects if r.key not in TOOLBAR]
            bar = [r for r in rects if r.key in TOOLBAR]
            assert min(r.h for r in keys) * h >= 24 and min(r.w for r in keys) * w >= 24
            assert min(r.h for r in bar) * h >= 17, (work, layout, size, width)


def test_clamp_keeps_window_inside_work_area():
    area = (0, 0, 1920, 1032)
    assert clamp_rect(1800, 900, 1248, 279, area) == (672, 753)
    assert clamp_rect(-500, -40, 1248, 279, area) == (0, 0)
    assert clamp_rect(2400, 300, 900, 200, (1920, 0, 2560, 1400)) == (2400, 300)   # second monitor


def test_hit_testing_matches_painted_keys_at_every_size():
    for layout in ("full", "compact"):
        rects = layout_rects(layout)
        for size in (40, 70, 130):
            for width in (60, 100, 150):
                w, h = keyboard_size(1920, 1032, layout, size, width)
                for r in rects:      # pixel centre of each painted key hits that key
                    cx, cy = (r.x + r.w / 2) * w, (r.y + r.h / 2) * h
                    assert hit(rects, cx / w, cy / h) is r


def test_toolbar_has_required_controls():
    for k in ("move", "size-", "sizepct", "size+", "width-", "width+", "reset", "hide"):
        assert k in TOOLBAR


def test_keyboard_settings_persist(tmp_path):
    s = Settings()
    s.keyboard.size_pct, s.keyboard.width_pct, s.keyboard.x, s.keyboard.y = 90, 120, 300, 640
    s.keyboard.layout = "compact"
    back = Settings.load(s.save(tmp_path / "s.json"))
    k = back.keyboard
    assert (k.size_pct, k.width_pct, k.x, k.y, k.layout) == (90, 120, 300, 640, "compact")


# ------------------------------------------------------------- Qt overlay
@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_overlay_title_bar_controls_resize_and_persist(qapp):
    from airdesk.ui.keyboard_overlay import KeyboardOverlay
    cfg = KeyboardConfig(suggestions=False)
    kb = KeyboardOverlay(cfg)            # never shown: no window appears, no input is sent
    saved = []
    kb.config_changed.connect(lambda: saved.append((cfg.size_pct, cfg.width_pct)))
    area = kb._work_area()
    w0, h0 = kb.width(), kb.height()
    assert (w0, h0) == keyboard_size(area.width(), area.height(), "full", 70, 100)
    kb._on_press("size-", 0)
    assert cfg.size_pct == 60 and kb.width() < w0 and kb.height() < h0
    kb._on_press("width+", 0)
    kb._on_press("width+", 0)
    assert cfg.width_pct == 120 and kb.height() == keyboard_size(area.width(), area.height(), "full", 60, 120)[1]
    for _ in range(20):
        kb._on_press("width+", 0)
    assert cfg.width_pct == WIDTH_RANGE[1] and kb.width() <= area.width()
    g = kb.frameGeometry()
    assert area.contains(g.topLeft()) and area.contains(g.bottomRight())
    kb._on_press("reset", 0)
    assert (cfg.size_pct, cfg.width_pct, cfg.x, cfg.y) == (70, 100, -1, -1)
    assert (kb.width(), kb.height()) == (w0, h0)
    assert saved and kb._label("sizepct") == "70%"
    kb._on_press("layout", 0)
    assert cfg.layout == "compact" and kb.width() < w0
    kb.deleteLater()
