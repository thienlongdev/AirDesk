"""v1.1 UI logic without showing windows: Gesture Mapping contact table and
the Benchmark Lab small-target test scoring."""

from types import SimpleNamespace

import pytest

from airdesk.config import Settings


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_contact_table_reflects_style(qapp):
    from airdesk.ui.pages import contact_rows
    g = Settings().gesture
    normal = dict(contact_rows("Index", "pinch_pointer", g))
    assert normal["Thumb + index (pointer)"].startswith("Left click")
    assert normal["Thumb + middle"] == "Right click" and normal["Thumb + ring"] == "Middle click"
    assert normal["Thumb + little"] == "not assigned"
    free = dict(contact_rows("Index", "pinch_free", g))
    assert free["Thumb + index (pointer)"].startswith("not assigned")
    assert free["Thumb + middle"].startswith("Left click")
    assert free["Thumb + ring"] == "Right click" and free["Thumb + little"] == "Middle click"
    assert len(contact_rows("Thumb", "pinch_free", g)) == 2


def test_precision_test_scores_hits_neighbours_and_misses(qapp):
    from PySide6.QtCore import QPointF
    from airdesk.ui.benchmark import PrecisionTest
    ctx = SimpleNamespace(s=Settings(), runtime=SimpleNamespace(engine=SimpleNamespace(armed=False, paused=False)))
    t = PrecisionTest(ctx, n=3)
    t.resize(1600, 900)                    # never shown
    out = []
    t.finished.connect(out.append)

    def press(pt):
        t.mousePressEvent(SimpleNamespace(position=lambda: pt))

    for _ in range(3):
        t._next()
        r = t.row
        press(r[t.target_i - 1].center())                  # neighbour: wrong target
        press(QPointF(r[0].left() - 30, r[0].top()))       # nothing: miss
        press(r[t.target_i].center())                      # correct
        press(r[t.target_i].center())                      # accidental second press, row already gone
        t._last_press = None
    t.finish()
    s = out[-1]["summary"]
    assert s["trials"] == 3 and s["wrong_target_clicks"] == 3 and s["missed_clicks"] == 3
    assert s["accidental_double_clicks"] == 3
    assert out[-1]["input_source"].startswith("physical")
    assert {x["size_px"] for x in out[-1]["trials"]} == {12, 16, 20}
    assert all(x["error_px"] < 1 for x in out[-1]["trials"])
