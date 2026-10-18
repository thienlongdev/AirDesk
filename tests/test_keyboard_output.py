from airdesk.core.keyboard_layout import COMPACT, FULL, SYMBOLS, hit, layout_rects
from airdesk.platform.keyboard_output import VK, KeyboardOutput


class FakeEditor:
    """Injector that applies events to a text buffer like a simple edit box."""

    def __init__(self):
        self.buf = ""
        self.chords = []

    def tap(self, vk, mods=()):
        if mods:
            self.chords.append((vk, tuple(mods)))
            return True
        if vk == VK["backspace"]:
            self.buf = self.buf[:-1]
        elif vk == VK["space"]:
            self.buf += " "
        elif vk == VK["enter"]:
            self.buf += "\n"
        else:
            self.chords.append((vk, ()))
        return True

    def text(self, s):
        self.buf += s
        return True

    def vk_for_char(self, ch):
        if ch.isalpha():
            return ord(ch.upper()), ch.isupper()
        if ch.isdigit():
            return ord(ch), False
        return None


def typed(keys, lang="en"):
    ed = FakeEditor()
    out = KeyboardOutput(ed, language=lang)
    for k in keys:
        out.press(k)
    return ed, out


def test_english_letters_shift_and_caps():
    ed, _ = typed(["shift", "h", "i", "space", "caps", "o", "k", "caps", "!"])
    assert ed.buf == "Hi OK!"
    ed, _ = typed(["shift", "1", "shift", "/"])
    assert ed.buf == "!?"


def test_shift_lock_cycle():
    ed, out = typed(["shift", "shift", "a", "b", "shift", "c"])
    assert ed.buf == "ABc"


def test_ctrl_shortcut_is_a_vk_chord_not_text():
    ed, out = typed(["ctrl", "c", "x"])
    assert ed.chords == [(ord("C"), (VK["ctrl"],))]
    assert ed.buf == "x"            # ctrl was one-shot


def test_named_shortcut_keys():
    ed, _ = typed(["copy", "paste", "undo", "selall"])
    assert [c[0] for c in ed.chords] == [ord("C"), ord("V"), ord("Z"), ord("A")]


def test_vietnamese_composition_emits_minimal_diffs():
    ed, out = typed(list("dduowngf") + ["space"] + list("Vieejt"), lang="vi")
    assert ed.buf == "đường Việt"


def test_vietnamese_backspace_during_composition():
    ed, out = typed(list("hoafn") + ["backspace", "backspace"], lang="vi")
    assert ed.buf == "hò"
    out.press("a")
    assert ed.buf == "hòa"


def test_vietnamese_backspace_after_commit_is_plain():
    ed, out = typed(list("vieejt") + ["space", "backspace", "backspace"], lang="vi")
    assert ed.buf == "việ"


def test_raw_mode_sends_virtual_keys_for_system_ime():
    ed, out = typed(["a", "shift", "b"], lang="raw")
    assert ed.chords == [(ord("A"), ()), (ord("B"), (VK["shift"],))]
    assert ed.buf == ""             # nothing composed by AirDesk itself


def test_suggestion_replaces_current_word():
    ed, out = typed(list("hel"))
    sug = out.suggestions()
    assert "hello" in sug or "help" in sug
    out.accept_suggestion("hello")
    assert ed.buf == "hello "


def test_vietnamese_suggestion_and_privacy():
    ed, out = typed(list("nguow"), lang="vi")
    assert "người" in out.suggestions()
    out.privacy = True
    assert out.suggestions() == []


def test_release_all_clears_latches():
    ed, out = typed(["ctrl", "alt"])
    out.release_all()
    out.press("a")
    assert ed.buf == "a" and ed.chords == []


def test_layout_rows_and_hit_testing():
    for name, rows in (("full", FULL), ("compact", COMPACT), ("symbols", SYMBOLS)):
        rects = layout_rects(name)
        assert len(rects) == sum(len(r) for r in rows)
        for r in rects:   # every key centre hits itself
            assert hit(rects, r.x + r.w / 2, r.y + r.h / 2) is r
        assert hit(rects, 1.5, 0.5) is None
    keys = {r.key for r in layout_rects("full")}
    for k in ("a", "z", "1", "space", "enter", "backspace", "del", "shift", "caps", "ctrl",
              "alt", "tab", "esc", "left", "right", "up", "down", "home", "end", ";", "/"):
        assert k in keys, k
