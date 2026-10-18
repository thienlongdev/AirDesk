import pytest

from airdesk.core.telex import TelexComposer, is_valid, telex

WORDS = [
    ("hoaf", "hòa"), ("hoafn", "hoàn"), ("dduowngf", "đường"), ("Vieejt Nam", "Việt Nam"),
    ("tieengs Vieetj", "tiếng Việt"), ("nguowif", "người"), ("quyeenf", "quyền"),
    ("gias", "giá"), ("giuwax", "giữa"), ("thuowr", "thuở"), ("muaf", "mùa"),
    ("khoer", "khỏe"), ("thuyr", "thủy"), ("nhw", "như"), ("owf", "ờ"), ("ddi", "đi"),
    ("chuyeenj", "chuyện"), ("ddepj", "đẹp"), ("trangw", "trăng"), ("Hoaf", "Hòa"),
    ("DDAAU", "ĐÂU"), ("luaatj", "luật"), ("toans", "toán"), ("hocj", "học"),
    ("quaf", "quà"), ("gif", "gì"), ("giowf", "giờ"), ("huowu", "hươu"),
    ("khuyur", "khuỷu"), ("tuwowng", "tương"), ("Truwowngf", "Trường"), ("nuwowcs", "nước"),
    ("cuar", "của"), ("mootj", "một"), ("xuaan", "xuân"), ("yeeu", "yêu"),
    ("dduwowcj", "được"), ("nawm", "năm"), ("ddaay", "đây"), ("phaanf", "phần"),
    ("mowis", "mới"), ("quoocs", "quốc"), ("giaf", "già"),
]


@pytest.mark.parametrize("keys,expected", WORDS)
def test_words(keys, expected):
    assert telex(keys) == expected


@pytest.mark.parametrize("keys,expected", [
    ("hoaf", "hoà"), ("khoer", "khoẻ"), ("thuyr", "thuỷ"), ("hoaj", "hoạ"), ("hoafn", "hoàn")])
def test_modern_tone_style(keys, expected):
    assert telex(keys, "modern") == expected


@pytest.mark.parametrize("keys,expected", [
    ("ass", "as"), ("aaa", "aa"), ("ddd", "dd"), ("tesst", "test"), ("aww", "aw")])
def test_repeat_undoes_modifier(keys, expected):
    assert telex(keys) == expected


@pytest.mark.parametrize("word", ["class", "xinh", "the", "window", "string"])
def test_non_vietnamese_words_are_not_mangled(word):
    assert telex(word) == word


def test_backspace_redraws_tone_position():
    c = TelexComposer()
    for ch in "dduowngf":
        c.feed(ch)
    seq = []
    while c.output:
        seq.append(c.backspace())
    assert seq == ["đườn", "đườ", "đư", "đ", ""]


def test_tone_moves_when_final_consonant_added():
    c = TelexComposer()
    for ch in "hoaf":
        c.feed(ch)
    assert c.output == "hòa"
    c.feed("n")
    assert c.output == "hoàn"


def test_z_removes_tone():
    assert telex("hoafz") == "hoa"


def test_validity():
    assert is_valid("nguoi") and is_valid("tr") and not is_valid("cl")
