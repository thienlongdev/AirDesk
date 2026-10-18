"""Vietnamese Telex composition engine (original AirDesk implementation).

The composer keeps the current word as *base letters* (with circumflex,
breve, horn and đ, but without tone) plus one separate tone. The visible
word is rendered from that state every time, so the tone mark moves to
the correct vowel automatically as letters are added:

    h o a f      -> hòa      (classic)  / hoà (modern style)
    h o a f n    -> hoàn     (final consonant: tone moves to the last vowel)
    d d u o w n g f -> đường

Telex keys
    aa ee oo -> â ê ô      aw ow uw -> ă ơ ư     uow -> ươ     dd -> đ
    w alone  -> ư          s f r x j -> sắc huyền hỏi ngã nặng   z -> remove tone
    Repeating a modifier undoes it and types the letter (ass -> as, aaa -> aa).

Words that cannot be Vietnamese syllables are left alone (``class`` stays
``class``), and a word that *becomes* invalid after a transformation is
restored to the raw keystrokes (``tesst`` style English input is not
mangled). The caller diffs `output` before and after each key and emits
backspaces + Unicode text, the same strategy used by common Vietnamese IMEs.
"""

from __future__ import annotations

TONE_KEYS = {"s": 1, "f": 2, "r": 3, "x": 4, "j": 5}
_TONED = {
    "a": "aáàảãạ", "ă": "ăắằẳẵặ", "â": "âấầẩẫậ", "e": "eéèẻẽẹ", "ê": "êếềểễệ",
    "i": "iíìỉĩị", "o": "oóòỏõọ", "ô": "ôốồổỗộ", "ơ": "ơớờởỡợ", "u": "uúùủũụ",
    "ư": "ưứừửữự", "y": "yýỳỷỹỵ",
}
VOWELS = set(_TONED)
_DECOMPOSE = {c: (base, tone) for base, row in _TONED.items() for tone, c in enumerate(row)}
_PLAIN = {"ă": "a", "â": "a", "ê": "e", "ô": "o", "ơ": "o", "ư": "u", "đ": "d"}
_DIACRITIC = {"ă", "â", "ê", "ô", "ơ", "ư"}

INITIALS = ("ngh", "ng", "nh", "ch", "gh", "gi", "kh", "ph", "qu", "th", "tr",
            "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "q", "r", "s",
            "t", "v", "x")
FINALS = {"", "c", "ch", "m", "n", "ng", "nh", "p", "t"}
# Vowel clusters with diacritics stripped (intermediate typing states included).
CLUSTERS = {
    "a", "e", "i", "o", "u", "y", "ai", "ao", "au", "ay", "eo", "eu", "ia", "ie",
    "iu", "oa", "oe", "oi", "oo", "ua", "ue", "ui", "uo", "uu", "uy", "ye", "ieu",
    "oai", "oay", "oeo", "uay", "uoi", "uya", "uye", "uyu", "uou", "yeu", "uau",
}


def plain(s: str) -> str:
    return "".join(_PLAIN.get(c, c) for c in s)


def split_syllable(base: str) -> tuple[str, str, str] | None:
    """Split lowercase base letters into (initial, vowels, final) or None."""
    initial = ""
    for ini in INITIALS:
        if base.startswith(ini):
            initial = ini
            break
    rest = base[len(initial):]
    if initial in ("gi", "qu") and not any(c in VOWELS for c in rest):
        # "gi"/"gin": the i is the vowel; "qu" alone keeps u as the vowel
        initial = initial[0]
        rest = base[1:]
    i = 0
    while i < len(rest) and rest[i] in VOWELS:
        i += 1
    vowels, final = rest[:i], rest[i:]
    if any(c in VOWELS for c in final):
        return None
    return initial, vowels, final


def is_valid(base: str) -> bool:
    """Could `base` be (a prefix of) a Vietnamese syllable?"""
    if not base:
        return True
    parts = split_syllable(base)
    if parts is None:
        return False
    initial, vowels, final = parts
    if not vowels:
        return not final and (initial in INITIALS or initial == "")
    if plain(vowels) not in CLUSTERS:
        return False
    if final not in FINALS:
        return False
    return True


def tone_index(base: str, style: str = "classic") -> int | None:
    """Index in `base` of the vowel that carries the tone mark."""
    parts = split_syllable(base)
    if parts is None:
        return None
    initial, vowels, final = parts
    if not vowels:
        return None
    start = len(initial)
    marked = [i for i, c in enumerate(vowels) if c in _DIACRITIC]
    if marked:
        return start + marked[-1]
    if final:
        return start + len(vowels) - 1
    if len(vowels) == 1:
        return start
    if len(vowels) == 3:
        return start + 1
    if style == "modern" and vowels in ("oa", "oe", "uy"):
        return start + 1
    return start


def render(base: list[str], upper: list[bool], tone: int, style: str = "classic",
           open_uo: bool = False) -> str:
    chars = list(base)
    parts = split_syllable("".join(base))
    if open_uo and parts is not None and parts[1] == "ươ" and not parts[2]:
        # A typed "uow" that ends the syllable is "uơ" (thuở, huơ); it
        # becomes "ươ" as soon as a final or u/i follows (người, hươu).
        chars[len(parts[0])] = "u"
    if tone:
        idx = tone_index("".join(chars), style)
        if idx is not None:
            chars[idx] = _TONED[chars[idx]][tone]
    return "".join(c.upper() if u else c for c, u in zip(chars, upper))


class TelexComposer:
    def __init__(self, style: str = "classic") -> None:
        self.style = style
        self.reset()

    def reset(self) -> None:
        self.base: list[str] = []
        self.upper: list[bool] = []
        self.tone = 0
        self.raw: list[str] = []
        self.literal: set[str] = set()
        self._transformed = False
        self._open_uo = False

    @property
    def output(self) -> str:
        return render(self.base, self.upper, self.tone, self.style, self._open_uo)

    @property
    def active(self) -> bool:
        return bool(self.base)

    # ---------------------------------------------------------------- feed
    def feed(self, ch: str) -> str:
        """Feed one ASCII letter; returns the new rendered word."""
        kl, up = ch.lower(), ch.isupper()
        self.raw.append(ch)
        applied = False
        if kl not in self.literal and is_valid("".join(self.base)):
            if kl in TONE_KEYS:
                applied = self._tone(TONE_KEYS[kl], kl, up)
            elif kl == "z" and self.tone:
                self.tone = 0
                applied = True
            elif kl in "aeo":
                applied = self._hat(kl, up)
            elif kl == "w":
                applied = self._horn(up)
            elif kl == "d":
                applied = self._dd(up)
            if applied:
                self._transformed = True
        if not applied:
            self._append(kl, up)
            if self._transformed and not is_valid("".join(self.base)):
                self._restore_raw()
        return self.output

    def _append(self, c: str, up: bool) -> None:
        self.base.append(c)
        self.upper.append(up)

    def _cluster_range(self) -> tuple[int, int] | None:
        parts = split_syllable("".join(self.base))
        if parts is None or not parts[1]:
            return None
        start = len(parts[0])
        return start, start + len(parts[1])

    def _tone(self, tone: int, key: str, up: bool) -> bool:
        if self._cluster_range() is None:
            return False
        if self.tone == tone:          # repeat = undo, then type the letter
            self.tone = 0
            self.literal.add(key)
            self._undo_raw(key)
            self._append(key, up)
            return True
        self.tone = tone
        return True

    def _hat(self, v: str, up: bool) -> bool:
        rng = self._cluster_range()
        if rng is None:
            return False
        hat = {"a": "â", "e": "ê", "o": "ô"}[v]
        variants = {"a": ("a", "ă"), "e": ("e",), "o": ("o", "ơ")}[v]
        for i in range(rng[1] - 1, rng[0] - 1, -1):
            c = self.base[i]
            if c in variants:
                self.base[i] = hat
                return True
            if c == hat:               # aaa -> aa
                self.base[i] = v
                self.literal.add(v)
                self._undo_raw(v)
                self._append(v, up)
                return True
        return False

    def _horn(self, up: bool) -> bool:
        rng = self._cluster_range()
        if rng is None:
            # "w" with no vowel yet types ư (nhw -> như, w -> ư)
            if all(c not in VOWELS for c in self.base):
                self._append("ư", up)
                return True
            return False
        s, e = rng
        cl = "".join(self.base[s:e])
        pc = plain(cl)
        idx = pc.find("uo")
        if idx >= 0 and not (self.base[s + idx] == "ư" and self.base[s + idx + 1] == "ơ"):
            self.base[s + idx] = "ư"       # uo / ưo / uơ -> ươ
            self.base[s + idx + 1] = "ơ"
            self._open_uo = True
            return True
        # Undo: horn/breve already applied -> plain letters + literal w.
        if any(c in ("ă", "ơ", "ư") for c in cl):
            for i in range(s, e):
                if self.base[i] in ("ă", "ơ", "ư"):
                    self.base[i] = _PLAIN[self.base[i]]
            self.literal.add("w")
            self._undo_raw("w")
            self._append("w", up)
            return True
        if pc.startswith("ua") or pc in ("ui", "uu"):
            self.base[s] = "ư"
            return True
        for target, rep in (("a", "ă"), ("o", "ơ"), ("u", "ư")):
            for i in range(e - 1, s - 1, -1):
                if self.base[i] == target:
                    self.base[i] = rep
                    return True
        return False

    def _dd(self, up: bool) -> bool:
        if self.base and self.base[0] == "d":
            self.base[0] = "đ"
            return True
        if self.base and self.base[0] == "đ":
            self.base[0] = "d"
            self.literal.add("d")
            self._undo_raw("d")
            self._append("d", up)
            return True
        return False

    def _undo_raw(self, key: str) -> None:
        """A repeated modifier undid a transformation: the earlier keystroke
        was a command, not a letter, so drop it from the raw record (ass ->
        as, tesst -> test when the word is later restored)."""
        for i in range(len(self.raw) - 2, -1, -1):
            if self.raw[i].lower() == key:
                del self.raw[i]
                return

    def _restore_raw(self) -> None:
        self.base = [c.lower() for c in self.raw]
        self.upper = [c.isupper() for c in self.raw]
        self.tone = 0
        self.literal = set("sfrxjzaeowd")  # stay literal for the rest of the word
        self._transformed = False

    # ----------------------------------------------------------- backspace
    def backspace(self) -> str | None:
        """Delete one visible character; returns the new word or None if empty."""
        out = self.output
        if not out:
            return None
        self.load(out[:-1])
        return self.output

    def load(self, word: str) -> None:
        """Rebuild composer state from visible text (used after backspace)."""
        self.reset()
        for ch in word:
            low = ch.lower()
            base, tone = _DECOMPOSE.get(low, (low, 0))
            if tone:
                self.tone = tone
            self.base.append(base)
            self.upper.append(ch != low)
        self.raw = list(word)


def telex(text: str, style: str = "classic") -> str:
    """Convenience: transform a whole string word by word (for tests/benchmarks)."""
    out: list[str] = []
    comp = TelexComposer(style)
    for ch in text:
        if ch.isascii() and ch.isalpha():
            comp.feed(ch)
        else:
            out.append(comp.output)
            comp.reset()
            out.append(ch)
    out.append(comp.output)
    return "".join(out)
