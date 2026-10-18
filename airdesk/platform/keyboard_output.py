"""Air Keyboard output: turns virtual key ids into real Windows keyboard input.

Three language modes, never combined, so accents are applied exactly once:

* ``en``  - characters are inserted with KEYEVENTF_UNICODE (independent of
  the active keyboard layout); control keys use virtual-key taps.
* ``vi``  - AirDesk's own Telex composer (`core.telex`). Each letter updates
  the composed word and only the *difference* is emitted (backspaces +
  Unicode), the strategy used by mainstream Vietnamese IMEs. If a system
  Vietnamese IME is also active, use ``raw`` instead.
* ``raw`` - letters are sent as virtual-key strokes so an installed IME
  (Windows Vietnamese Telex, Unikey, EVKey, Japanese IME...) composes them.

Modifiers are *one-shot* (v1.1.3 default): a tap arms Ctrl / Shift / Alt /
Win for the next key, which then goes out as one real virtual-key chord
(Ctrl down, C down, C up, Ctrl up - a single SendInput batch) and the
modifiers clear; tapping an armed modifier again cancels it; several can be
armed together (Ctrl+Shift+S). An armed modifier nobody uses expires after
``modifier_timeout_s``. The v1.0 latch cycle (one-shot -> locked -> off) is
kept behind ``one_shot=False``. Modifiers are never held down between
events, so they cannot get stuck; a few OS-wide shortcuts (Win+L, Ctrl+Esc,
Alt+Tab, ...) are never sent from the Air Keyboard.

Privacy: typed text is never logged or written anywhere. The current
word is held in memory only for composition and suggestions, and is
cleared on focus change; suggestions are disabled in privacy mode and in
detected password fields.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Protocol

from ..core.keyboard_layout import MODIFIERS, char_for
from ..core.suggest import Suggester
from ..core.telex import TelexComposer

log = logging.getLogger(__name__)

VK = {
    "backspace": 0x08, "tab": 0x09, "enter": 0x0D, "esc": 0x1B, "space": 0x20,
    "pgup": 0x21, "pgdn": 0x22, "end": 0x23, "home": 0x24, "left": 0x25, "up": 0x26,
    "right": 0x27, "down": 0x28, "del": 0x2E, "shift": 0x10, "ctrl": 0x11, "alt": 0x12,
    "win": 0x5B, "caps": 0x14,
}
VK.update({f"f{i}": 0x6F + i for i in range(1, 13)})
SHORTCUTS = {"copy": ("ctrl", "c"), "paste": ("ctrl", "v"), "cut": ("ctrl", "x"),
             "undo": ("ctrl", "z"), "selall": ("ctrl", "a"), "zoomin": ("ctrl", 0xBB),
             "zoomout": ("ctrl", 0xBD)}
# OS-wide shortcuts the Air Keyboard never sends (lock screen, Start menu,
# window switching, Task Manager): an armed modifier plus a stray key must
# not take the user out of the app they are typing in.
BLOCKED_COMBOS = {
    (frozenset({"win"}), "l"), (frozenset({"ctrl"}), "esc"), (frozenset({"alt"}), "tab"),
    (frozenset({"alt"}), "esc"), (frozenset({"ctrl", "shift"}), "esc"),
    (frozenset({"ctrl", "alt"}), "del"),
}
MOD_ORDER = ("ctrl", "alt", "shift", "win")
MOD_LABEL = {"ctrl": "Ctrl", "alt": "Alt", "shift": "Shift", "win": "Win"}
# Keys that end a composed word / suggestion context.
WORD_BREAK = {"space", "enter", "tab", "esc", "left", "right", "up", "down", "home",
              "end", "pgup", "pgdn", "del"}


class Injector(Protocol):
    def tap(self, vk: int, mods: tuple[int, ...] = ()) -> bool: ...
    def text(self, s: str) -> bool: ...
    def vk_for_char(self, ch: str) -> tuple[int, bool] | None: ...


class Win32Injector:
    def __init__(self) -> None:
        from . import win32
        self.w = win32

    def tap(self, vk: int, mods: tuple[int, ...] = ()) -> bool:
        return self.w.key_tap(vk, mods)

    def text(self, s: str) -> bool:
        return self.w.type_unicode(s)

    def vk_for_char(self, ch: str) -> tuple[int, bool] | None:
        return self.w.vk_for_char(ch)


@dataclass
class KeyResult:
    ok: bool = True
    kind: str = "key"            # key | char | modifier | none | blocked
    message: str = ""
    combo: str = ""              # e.g. "Ctrl+C" when a modifier chord was sent


@dataclass
class ModState:
    """0 = off, 1 = one-shot, 2 = locked (legacy cycle only)."""

    levels: dict[str, int] = field(default_factory=lambda: {m: 0 for m in MODIFIERS})
    one_shot: bool = False                  # True: tap = arm / cancel (v1.1.3 app default)
    since: dict[str, float] = field(default_factory=dict)

    def tap(self, m: str, now: float = 0.0) -> None:
        if self.one_shot:
            self.levels[m] = 0 if self.levels[m] else 1
        else:
            self.levels[m] = (self.levels[m] + 1) % 3
        self.since[m] = now

    def armed(self) -> list[str]:
        return [m for m in MOD_ORDER if self.levels.get(m)]

    def expire(self, now: float, timeout: float) -> bool:
        """Clear one-shot modifiers armed longer than `timeout` ago."""
        changed = False
        if timeout <= 0:
            return False
        for m, v in self.levels.items():
            if v == 1 and now - self.since.get(m, now) >= timeout:
                self.levels[m] = 0
                changed = True
        return changed

    def active(self, m: str) -> bool:
        return self.levels[m] > 0

    def consume(self) -> None:
        for m, v in self.levels.items():
            if v == 1:
                self.levels[m] = 0

    def clear(self) -> None:
        for m in self.levels:
            self.levels[m] = 0


class KeyboardOutput:
    def __init__(self, injector: Injector | None = None, language: str = "en",
                 tone_style: str = "classic", one_shot: bool = False) -> None:
        self.inj: Injector = injector or Win32Injector()
        self.language = language
        self.mods = ModState(one_shot=one_shot)
        self.caps = False
        self.composer = TelexComposer(tone_style)
        self.word = ""                    # current English word (suggestions only)
        self.suggester = Suggester()
        self.suggestions_enabled = True
        self.privacy = False
        self.sensitive = False            # target is a password field
        self.keys_sent = 0
        self.last_key_t = 0.0

    # ---------------------------------------------------------------- state
    def set_language(self, lang: str) -> None:
        self.commit()
        self.language = lang

    def set_tone_style(self, style: str) -> None:
        self.composer.style = style

    def commit(self) -> None:
        """End the current word (focus change, click, separator)."""
        self.composer.reset()
        self.word = ""

    def release_all(self) -> None:
        """Nothing is ever held down; clear latches and composition."""
        self.mods.clear()
        self.commit()

    @property
    def shift(self) -> bool:
        return self.mods.active("shift")

    def current_word(self) -> str:
        return self.composer.output if self.language == "vi" else self.word

    def suggestions(self, n: int = 3) -> list[str]:
        if not self.suggestions_enabled or self.privacy or self.sensitive:
            return []
        return self.suggester.complete(self.current_word(), self.language, n)

    # ---------------------------------------------------------------- input
    def clear_modifiers(self) -> None:
        self.mods.clear()

    def press(self, key: str) -> KeyResult:
        self.last_key_t = time.monotonic()
        if key in MODIFIERS:
            self.mods.tap(key, self.last_key_t)
            return KeyResult(kind="modifier")
        if key == "caps":
            self.caps = not self.caps
            return KeyResult(kind="modifier")
        if key in SHORTCUTS:
            mod, k = SHORTCUTS[key]
            vk = k if isinstance(k, int) else ord(k.upper())
            self.commit()
            ok = self.inj.tap(vk, (VK[mod],))
            self.mods.consume()
            return self._sent(ok)
        if len(key) == 1:
            return self._char(key)
        if key == "backspace":
            return self._backspace()
        if key in VK:
            if key in WORD_BREAK:
                self.commit()
            return self._chord(VK[key], key)
        return KeyResult(ok=False, kind="none")

    def accept_suggestion(self, word: str) -> KeyResult:
        cur = self.current_word()
        ok = True
        if cur:
            ok = self._backspaces(len(cur))
        ok = self.inj.text(word + " ") and ok
        self.commit()
        return self._sent(ok)

    # -------------------------------------------------------------- helpers
    def _mod_vks(self) -> tuple[int, ...]:
        return tuple(VK[m] for m in MOD_ORDER if self.mods.active(m))

    def _chord(self, vk: int, name: str) -> KeyResult:
        """One virtual key with every armed modifier, as one ordered batch
        (modifiers down, key down, key up, modifiers up). The one-shot
        modifiers clear whatever happens - sent, refused or blocked - so a
        failed dispatch can never leave Ctrl armed."""
        names = self.mods.armed()
        combo = "+".join([MOD_LABEL[m] for m in names] + [name.upper() if len(name) == 1 else name.capitalize()])
        if (frozenset(names), name) in BLOCKED_COMBOS:
            self.mods.consume()
            return KeyResult(ok=False, kind="blocked", combo=combo,
                             message=f"{combo} is a system shortcut - not sent from the Air Keyboard")
        ok = self.inj.tap(vk, tuple(VK[m] for m in names))
        self.mods.consume()
        r = self._sent(ok)
        r.combo = combo if names else ""
        return r

    def _sent(self, ok: bool) -> KeyResult:
        if ok:
            self.keys_sent += 1
            return KeyResult()
        return KeyResult(ok=False, kind="blocked",
                         message="Windows rejected the input (the target app may run as administrator)")

    def _backspaces(self, n: int) -> bool:
        ok = True
        for _ in range(n):
            ok = self.inj.tap(VK["backspace"]) and ok
        return ok

    def _diff(self, old: str, new: str) -> bool:
        p = 0
        while p < len(old) and p < len(new) and old[p] == new[p]:
            p += 1
        ok = self._backspaces(len(old) - p)
        if new[p:]:
            ok = self.inj.text(new[p:]) and ok
        return ok

    def _char(self, key: str) -> KeyResult:
        shortcut = self.mods.active("ctrl") or self.mods.active("alt") or self.mods.active("win")
        if shortcut:
            # Ctrl+C etc. must be real virtual-key chords, never Unicode.
            self.commit()
            vkr = self.inj.vk_for_char(key)
            if vkr is None:
                self.mods.consume()
                return KeyResult(ok=False, kind="none", message=f"no key for {key!r}")
            return self._chord(vkr[0], key)

        ch = char_for(key, self.shift, self.caps) or key
        self.mods.consume()
        if self.language == "vi" and ch.isascii() and ch.isalpha():
            old = self.composer.output
            new = self.composer.feed(ch)
            return self._sent(self._diff(old, new))
        if self.language == "raw":
            vkr = self.inj.vk_for_char(ch)
            if vkr is not None:
                vk, need_shift = vkr
                return self._sent(self.inj.tap(vk, (VK["shift"],) if need_shift else ()))
            return self._sent(self.inj.text(ch))
        # English (or a non-letter in Vietnamese mode).
        if ch.isalnum() or ch in "'-":
            self.word += ch
        else:
            self.commit()
        if self.language == "vi":
            self.composer.reset()
        return self._sent(self.inj.text(ch))

    def _backspace(self) -> KeyResult:
        mods = self._mod_vks()
        if self.language == "vi" and self.composer.active and not mods:
            old = self.composer.output
            new = self.composer.backspace() or ""
            if not new:
                self.composer.reset()
            return self._sent(self._diff(old, new))
        if mods:
            self.commit()
        else:
            self.word = self.word[:-1]
        return self._chord(VK["backspace"], "backspace")
