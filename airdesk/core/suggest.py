"""Offline word completion for the Air Keyboard.

Static frequency-ordered word lists shipped with the app (no learning, no
storage of what the user types). Vietnamese matching is accent-insensitive
on the prefix so "nguoi" and "ngườ" both suggest "người".
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache
from pathlib import Path

from ..resources import resource_path


def fold(s: str) -> str:
    """Lowercase and strip Vietnamese diacritics (đ -> d)."""
    s = s.lower().replace("đ", "d")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


@lru_cache(maxsize=4)
def _words(lang: str) -> tuple[str, ...]:
    path: Path = resource_path("data", f"words_{lang}.txt")
    try:
        lines = path.read_text(encoding="utf-8").split()
    except OSError:
        return ()
    seen: dict[str, None] = {}
    for w in lines:
        seen.setdefault(w.strip(), None)
    return tuple(w for w in seen if w)


class Suggester:
    def complete(self, prefix: str, lang: str, n: int = 3) -> list[str]:
        if not prefix or not prefix.isalpha() and lang == "en":
            return []
        words = _words("vi" if lang == "vi" else "en")
        out: list[str] = []
        low = prefix.lower()
        if lang == "vi":
            f = fold(prefix)
            exact = [w for w in words if w.startswith(low) and w != low]
            loose = [w for w in words if fold(w).startswith(f) and w != low and w not in exact]
            cands = exact + loose
        else:
            cands = [w for w in words if w.startswith(low) and w != low]
        for w in cands[:n]:
            out.append(_match_case(w, prefix))
        return out


def _match_case(word: str, prefix: str) -> str:
    if prefix.isupper() and len(prefix) > 1:
        return word.upper()
    if prefix[:1].isupper():
        return word[:1].upper() + word[1:]
    return word
