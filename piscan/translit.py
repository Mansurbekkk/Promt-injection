"""Uzbek Latin <-> Cyrillic transliteration (rule-based, deterministic).

Used to build the UZ-Cyrillic variant of every payload from its UZ-Latin form,
so the two differ only in script (Sec. 5 of the companion paper). Round-trip
conversion is used as a sanity check, as in the user's prior work.
"""
from __future__ import annotations

# Order matters: multi-character sequences first.
_LAT2CYR = [
    ("o‘", "ў"), ("o'", "ў"), ("oʻ", "ў"), ("g‘", "ғ"), ("g'", "ғ"), ("gʻ", "ғ"),
    ("sh", "ш"), ("ch", "ч"), ("ng", "нг"), ("yo", "ё"), ("yu", "ю"), ("ya", "я"),
    ("ye", "е"), ("ts", "ц"),
    ("a", "а"), ("b", "б"), ("d", "д"), ("e", "э"), ("f", "ф"), ("g", "г"),
    ("h", "ҳ"), ("i", "и"), ("j", "ж"), ("k", "к"), ("l", "л"), ("m", "м"),
    ("n", "н"), ("o", "о"), ("p", "п"), ("q", "қ"), ("r", "р"), ("s", "с"),
    ("t", "т"), ("u", "у"), ("v", "в"), ("x", "х"), ("y", "й"), ("z", "з"),
    ("’", "ъ"), ("'", "ъ"),
]


def _apply(text: str, table, lower_src: bool) -> str:
    out = []
    i, n = 0, len(text)
    src = text.lower() if lower_src else text
    while i < n:
        for a, b in table:
            if src.startswith(a, i):
                chunk = b
                # keep capitalisation of the first letter
                if text[i].isupper():
                    chunk = chunk[:1].upper() + chunk[1:]
                out.append(chunk)
                i += len(a)
                break
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


import re as _re

# Spans that a bilingual attacker keeps in Latin even when writing Cyrillic:
# URLs, e-mail addresses, and text inside quotes (exact literals the model must echo).
_PROTECT = _re.compile(
    r"""(https?://\S+ | www\.\S+ | [\w.+-]+@[\w.-]+\.\w+ | "[^"]*" | '[^']*')""",
    _re.VERBOSE,
)


def latin_to_cyrillic(text: str, preserve_literals: bool = True) -> str:
    """Transliterate Latin Uzbek to Cyrillic. With preserve_literals (default),
    URLs, e-mail addresses and quoted strings are left in Latin, as a real
    bilingual author would write them."""
    if not preserve_literals:
        return _apply(text, _LAT2CYR, lower_src=True)
    out, last = [], 0
    for m in _PROTECT.finditer(text):
        out.append(_apply(text[last:m.start()], _LAT2CYR, lower_src=True))
        out.append(m.group(0))          # keep literal as-is
        last = m.end()
    out.append(_apply(text[last:], _LAT2CYR, lower_src=True))
    return "".join(out)


_CYR2LAT = [
    ("ў", "o‘"), ("ғ", "g‘"), ("ш", "sh"), ("ч", "ch"), ("ё", "yo"), ("ю", "yu"),
    ("я", "ya"), ("ц", "ts"),
    ("а", "a"), ("б", "b"), ("д", "d"), ("э", "e"), ("ф", "f"), ("г", "g"),
    ("ҳ", "h"), ("и", "i"), ("ж", "j"), ("к", "k"), ("л", "l"), ("м", "m"),
    ("н", "n"), ("о", "o"), ("п", "p"), ("қ", "q"), ("р", "r"), ("с", "s"),
    ("т", "t"), ("у", "u"), ("в", "v"), ("х", "x"), ("й", "y"), ("з", "z"),
    ("ъ", "’"), ("е", "e"),
]


def cyrillic_to_latin(text: str) -> str:
    return _apply(text, _CYR2LAT, lower_src=True)


def round_trip_ok(latin: str) -> bool:
    """True if Latin -> Cyrillic -> Latin returns the input (ignoring case/spacing
    and the known Cyrillic 'е' = e/ye ambiguity)."""
    back = cyrillic_to_latin(latin_to_cyrillic(latin))
    norm = lambda s: s.lower().replace("ye", "e").replace("‘", "'").split()
    return norm(back) == norm(latin)
