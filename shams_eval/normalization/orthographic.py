"""Orthographic normalization tiers for the Arabic-script tiers of SHAMS.

Two tiers, applied symmetrically to references and hypotheses:

``strict``
    Surface-faithful. Unicode canonicalization (NFKC, which folds presentation
    forms and ligatures onto their canonical letters) and tatweel removal.
    Every orthographic contrast the reference makes is scored.

``neutral``
    ``strict`` plus the standard Arabic orthographic fold, which collapses the
    spelling conventions that carry no phonemic contrast in Levantine and that
    models spell inconsistently:

    ========================  ==============
    fold                      example
    ========================  ==============
    alif variants             أ إ آ ٱ → ا
    alif maqsuura             ى → ي
    taa marbuuta              ة → ه
    residual diacritics       ـً ـّ ـْ → ∅
    ========================  ==============

The strict–neutral gap is therefore a measurement in its own right: how much of
a system's error is spelling convention rather than recognition. Whisper-
Levantine, trained on normalized text, emits no أ إ آ ة ى at all, so under
``strict`` each collapsed grapheme costs a whole word and under ``neutral`` the
penalty disappears.
"""
from __future__ import annotations

import re
import unicodedata

TATWEEL = "ـ"

#: U+064B to U+0652 (fathatan to sukun) and U+0670 (superscript alif).
_DIACRITICS = re.compile(r"[ً-ْٰ]")
_ALEF = str.maketrans({c: "ا" for c in "إأٱآ"})
_ALEF_MAKSURA = str.maketrans({"ى": "ي"})
_TEH_MARBUTA = str.maketrans({"ة": "ه"})
#: Two compatibility characters NFKC does not decompose.
_CHAR_FIX = {"﷼": "ريال", "﷽": "بسم الله الرحمن الرحيم"}


def normalize_unicode(s: str) -> str:
    """Canonical Unicode form: presentation forms and ligatures onto their letters."""
    for src, dst in _CHAR_FIX.items():
        s = s.replace(src, dst)
    return unicodedata.normalize("NFKC", s)


def dediac(s: str) -> str:
    return _DIACRITICS.sub("", s)


def fold_letters(s: str) -> str:
    """The letter fold both Arabic-script tiers share.

    Alif variants, alif maqsuura and taa marbuuta, and nothing else. The
    diacritized tier applies this too; it cannot strip diacritics, because
    there the diacritics are the target.
    """
    return s.translate(_ALEF).translate(_ALEF_MAKSURA).translate(_TEH_MARBUTA)


def strict(s: str | None) -> str:
    """Surface-faithful pre-normalization: Unicode canonicalization, tatweel."""
    if not s:
        return ""
    return normalize_unicode(s).replace(TATWEEL, "")


def neutral(s: str | None) -> str:
    """``strict`` plus diacritic stripping and the Arabic orthographic fold.

    Diacritics are stripped before the letter fold: alif variants are single
    codepoints after Unicode normalization, so their hamza is never stripped.
    """
    if not s:
        return ""
    return fold_letters(dediac(strict(s)))


TIERS = {"strict": strict, "neutral": neutral}
