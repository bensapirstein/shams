"""Normalization for the diacritized tier, implementing Appendix B.1 step by step.

The first two operations are the ones every Arabic-script tier shares, called
from :mod:`shams_eval.normalization.orthographic` so the two tiers cannot drift:

1. Unicode canonicalization and *tatweel* removal.
2. The letter fold: alif variants, alif maqsuura and taa marbuuta
   (``orthographic.fold_letters``). The orthographic tier's ``neutral`` fold
   also strips residual diacritics; this tier cannot, because here the
   diacritics are the target. That is the only difference between the two
   letter-level folds.

The rest is what the diacritized tier adds:

3. Strip *sukun* (U+0652). Gold labels are sukun-free by construction, so a
   model that emits it would otherwise be charged for a mark the reference
   cannot carry.
4. Canonicalize accusative *tanwin* onto the consonant preceding the seat alif,
   so كتاباً and كتابًا score as identical. This is the Unicode-recommended
   order; the two spellings are the same word and differ only in which base the
   mark is attached to, which would otherwise register as two errors.
5. Drop the MSA case-*tanwin* — *dammatan* (U+064C, ``-un``) and *kasratan*
   (U+064D, ``-in``) — from both sides while keeping *fathatan* (U+064B, the
   realized ``-an``). Spoken Levantine does not realize MSA case endings, so a
   system that supplies them is displaying MSA register rather than making a
   diacritization error. They are all but absent from the reference (1 of
   15,506 tokens).

Each step is separately switchable through :class:`DiacriticProfile`:

``PAPER``
    All five steps. The DER reported in §3.1.
``TANWIN_INCLUDED``
    Steps 1–4, keeping the case-tanwin. The §3.1 ablation, reported in the
    prose rather than in the table: it is an ablation of the scheme, not a
    second fold of notation, so it does not belong beside a printed figure the
    way the strict–neutral gap does.

Keeping the profiles as data rather than as functions makes the ablation a
one-line change and lets a table caption name the profile it was scored under.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .orthographic import TATWEEL, fold_letters, normalize_unicode

FATHATAN, DAMMATAN, KASRATAN = "ً", "ٌ", "ٍ"
FATHA, DAMMA, KASRA = "َ", "ُ", "ِ"
SHADDA, SUKUN = "ّ", "ْ"
SUPERSCRIPT_ALEF = "ٰ"
ALEF = "ا"

#: Every mark the diacritized tier can carry, sukun included so it can be
#: stripped: U+064B to U+0652, U+0670, plus the Quranic marks that occasionally
#: leak out of a model.
DIACRITICS = frozenset(
    "ًٌٍَُِّْ"
    "ٕٖٜٟٓٔٗ٘ٙٚٛٝٞ"
    "ٰ"
)

CASE_TANWIN = frozenset({DAMMATAN, KASRATAN})

_ARABIC_LETTER_RE = re.compile(r"[ء-غف-يٱ-ۓ]")

#: fathatan sitting *on* a seat alif, at the end of a token: كتاباً
#: Rewritten so the mark precedes the alif: كتابًا
_TANWIN_ON_SEAT = re.compile(r"اً")


def is_arabic_letter(ch: str) -> bool:
    """True for an Arabic base letter, not a mark, digit or punctuation."""
    return bool(_ARABIC_LETTER_RE.match(ch))


@dataclass(frozen=True)
class DiacriticProfile:
    """One named configuration of the Appendix B.1 pipeline."""

    name: str
    unicode_norm: bool = True
    strip_tatweel: bool = True
    strip_sukun: bool = True
    fold_letters: bool = True
    canonicalize_tanwin_seat: bool = True
    drop_case_tanwin: bool = True

    def __call__(self, text: str | None) -> str:
        if not text:
            return ""
        if self.unicode_norm:
            text = normalize_unicode(text)
        if self.strip_tatweel:
            text = text.replace(TATWEEL, "")
        if self.strip_sukun:
            text = text.replace(SUKUN, "")
        if self.fold_letters:
            text = fold_letters(text)
        if self.canonicalize_tanwin_seat:
            # Applied after the letter fold so آً has already become اً.
            text = _TANWIN_ON_SEAT.sub("ًا", text)
        if self.drop_case_tanwin:
            text = "".join(c for c in text if c not in CASE_TANWIN)
        return re.sub(r"[^\S\n]+", " ", text).strip()


PAPER = DiacriticProfile("paper")
TANWIN_INCLUDED = DiacriticProfile("tanwin_included", drop_case_tanwin=False)

#: The normalization tiers this dataset tier is scored under. Unlike the
#: orthographic and phonetic tiers, whose second tier is a *fold* of notation,
#: the diacritized tier's second configuration is the paper's own ablation:
#: §3.1 reports a primary DER that drops the MSA case-tanwin and a
#: tanwin-included figure beside it. Naming the ablation as a tier is what lets
#: the evaluator compute both in one pass.
TIERS = {"paper": PAPER, "tanwin_included": TANWIN_INCLUDED}
