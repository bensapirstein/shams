"""Normalization for the DIN-31635 phonetic tier, shared by G2P and A2P.

Two tiers, as in Appendix B.1:

``strict``
    Map every system's output onto the reference transcription alphabet with
    deterministic character rules (ɣ→ġ, ʃ→š, aː→ā …) and strip segmentation
    hyphens, punctuation and casing. Answers "how faithfully does the system
    reproduce the SHAMS annotation?", which favours the prompted LLMs.

``neutral``
    ``strict`` plus a symmetric fold of the notation distinctions not every
    system's inventory encodes: the definite-article vowel and sun-letter
    assimilation (āl-/il-), the tāʾ marbūṭa endings (-eh/-e, -ah/-a), the
    affricate/fricative pairs ǧ/j and č/k, and the reduced vowel ə. The ey/ē
    monophthong spelling is a Levanti notation and is mapped in that system's
    entry normalizer, before the fold. Phonological contrasts — vowel length and quality,
    emphatics, gemination, qaf realization — are preserved and still score.
    This is the primary tier for G2P and A2P.

A2P and G2P deliberately share one implementation: the two tasks target the
same tier, so scoring them through different code would make their numbers
incomparable, and the direct-vs-cascade contrast in §3.4 rests on their being
comparable.

Per-system entry normalizers (``normalize_levanti``, ``normalize_charsiu``,
``normalize_llm``) map a particular system's inventory into the SHAMS alphabet
before the shared fold. IPA-emitting phoneme recognizers go through
:func:`from_ipa` first. ``TIERS`` applies the ``llm`` entry, which is what every
prompted system needs; a dedicated G2P system is scored through its own entry
in :data:`SYSTEM_TIERS`.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

# --------------------------------------------------------------------------
# shared cleaning
# --------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[.,!?;:()\[\]{}؟،؛…\"'«»“”‘’—–]")
_TRIPLE_RE = re.compile(r"(.)\1{2,}")


def _nfc(t: str) -> str:
    return unicodedata.normalize("NFC", t)


def _clean(t: str) -> str:
    t = t.lower().replace("-", "")
    t = _PUNCT_RE.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


# --------------------------------------------------------------------------
# per-system alphabet mapping (system inventory -> SHAMS DIN symbol set)
# --------------------------------------------------------------------------

_LEV = [("θ", "ṯ"), ("ž", "ǧ"), ("ħ", "ḥ"), ("x", "ḫ"),
        ("ð", "ḏ"), ("ʕ", "ʿ"), ("ɣ", "ġ"), ("ʔ", "ʾ")]


def normalize_shams(t: str) -> str:
    """The reference tier's own alphabet: cleaning only."""
    return _clean(_nfc(t))


def normalize_levanti(t: str) -> str:
    t = _nfc(t).replace("ð̣", "ẓ")
    for s, g in _LEV:
        t = t.replace(s, g)
    t = t.replace("ً", "n").replace("ْ", "")   # leaked tanwin / sukun
    t = t.replace("ey", "ē")                             # monophthong notation
    t = _TRIPLE_RE.sub(r"\1\1", t)
    return _clean(t)


def normalize_charsiu(t: str) -> str:
    t = _nfc(t).replace("/", "").replace("ˈ", "").replace("ˌ", "")
    t = (t.replace("d͡ʒ", "ǧ").replace("dʒ", "ǧ")
          .replace("t͡ʃ", "č").replace("tʃ", "č"))
    for s, g in [("sˤ", "ṣ"), ("dˤ", "ḍ"), ("tˤ", "ṭ"), ("ðˤ", "ẓ"), ("zˤ", "ẓ")]:
        t = t.replace(s, g)
    t = t.replace("eɪ", "ē").replace("aɪ", "ay").replace("aʊ", "aw")
    for s, g in [("aː", "ā"), ("iː", "ī"), ("uː", "ū"), ("eː", "ē"), ("oː", "ō")]:
        t = t.replace(s, g)
    t = re.sub(r"(.)ː", r"\1\1", t).replace("ˤ", "")
    for s, g in [("j", "y"), ("ʃ", "š"), ("ʒ", "ǧ"), ("θ", "ṯ"), ("ð", "ḏ"),
                 ("ħ", "ḥ"), ("χ", "ḫ"), ("x", "ḫ"), ("ʕ", "ʿ"), ("ɣ", "ġ"),
                 ("ʔ", "ʾ")]:
        t = t.replace(s, g)
    return _clean(t)


def normalize_llm(t: str) -> str:
    t = _nfc(t).replace("ʕ", "ʿ").replace("ʔ", "ʾ").replace("ħ", "ḥ").replace("ḷ", "l")
    return _clean(t)


#: IPA (and common espeak / Allosaurus symbols) -> DIN 31635, applied before the
#: shared fold for phoneme-recognition models whose raw output is IPA. Longest
#: source first, so digraphs win over their parts.
IPA_TO_DIN = {
    "d͡ʒ": "ǧ", "dʒ": "ǧ", "t͡ʃ": "č", "tʃ": "č",
    "aː": "ā", "iː": "ī", "uː": "ū", "eː": "ē", "oː": "ō",
    "sˤ": "ṣ", "dˤ": "ḍ", "tˤ": "ṭ", "ðˤ": "ẓ", "zˤ": "ẓ",
    "ʃ": "š", "ʒ": "ž", "ɣ": "ġ", "x": "ḫ", "χ": "ḫ",
    "θ": "ṯ", "ð": "ḏ", "ħ": "ḥ", "ʕ": "ʿ", "ʔ": "ʾ",
    "ŋ": "n", "ɡ": "g", "j": "y", "q": "q",
    "ɛ": "e", "ɪ": "i", "ʊ": "u", "ɔ": "o", "ə": "e", "ɑ": "a",
    "ː": "",
}


def from_ipa(text: str) -> str:
    text = _nfc(text)
    for src in sorted(IPA_TO_DIN, key=len, reverse=True):
        text = text.replace(src, IPA_TO_DIN[src])
    return text


ENTRY_NORMALIZERS = {
    "shams": normalize_shams,
    "levanti": normalize_levanti,
    "charsiu": normalize_charsiu,
    "llm": normalize_llm,
    "ipa": lambda t: normalize_llm(from_ipa(t)),
}


# --------------------------------------------------------------------------
# symmetric fold
# --------------------------------------------------------------------------

def _article(tok: str) -> str:
    tok = re.sub(r"^(āl|al|el|il)", "il", tok)    # canonical article vowel
    tok = re.sub(r"^il(?=(.)\1)", "i", tok)        # drop l before an assimilated geminate
    return tok


def fold(t: str, level: str) -> str:
    """The symmetric fold. ``strict`` is the identity; ``neutral`` folds notation."""
    if level == "strict":
        return t
    t = re.sub(r"eh(?=\s|$)", "e", t)
    t = re.sub(r"ah(?=\s|$)", "a", t)
    t = " ".join(_article(x) for x in t.split())
    t = t.replace("ǧ", "j").replace("č", "k").replace("ə", "i")
    return t


def normalize(text: str | None, level: str = "neutral", entry: str = "llm") -> str:
    """Entry normalizer, then the fold. Apply symmetrically to ref and hyp."""
    return fold(ENTRY_NORMALIZERS[entry](text or ""), level)



def tiers_for(entry: str) -> dict[str, Callable[[str | None], str]]:
    """The two folds, with one system's entry normalizer in front of them."""
    return {level: (lambda t, _level=level, _entry=entry: normalize(t, _level, _entry))
            for level in ("strict", "neutral")}


TIERS = tiers_for("llm")
"""The gold's normalizers, and the default for a hypothesis.

Applied to the reference and to any system not named in :data:`SYSTEM_TIERS`.
"""

#: model key -> that system's normalizers, for a system that does not emit the
#: SHAMS symbol set.
#:
#: The fold that makes two systems comparable is symmetric and lives in
#: :data:`TIERS`. Reaching the alphabet the fold operates on is not: CharsiuG2P
#: emits IPA and Levanti its own romanization, so mapping those inventories onto
#: the SHAMS set is a per-system step on the hypothesis alone. Scoring
#: CharsiuG2P through the reference's own normalizer charges it about 25 CER
#: points for its alphabet rather than its phonology, which is the opposite of
#: what the neutral fold exists to measure.
#:
#: Every A2P system and every prompted LLM emits DIN-31635 or close to it and is
#: absent from this map, so it is read only by the dedicated G2P systems.
SYSTEM_TIERS: dict[str, dict[str, Callable[[str | None], str]]] = {
    "levanti": tiers_for("levanti"),
    "charsiu-byt5": tiers_for("charsiu"),
}
