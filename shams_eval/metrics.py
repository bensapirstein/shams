"""Error counts and the metrics built on them.

Every metric SHAMS reports — WER on the orthographic tier, CER on the phonetic
tier, DER on the diacritized tier — reduces to the same shape: a count of errors
over a count of reference units, per utterance. That is the whole point of this
module. Alignment differs by metric, because a word, a character and a diacritic
slot are aligned differently; aggregation and stratification do not, because
once you hold (errors, units) per utterance, a corpus score is a sum.

Keeping the numerator and the denominator apart is what makes that true. A
corpus WER is not the mean of per-utterance WERs, so a design that stores rates
cannot aggregate; one that stores counts computes every stratum by summing a
subset of a single alignment pass.

    counts = [align(r, h) for r, h in pairs]     # one pass, the expensive part
    overall = rate(counts)                        # a sum
    by_variety = {v: rate([counts[i] for i in idx]) for v, idx in strata}

The one metric that does not fit a plain edit distance is the per-utterance
capped CER used for A2P, where a looping audio-LLM can produce a raw CER of
2,500% that would otherwise dominate a micro-average. It is expressed here as a
capped :class:`ErrorCounts` — errors clipped to the reference length — so that
it still aggregates like everything else.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import jiwer

from .normalization import diacritic
from .normalization.diacritic import SHADDA, SUKUN

_BASE = [jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
         jiwer.RemoveMultipleSpaces(), jiwer.Strip()]
WORD_TRANSFORM = jiwer.Compose([*_BASE, jiwer.ReduceToListOfListOfWords()])
CHAR_TRANSFORM = jiwer.Compose([*_BASE, jiwer.ReduceToListOfListOfChars()])


@dataclass(frozen=True)
class ErrorCounts:
    """Edit counts for one utterance under one metric.

    ``reference_units`` is substitutions + deletions + hits, i.e. the length of
    the reference in whatever unit the metric counts. Insertions are errors but
    are not reference units, which is why the two properties are not symmetric.
    """

    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def reference_units(self) -> int:
        return self.substitutions + self.deletions + self.hits

    def __add__(self, other: ErrorCounts) -> ErrorCounts:
        return ErrorCounts(
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.hits + other.hits,
        )

    def capped(self) -> ErrorCounts:
        """Clip errors at the reference length: a fully wrong utterance is 100%.

        Used for A2P, where runaway generations would otherwise dominate.
        Expressed as substitutions so the sum stays a valid count vector.
        """
        n = self.reference_units
        if self.errors <= n:
            return self
        return ErrorCounts(substitutions=n, deletions=0, insertions=0, hits=0)

    def as_dict(self) -> dict[str, int]:
        return {"substitutions": self.substitutions, "deletions": self.deletions,
                "insertions": self.insertions, "hits": self.hits}


ZERO = ErrorCounts()


def total(counts: Iterable[ErrorCounts]) -> ErrorCounts:
    acc = ZERO
    for c in counts:
        acc = acc + c
    return acc


def rate(counts: Iterable[ErrorCounts]) -> float:
    """Corpus-level error rate: summed errors over summed reference units.

    The micro-average, and the default everywhere. A long utterance counts for
    more than a short one because it contains more of the material being scored.
    """
    t = total(counts)
    return t.errors / t.reference_units if t.reference_units else float("nan")


def utterance_rates(counts: Iterable[ErrorCounts]) -> list[float]:
    """Per-utterance error rate, nan where the reference is empty."""
    return [c.errors / c.reference_units if c.reference_units else float("nan")
            for c in counts]


def macro_rate(counts: Iterable[ErrorCounts]) -> float:
    """Mean of the per-utterance rates, ignoring empty references.

    Every utterance counts once regardless of length. This is only used for
    A2P, and only because the paper's A2P numbers are defined that way: the
    metric there is a per-utterance-capped CER, and capping is a per-utterance
    operation, so a per-utterance mean is the aggregation that matches it. A
    micro-average of the same capped counts is a different and equally
    defensible number; it is not the published one.
    """
    rs = [r for r in utterance_rates(counts) if r == r]
    return sum(rs) / len(rs) if rs else float("nan")


#: Metrics whose corpus score is a mean of per-utterance rates rather than a
#: ratio of sums. Consulted by :func:`score`, so a stratum and the headline
#: number always agree on the definition.
MACRO_METRICS: frozenset[str] = frozenset({"cer_capped"})


def score(counts: Iterable[ErrorCounts], metric: str) -> float:
    """The corpus score for one metric, under that metric's own aggregation."""
    counts = list(counts)
    return macro_rate(counts) if metric in MACRO_METRICS else rate(counts)


# ---------------------------------------------------------------------------
# aligners
# ---------------------------------------------------------------------------

def word_counts(reference: str, hypothesis: str) -> ErrorCounts:
    """Word-level edit counts (WER)."""
    o = jiwer.process_words(reference, hypothesis,
                            reference_transform=WORD_TRANSFORM,
                            hypothesis_transform=WORD_TRANSFORM)
    return ErrorCounts(o.substitutions, o.deletions, o.insertions, o.hits)


def char_counts(reference: str, hypothesis: str) -> ErrorCounts:
    """Character-level edit counts (CER)."""
    o = jiwer.process_characters(reference, hypothesis,
                                 reference_transform=CHAR_TRANSFORM,
                                 hypothesis_transform=CHAR_TRANSFORM)
    return ErrorCounts(o.substitutions, o.deletions, o.insertions, o.hits)


# --- DER as CATT computes it -----------------------------------------------

#: The six vowels, in the order the reference implementation moves them behind
#: a shadda. The order is load-bearing. On a three-mark stack, swapping an inner
#: pair first would produce a pair an earlier entry would then swap back, and
#: the letter would come out carrying a different mark.
_CATT_SWAP_ORDER = (diacritic.FATHA, diacritic.DAMMA, diacritic.KASRA,
                    diacritic.FATHATAN, diacritic.DAMMATAN, diacritic.KASRATAN)
_CATT_VOWELS = frozenset(_CATT_SWAP_ORDER)

#: The eight marks CATT's tokenizer recognizes. Every other mark falls outside
#: its character set and is dropped before scoring. Order does not matter here:
#: these are only tested for membership and used to collapse a doubled mark,
#: and collapsing one mark cannot produce a double of another.
_CATT_MARKS = _CATT_SWAP_ORDER + (SHADDA, SUKUN)

#: CATT keeps Arabic letters, the marks, dagger alif and wasla, and turns
#: everything else into a space. A Latin word or a Farsi yeh therefore opens a
#: word boundary rather than counting as a letter.
_CATT_KEEP_RE = re.compile(r"[^\u0621-\u063A\u0640-\u0652\u0670\u0671 ]")

#: Lam-alef presentation forms, which CATT's transliteration writes as two
#: letters and so scores as two slots.
_CATT_LIGATURES = {"\ufefb": "لا", "\ufef7": "لأ", "\ufef5": "لآ", "\ufef9": "لإ"}

_WASLA = "ٱ"


def _catt_normalize(text: str) -> str:
    """CATT's own preprocessing, which its DER is defined on top of.

    This runs after the tier normalizer and does not duplicate it: Appendix B.1
    folds hamza and drops the case-tanwin, while these steps decide what CATT
    counts as a letter at all. The released predictions carry punctuation and
    stretches of Latin script that B.1 leaves in place, and CATT scores those as
    word boundaries, so the difference is not hypothetical.
    """
    text = text.replace(diacritic.TATWEEL, "").replace(_WASLA, diacritic.ALEF)
    for ligature, letters in _CATT_LIGATURES.items():
        text = text.replace(ligature, letters)
    text = " ".join(_CATT_KEEP_RE.sub(" ", text).split())
    text = text.replace(diacritic.SUPERSCRIPT_ALEF, "")
    for vowel in _CATT_SWAP_ORDER:
        text = text.replace(vowel + SHADDA, SHADDA + vowel)
    for mark in _CATT_MARKS:
        text = text.replace(mark * 2, mark)
    return text


def catt_slots(text: str) -> list[tuple[str, str]]:
    """(base, mark) per scored position, bracketed by sentence boundaries.

    A position is an Arabic letter *or a space*: CATT counts the word boundary
    as a slot of its own, always unmarked, and brackets the utterance with two
    more. Its mark is the one mark that follows it, or a shadda plus the vowel
    after that. A third stacked mark is dropped, as it is by the scan this
    follows. The dagger alif is dropped earlier still, so whether a system
    writes one is invisible here.
    """
    text = _catt_normalize(text or "")
    slots = [("<BOS>", "<BOS>")]
    n = len(text)
    for i, ch in enumerate(text):
        if ch in _CATT_MARKS:
            continue
        nxt = text[i + 1] if i + 1 < n else ""
        if nxt not in _CATT_MARKS:
            slots.append((ch, "∅"))
        elif nxt == SHADDA and i + 2 < n and text[i + 2] in _CATT_VOWELS:
            slots.append((ch, SHADDA + text[i + 2]))
        else:
            slots.append((ch, nxt))
    slots.append(("<EOS>", "<EOS>"))
    return slots


def _catt_tags(text: str, *, case_ending: bool) -> list[str]:
    """The mark stream alone, which is what CATT aligns.

    Without ``case_ending``, the mark on the last letter of a word is cleared.
    CATT finds that letter by looking for an unmarked space after it, so a word
    at the end of the utterance keeps its mark: nothing follows it but the
    sentence boundary. That is reproduced here rather than corrected.
    """
    slots = catt_slots(text)
    if not case_ending:
        boundaries = {i for i, slot in enumerate(slots) if slot == (" ", "∅")}
        slots = [(base, "∅" if (i + 1) in boundaries else mark)
                 for i, (base, mark) in enumerate(slots)]
    return [mark for _, mark in slots]


_CATT_TAG_TRANSFORM = jiwer.Compose([jiwer.ReduceToListOfListOfWords()])


def catt_der_counts(reference: str, hypothesis: str, *,
                    case_ending: bool = True) -> ErrorCounts:
    """DER as the CATT reference implementation computes it.

    CATT reduces each side to one mark per letter and per space, then scores the
    two mark streams against each other as a WER: a full edit-distance
    alignment, which absorbs a length difference as insertions and deletions
    rather than shifting every later position. The corpus score is a ratio of
    sums, so no macro entry is needed.

    Two properties follow from that and are worth stating, because both lower
    the rate against a definition that counts letters alone. Every space is a
    scored position that agrees whenever both sides put a word boundary there,
    and two more agree at the sentence boundaries, so the denominator holds
    positions that cannot be wrong. And a word the system respelled is not
    excluded: its marks are aligned against whatever the reference had there.

    Verified against ``compute_der.py`` and ``TashkeelTokenizer`` from
    abjadai/catt over every released diacritization prediction under both
    normalization profiles and both case-ending settings: the rate agrees in
    all 34,528 comparisons. The split across substitutions, deletions and
    insertions can differ on an utterance where the two sides are not the same
    length, because that implementation breaks alignment ties through
    ``kaldialign`` and this one through ``jiwer``. The totals the rate is built
    from do not move.
    """
    ref_marks = _catt_tags(reference, case_ending=case_ending)
    hyp_marks = _catt_tags(hypothesis, case_ending=case_ending)
    o = jiwer.process_words(" ".join(ref_marks), " ".join(hyp_marks),
                            reference_transform=_CATT_TAG_TRANSFORM,
                            hypothesis_transform=_CATT_TAG_TRANSFORM)
    return ErrorCounts(o.substitutions, o.deletions, o.insertions, o.hits)


Aligner = Callable[[str, str], ErrorCounts]

#: name -> (aligner, display label). A task declares which of these it reports;
#: nothing else in the package knows what a metric is.
METRICS: dict[str, tuple[Aligner, str]] = {
    "wer": (word_counts, "WER"),
    "cer": (char_counts, "CER"),
    "der": (catt_der_counts, "DER"),
    "cer_capped": (lambda r, h: char_counts(r, h).capped(), "CER"),
}
