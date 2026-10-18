"""Inter-annotator agreement on the diacritization subset (§2.4, Table 3).

Two independent annotators, both native Palestinian Arabic speakers, diacritized
the same 150 utterances (30 per variety) from the audio. Annotator A's labels
are the released gold; annotator B re-diacritized the same transcription. This
module scores the agreement between them and is what produces the released
table.

**The unit is the diacritic slot**: one per Arabic letter, labelled with the set
of marks that letter carries, normalized to a single symbol (∅, a, i, u, aN, ~a,
…). Utterance-level agreement would give 150 observations; word-level a few
thousand; slots give 7,290, which is what lets a 150-utterance study support a
per-variety breakdown at all.

**Words whose consonantal skeletons differ are excluded.** If B respelled a
word, that is an orthographic divergence, not a diacritic disagreement, and
counting it would charge a single respelling as a run of errors down the rest of
the word. Those words are aligned out with difflib and reported separately.

**Statistics.** Raw agreement, Cohen's κ, Krippendorff's α (nominal, the more
defensible of the two for this design), DER and DER excluding word-final
letters, and word-level exact agreement. Intervals are cluster bootstraps
resampling *utterances*: slots within an utterance are not independent, and a
naive slot bootstrap would give intervals several times too narrow. A
permutation test against κ = 0 is reported for completeness.

**Breakdowns** by variety, gender, source, and by whether annotator A edited the
silver label or approved it — higher agreement on the approved stratum would
mean part of the agreement is both annotators deferring to the same automatic
label rather than converging independently. And one direct contrast: agreement on
Palestinian varieties against the rest, with a bootstrap interval on the
difference.
"""
from __future__ import annotations

import json
import math
import random
import zlib
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from .normalization.diacritic import (
    DIACRITICS,
    SHADDA,
    SUKUN,
    DiacriticProfile,
    is_arabic_letter,
)

#: NFC, tatweel and sukun only. The orthographic folds of the scoring profile are
#: deliberately *not* applied: if one annotator writes أ and the other ا, that is
#: a disagreement about the text, and the skeleton alignment is where it belongs.
IAA_NORMALIZE = DiacriticProfile(
    "iaa", unicode_norm=True, strip_tatweel=True, strip_sukun=True,
    fold_letters=False, canonicalize_tanwin_seat=False, drop_case_tanwin=False,
)

DEFAULT_SEED = 20260727
DEFAULT_BOOTSTRAP = 5000
DEFAULT_PERMUTATIONS = 2000

Pair = tuple[str, str, bool]


# ---------------------------------------------------------------------------
# diacritic slots
# ---------------------------------------------------------------------------
# One label per letter is the unit two annotators agree or disagree on; κ is
# computed over the confusion matrix of these labels. This is not the DER of the
# results tables, which :func:`shams_eval.metrics.catt_der_counts` scores.

_SLOT_SYMBOL = {
    "َ": "a", "ُ": "u", "ِ": "i",
    "ً": "aN", "ٌ": "uN", "ٍ": "iN",
    "ٰ": "A", SHADDA: "~",
}


def word_slots(word: str) -> tuple[str, list[str]]:
    """(consonantal skeleton, one diacritic label per letter).

    Marks are canonically ordered — shadda first, then the vowel — so ـّـَ and
    ـَـّ cannot count as a disagreement. An unmarked letter is ``∅``, which is a
    label like any other: an annotator who leaves a letter bare and one who does
    the same agree, and κ is computed over the confusion matrix of these labels.
    """
    skeleton: list[str] = []
    marks: list[list[str]] = []
    for ch in word:
        if is_arabic_letter(ch):
            skeleton.append(ch)
            marks.append([])
        elif ch in DIACRITICS and ch != SUKUN and marks:
            marks[-1].append(ch)
    labels = []
    for group in marks:
        if not group:
            labels.append("∅")
            continue
        vowels = sorted(m for m in group if m != SHADDA)
        ordered = ([SHADDA] if SHADDA in group else []) + vowels
        labels.append("".join(_SLOT_SYMBOL.get(m, m) for m in ordered))
    return "".join(skeleton), labels


def utterance_slots(text: str) -> list[tuple[str, list[str]]]:
    keep = lambda w: "".join(c for c in w if is_arabic_letter(c) or c in DIACRITICS)  # noqa: E731
    return [word_slots(keep(w)) for w in (text or "").split() if keep(w)]


@dataclass
class Unit:
    """One scored utterance and its aligned slot pairs."""

    seg_id: str
    subdialect: str
    gender: str
    source: str
    pairs: list[Pair]
    words_compared: int
    words_divergent: int
    words_agree: int
    first_pass_edited: bool = False
    second_pass_edited: bool = True


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def agreement(pairs: Sequence[Pair]) -> float:
    return sum(1 for x, y, _ in pairs if x == y) / len(pairs) if pairs else float("nan")


def cohen_kappa(pairs: Sequence[Pair]) -> float:
    n = len(pairs)
    if n == 0:
        return float("nan")
    obs = sum(1 for x, y, _ in pairs if x == y) / n
    ca = Counter(x for x, _, _ in pairs)
    cb = Counter(y for _, y, _ in pairs)
    exp = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if exp == 1 else (obs - exp) / (1 - exp)


def krippendorff_alpha(pairs: Sequence[Pair]) -> float:
    """Nominal α for two coders with complete data."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    counts: Counter = Counter()
    for x, y, _ in pairs:
        counts[x] += 1
        counts[y] += 1
    total = 2 * n
    do = sum(1 for x, y, _ in pairs if x != y) / n
    de = 1 - sum(c * (c - 1) for c in counts.values()) / (total * (total - 1))
    return 1.0 if de == 0 else 1 - do / de


def der(pairs: Sequence[Pair], exclude_final: bool = False) -> float:
    sel = [p for p in pairs if not (exclude_final and p[2])]
    return sum(1 for x, y, _ in sel if x != y) / len(sel) if sel else float("nan")


def pairs_of(units: Iterable[Unit]) -> list[Pair]:
    return [p for u in units for p in u.pairs]


def kappa_of(units: Iterable[Unit]) -> float:
    return cohen_kappa(pairs_of(units))


def alpha_of(units: Iterable[Unit]) -> float:
    return krippendorff_alpha(pairs_of(units))


def metrics(units: Sequence[Unit]) -> dict:
    pairs = pairs_of(units)
    words_total = sum(u.words_compared for u in units)
    return {
        "utterances": len(units),
        "slots": len(pairs),
        "agreement": agreement(pairs),
        "kappa": cohen_kappa(pairs),
        "alpha": krippendorff_alpha(pairs),
        "der": der(pairs),
        "der_no_final": der(pairs, exclude_final=True),
        "word_agreement": (sum(u.words_agree for u in units) / words_total
                           if words_total else float("nan")),
        "words_compared": words_total,
        "words_orthographically_divergent": sum(u.words_divergent for u in units),
    }


# ---------------------------------------------------------------------------
# resampling
# ---------------------------------------------------------------------------

def stratum_seed(key: object) -> int:
    """Stable per-stratum bootstrap offset.

    CRC32, not ``hash()``: Python randomizes string hashing per process unless
    PYTHONHASHSEED is set, so a bootstrap seeded with ``hash`` would give a
    different interval on every run while the point estimate stayed put.
    """
    return zlib.crc32(str(key).encode("utf-8")) % 1000


def cluster_bootstrap(units: Sequence[Unit], stat: Callable[[Sequence[Unit]], float],
                      reps: int = DEFAULT_BOOTSTRAP, seed: int = 1,
                      alpha: float = 0.05) -> tuple[float, float]:
    """Percentile interval, resampling utterances (the cluster) with replacement."""
    if not units:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(units)
    draws = []
    for _ in range(reps):
        sample = [units[rng.randrange(n)] for _ in range(n)]
        v = stat(sample)
        if not math.isnan(v):
            draws.append(v)
    if not draws:
        return (float("nan"), float("nan"))
    draws.sort()
    lo = draws[int((alpha / 2) * len(draws))]
    hi = draws[min(len(draws) - 1, int((1 - alpha / 2) * len(draws)))]
    return (lo, hi)


def difference_ci(units_a: Sequence[Unit], units_b: Sequence[Unit],
                  stat: Callable[[Sequence[Unit]], float],
                  reps: int = DEFAULT_BOOTSTRAP, seed: int = 3,
                  alpha: float = 0.05) -> tuple[float, float]:
    """Interval on stat(A) − stat(B), resampling within each group."""
    if not units_a or not units_b:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    na, nb = len(units_a), len(units_b)
    draws = []
    for _ in range(reps):
        sa = [units_a[rng.randrange(na)] for _ in range(na)]
        sb = [units_b[rng.randrange(nb)] for _ in range(nb)]
        va, vb = stat(sa), stat(sb)
        if not (math.isnan(va) or math.isnan(vb)):
            draws.append(va - vb)
    if not draws:
        return (float("nan"), float("nan"))
    draws.sort()
    lo = draws[int((alpha / 2) * len(draws))]
    hi = draws[min(len(draws) - 1, int((1 - alpha / 2) * len(draws)))]
    return (lo, hi)


def permutation_p(units: Sequence[Unit], reps: int = DEFAULT_PERMUTATIONS,
                  seed: int = 2) -> float:
    """H0: the two annotators' labels are independent, κ = 0.

    Shuffles B's labels across all slots and recomputes κ; returns the
    add-one-smoothed proportion of permutations reaching the observed value.
    """
    pairs = pairs_of(units)
    if not pairs:
        return float("nan")
    obs = cohen_kappa(pairs)
    rng = random.Random(seed)
    bs = [y for _, y, _ in pairs]
    hits = 0
    for _ in range(reps):
        rng.shuffle(bs)
        if cohen_kappa([(x, y, f) for (x, _, f), y in zip(pairs, bs)]) >= obs:
            hits += 1
    return (hits + 1) / (reps + 1)


# ---------------------------------------------------------------------------
# building units
# ---------------------------------------------------------------------------

def align_labels(a_text: str, b_text: str) -> tuple[list[Pair], int, int, int]:
    """(slot pairs, words compared, words orthographically divergent, words agreeing)."""
    a = utterance_slots(IAA_NORMALIZE(a_text))
    b = utterance_slots(IAA_NORMALIZE(b_text))
    sm = SequenceMatcher(None, [w[0] for w in a], [w[0] for w in b], autojunk=False)

    pairs: list[Pair] = []
    compared = divergent = agree = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                la, lb = a[i][1], b[j][1]
                if len(la) != len(lb):        # unreachable when skeletons match
                    continue
                compared += 1
                if la == lb:
                    agree += 1
                last = len(la) - 1
                for k, (x, y) in enumerate(zip(la, lb)):
                    pairs.append((x, y, k == last))
        else:
            divergent += max(i2 - i1, j2 - j1)
    return pairs, compared, divergent, agree


def read_jsonl(path: str | Path) -> dict[str, dict]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["seg_id"]] = r
    return out


def build_units(subset: dict[str, dict], annotator_b: dict[str, dict]) -> list[Unit]:
    """Join the study frame to the second pass. The only place the two are matched."""
    units: list[Unit] = []
    for seg_id, m in subset.items():
        b = annotator_b.get(seg_id)
        if not b:
            continue
        b_text = b.get("vocalized") or b.get("text") or ""
        pairs, compared, divergent, agree = align_labels(m["reference_vocalized"], b_text)
        units.append(Unit(
            seg_id=seg_id,
            subdialect=m.get("subdialect", "unknown"),
            gender=m.get("gender", "?"),
            source=m.get("source", "unknown"),
            pairs=pairs,
            words_compared=compared,
            words_divergent=divergent,
            words_agree=agree,
            first_pass_edited=bool(m.get("first_pass_edited")),
            second_pass_edited=bool(b.get("edited", True)),
        ))
    return units


def load_units(subset_path: str | Path, annotator_b_path: str | Path) -> list[Unit]:
    return build_units(read_jsonl(subset_path), read_jsonl(annotator_b_path))


# ---------------------------------------------------------------------------
# the study
# ---------------------------------------------------------------------------

@dataclass
class Study:
    overall: dict
    strata: dict
    palestinian_contrast: dict
    disagreements: list[tuple[str, int]]
    per_item: list[dict] = field(default_factory=list)
    settings: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "overall": self.overall,
            "strata": self.strata,
            "palestinian_contrast": self.palestinian_contrast,
            "disagreements": [{"pair": p, "count": n} for p, n in self.disagreements],
            "settings": self.settings,
        }


def run(units: Sequence[Unit], *, seed: int = DEFAULT_SEED,
        bootstrap: int = DEFAULT_BOOTSTRAP,
        permutations: int = DEFAULT_PERMUTATIONS) -> Study:
    """Every statistic in Table 3, with the seeds the released table was built on."""
    overall = metrics(units)
    overall["kappa_ci"] = list(cluster_bootstrap(units, kappa_of, reps=bootstrap, seed=seed))
    overall["alpha_ci"] = list(cluster_bootstrap(units, alpha_of, reps=bootstrap,
                                                 seed=seed + 1))
    overall["permutation_p"] = permutation_p(units, reps=permutations, seed=seed + 2)

    def section(key_fn) -> dict:
        buckets: dict[str, list[Unit]] = {}
        for u in units:
            buckets.setdefault(str(key_fn(u)), []).append(u)
        out = {}
        for k in sorted(buckets):
            g = buckets[k]
            m = metrics(g)
            m["kappa_ci"] = list(cluster_bootstrap(
                g, kappa_of, reps=max(1000, bootstrap // 2),
                seed=seed + stratum_seed(k)))
            out[k] = m
        return out

    strata = {
        "subdialect": section(lambda u: u.subdialect),
        "gender": section(lambda u: u.gender),
        "source": section(lambda u: u.source),
        "first_pass_provenance": section(
            lambda u: "silver edited in round 1" if u.first_pass_edited
            else "silver approved in round 1"),
    }

    pal = [u for u in units if u.subdialect.startswith("Palestinian")]
    non = [u for u in units if not u.subdialect.startswith("Palestinian")]
    contrast = {
        "palestinian": metrics(pal),
        "non_palestinian": metrics(non),
        "difference": kappa_of(pal) - kappa_of(non),
        "difference_ci": list(difference_ci(pal, non, kappa_of, reps=bootstrap,
                                            seed=seed + 5)),
    }
    contrast["palestinian"]["kappa_ci"] = list(
        cluster_bootstrap(pal, kappa_of, reps=bootstrap, seed=seed + 6))
    contrast["non_palestinian"]["kappa_ci"] = list(
        cluster_bootstrap(non, kappa_of, reps=bootstrap, seed=seed + 7))

    conf: Counter = Counter()
    for u in units:
        for x, y, _ in u.pairs:
            if x != y:
                conf[" ↔ ".join(sorted((x, y)))] += 1

    per_item = []
    for u in units:
        m = metrics([u])
        per_item.append({
            "seg_id": u.seg_id, "subdialect": u.subdialect, "gender": u.gender,
            "source": u.source, "slots": m["slots"], "agreement": m["agreement"],
            "der": m["der"],
            "words_orthographically_divergent": u.words_divergent,
            "first_pass_edited": u.first_pass_edited,
            "second_pass_edited": u.second_pass_edited,
        })

    return Study(overall=overall, strata=strata, palestinian_contrast=contrast,
                 disagreements=conf.most_common(12), per_item=per_item,
                 settings={"seed": seed, "bootstrap": bootstrap,
                           "permutations": permutations})
