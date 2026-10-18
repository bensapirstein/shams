r"""The paper's example tables, filled from the release.

A qualitative table quotes utterances and system outputs verbatim. Typed in by
hand, a table comes to quote a reference the release no longer carries, and
nothing says so. Here no string is typed in: each table is a template under
``templates/`` in which every quoted string is a placeholder, resolved from the
released dataset and from ``predictions/`` when the table is generated.

Placeholders::

    {{transcription LEB_0}}                a released tier, verbatim
    {{subdialect LEB_0}}                   the variety label, as the tables print it
    {{asr/whisper-large-v3 7562_1}}        a system's raw output, from predictions/asr/
    {{asr/whisper-large-v3 7562_1 bold}}   ... with the words that differ from the
                                           reference set in \textbf{}
    {{diacritization/catt 7896_1 bold=كَيْفَ,يُعْرَفُ}}
                                           ... with these words set in \textbf{}
    {{g2p/levanti LEB_24 norm}}            ... after normalization
    {{transcription LEB_6 4:}}             a word window (a Python slice); the cut
                                           ends get \dots

Options combine. Arabic-script runs are set in ``\ar{}`` and Latin runs are left
bare, so a code-switched line reads in both directions.

``bold`` and ``norm`` use one fold per tier, the one that folds the most
notation (:data:`FOLD`), and ``bold`` compares words with their punctuation
removed. A bold word is therefore one that still differs after every
normalization the paper applies, not a spelling or punctuation convention.
``bold=`` names the words instead, for a table whose caption gives the marked
words a narrower meaning than "differs from the reference".
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import jiwer

from ..dataset import Dataset
from ..predictions import load
from ..tiers import TASKS, TIERS
from . import names

TEMPLATES = Path(__file__).with_name("templates")
PREDICTIONS = Path(__file__).resolve().parents[2] / "predictions"

#: The normalization fold ``bold`` and ``norm`` use, per tier: the one that
#: folds the most notation, so that a bold word is an error under every fold.
FOLD = {"orthographic": "neutral", "diacritized": "paper", "phonetic": "neutral"}

PLACEHOLDER = re.compile(r"\{\{([^\s{}]+) ([^\s{}]+)((?: [^\s{}]+)*)\}\}")
ARABIC = re.compile(r"[\u0600-\u06FF]")
LATIN = re.compile(r"[A-Za-z]")
PUNCT = re.compile(r"[.,!?;:()\[\]\"'«»“”‘’…؟،؛—–]")
FIELD_TIER = {t.field: t for t in TIERS.values()}


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _escape(word: str) -> str:
    return re.sub(r"([&%$#_{}])", r"\\\1", word)


def _differs(ref: list[str], hyp: list[str]) -> list[bool]:
    """For each hypothesis word, whether the alignment puts it outside an equal chunk.

    A word that normalizes to nothing (punctuation) is not aligned, and so is
    never bold: the fold does not score it.
    """
    kept = [i for i, w in enumerate(hyp) if w]
    flags = [False] * len(hyp)
    if not kept:
        return flags
    ref = [w for w in ref if w]
    if not ref:
        return [bool(w) for w in hyp]
    out = jiwer.process_words(" ".join(ref), " ".join(hyp[i] for i in kept))
    for chunk in out.alignments[0]:
        if chunk.type != "equal":
            for j in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                flags[kept[j]] = True
    return flags


def _tex(words: list[str], bold: list[bool]) -> str:
    r"""Set the words, bold where flagged, each Arabic-script run inside \ar{}.

    A token with no letters (punctuation, digits, \dots) belongs to the run of
    the next lettered token, so a leading ellipsis stays inside the Arabic run
    and reads at the right, where the sentence begins.
    """
    runs: list[tuple[bool, list[str]]] = []
    pending: list[str] = []
    for word, flag in zip(words, bold, strict=True):
        tex = _escape(word)
        tex = f"\\textbf{{{tex}}}" if flag else tex
        arabic = bool(ARABIC.search(word))
        lettered = arabic or (LATIN.search(word) and not word.startswith("\\"))
        if not lettered:
            pending.append(tex)
            continue
        if not runs or runs[-1][0] != arabic:
            runs.append((arabic, []))
        runs[-1][1].extend(pending + [tex])
        pending = []
    if pending:
        if not runs:
            runs.append((False, []))
        runs[-1][1].extend(pending)
    return " ".join(f"\\ar{{{' '.join(r)}}}" if arabic else " ".join(r)
                    for arabic, r in runs)


def _resolve(spec: str, seg_id: str, opts: list[str], data: Dataset, cache: dict) -> str:
    if spec == "subdialect":
        variety = data.get(seg_id).subdialect
        return names.VARIETY.get(variety, variety)
    if "/" in spec:
        task_key, model = spec.split("/", 1)
        task = TASKS[task_key]
        tier = task.tier
        if spec not in cache:
            preds = load(PREDICTIONS / task_key / f"{model}.jsonl", task, data)
            cache[spec] = dict(zip(preds.seg_ids, preds.hypotheses, strict=True))
        text = cache[spec].get(seg_id)
        if text is None:
            raise KeyError(f"{spec} has no output for {seg_id}")
    else:
        tier, model = FIELD_TIER[spec], None
        text = data.reference(seg_id, spec)
    normalize = tier.normalizer(FOLD[tier.key], model)
    gold = tier.normalizer(FOLD[tier.key])
    if "norm" in opts:
        text = normalize(text)
    words = text.split()
    bold = [False] * len(words)
    marked = next((o[5:].split(",") for o in opts if o.startswith("bold=")), None)
    if marked is not None:
        bold = [_nfc(w) in {_nfc(m) for m in marked} for w in words]
    elif "bold" in opts:
        reference = data.reference(seg_id, tier.field).split()
        bold = _differs([PUNCT.sub("", gold(w)) for w in reference],
                        [PUNCT.sub("", normalize(w)) for w in words])
    window = next((o for o in opts if ":" in o), None)
    if window:
        start, stop = (int(x) if x else None for x in window.split(":"))
        head, tail = words[:start] if start else [], words[stop:] if stop else []
        words, bold = words[start:stop], bold[start:stop]
        words = ([r"\dots"] if head else []) + words + ([r"\dots"] if tail else [])
        bold = ([False] if head else []) + bold + ([False] if tail else [])
    return _tex(words, bold)


def render(template: Path, data: Dataset) -> str:
    cache: dict = {}

    def fill(m: re.Match) -> str:
        spec, seg_id, opts = m.group(1), m.group(2), m.group(3).split()
        if seg_id not in data:
            raise KeyError(f"{template.name}: {seg_id} is not in the release")
        return _resolve(spec, seg_id, opts, data, cache)

    return PLACEHOLDER.sub(fill, template.read_text(encoding="utf-8"))


def write_all(data: Dataset, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for template in sorted(TEMPLATES.glob("*.tex")):
        p = out_dir / template.name
        p.write_text(render(template, data), encoding="utf-8")
        written.append(p)
    return written


def _plain(tex: str) -> str:
    """A table as text: comments dropped, the markup around a quoted string removed.

    Canonically composed, so that two orderings of the same combining marks,
    which are one string to a reader and to the fonts, compare equal.
    """
    tex = _nfc(tex)
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    tex = re.sub(r"\\dots|\.\.\.|…", "", tex)
    tex = re.sub(r"\\(?:ar|RL|textbf|textit|texttt|foreignlanguage\{arabic\})\s*\{", "", tex)
    return " ".join(tex.replace("{", "").replace("}", "").split())


def verify(data: Dataset, tables_dir: Path) -> list[tuple[str, str, bool]]:
    """Whether each string a template draws is what a manuscript's copy quotes.

    Returns (table, placeholder, found) per placeholder, ``bold`` set aside so a
    hand-bolded word is not reported as drift. A string not found is one the
    manuscript quotes from an earlier revision, or with an edit.
    """
    rows = []
    for template in sorted(TEMPLATES.glob("*.tex")):
        copy = Path(tables_dir) / template.name
        if not copy.exists():
            continue
        plain = _plain(copy.read_text(encoding="utf-8"))
        cache: dict = {}
        for m in PLACEHOLDER.finditer(template.read_text(encoding="utf-8")):
            spec, seg_id, opts = m.group(1), m.group(2), m.group(3).split()
            opts = [o for o in opts if not o.startswith("bold")]
            text = _plain(_resolve(spec, seg_id, opts, data, cache))
            rows.append((template.name, f"{spec} {seg_id}", text in plain))
    return rows
