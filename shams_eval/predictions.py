"""Reading a prediction file, and checking it before it is scored.

A prediction file is JSON Lines, one record per utterance, carrying at minimum a
``seg_id`` and a hypothesis. Everything else in the record is provenance, not
input: references and stratum labels are resolved from the release (see
:mod:`shams_eval.dataset`) so that every system is scored against the same gold
and cut by the same labels, whatever each runner happened to write.

Three things are checked and reported before scoring.

**Coverage.** Which released utterances the file covers, and whether it carries a
``seg_id`` the release does not.

**Reference drift.** How far the frozen ``reference`` in the file has moved from
the release. Not an error: a run made before a corpus revision will differ from
it. It is reported because it is the difference between a number recomputed
today and the number printed alongside the run.

**Degenerate output.** Two kinds. A hypothesis more than three times the
reference length is a looping or runaway generation: a handful of them dominates
a corpus score far more than the system's ordinary behaviour warrants, which is
worth knowing before reading the score as a statement about the system. And an
empty hypothesis, which means the system
recognized nothing: the default is to score it as a full deletion, since
dropping such records would remove the utterances a system failed hardest on
from its numerator and denominator alike. ``on_empty="skip"`` drops them
instead.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .dataset import Dataset
from .tiers import Task


@dataclass
class PredictionSet:
    model_key: str
    path: Path
    seg_ids: list[str]
    hypotheses: list[str]
    records: list[dict] = field(repr=False, default_factory=list)
    skipped_empty: int = 0
    unknown_seg_ids: list[str] = field(default_factory=list)
    duplicate_seg_ids: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.seg_ids)


def load(path: str | Path, task: Task, dataset: Dataset, *,
         on_empty: str = "score") -> PredictionSet:
    """Read one prediction file and align it to the release."""
    path = Path(path)
    seg_ids: list[str] = []
    hyps: list[str] = []
    records: list[dict] = []
    seen: set[str] = set()
    unknown: list[str] = []
    duplicates: list[str] = []
    skipped = 0
    model_key = path.stem

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        seg_id = rec.get("seg_id")
        if not seg_id:
            continue
        if not records:
            model_key = rec.get("model_key") or path.stem
        if seg_id in seen:
            duplicates.append(seg_id)
            continue
        seen.add(seg_id)
        if seg_id not in dataset:
            unknown.append(seg_id)
            continue
        hyp = task.hypothesis(rec)
        if not hyp:
            if on_empty == "skip":
                skipped += 1
                continue
            hyp = ""
        seg_ids.append(seg_id)
        hyps.append(hyp)
        records.append(rec)

    return PredictionSet(model_key=model_key, path=path, seg_ids=seg_ids,
                         hypotheses=hyps, records=records, skipped_empty=skipped,
                         unknown_seg_ids=unknown, duplicate_seg_ids=duplicates)


def provenance(preds: PredictionSet, task: Task, dataset: Dataset) -> dict:
    """Coverage, reference drift and degenerate-output counts, for the report."""
    covered = set(preds.seg_ids)
    released = {u.seg_id for u in dataset}
    drift = 0
    frozen_present = 0
    for seg_id, rec in zip(preds.seg_ids, preds.records):
        frozen = rec.get("reference")
        if frozen is None:
            continue
        frozen_present += 1
        if frozen != dataset.reference(seg_id, task.reference_field):
            drift += 1
    empty = sum(1 for h in preds.hypotheses if not h.strip())
    # A hypothesis several times the reference length is a looping or runaway
    # generation, not a bad transcription. A handful of them dominates a corpus
    # score far more than the system's typical behaviour warrants, so the count
    # belongs beside the number.
    runaway = 0
    for seg_id, hyp in zip(preds.seg_ids, preds.hypotheses):
        ref_words = len((dataset.reference(seg_id, task.reference_field) or "").split())
        if ref_words and len(hyp.split()) > 3 * ref_words:
            runaway += 1
    return {
        "model_key": preds.model_key,
        "file": preds.path.name,
        "n_scored": len(preds),
        "n_released": len(released),
        "missing_from_predictions": sorted(released - covered),
        "unknown_seg_ids": preds.unknown_seg_ids,
        "duplicate_seg_ids": preds.duplicate_seg_ids,
        "frozen_reference_rows": frozen_present,
        "frozen_reference_drift": drift,
        "empty_hypotheses": empty,
        "runaway_hypotheses": runaway,
        "skipped_empty": preds.skipped_empty,
        "dataset": dataset.provenance,
    }


def summarize_provenance(p: dict) -> str:
    bits = [f"{p['n_scored']}/{p['n_released']} utterances"]
    if p["missing_from_predictions"]:
        bits.append(f"{len(p['missing_from_predictions'])} missing")
    if p["unknown_seg_ids"]:
        bits.append(f"{len(p['unknown_seg_ids'])} unknown seg_id")
    if p["duplicate_seg_ids"]:
        bits.append(f"{len(p['duplicate_seg_ids'])} duplicate seg_id")
    if p["frozen_reference_drift"]:
        bits.append(f"{p['frozen_reference_drift']} frozen refs stale")
    if p["empty_hypotheses"]:
        bits.append(f"{p['empty_hypotheses']} empty output")
    if p.get("runaway_hypotheses"):
        bits.append(f"{p['runaway_hypotheses']} runaway output")
    if p["skipped_empty"]:
        bits.append(f"{p['skipped_empty']} empty SKIPPED")
    return ", ".join(bits)
