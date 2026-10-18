"""The evaluator. One function, every task.

The shape of the computation is the same whatever tier is being scored:

1. Resolve references for the task's target tier from the release.
2. Normalize both sides under each normalization tier of that target tier.
3. Align once per (normalization tier, metric) pair, keeping per-utterance
   :class:`~shams_eval.metrics.ErrorCounts`.
4. Every stratum is then a sum over a subset of those counts. No further
   alignment happens.

Step 3 is the only expensive part and it runs once. That is what makes a
nine-system table with per-variety, per-source and variety x source cells finish
in seconds rather than minutes, and it is why two cells of the same table are
guaranteed to be measured the same way: they come from one alignment pass.

Interval estimation lives in :mod:`shams_eval.iaa` and nowhere else. The
agreement study is the only place the package makes an inferential claim; a task
score here is a descriptive rate over a fixed benchmark, reported with the
utterance count of the cell it was computed on.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from . import predictions as pred_io
from .dataset import Dataset, Utterance
from .metrics import METRICS, ErrorCounts, total
from .metrics import score as metric_score
from .strata import STRATA
from .tiers import TASKS, Task


@dataclass
class Evaluation:
    """Everything computed for one model on one task."""

    task: str
    model_key: str
    n: int
    normalization_tiers: tuple[str, ...]
    metrics: tuple[str, ...]
    overall: dict
    by: dict
    provenance: dict

    def as_dict(self) -> dict:
        return {
            "task": self.task,
            "model_key": self.model_key,
            "n": self.n,
            "normalization_tiers": list(self.normalization_tiers),
            "metrics": list(self.metrics),
            "overall": self.overall,
            "by": self.by,
            "provenance": self.provenance,
        }

    def score(self, metric: str, norm_tier: str | None = None) -> float:
        norm_tier = norm_tier or self.normalization_tiers[0]
        return self.overall[norm_tier][metric]["rate"]


def _block(counts: dict[str, dict[str, list[ErrorCounts]]], idx: Sequence[int],
           utterances: Sequence[Utterance]) -> dict:
    """Rate and descriptive counts for one subset."""
    out: dict = {
        "n": len(idx),
        "duration_min": round(sum(utterances[i].duration for i in idx) / 60.0, 2),
        "gender": {},
    }
    for i in idx:
        g = utterances[i].gender
        out["gender"][g] = out["gender"].get(g, 0) + 1
    out["gender"] = dict(sorted(out["gender"].items()))

    for norm_tier, per_metric in counts.items():
        subset = {m: [c[i] for i in idx] for m, c in per_metric.items()}
        out[norm_tier] = {
            m: {
                "rate": metric_score(subset[m], m),
                "counts": total(subset[m]).as_dict(),
            }
            for m in subset
        }
    return out


def evaluate(path: str | Path, task: str | Task, dataset: Dataset, *,
             on_empty: str = "score",
             normalization_tiers: Sequence[str] | None = None,
             metrics: Sequence[str] | None = None) -> Evaluation:
    task = TASKS[task] if isinstance(task, str) else task
    tier = task.tier
    norm_tiers = tuple(normalization_tiers or task.scored_normalization_tiers)
    metric_names = tuple(metrics or task.scored_metrics)

    preds = pred_io.load(path, task, dataset, on_empty=on_empty)
    prov = pred_io.provenance(preds, task, dataset)
    if not preds.seg_ids:
        raise ValueError(f"{path}: no scorable records")

    utterances = [dataset.get(s) for s in preds.seg_ids]
    references = [u.tiers.get(tier.field) or "" for u in utterances]

    # The single alignment pass.
    counts: dict[str, dict[str, list[ErrorCounts]]] = {}
    for norm_tier in norm_tiers:
        # The gold takes the tier's own normalizer; the hypothesis takes its
        # system's, which differs only for a system that emits another alphabet.
        refs = [tier.normalizer(norm_tier)(r) for r in references]
        hyp_norm = tier.normalizer(norm_tier, preds.model_key)
        hyps = [hyp_norm(h) for h in preds.hypotheses]
        counts[norm_tier] = {
            m: [METRICS[m][0](r, h) for r, h in zip(refs, hyps)]
            for m in metric_names
        }

    all_idx = list(range(len(preds.seg_ids)))
    overall = _block(counts, all_idx, utterances)

    # Strata. A key is a label, or a pair of labels for the nested cut.
    groups: dict[str, dict] = {}
    for name, key_fn in STRATA.items():
        buckets: dict[object, list[int]] = {}
        for i, u in enumerate(utterances):
            buckets.setdefault(key_fn(u), []).append(i)
        out: dict = {}
        for k, idx in sorted(buckets.items(), key=lambda kv: str(kv[0])):
            block = _block(counts, idx, utterances)
            if isinstance(k, tuple):
                out.setdefault(k[0], {})[k[1]] = block
            else:
                out[k] = block
        groups[name] = out

    return Evaluation(
        task=task.key,
        model_key=preds.model_key,
        n=len(preds.seg_ids),
        normalization_tiers=norm_tiers,
        metrics=metric_names,
        overall=overall,
        by=groups,
        provenance=prov,
    )
