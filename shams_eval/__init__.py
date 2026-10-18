"""SHAMS evaluation: one metric implementation across all four benchmark tiers.

    from shams_eval import dataset, evaluate
    ds = dataset.Dataset.from_file("shams_v2.jsonl")
    result = evaluate.evaluate("predictions/asr/gemini-asr.jsonl", "asr", ds)
    result.score("wer")

See :mod:`shams_eval.tiers` for how tasks and tiers relate, and the README for
the command-line entry points.
"""
__version__ = "1.0.0"

from . import dataset, evaluate, iaa, metrics, predictions, strata, tiers

__all__ = ["dataset", "evaluate", "iaa", "metrics", "predictions",
           "strata", "tiers"]
