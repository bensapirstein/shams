"""The one place a SHAMS reference or a stratum label is read.

References and strata are resolved from the released dataset, never from the
prediction file. A prediction file carries a ``reference`` snapshot and stratum
labels as they stood when the model was run, and a run made before a corpus
revision carries the older ones; scoring against them would grade each system
against a different answer key and cut each table by different labels. Resolving
from the release makes the comparison uniform, and any drift is reported instead
of applied (see :func:`shams_eval.predictions.provenance`).

Two sources, same schema:

``hub``    ``shams-nlp/shams`` at a pinned revision. The canonical path.
``file``   A local JSONL snapshot with the same columns, for offline use.

The pin is deliberate: an unpinned ``load_dataset`` silently re-scores against
whatever the Hub holds that morning.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

HUB_DATASET = "shams-nlp/shams"
HUB_SPLIT = "test"

#: Pinned so results are reproducible. Bump deliberately and re-run every table:
#: this constant selects the gold every number is measured against.
HUB_REVISION = "f2715626f12ec17048fdad76e74101bb0fd31bb8"  # tag v1.0

TEXT_FIELDS = ("transcription", "vocalized_transcript", "translit")
# Scoring reads the first five. The rest are provenance, and the last four are
# what `shams-eval build-audio` needs to cut a segment back out of its source.
META_FIELDS = ("subdialect", "source", "gender", "country", "duration",
               "audio_file", "original_file", "title",
               "cut_start_ms", "cut_end_ms")


@dataclass(frozen=True)
class Utterance:
    seg_id: str
    tiers: dict[str, str]
    meta: dict[str, object]

    @property
    def subdialect(self) -> str:
        return str(self.meta.get("subdialect") or "unknown")

    @property
    def source(self) -> str:
        return str(self.meta.get("source") or "unknown")

    @property
    def gender(self) -> str:
        return str(self.meta.get("gender") or "?")

    @property
    def duration(self) -> float:
        try:
            return float(self.meta.get("duration") or 0.0)
        except (TypeError, ValueError):
            return 0.0


class Dataset:
    """The released benchmark, keyed by ``seg_id``."""

    def __init__(self, utterances: dict[str, Utterance], provenance: str):
        self._u = utterances
        self.provenance = provenance

    def __len__(self) -> int:
        return len(self._u)

    def __contains__(self, seg_id: str) -> bool:
        return seg_id in self._u

    def __iter__(self) -> Iterator[Utterance]:
        return iter(self._u.values())

    def get(self, seg_id: str) -> Utterance | None:
        return self._u.get(seg_id)

    def reference(self, seg_id: str, field: str) -> str:
        u = self._u.get(seg_id)
        return (u.tiers.get(field) or "") if u else ""

    @classmethod
    def from_records(cls, records, provenance: str) -> Dataset:
        out: dict[str, Utterance] = {}
        for r in records:
            seg_id = r["seg_id"]
            out[seg_id] = Utterance(
                seg_id=seg_id,
                tiers={f: r.get(f) or "" for f in TEXT_FIELDS},
                meta={f: r.get(f) for f in META_FIELDS},
            )
        return cls(out, provenance)

    @classmethod
    def from_file(cls, path: str | Path) -> Dataset:
        """Read a local snapshot, recording which snapshot rather than where it sat.

        Provenance names the file and a digest of its contents, not the path it
        was read from. A path is a fact about one machine: it puts the caller's
        home directory into ``results/`` and makes a committed results file
        impossible for anyone else to reproduce. A digest says the thing the
        reader actually needs, which is whether these numbers came from the same
        references theirs will.
        """
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        return cls.from_records(rows, provenance=f"file:{path.name}@{digest}")

    @classmethod
    def from_hub(cls, name: str = HUB_DATASET, split: str = HUB_SPLIT,
                 revision: str = HUB_REVISION) -> Dataset:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "reading references from the Hub needs `datasets`: either "
                "`pip install datasets`, or pass --dataset with a local JSONL "
                "snapshot of the release."
            ) from exc

        ds = load_dataset(name, split=split, revision=revision)
        # Drop the audio column before materializing: the text tiers are all the
        # evaluator needs and decoding 1,300 waveforms to score a table is waste.
        keep = [c for c in ds.column_names if c != "audio"]
        rows = ds.select_columns(keep).to_list()
        return cls.from_records(rows, provenance=f"hub:{name}@{revision}:{split}")

