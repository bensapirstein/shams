r"""The four aligned tiers of SHAMS, and the four tasks defined over them.

Every utterance in SHAMS carries the same content at four levels:

    audio  ──▶  orthographic  ──▶  diacritized  ──▶  phonetic
               (transcription)  (vocalized_       (translit,
                                 transcript)       DIN-31635)

A task is a pair of tiers. That is the whole taxonomy, and it is why one
evaluator serves all four:

    ================  ==============  ==============  =======================
    task              source          target          metrics
    ================  ==============  ==============  =======================
    ASR               audio           orthographic    WER, CER
    diacritization    orthographic    diacritized     DER
    G2P               orthographic    phonetic        CER
    A2P               audio           phonetic        CER (capped)
    ================  ==============  ==============  =======================

G2P skips the diacritized tier: the systems are given the undiacritized
orthography, as \S\ref{sec:g2p} defines the task, so it is the one task whose
two tiers are not adjacent in the chain above.

The target tier fixes three things at once: which dataset column is the gold,
which normalizer family applies, and which metrics are meaningful. Nothing else
in the package needs to know what task it is running — the evaluator takes a
:class:`Task`, resolves its target :class:`Tier`, and the rest is shared. Adding
a fifth task means adding a row here.

Two tasks sharing a target tier are directly comparable, which is what the
§3.4 direct-A2P versus ASR→G2P-cascade contrast rests on: both land on the
phonetic tier and are scored by the same function under the same fold.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from .normalization import diacritic, orthographic, phonetic


@dataclass(frozen=True)
class Tier:
    """One representation level, and how text at that level is scored."""

    key: str
    field: str
    """The column of the released dataset that carries the gold for this tier."""
    normalizers: Mapping[str, Callable[[str | None], str]]
    """Normalization tier name -> a function applied to the gold, and to the
    hypothesis of any system not named in :attr:`system_normalizers`."""
    metrics: tuple[str, ...]
    """Metric names from :data:`shams_eval.metrics.METRICS`, primary first."""
    system_normalizers: Mapping[str, Mapping[str, Callable[[str | None], str]]] = (
        field(default_factory=dict))
    """model key -> that system's normalizers, for the hypothesis side only.

    The fold that makes two systems comparable is symmetric and lives in
    :attr:`normalizers`. Reaching the alphabet it operates on is not, because a
    system can emit its own inventory: mapping IPA or a house romanization onto
    the reference symbol set is a property of the system, not of the tier. The
    gold always goes through :attr:`normalizers`, so adding a system here cannot
    move another system's number.
    """

    @property
    def normalization_tiers(self) -> tuple[str, ...]:
        return tuple(self.normalizers)

    def normalizer(self, norm_tier: str, model_key: str | None = None
                   ) -> Callable[[str | None], str]:
        """The hypothesis normalizer for this system, or the gold's when None."""
        if model_key is None:
            return self.normalizers[norm_tier]
        return self.system_normalizers.get(model_key, self.normalizers)[norm_tier]


ORTHOGRAPHIC = Tier(
    key="orthographic",
    field="transcription",
    normalizers=orthographic.TIERS,
    metrics=("wer", "cer"),
)

#: DER is the definition the CATT reference implementation uses: the two sides
#: are reduced to one mark per letter and per word boundary, and the mark
#: streams are aligned by edit distance. It is the only DER reported here, so a
#: figure in this table can be read against a figure published with CATT without
#: a conversion.
DIACRITIZED = Tier(
    key="diacritized",
    field="vocalized_transcript",
    normalizers=diacritic.TIERS,
    metrics=("der",),
)

PHONETIC = Tier(
    key="phonetic",
    field="translit",
    normalizers=phonetic.TIERS,
    metrics=("cer",),
    system_normalizers=phonetic.SYSTEM_TIERS,
)

TIERS: dict[str, Tier] = {t.key: t for t in (ORTHOGRAPHIC, DIACRITIZED, PHONETIC)}


@dataclass(frozen=True)
class Task:
    """A benchmark task: predict one tier from another."""

    key: str
    source: str
    target: str
    prediction_fields: tuple[str, ...]
    """JSON keys a prediction file may use for the hypothesis, in priority order.

    Runners in different sub-benchmarks named this field differently
    (``prediction``, ``prediction_raw``, ``pred_diacritics``,
    ``prediction_translit``). Reading them all here, rather than rewriting the
    files, keeps the released predictions byte-identical to what the runners
    emitted — a prediction file that has been reformatted is no longer evidence.
    """
    metrics: tuple[str, ...] = ()
    """Overrides the target tier's metrics when the task scores differently."""
    normalization_tiers: tuple[str, ...] = ()
    """Overrides the order of the target tier's normalizations.

    The first entry is the primary one: the figure a table prints, with the gap
    to the second set beside it. ASR and both phonetic tasks print the
    ``neutral`` fold, the convention in Arabic ASR evaluation and in phone
    recognition: a system is not charged for a spelling or notation choice that
    carries no contrast in the language. The ``strict`` gap beside it is what
    says how far the system's conventions are from the annotation's."""
    display: str = ""

    @property
    def tier(self) -> Tier:
        return TIERS[self.target]

    @property
    def scored_metrics(self) -> tuple[str, ...]:
        return self.metrics or self.tier.metrics

    @property
    def reference_field(self) -> str:
        return self.tier.field

    @property
    def scored_normalization_tiers(self) -> tuple[str, ...]:
        return self.normalization_tiers or self.tier.normalization_tiers

    def hypothesis(self, record: Mapping) -> str | None:
        for key in self.prediction_fields:
            if key in record:
                return record[key]
        return None


TASKS: dict[str, Task] = {
    "asr": Task(
        key="asr", source="audio", target="orthographic",
        prediction_fields=("prediction", "transcription_pred"),
        normalization_tiers=("neutral", "strict"),
        display="ASR",
    ),
    "diacritization": Task(
        key="diacritization", source="orthographic", target="diacritized",
        prediction_fields=("prediction_raw", "pred_diacritics", "prediction"),
        display="Diacritization",
    ),
    "g2p": Task(
        key="g2p", source="orthographic", target="phonetic",
        prediction_fields=("prediction_translit", "pred_translit", "prediction"),
        normalization_tiers=("neutral", "strict"),
        display="G2P",
    ),
    "a2p": Task(
        key="a2p", source="audio", target="phonetic",
        prediction_fields=("prediction_translit", "prediction"),
        metrics=("cer_capped",),
        normalization_tiers=("neutral", "strict"),
        display="A2P",
    ),
}
