# Inter-annotator agreement — the released study

The artifact behind §2.4 and Table 3. Two files, joined on `seg_id`, plus the
scorer in `shams_eval/iaa.py`. Together they reproduce the table end to end:

```bash
shams-eval iaa
```

No dataset access and no optional dependency needed; the study is
self-contained, and the intervals are seeded, so the table is byte-reproducible.

## `iaa_subset.jsonl` — 150 items, 30 per variety

| field | meaning |
|---|---|
| `seg_id` | utterance identifier; joins to `shams-nlp/shams` for the audio and the remaining tiers |
| `subdialect` | Levantine variety, the stratification of the sample |
| `gender` | speaker gender |
| `source` | source corpus, `casablanca` or `langmedia` |
| `transcription` | unvocalized Arabic orthography, the text both annotators diacritized |
| `reference_vocalized` | annotator A's diacritized label, identical to the released gold |

`reference_vocalized` is called the *reference* because the scorer takes it as
the first row of the confusion matrix, not because it is more correct than
annotator B's. Cohen's κ is symmetric.

Every `reference_vocalized` here equals the released dataset's
`vocalized_transcript` for the same `seg_id`, so the two artifacts can be joined
without disagreeing.

## `iaa_annotator_b.jsonl` — 150 items

`seg_id` and `vocalized`: annotator B's label for the same utterance, and
nothing else. The Label Studio platform metadata — annotator ids, task ids,
timestamps, lead times — is not part of the release, and neither annotator is
identifiable from these files.

Annotator B saw annotator A's consonantal skeleton and edited the diacritics on
it, so the two files carry the same word sequence except where B respelled a
word. Those cases are an orthographic divergence, not a diacritic disagreement:
the scorer aligns them out with difflib and counts them separately, because
otherwise one respelling would register as a run of errors down the rest of the
word.

## What the study measures

Both annotators are native Palestinian Arabic speakers who diacritized from the
audio. The unit of agreement is the **diacritic slot** — one per Arabic letter,
labelled with the set of marks it carries. Utterance-level agreement would give
150 observations; slots give more than seven thousand, which is what lets a
150-utterance study support a per-variety breakdown at all.

The `DER` column of `results/iaa.md` is the share of compared slots on which
the two annotators disagree. It is not the DER of the results tables, which
follows the CATT implementation and counts word boundaries as positions.

Confidence intervals are cluster bootstraps resampling **utterances**, not
slots: slots within an utterance are not independent, and a naive slot bootstrap
would give intervals several times too narrow.

The design has a stated limit, and the report states it: B re-diacritized A's
transcription rather than transcribing from scratch, so this measures the
reliability of the diacritization decision *given* the transcription, not of the
transcription itself.
