# Released model outputs

One JSON Lines file per system per task, as each runner emitted it. They are not
reformatted or re-keyed, so they remain a record of what the model produced.

```
asr/               9 systems x 1,300 utterances
diacritization/    7 systems x 1,300 utterances
g2p/               6 systems x 1,300 utterances
a2p/               5 systems x 1,300 utterances
```

`MANIFEST.json` records, per file, the utterance count, how many frozen
`reference` snapshots were refreshed to the released text, and a note where a
file needs one.

Every record carries a `seg_id` that joins to `shams-nlp/shams` and a hypothesis
under one of the field names listed in the root README. Everything else in the
record is provenance: `reference`, `subdialect`, `source`, `gender`, timestamps,
model version. It is reported by `shams-eval` and not scored, because these runs
span several revisions of the corpus and a file's own snapshot of the gold or of
a variety label may predate the release. Both are resolved from the dataset, and
any drift is reported.

Run `shams-eval score <task>` and read the provenance block above the table.
