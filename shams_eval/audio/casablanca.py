"""The Casablanca half: a lookup, not a cut.

SHAMS inherits Casablanca's segmentation and identifiers whole. A released
Casablanca row carries no cut, and its audio is the corpus's own bytes, so this
step pulls the rows whose ``seg_id`` SHAMS uses out of the corpus and writes
them unchanged.

The corpus is ``UBC-NLP/Casablanca`` on the Hub, one config per country, and it
is access-gated: accept the terms on the dataset page and log in with
``huggingface-cli login`` first. SHAMS draws on the Palestine and Jordan
configs. Splits are searched rather than assumed, so a row moving between splits
upstream does not break the lookup, though it would mean the corpus had changed
under the release, which ``verify-audio`` then shows.
"""
from __future__ import annotations

import sys
from pathlib import Path

CORPUS = "UBC-NLP/Casablanca"
CONFIGS = ["Palestine", "Jordan"]
SPLITS = ["validation", "test"]


def fetch(seg_ids: set[str], out_dir: Path) -> dict:
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise ImportError(
            "fetching the Casablanca segments needs `datasets`: "
            "`pip install datasets`, then accept the terms at "
            f"https://huggingface.co/datasets/{CORPUS} and run `huggingface-cli login`."
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(seg_ids)
    written = 0
    for config in CONFIGS:
        if not wanted:
            break
        ds = load_dataset(CORPUS, config, streaming=True)
        for split in SPLITS:
            if split not in ds:
                continue
            # decode=False keeps the corpus's own bytes: decoding and re-encoding
            # would change the file while leaving it audibly identical
            stream = ds[split].cast_column("audio", Audio(decode=False))
            for ex in stream:
                seg_id = ex.get("seg_id")
                if seg_id not in wanted:
                    continue
                (out_dir / f"{seg_id}.wav").write_bytes(ex["audio"]["bytes"])
                wanted.discard(seg_id)
                written += 1
            if not wanted:
                break

    print(f"{written} of {len(seg_ids)} Casablanca segments written")
    if wanted:
        print(f"WARNING {len(wanted)} seg_ids not found in the corpus "
              f"(first: {', '.join(sorted(wanted)[:5])})", file=sys.stderr)
    return {"written": written, "missing": sorted(wanted)}
