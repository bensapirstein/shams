"""Command line entry points.

    shams-eval score asr                     # every ASR prediction file
    shams-eval score asr predictions/asr/gemini-asr.jsonl
    shams-eval iaa
    shams-eval tables results/asr.json
    shams-eval examples                      # the qualitative tables, from the release
    shams-eval examples --verify ../paper/tables
    shams-eval build-audio                   # rebuild the audio from its sources
    shams-eval verify-audio audio/

Defaults are chosen so that the bare command reproduces the paper: all
prediction files for the task, references from the pinned release. Everything a
table depends on is written into the results JSON beside the numbers.

``iaa`` is the one subcommand that estimates intervals, and it has its own
resampling options. A task score is a descriptive rate over a fixed benchmark.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import dataset as ds_mod
from . import iaa as iaa_mod
from .evaluate import evaluate
from .report import console
from .tiers import TASKS

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "predictions"
RESULTS = ROOT / "results"
IAA_DIR = ROOT / "iaa"


def _dataset(args) -> ds_mod.Dataset:
    if args.dataset:
        return ds_mod.Dataset.from_file(args.dataset)
    if args.references == "file":
        sys.exit("--references file needs --dataset PATH")
    return ds_mod.Dataset.from_hub(revision=args.revision)


def _add_reference_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--references", choices=["hub", "file"], default="hub",
                   help="read the release from the pinned Hub revision (default) or "
                        "from a local JSONL snapshot")
    p.add_argument("--dataset", type=Path,
                   help="path to a local dataset snapshot; implies --references file")
    p.add_argument("--revision", default=ds_mod.HUB_REVISION,
                   help=f"Hub revision to pin (default: {ds_mod.HUB_REVISION})")


def cmd_score(args) -> None:
    task = TASKS[args.task]
    files = [Path(f) for f in args.files] or sorted((PREDICTIONS / args.task).glob("*.jsonl"))
    if not files:
        sys.exit(f"no prediction files for task {args.task!r}")

    data = _dataset(args)
    print(f"references: {data.provenance} ({len(data)} utterances)", file=sys.stderr)

    results = []
    for f in files:
        print(f"scoring {f.name} ...", file=sys.stderr)
        results.append(evaluate(f, task, data, on_empty=args.on_empty))

    print(console.headline(results))
    print(console.provenance(results))
    for r in results:
        print(console.breakdown(r))
    if args.by_source:
        for r in results:
            print(console.source_contrast(r))

    out = Path(args.out) if args.out else RESULTS / f"{args.task}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([r.as_dict() for r in results], ensure_ascii=False,
                              indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}", file=sys.stderr)

    if not args.no_tables:
        from .report import latex

        written = latex.write_all(results, RESULTS / "tables", task=args.task)
        for p in written:
            print(f"wrote {p}", file=sys.stderr)


def cmd_iaa(args) -> None:
    units = iaa_mod.load_units(args.subset, args.annotator_b)
    if not units:
        sys.exit("no overlapping items between the subset and the second pass")
    study = iaa_mod.run(units, seed=args.seed, bootstrap=args.bootstrap,
                        permutations=args.permutations)

    from .report import iaa_report, latex

    report = iaa_report.markdown(study, n_subset=len(units))
    print(report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n", encoding="utf-8")
    (out.parent / "iaa.json").write_text(
        json.dumps(study.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out.parent / "iaa_per_item.jsonl", "w", encoding="utf-8") as f:
        for row in study.per_item:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tex = latex.iaa_table(study)
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    (RESULTS / "tables" / "iaa.tex").write_text(tex, encoding="utf-8")
    print(f"\nwrote {out}, {out.parent / 'iaa.json'}, "
          f"{RESULTS / 'tables' / 'iaa.tex'}", file=sys.stderr)


def cmd_examples(args) -> None:
    from .report import examples

    data = _dataset(args)
    print(f"references: {data.provenance} ({len(data)} utterances)", file=sys.stderr)
    if args.verify:
        stale = 0
        for table, what, found in examples.verify(data, args.verify):
            print(f"{'ok   ' if found else 'STALE'}  {table}  {what}")
            stale += not found
        print(f"\n{stale} quoted strings differ from the release", file=sys.stderr)
        return
    for p in examples.write_all(data, args.out_dir):
        print(f"wrote {p}", file=sys.stderr)


def cmd_build_audio(args) -> None:
    from .audio import casablanca, langmedia
    from .audio import checksums as load_checksums
    from .audio import clips as load_clips

    out = Path(args.out)
    clip_dir = Path(args.clips_dir) if args.clips_dir else out / ".clips"
    data = _dataset(args)
    print(f"metadata: {data.provenance} ({len(data)} utterances)", file=sys.stderr)

    rows = list(data)
    lang = [u for u in rows if u.source == "langmedia"]
    casa = {u.seg_id for u in rows if u.source == "casablanca"}

    n_frames = ({k: int(v["n_frames"]) for k, v in load_checksums().items()}
                if args.exact_length else None)

    manifest = load_clips()
    print(f"\nfetching {len(manifest)} LangMedia clips -> {clip_dir}")
    fetched = langmedia.fetch(manifest, clip_dir, clip_dir / "media",
                              only=set(args.only) if args.only else None,
                              keep_media=args.keep_media)

    print(f"\ncutting {len(lang)} LangMedia segments -> {out}")
    cut = langmedia.cut(lang, manifest, clip_dir, out, n_frames=n_frames)

    casa_result = {"written": 0, "missing": []}
    if args.skip_casablanca:
        print(f"\nskipping {len(casa)} Casablanca segments")
    else:
        print(f"\nfetching {len(casa)} Casablanca segments -> {out}")
        casa_result = casablanca.fetch(casa, out)

    print(f"\n{cut['written'] + casa_result['written']} segments in {out}")
    if fetched["failed"]:
        print(f"\n{len(fetched['failed'])} clips failed:", file=sys.stderr)
        for f_ in fetched["failed"]:
            print(f"  {f_['trackname']}  {f_['media_url']}\n    {f_['error']}",
                  file=sys.stderr)
    print(f"source clips are cached in {clip_dir}; delete it once the build is "
          f"verified.\nnext: shams-eval verify-audio {out}", file=sys.stderr)
    if fetched["failed"] or cut["unresolved"] or cut["no_clip"] or casa_result["missing"]:
        sys.exit(1)


def cmd_verify_audio(args) -> None:
    from .audio import verify as verify_mod

    audio_dir = Path(args.audio_dir)
    if not audio_dir.is_dir():
        sys.exit(f"{audio_dir} is not a directory; run `shams-eval build-audio` first")
    tolerance = (verify_mod.DEFAULT_TOLERANCE_DB if args.tolerance_db is None
                 else args.tolerance_db)
    report = verify_mod.verify(audio_dir, reference=args.reference,
                               tolerance_db=tolerance, limit=args.limit)
    print(verify_mod.format_report(report))
    if args.report:
        for p_ in verify_mod.write_report(report, Path(args.report)):
            print(f"wrote {p_}", file=sys.stderr)
    sys.exit(1 if report["failures"] or report["verdicts"]["not_reconstructed"] else 0)


def cmd_tables(args) -> None:
    from .evaluate import Evaluation
    from .report import latex

    results = []
    for r in json.loads(Path(args.results).read_text(encoding="utf-8")):
        r["normalization_tiers"] = tuple(r["normalization_tiers"])
        r["metrics"] = tuple(r["metrics"])
        results.append(Evaluation(**r))
    task = results[0].task
    for p in latex.write_all(results, Path(args.out_dir), task=task):
        print(f"wrote {p}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="shams-eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("score", help="score prediction files for one task")
    s.add_argument("task", choices=sorted(TASKS))
    s.add_argument("files", nargs="*", help="default: every .jsonl under predictions/<task>/")
    _add_reference_args(s)
    s.add_argument("--on-empty", choices=["score", "skip"], default="score",
                   help="an empty hypothesis is a full deletion (default) or dropped "
                        "from the corpus")
    s.add_argument("--by-source", action="store_true",
                   help="print the variety x source breakdown")
    s.add_argument("--out", help="results JSON (default: results/<task>.json)")
    s.add_argument("--no-tables", action="store_true")
    s.set_defaults(func=cmd_score)

    i = sub.add_parser("iaa", help="score the inter-annotator agreement study")
    i.add_argument("--subset", type=Path, default=IAA_DIR / "iaa_subset.jsonl")
    i.add_argument("--annotator-b", type=Path, default=IAA_DIR / "iaa_annotator_b.jsonl")
    i.add_argument("--out", type=Path, default=RESULTS / "iaa.md")
    i.add_argument("--seed", type=int, default=iaa_mod.DEFAULT_SEED)
    i.add_argument("--bootstrap", type=int, default=iaa_mod.DEFAULT_BOOTSTRAP)
    i.add_argument("--permutations", type=int, default=iaa_mod.DEFAULT_PERMUTATIONS)
    i.set_defaults(func=cmd_iaa)

    e = sub.add_parser("examples", help="fill the qualitative tables from the release")
    _add_reference_args(e)
    e.add_argument("--out-dir", type=Path, default=RESULTS / "tables")
    e.add_argument("--verify", type=Path, metavar="DIR",
                   help="instead of writing, report which quoted strings a manuscript's "
                        "tables/ directory no longer matches")
    e.set_defaults(func=cmd_examples)

    t = sub.add_parser("tables", help="regenerate LaTeX from a results JSON")
    t.add_argument("results")
    t.add_argument("--out-dir", default=str(RESULTS / "tables"))
    t.set_defaults(func=cmd_tables)

    b = sub.add_parser("build-audio", help="rebuild the benchmark audio from its sources")
    b.add_argument("--out", type=Path, default=Path("audio"),
                   help="where the segments are written (default: audio/)")
    _add_reference_args(b)
    b.add_argument("--clips-dir", type=Path,
                   help="cache for the LangMedia source clips (default: <out>/.clips)")
    b.add_argument("--only", nargs="*", metavar="TRACK",
                   help="fetch only these clips from clips.csv")
    b.add_argument("--skip-casablanca", action="store_true",
                   help="build the LangMedia half alone, without the access-gated corpus")
    b.add_argument("--keep-media", action="store_true",
                   help="keep each downloaded media file beside its decoded clip")
    b.add_argument("--no-exact-length", dest="exact_length", action="store_false",
                   help="cut to the published end millisecond instead of taking each "
                        "segment's frame count from checksums.csv")
    b.set_defaults(func=cmd_build_audio)

    v = sub.add_parser("verify-audio", help="check a rebuild against checksums.csv")
    v.add_argument("audio_dir", type=Path, help="directory of rebuilt segments")
    v.add_argument("--reference", type=Path, metavar="DIR",
                   help="a directory of released wavs to measure against, instead of "
                        "checksums.csv; reports the signal to difference ratio of "
                        "every mismatching segment")
    v.add_argument("--tolerance-db", type=float, default=None,
                   help="with --reference, accept a difference quieter than this "
                        "(default: 55, the measured separation between decoder "
                        "noise and a real difference)")
    v.add_argument("--limit", type=int, help="check only the first N segments")
    v.add_argument("--report", type=Path, help="also write a markdown report here")
    v.set_defaults(func=cmd_verify_audio)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
