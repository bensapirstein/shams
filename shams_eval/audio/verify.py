"""Is a rebuild the same audio as the release?

Every rebuilt segment is checked against ``checksums.csv``, which carries the
format, the frame count and a SHA-256 of the decoded PCM of each released
segment. No reference audio is needed, so anyone can run it.

Three verdicts matter:

``identical``         the decoded PCM hashes equal. The rebuild is the released
                      audio, bit for bit.
``within_tolerance``  same sample rate, channels, sample width and frame count,
                      different bytes. This is the expected result for a rebuild
                      made with a different ffmpeg than the release: two builds
                      decoding one AAC stream disagree by a least significant
                      bit on about half the samples, so hash equality only holds
                      within a single build. The segmentation is right and the
                      audio is the same audio.
``length_mismatch`` / ``format_mismatch``
                      a real defect: the wrong cut boundary, or a clip decoded
                      at the wrong profile. Counted as a failure.

The separation between decoder noise and a real difference was measured when
the release was cut, against the released audio itself; ``--reference DIR``
reproduces that stronger check by measuring the signal to difference ratio of
each mismatching row and accepting it only above ``--tolerance-db``. Against
checksums alone there is no signal to measure, so ``within_tolerance`` there
rests on format and length agreeing rather than on a measured ratio.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from . import checksums as load_checksums
from .wav import compare, md5, pcm_sha256, read_wav, snr_db

PASSING = ("identical", "within_tolerance")
DEFAULT_TOLERANCE_DB = 55.0


def _against_checksums(clip, exp: dict) -> dict:
    res = {"a_frames": clip.n_frames, "b_frames": int(exp["n_frames"]),
           "frame_delta": clip.n_frames - int(exp["n_frames"]),
           "a_format": f"{clip.framerate}Hz/{clip.nchannels}ch/{clip.sampwidth*8}bit",
           "b_format": f"{exp['sample_rate']}Hz/{exp['channels']}ch"
                       f"/{int(exp['sample_width'])*8}bit"}
    if (str(clip.framerate) != exp["sample_rate"]
            or str(clip.nchannels) != exp["channels"]
            or str(clip.sampwidth) != exp["sample_width"]):
        res["verdict"] = "format_mismatch"
    elif clip.n_frames != int(exp["n_frames"]):
        res["verdict"] = "length_mismatch"
    elif pcm_sha256(clip) == exp["pcm_sha256"]:
        res["verdict"] = "identical"
    else:
        res["verdict"] = "within_tolerance"
    return res


def verify(audio_dir: Path, *, reference: Path | None = None,
           tolerance_db: float | None = None, limit: int | None = None) -> dict:
    expected = load_checksums()
    verdicts: Counter = Counter()
    by_source: dict[str, Counter] = defaultdict(Counter)
    failures: list[dict] = []
    n = 0

    for seg_id, exp in sorted(expected.items()):
        if limit and n >= limit:
            break
        src = exp["source"]
        got = audio_dir / f"{seg_id}.wav"
        if not got.exists():
            verdicts["not_reconstructed"] += 1
            by_source[src]["not_reconstructed"] += 1
            continue
        n += 1
        clip = read_wav(got)
        if reference is not None:
            ref_path = reference / f"{seg_id}.wav"
            if not ref_path.exists():
                verdicts["no_reference"] += 1
                by_source[src]["no_reference"] += 1
                continue
            ref = read_wav(ref_path)
            res = compare(clip, ref)
            res["bytes_identical"] = md5(got) == md5(ref_path)
            if tolerance_db is not None and res["verdict"] == "sample_mismatch":
                res["snr_db"] = round(snr_db(clip, ref), 2)
                if res["snr_db"] >= tolerance_db:
                    res["verdict"] = "within_tolerance"
        else:
            res = _against_checksums(clip, exp)
        verdicts[res["verdict"]] += 1
        by_source[src][res["verdict"]] += 1
        if res["verdict"] not in PASSING:
            failures.append({"seg_id": seg_id, "source": src, **res})

    return {"checked": sum(verdicts.values()), "verdicts": verdicts,
            "by_source": by_source, "failures": failures}


def format_report(report: dict) -> str:
    lines = [f"{report['checked']} segments checked", ""]
    for v, c in report["verdicts"].most_common():
        lines.append(f"  {v:20s} {c:5d}")
    lines.append("")
    for src, counts in sorted(report["by_source"].items()):
        lines.append(f"  {src:12s} " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))
    failures = report["failures"]
    if failures:
        lines += ["", f"First {min(20, len(failures))} of {len(failures)} failures:"]
        for f_ in failures[:20]:
            lines.append(f"  {f_['seg_id']:12s} {f_['verdict']:18s} "
                         f"frames {f_.get('a_frames')} vs {f_.get('b_frames')} "
                         f"({f_.get('frame_delta')})  "
                         f"{f_.get('a_format')} vs {f_.get('b_format')}")
    return "\n".join(lines)


def write_report(report: dict, path: Path) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# Audio reconstruction check\n\n```\n{format_report(report)}\n```\n",
                    encoding="utf-8")
    csv_path = path.with_suffix(".failures.csv")
    failures = report["failures"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if failures:
            keys = sorted({k for f_ in failures for k in f_})
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(failures)
        else:
            # write the empty file too, so a stale one from an earlier run cannot
            # be mistaken for the current result
            csv.writer(f).writerow(["seg_id", "verdict", "note"])
    return [path, csv_path]
