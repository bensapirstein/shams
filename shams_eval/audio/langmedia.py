"""The LangMedia half: fetch each source clip, then cut the segments out of it.

Two steps over one manifest. ``fetch`` downloads every clip in ``clips.csv``
and decodes it to WAV with that clip's recorded profile; ``cut`` resolves each
released row to its clip and writes one WAV per ``seg_id``.

The decode profile matters. The two acquisition rounds decoded differently and
a cut inherits the format of the clip it came from, so the profile is carried
per clip rather than assumed:

``native``     ``-vn -acodec pcm_s16le``, keeping the source's own rate and
               channel count
``16k-mono``   the same, resampled to 16 kHz mono

Using the wrong one yields audio that sounds right and matches nothing.

LangMedia is a small academic site. The default is one request every half
second with an identifying User-Agent; please leave it that way.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .wav import read_wav, slice_ms, write_wav

USER_AGENT = "SHAMS-reconstruction (https://github.com/bensapirstein/shams)"
DELAY_S = 0.5


def require_ffmpeg() -> str:
    """The path to ffmpeg, or a clear failure. It is the one non-Python need."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        sys.exit(
            "ffmpeg is not on PATH, and rebuilding the LangMedia audio needs it "
            "to decode the source clips.\n"
            "  macOS          brew install ffmpeg\n"
            "  Debian/Ubuntu  apt install ffmpeg\n"
            "Then run `shams-eval build-audio` again. Already-fetched clips are "
            "kept, so the run resumes rather than restarts."
        )
    return exe


def decode(src: Path, dst: Path, profile: str) -> None:
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
           "-vn", "-acodec", "pcm_s16le"]
    if profile == "16k-mono":
        cmd += ["-ar", "16000", "-ac", "1"]
    elif profile != "native":
        rate, ch = profile.split(":")[1].split("/")
        cmd += ["-ar", rate, "-ac", ch]
    cmd.append(str(dst))
    subprocess.run(cmd, check=True)


def fetch(clips: list[dict], clip_dir: Path, media_dir: Path, *,
          only: set[str] | None = None, keep_media: bool = False) -> dict:
    """Download and decode the source clips. One bad clip does not stop the run."""
    require_ffmpeg()
    if only:
        clips = [c for c in clips if c["trackname"] in only]
    clip_dir.mkdir(parents=True, exist_ok=True)

    done = skipped = 0
    failed: list[dict] = []
    for c in clips:
        track = c["trackname"]
        dst = clip_dir / f"{track}.wav"
        if dst.exists():
            skipped += 1
            continue
        try:
            media_dir.mkdir(parents=True, exist_ok=True)
            src = media_dir / f"{track}_{Path(c['media_url']).name}"
            if not src.exists():
                req = urllib.request.Request(
                    c["media_url"], headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=120) as r, open(src, "wb") as f:
                    shutil.copyfileobj(r, f)
                time.sleep(DELAY_S)
            decode(src, dst, c["decode"])
            if not keep_media:
                src.unlink(missing_ok=True)
            done += 1
            print(f"  ok   {track} ({c['decode']})")
        except Exception as e:  # noqa: BLE001 - one bad clip must not stop the run
            dst.unlink(missing_ok=True)
            failed.append({"trackname": track, "media_url": c["media_url"],
                           "error": str(e)})
            print(f"  FAIL {track}: {e}\n       {c['media_url']}", file=sys.stderr)

    print(f"{done} decoded, {skipped} already present, {len(failed)} failed")
    return {"decoded": done, "skipped": skipped, "failed": failed}


def _index(clips: list[dict]) -> tuple[dict, dict]:
    by_key = {(c["title"], c["original_file"]): c for c in clips}
    if len(by_key) != len(clips):
        sys.exit("clips.csv: (title, original_file) does not identify one clip")
    # `original_file` may cite either the LangMedia page or the media file
    # depending on when a release was cut; accept both, and title alone where it
    # is unambiguous, rather than failing on a changed citation handle.
    by_key.update({(c["title"], c["media_url"]): c for c in clips})
    seen: dict[str, list[dict]] = {}
    for c in clips:
        seen.setdefault(c["title"], []).append(c)
    return by_key, {t: v[0] for t, v in seen.items() if len(v) == 1}


def cut(rows: list, clips: list[dict], clip_dir: Path, out_dir: Path, *,
        n_frames: dict[str, int] | None = None) -> dict:
    """Cut every released LangMedia segment out of its clip.

    ``n_frames`` takes each segment's length from ``checksums.csv`` instead of
    the published end millisecond. The start of every segment is exact, and so
    is the end of all but the rows whose clips were segmented on sample indices:
    there the released boundary falls between two milliseconds and the published
    value is the rounded one, so a cut made from it runs a few samples long or
    short.
    """
    by_key, by_title = _index(clips)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_frames = n_frames or {}

    cache: dict[str, object] = {}
    written = unresolved = no_clip = trimmed = 0
    for r in rows:
        meta = r.meta
        c = (by_key.get((meta.get("title"), meta.get("original_file")))
             or by_title.get(meta.get("title")))
        if c is None:
            unresolved += 1
            print(f"  FAIL {r.seg_id}: no clip for "
                  f"({meta.get('title')!r}, {meta.get('original_file')})", file=sys.stderr)
            continue
        track = c["trackname"]
        if track not in cache:
            p = clip_dir / f"{track}.wav"
            if not p.exists():
                no_clip += 1
                print(f"  FAIL {r.seg_id}: clip {track} was not fetched", file=sys.stderr)
                continue
            cache[track] = read_wav(p)
        clip = cache[track]
        seg = slice_ms(clip, meta["cut_start_ms"], meta["cut_end_ms"])
        want = n_frames.get(r.seg_id)
        if want is not None and want != seg.n_frames:
            start = int(meta["cut_start_ms"] * clip.framerate / 1000.0) * clip.frame_width
            seg = seg._replace(frames=clip.frames[start:start + want * clip.frame_width])
            trimmed += 1
        write_wav(out_dir / f"{r.seg_id}.wav", seg)
        written += 1

    if trimmed:
        print(f"{trimmed} segments took their length from checksums.csv")
    print(f"{written} segments cut, {unresolved} unresolved, {no_clip} missing clips")
    return {"written": written, "unresolved": unresolved, "no_clip": no_clip}
