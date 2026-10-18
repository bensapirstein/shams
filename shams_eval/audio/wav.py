"""WAV read, millisecond slice and write, on the standard library.

The released LangMedia segments were cut by converting a millisecond position
to a frame index as ``int(ms * frame_rate / 1000.0)`` and slicing the raw PCM
buffer. For 16-bit PCM that is a byte-range slice, so reproducing it needs no
audio library and no third-party dependency.

Nothing here resamples, dithers or converts format. A cut inherits the sample
rate and channel count of the clip it was taken from, which is why each clip
has to be decoded with the profile recorded for it in ``clips.csv``.
"""
from __future__ import annotations

import hashlib
import wave
from pathlib import Path
from typing import NamedTuple


class Clip(NamedTuple):
    nchannels: int
    sampwidth: int
    framerate: int
    frames: bytes

    @property
    def frame_width(self) -> int:
        return self.nchannels * self.sampwidth

    @property
    def n_frames(self) -> int:
        return len(self.frames) // self.frame_width

    @property
    def duration_ms(self) -> float:
        return self.n_frames / self.framerate * 1000.0


def read_wav(path: str | Path) -> Clip:
    with wave.open(str(path), "rb") as w:
        if w.getcomptype() != "NONE":
            raise ValueError(f"{path}: compressed WAV not supported")
        return Clip(w.getnchannels(), w.getsampwidth(), w.getframerate(),
                    w.readframes(w.getnframes()))


def write_wav(path: str | Path, clip: Clip) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(clip.nchannels)
        w.setsampwidth(clip.sampwidth)
        w.setframerate(clip.framerate)
        w.writeframes(clip.frames)


def slice_ms(clip: Clip, start_ms: float, end_ms: float) -> Clip:
    """Cut ``[start_ms, end_ms)`` out of ``clip`` on the raw buffer.

    Both bounds are clamped to the clip length in whole milliseconds before
    conversion, and the conversion truncates rather than rounds. A cut that
    runs past the end of a clip therefore yields the same short segment the
    release carries, rather than a longer one.
    """
    length_ms = round(clip.duration_ms)
    start_ms = min(max(start_ms, 0), length_ms)
    end_ms = min(max(end_ms, 0), length_ms)
    fw = clip.frame_width
    start = int(start_ms * clip.framerate / 1000.0) * fw
    end = int(end_ms * clip.framerate / 1000.0) * fw
    return clip._replace(frames=clip.frames[start:end])


def md5(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pcm_sha256(clip: Clip) -> str:
    """Hash of the decoded audio only, header and metadata excluded."""
    h = hashlib.sha256()
    h.update(f"{clip.nchannels}/{clip.sampwidth}/{clip.framerate}/".encode())
    h.update(clip.frames)
    return h.hexdigest()


def compare(a: Clip, b: Clip) -> dict:
    """Sample-exact comparison. Returns a verdict dict, never raises."""
    out = {
        "format_match": (a.nchannels, a.sampwidth, a.framerate)
                        == (b.nchannels, b.sampwidth, b.framerate),
        "a_format": f"{a.framerate}Hz/{a.nchannels}ch/{a.sampwidth*8}bit",
        "b_format": f"{b.framerate}Hz/{b.nchannels}ch/{b.sampwidth*8}bit",
        "a_frames": a.n_frames,
        "b_frames": b.n_frames,
        "frame_delta": a.n_frames - b.n_frames,
    }
    if not out["format_match"]:
        out["verdict"] = "format_mismatch"
        return out
    if a.frames == b.frames:
        out["verdict"] = "identical"
        out["max_abs_delta"] = 0
        return out
    if a.n_frames != b.n_frames:
        out["verdict"] = "length_mismatch"
        return out
    if a.sampwidth == 2:
        import array
        xa, xb = array.array("h"), array.array("h")
        xa.frombytes(a.frames)
        xb.frombytes(b.frames)
        out["max_abs_delta"] = max(abs(i - j) for i, j in zip(xa, xb))
        out["n_differing_samples"] = sum(1 for i, j in zip(xa, xb) if i != j)
    out["verdict"] = "sample_mismatch"
    return out


def snr_db(a: Clip, b: Clip) -> float:
    """Signal to difference ratio over the overlap, in dB."""
    import array
    import math
    xa, xb = array.array("h"), array.array("h")
    xa.frombytes(a.frames)
    xb.frombytes(b.frames)
    n = min(len(xa), len(xb))
    if n == 0:
        return float("-inf")
    mean = sum(xa[:n]) / n
    var = sum((v - mean) ** 2 for v in xa[:n]) / n
    err = sum((xa[i] - xb[i]) ** 2 for i in range(n)) / n
    return float("inf") if err == 0 else 10 * math.log10(max(var, 1e-12) / err)
