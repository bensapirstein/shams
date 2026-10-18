"""Rebuilding the SHAMS audio from its sources.

The benchmark distributes text, not audio: the Casablanca half is access-gated
upstream and the LangMedia half belongs to LangMedia. What travels with the
package is the metadata needed to fetch each source and cut it back to the
released segmentation, in three manifests read from the wheel:

``clips.csv``      one row per LangMedia source clip: where to fetch it and the
                   ffmpeg profile its segments were cut at
``checksums.csv``  one row per released segment: format, frame count and a
                   SHA-256 of the decoded PCM
``BUILD.json``     the decoder and cut arithmetic the release was made with

Two entry points, ``shams-eval build-audio`` and ``shams-eval verify-audio``.
Segment metadata is read through :mod:`shams_eval.dataset`, the same release the
scorer grades against, so a rebuild and a score always describe the same rows.
"""
from __future__ import annotations

import csv
import json
from importlib.resources import files

__all__ = ["build_info", "checksums", "clips", "manifest_path"]


def manifest_path(name: str):
    """A manifest as a traversable, wherever the package is installed from."""
    return files(__package__).joinpath("manifests", name)


def _rows(name: str) -> list[dict]:
    with manifest_path(name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def clips() -> list[dict]:
    """The LangMedia source clips."""
    return _rows("clips.csv")


def checksums() -> dict[str, dict]:
    """Released segments keyed by ``seg_id``."""
    return {r["seg_id"]: r for r in _rows("checksums.csv")}


def build_info() -> dict:
    """How the released audio was produced."""
    with manifest_path("BUILD.json").open(encoding="utf-8") as f:
        return json.load(f)
