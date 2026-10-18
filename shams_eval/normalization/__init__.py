"""Normalization, one module per SHAMS text tier.

Each module exposes a ``TIERS`` mapping from a normalization-tier name to a
callable applied symmetrically to reference and hypothesis. Every tier of the
dataset is scored through exactly one of these, and nothing else in the package
touches raw text, so a change to how a tier is normalized has one place to
happen and one place to be reviewed.
"""
from . import diacritic, orthographic, phonetic

__all__ = ["orthographic", "diacritic", "phonetic"]
