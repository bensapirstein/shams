"""How results are broken down, defined once for every task.

SHAMS is variety-stratified by design, so a headline number is the least
interesting thing it produces. These are the cuts the paper reports,
named in one place so the ASR table, the diacritization table and the
G2P table are cut identically and can be read against each other.

``subdialect_source`` is the one that does analytical work rather than
description. Variety and source corpus are entangled — Casablanca supplies
Palestinian Rural, Jordanian and part of Palestinian Urban; LangMedia supplies
Lebanese, Syrian and the rest — so a per-variety difference could be a corpus
difference in disguise. Only a variety drawn from both corpora can separate
them, which after the expansion means Jordanian (137 Casablanca against 168
LangMedia) and, far more weakly, Palestinian Urban (22 against 208).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping

from .dataset import Utterance

StratumFn = Callable[[Utterance], object]

STRATA: Mapping[str, StratumFn] = {
    "subdialect": lambda u: u.subdialect,
    "source": lambda u: u.source,
    "gender": lambda u: u.gender,
    "subdialect_source": lambda u: (u.subdialect, u.source),
}

#: Varieties are listed in this order wherever they appear, so the reader can
#: track a row across tables.
SUBDIALECT_ORDER = ["Palestinian/Urban", "Palestinian/Rural", "Jordanian",
                    "Lebanese", "Syrian"]


def order_key(label: str) -> int:
    return SUBDIALECT_ORDER.index(label) if label in SUBDIALECT_ORDER else 99
