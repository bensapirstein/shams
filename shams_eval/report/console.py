"""Human-readable reports: the markdown tables printed by the CLI."""
from __future__ import annotations

from collections.abc import Sequence

from ..evaluate import Evaluation
from ..strata import order_key
from . import names


def pct(v, width: int = 6) -> str:
    return f"{'N/A':>{width}}" if v != v else f"{v * 100:{width}.1f}"


def headline(evals: Sequence[Evaluation]) -> str:
    if not evals:
        return "(no results)"
    e0 = evals[0]
    tiers, metrics = e0.normalization_tiers, e0.metrics
    head = ["Model", "N"]
    for m in metrics:
        head += [f"{names.METRIC.get(m, m)} {t}" for t in tiers]
        if len(tiers) == 2:
            head.append("Δ")
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for e in evals:
        row = [e.model_key, str(e.n)]
        for m in metrics:
            vals = [e.overall[t][m]["rate"] for t in tiers]
            row += [pct(v).strip() for v in vals]
            if len(tiers) == 2:
                d = vals[0] - vals[1]
                row.append("N/A" if d != d else f"{d * 100:+.1f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def breakdown(e: Evaluation, stratum: str = "subdialect",
              norm_tier: str | None = None) -> str:
    norm_tier = norm_tier or e.normalization_tiers[0]
    blocks = e.by[stratum]
    keys = sorted(blocks, key=order_key) if stratum == "subdialect" else sorted(blocks)
    metrics = e.metrics
    width = max(9, max(len(names.METRIC.get(m, m)) for m in metrics) + 2)
    out = [f"\n### {e.model_key} — by {stratum} ({norm_tier})",
           f"{'':<20}{'N':>5}" + "".join(f"{names.METRIC.get(m, m):>{width}}" for m in metrics)]
    out.append("-" * (25 + width * len(metrics)))
    for k in keys:
        b = blocks[k]
        out.append(f"{names.VARIETY.get(k, k):<20}{b['n']:>5}"
                   + "".join(pct(b[norm_tier][m]["rate"], width) for m in metrics))
    return "\n".join(out)


def source_contrast(e: Evaluation, norm_tier: str | None = None) -> str:
    """Variety x source cells, with the difference where both corpora are present.

    The difference is arithmetic: the second corpus minus the first, on the same
    fold. It carries no interval, so read it against the two utterance counts on
    the rows above it.
    """
    norm_tier = norm_tier or e.normalization_tiers[0]
    cells = e.by["subdialect_source"]
    header = (f"{'Variety':<14}{'Source':<12}{'N':>5}{'min':>7}  "
              + "".join(f"{names.METRIC.get(m, m):>9}" for m in e.metrics)
              + "  Gender")
    out = [f"\n### {e.model_key} — variety x source ({norm_tier})", header,
           "-" * len(header)]
    for sd in sorted(cells, key=order_key):
        srcs = sorted(cells[sd])
        for src in srcs:
            b = cells[sd][src]
            gender = " ".join(f"{k}:{v}" for k, v in b["gender"].items())
            row = f"{names.VARIETY.get(sd, sd):<14}{names.SOURCE.get(src, src):<12}" \
                  f"{b['n']:>5}{b['duration_min']:>7.1f}  "
            for m in e.metrics:
                row += pct(b[norm_tier][m]["rate"], 9)
            out.append(row + "  " + gender)
        if len(srcs) == 2:
            a_key, b_key = srcs
            row = f"{'':<14}{'Δ (' + b_key[:3] + '−' + a_key[:3] + ')':<12}{'':>5}{'':>7}  "
            for m in e.metrics:
                d = (cells[sd][b_key][norm_tier][m]["rate"]
                     - cells[sd][a_key][norm_tier][m]["rate"])
                row += f"{d * 100:>+9.1f}" if d == d else f"{'N/A':>9}"
            out.append(row)
    return "\n".join(out)


def provenance(evals: Sequence[Evaluation]) -> str:
    from ..predictions import summarize_provenance

    lines = ["\n### Prediction-file provenance"]
    for e in evals:
        lines.append(f"  {e.model_key:<28} {summarize_provenance(e.provenance)}")
    return "\n".join(lines)
