"""The markdown IAA report."""
from __future__ import annotations

import math

from ..strata import order_key
from . import names

HEADER = ("| Stratum | Utt. | Slots | Agree | κ [95% CI] | α | DER | "
          "DER −final | Word agree |\n|---|---|---|---|---|---|---|---|---|")


def _f(v, pct=False):
    if isinstance(v, float) and math.isnan(v):
        return "—"
    if isinstance(v, float):
        return f"{100*v:.1f}%" if pct else f"{v:.3f}"
    return str(v)


def _row(name, m):
    ci = m.get("kappa_ci")
    ci_s = f" [{_f(ci[0])}, {_f(ci[1])}]" if ci else ""
    return (f"| {name} | {m['utterances']} | {m['slots']} | {_f(m['agreement'], True)} "
            f"| {_f(m['kappa'])}{ci_s} | {_f(m['alpha'])} | {_f(m['der'], True)} "
            f"| {_f(m['der_no_final'], True)} | {_f(m['word_agreement'], True)} |")


def markdown(study, n_subset: int) -> str:
    o, s = study.overall, study.settings
    p = o["permutation_p"]
    lines = [
        "# Inter-annotator agreement — diacritization subset",
        "",
        f"Items scored: **{o['utterances']}** of {n_subset} in the subset  ",
        f"Slots compared: **{o['slots']}**  ",
        f"Bootstrap: {s['bootstrap']} cluster resamples over utterances; "
        f"permutation test: {s['permutations']} shuffles; seed {s['seed']}",
        "",
        "## Overall", "", HEADER, _row("All", o), "",
        f"- Krippendorff's α 95% CI: [{_f(o['alpha_ci'][0])}, {_f(o['alpha_ci'][1])}]",
        f"- Permutation test against κ = 0: **p "
        f"{'< ' + _f(1 / s['permutations']) if p <= 1 / s['permutations'] else '= ' + _f(p)}**",
        f"- Words compared: {o['words_compared']}; words diverging orthographically "
        f"(excluded from the slot statistics): {o['words_orthographically_divergent']}",
        "",
    ]
    titles = {"subdialect": "subdialect", "gender": "gender", "source": "source",
              "first_pass_provenance": "first-pass provenance"}
    for key, title in titles.items():
        lines += [f"## By {title}", "", HEADER]
        keys = (sorted(study.strata[key], key=order_key) if key == "subdialect"
                else sorted(study.strata[key]))
        for k in keys:
            lines.append(_row(names.VARIETY.get(k, k), study.strata[key][k]))
        lines.append("")

    c = study.palestinian_contrast
    lines += [
        "## Annotator variety vs utterance variety", "",
        "The annotators are speakers of Palestinian Arabic. If a Palestinian annotator "
        "cannot reliably diacritize another variety's phonology, agreement should be "
        "measurably lower off Palestinian material.", "",
        HEADER,
        _row("Palestinian (Urban + Rural)", c["palestinian"]),
        _row("Non-Palestinian (Jor/Leb/Syr)", c["non_palestinian"]), "",
        f"Difference in κ (Palestinian − non-Palestinian): **{_f(c['difference'])}**, "
        f"95% CI [{_f(c['difference_ci'][0])}, {_f(c['difference_ci'][1])}]", "",
        "An interval covering zero would mean the data show no variety-dependent drop in "
        "reliability at this sample size. It is not evidence of equivalence either way; "
        "state the interval, not just the sign.", "",
        "## Most frequent disagreements", "", "| Label pair | Count |", "|---|---|",
    ]
    for pair, n in study.disagreements:
        lines.append(f"| {pair} | {n} |")
    lines.append("")
    return "\n".join(lines)
