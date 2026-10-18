r"""LaTeX table generation.

Every table is generated from a results JSON and nothing else, so a number in
the paper can always be traced to a file that a command produced. Labels and
captions match the manuscript's ``\ref`` targets, so regenerating a table is a
file replacement and never an edit to the surrounding prose.

Two conventions worth knowing when reading the output:

``\dgap``
    The small grey figure beside a printed score is its difference from the
    task's second normalization tier. Printing it inline rather than as a
    second table lets a reader see recognition error and notation convention
    in one cell. The macro is emitted with the table, via ``\providecommand``,
    so the table compiles without touching ``preamble.tex``.

``$^{*}$``, ``$^{\dagger}$``
    The system's row is not what the rest of the table is: its score is not an
    independent measurement, or it is not a system of the kind being compared.
    Symbols are handed out in column order and the caption pairs each with the
    sentence that explains it.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..evaluate import Evaluation
from ..strata import order_key
from . import names

DGAP_MACRO = (
    r"% \dgap{d}: gap to the second normalization tier in points, set small and"
    "\n"
    r"% grey next to the primary figure. Requires xcolor (already loaded by acl.sty)."
    "\n"
    r"\providecommand{\dgap}[1]{\,\raisebox{0.2ex}{\scriptsize\textcolor{gray}{#1}}}"
)

DGAPSTACK_MACRO = (
    r"% \dgapstack{figure}{d}: the same gap as \dgap, set under the figure"
    "\n"
    r"% rather than beside it. A cell then costs the width of the figure alone,"
    "\n"
    r"% which is what lets the table hold one column."
    "\n"
    r"\providecommand{\dgapstack}[2]{\begin{tabular}{@{}c@{}}#1\\[-2.5pt]"
    r"{\scriptsize\textcolor{gray}{#2}}\end{tabular}}"
)

#: A caption counts in words, the way the prose around it does.
COUNT_WORD = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
              7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def _open(env: str, spec: str, *, colsep: str | None = None,
          fill: bool = False) -> list[str]:
    r"""The float and tabular opening every table shares.

    ``fill`` sets ``tabular*`` against the width of the column the float sits
    in, with ``\extracolsep{\fill}`` spreading the slack between the columns
    and ``@{}`` dropping the pad outside the first and last. The rules then run
    the full measure instead of sitting in a margin of white on both sides.

    It suits a table close to that width and not one far short of it: the slack
    goes into the gaps, so on a narrow table a row of wide gaps is what the
    reader sees. Measure the body before turning it on.
    """
    width = r"\textwidth" if env == "table*" else r"\columnwidth"
    lines = [f"\\begin{{{env}}}[t]", r"\centering", r"\small"]
    if colsep:
        lines.append(f"\\setlength{{\\tabcolsep}}{{{colsep}}}")
    if fill:
        lines.append(f"\\begin{{tabular*}}{{{width}}}"
                     f"{{@{{\\extracolsep{{\\fill}}}}{spec}@{{}}}}")
    else:
        lines.append(f"\\begin{{tabular}}{{{spec}}}")
    return lines


def _close(fill: bool = False) -> str:
    return r"\end{tabular*}" if fill else r"\end{tabular}"


def fmt(v: float, bold: bool = False) -> str:
    if v != v:
        return "N/A"
    s = f"{v * 100:.1f}"
    return f"\\textbf{{{s}}}" if bold else s


def _cell(block: dict, metric: str, tiers: Sequence[str], bold: bool = False,
          stack: bool = False) -> str:
    strict_v = block[tiers[0]][metric]["rate"]
    out = fmt(strict_v, bold)
    if len(tiers) < 2:
        return out
    neutral_v = block[tiers[1]][metric]["rate"]
    if strict_v != strict_v or neutral_v != neutral_v:
        return out
    # A gap under half a point prints as zero, and "-0.0" beside one cell and
    # "0.0" beside the next reads as a difference the reader cannot see.
    gap = f"{(strict_v - neutral_v) * 100:.1f}"
    if gap == "-0.0":
        gap = "0.0"
    if stack:
        return f"\\dgapstack{{{out}}}{{{gap}}}"
    return f"{out}\\dgap{{{gap}}}"


def _rows(evals: Sequence[Evaluation]) -> list[tuple[str, dict, bool]]:
    """(row label, model_key -> block, is_overall) in variety order."""
    by_model = {e.model_key: e for e in evals}
    rows = [("Overall", {m: e.overall for m, e in by_model.items()}, True)]
    varieties = sorted({v for e in evals for v in e.by["subdialect"]}, key=order_key)
    for v in varieties:
        rows.append((names.VARIETY.get(v, v),
                     {m: e.by["subdialect"].get(v) for m, e in by_model.items()}, False))
    return rows


def _scope(evals: Sequence[Evaluation]) -> str:
    """The caption clause naming how many utterances the table is scored on.

    Not every system need cover the whole benchmark: a run can be short of a
    few utterances. Saying "all 1300" when one column is scored on fewer is the
    kind of caption a reviewer checks.
    """
    ns = sorted({e.n for e in evals})
    return (f"on all {ns[0]} utterances" if len(ns) == 1 else
            f"on up to {ns[-1]} utterances ({ns[0]} for the shortest run)")


def results_table(evals: Sequence[Evaluation], *, tier: str | None = None,
                  label: str = "tab:asr-results", task: str = "asr",
                  models: Sequence[str] | None = None, wide: bool = True,
                  colsep: str = "4pt", fill: bool | None = None,
                  stack_gap: bool = False) -> str:
    """Per-variety results, one column per system, WER block above CER block.

    ``models`` restricts the columns to those systems, in the usual order. A
    system scored but not listed is absent from the table, from its bold
    comparison and from its caption, the same as one that was never run.
    ``wide`` sets ``table*``; a table narrow enough for one column sets it
    False. ``colsep`` is the inter-column space that form was measured to need.
    ``fill`` runs the body out to the measure and defaults to on for a narrow
    table, which is right while the body nearly fills the column and wrong once
    it does not. ``stack_gap`` sets the grey gap figure under its score instead
    of beside it, which is what buys a crowded table the width to fit.
    """
    if fill is None:
        fill = not wide
    scored = {e.model_key for e in evals}
    present = names.order(scored if models is None else [m for m in models if m in scored])
    shown = [e for e in evals if e.model_key in set(present)]
    env = "table*" if wide else "table"
    norm_tiers = evals[0].normalization_tiers
    show_gap = tier is None and len(norm_tiers) >= 2
    tiers = norm_tiers if show_gap else (tier or norm_tiers[0],)
    metrics = evals[0].metrics

    stack = show_gap and stack_gap
    lines = []
    if show_gap:
        lines.append(DGAPSTACK_MACRO if stack else DGAP_MACRO)
    lines += _open(env, f"l {'c' * len(present)}", colsep=colsep, fill=fill)
    lines.append(r"\toprule")
    marks = names.marks(present, task)
    # A footnoted system is not what the rest of the table is, so it does not
    # take the bold: the best-per-row is chosen among the systems being compared.
    compared = [m for m in present if m not in marks]
    for m in present:
        lines.append(f"& \\rotatebox{{60}}"
                     f"{{{names.display(m, 'short', task)}{marks.get(m, '')}}}")
    lines[-1] += r" \\"
    lines.append(r"\midrule")

    ncols = len(present) + 1
    # The metric heading spans the table, so it carries the row's left edge. On
    # a filled body that edge has no pad, and the heading has to lose it too or
    # it sits a couple of points right of the rules above and below it.
    head_spec = "@{}l" if fill else "l"
    for i, metric in enumerate(metrics):
        if i:
            lines.append(r"\addlinespace[2pt]\midrule\addlinespace[2pt]")
        lines.append(f"\\multicolumn{{{ncols}}}{{{head_spec}}}"
                     f"{{\\textit{{{names.METRIC_TEX.get(metric, metric)} (\\%)}}}} \\\\")
        lines.append(r"\addlinespace[2pt]")
        for label_txt, blocks, is_overall in _rows(evals):
            vals = {m: (blocks[m][tiers[0]][metric]["rate"] if blocks.get(m) else float("nan"))
                    for m in present}
            finite = [vals[m] for m in compared if vals[m] == vals[m]]
            best = min(finite) if finite else None
            cells = []
            for m in present:
                b = blocks.get(m)
                if not b:
                    cells.append("N/A")
                    continue
                bold = best is not None and abs(vals[m] - best) < 1e-9
                cells.append(_cell(b, metric, tiers, bold, stack=stack))
            prefix = f"\\textbf{{{label_txt}}}" if is_overall else f"~ {label_txt}"
            lines.append(f"{prefix} & " + " & ".join(cells) + r" \\")

    scope = _scope(shown)
    # One clause, not the whole scheme. The folds are defined once, in the
    # appendix normalization subsection, and every table points at it. The
    # second tier is named rather than hard-coded: the diacritization gap is
    # paper against tanwin-included, not strict against neutral, and A2P prints
    # the neutral figure with the gap to strict.
    tier_note = (
        f", under \\textit{{{tiers[0]}}} scoring. Small grey figures give the "
        f"difference from the \\textit{{{tiers[1].replace('_', '-')}}} tier "
        r"(\S\ref{sec:app_normalization})." if show_gap else ".")
    notes = names.caption_notes(present, task)
    caveat_note = f" {notes}" if notes else ""
    lines += [
        r"\bottomrule", _close(fill),
        (f"\\caption{{{names.TASK.get(task, task)} results per variety "
         f"({'/'.join(names.METRIC_TEX.get(m, m) for m in metrics)} \\%, $\\downarrow$) "
         f"{scope}{tier_note} "
         r"\textbf{Bold} = best per row." + caveat_note + "}"),
        f"\\label{{{label}}}", f"\\end{{{env}}}"]
    return "\n".join(lines)


def cascade_table(evals: Sequence[Evaluation],
                  paths: Sequence[tuple[str, str, str]], *,
                  tier: str | None = None, task: str = "a2p",
                  label: str | None = None) -> str:
    """Direct against cascade, grouped by audio system with varieties across.

    The transpose of :func:`results_table`. Here the question is what one audio
    system loses by going straight to phonemes rather than through a
    transcript, so the two paths sit next to each other under the system they
    share and the varieties become columns.
    """
    label = label or f"tab:{task}-cascade-appendix"
    by_model = {e.model_key: e for e in evals}
    rows = [(audio, path, key) for audio, path, key in paths if key in by_model]
    if not rows:
        return f"% no {task} cascade predictions scored; table not generated\n"

    metric = evals[0].metrics[0]
    tier = tier or evals[0].normalization_tiers[0]
    varieties = sorted({v for e in evals for v in e.by["subdialect"]}, key=order_key)
    keys = [k for _, _, k in rows]
    marks = names.marks(keys, task)
    # The system that carries a footnote is the one that is not what the rest
    # of the table is, so it neither takes the bold nor sets the best.
    compared = [k for k in keys if k not in marks]

    def value(key: str, variety: str | None) -> float:
        e = by_model[key]
        block = e.overall if variety is None else e.by["subdialect"].get(variety)
        return block[tier][metric]["rate"] if block else float("nan")

    columns: list[str | None] = [None, *varieties]
    best = {}
    for c in columns:
        finite = [v for v in (value(k, c) for k in compared) if v == v]
        best[c] = min(finite) if finite else None

    heads = " & ".join(f"\\rotatebox{{60}}{{{names.VARIETY.get(v, v)}}}"
                       for v in varieties)
    # Two label columns, an overall column and one per variety: measured at
    # 324pt against a 219pt column, so this one spans the page.
    lines = _open("table*", f"ll r {'r' * len(varieties)}", colsep="3.5pt") + [
             r"\toprule",
             r"\textbf{Audio} & \textbf{Path} & \textbf{All} & " + heads + r" \\",
             r"\midrule"]

    previous = None
    for audio, path, key in rows:
        if previous is not None and audio != previous:
            lines.append(r"\addlinespace[2pt]")
        cells = []
        for c in columns:
            v = value(key, c)
            cells.append(fmt(v, bold=(best[c] is not None and v == v
                                      and abs(v - best[c]) < 1e-9)))
        head = "" if audio == previous else f"{audio}{marks.get(key, '')}"
        lines.append(f"{head} & {path} & " + " & ".join(cells) + r" \\")
        previous = audio

    notes = names.caption_notes(keys, task)
    caveat_note = f" {notes}" if notes else ""
    lines += [
        r"\bottomrule", r"\end{tabular}",
        (f"\\caption{{Per-variety {names.METRIC_TEX.get(metric, metric)} "
         r"(\%, $\downarrow$) for direct A2P against an "
         r"ASR\,$\rightarrow$\,G2P cascade and a "
         r"gold-transcript\,$\rightarrow$\,G2P oracle, "
         f"{_scope([by_model[k] for k in keys])}. The G2P stage is "
         r"Gemini~3.5~Flash in every cascade row, so the GPT-Audio cascade is a "
         r"hybrid and its margin over GPT-Audio's direct A2P is an upper bound. "
         f"Scored with the G2P \\textit{{{tier}}} normalization and the 100\\% "
         r"per-utterance cap (\S\ref{sec:app_normalization}); figures are macro "
         f"means over utterances. \\textbf{{Bold}} = best per column among the "
         f"{COUNT_WORD.get(len(compared), len(compared))} audio systems."
         + caveat_note + "}"),
        f"\\label{{{label}}}", r"\end{table*}"]
    return "\n".join(lines)


def source_contrast_table(evals: Sequence[Evaluation], variety: str,
                          *, tier: str | None = None,
                          label: str | None = None) -> str:
    """Within-variety source contrast: the same variety read from both corpora.

    Only a variety drawn from both Casablanca and LangMedia supports a
    within-variety estimate of the source artifact. After the expansion that is
    Jordanian at 137 against 168, which is properly powered, and Palestinian
    Urban at 22 against 208, which is not and is reported for continuity with
    the submitted analysis.
    """
    tier = tier or evals[0].normalization_tiers[0]
    task = evals[0].task
    label = label or f"tab:asr-source-{variety.split('/')[-1].lower()}"
    present = [e.model_key for e in evals
               if len(e.by["subdialect_source"].get(variety) or {}) == 2]
    present = names.order(present)
    if not present:
        return f"% {variety} is not drawn from two sources; table not generated\n"

    by_model = {e.model_key: e for e in evals}
    cells0 = by_model[present[0]].by["subdialect_source"][variety]
    a_key, b_key = sorted(cells0)
    ns = {s: cells0[s]["n"] for s in (a_key, b_key)}
    mins = {s: cells0[s]["duration_min"] for s in (a_key, b_key)}
    metrics = evals[0].metrics

    lines = _open("table", "l rr r", colsep="4pt", fill=True) + [
             r"\toprule",
             f"& \\multicolumn{{1}}{{c}}{{{names.SOURCE.get(a_key, a_key)}}} "
             f"& \\multicolumn{{1}}{{c}}{{{names.SOURCE.get(b_key, b_key)}}} "
             r"& $\Delta$ \\",
             f"& \\multicolumn{{1}}{{c}}{{\\scriptsize $n={ns[a_key]}$, "
             f"{mins[a_key]:.1f}\\,min}} "
             f"& \\multicolumn{{1}}{{c}}{{\\scriptsize $n={ns[b_key]}$, "
             f"{mins[b_key]:.1f}\\,min}}" + r" & \\",
             r"\midrule"]

    for i, metric in enumerate(metrics):
        if i:
            lines.append(r"\addlinespace[2pt]\midrule\addlinespace[2pt]")
        lines.append(f"\\multicolumn{{4}}{{l}}"
                     f"{{\\textit{{{names.METRIC_TEX.get(metric, metric)} (\\%)}}}} \\\\")
        lines.append(r"\addlinespace[2pt]")
        row_marks = names.marks(present, task)
        for m in present:
            cells = by_model[m].by["subdialect_source"][variety]
            a = cells[a_key][tier][metric]["rate"]
            b = cells[b_key][tier][metric]["rate"]
            lines.append(f"~ {names.display(m, 'short', task)}{row_marks.get(m, '')} "
                         f"& {a*100:.1f} & {b*100:.1f} & {(b-a)*100:+.1f} \\\\")

    powered = ("the within-variety comparison the expanded dataset was selected to "
               "support"
               if min(ns.values()) >= 100 else
               "small on the "
               f"{names.SOURCE.get(min(ns, key=ns.get))} side and therefore indicative "
               "rather than conclusive")
    notes = names.caption_notes(present, task)
    lines += [
        r"\bottomrule", _close(True),
        (r"\caption{Within-variety source contrast for "
         f"{names.VARIETY.get(variety, variety)}, scored as in "
         f"\\Cref{{tab:{task}-results}}, {powered}. $\\Delta$ is "
         f"{names.SOURCE.get(b_key, b_key)} $-$ {names.SOURCE.get(a_key, a_key)}, "
         r"and is a difference of rates rather than an estimate with an interval; "
         r"read it against the two utterance counts in the header. "
         r"Source and speaker gender remain partly entangled "
         r"(\S\ref{sec:limitations}), so this bounds rather than isolates the "
         r"source effect." + (f" {notes}" if notes else "") + "}"),
        f"\\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)


#: The manuscript labels the ASR model roster ``tab:models``, from before there
#: was more than one roster. Kept, so existing \ref targets keep resolving.
MODELS_TABLE_LABEL = {"asr": "tab:asr-models", "diacritization": "tab:diac-models",
                      "a2p": "tab:a2p-models", "g2p": "tab:g2p-models"}


def models_table(evals: Sequence[Evaluation], task: str = "asr") -> str:
    present = names.order([e.model_key for e in evals])
    # The regime column makes the roster wider than one manuscript column.
    lines = _open("table*", "l l l") + [r"\toprule",
             r"\textbf{System} & \textbf{Regime} & \textbf{Configuration} \\",
             r"\midrule"]
    marks = names.marks(present, task)
    for m in present:
        info = names.info(m, task)
        display = info.get("display", m)
        url = info.get("url")
        cell = f"\\href{{{url}}}{{{display}}}" if url else display
        lines.append(f"{cell}{marks.get(m, '')} & {info.get('regime', '---')} "
                     f"& {info.get('config', '---')} \\\\")
    notes = names.caption_notes(present, task)
    lines += [r"\bottomrule", r"\end{tabular}",
              f"\\caption{{{names.TASK.get(task, task)} systems evaluated, with "
              "regime, configuration and model URLs." +
              (f" {notes}" if notes else "") + "}",
              f"\\label{{{MODELS_TABLE_LABEL.get(task, f'tab:{task}-models')}}}",
              r"\end{table*}"]
    return "\n".join(lines)


def iaa_table(study) -> str:
    """Table 3: agreement overall and per variety, plus the Palestinian contrast."""
    o = study.overall
    lines = _open("table", "l rr rr", colsep="3.5pt", fill=True) + [
             r"\toprule",
             r"& \textbf{Utt.} & \textbf{Slots} & \textbf{Agree} "
             r"& \boldmath$\kappa$ \textbf{[95\% CI]} \\", r"\midrule"]

    def row(label, m, indent=False):
        pre = f"~ {label}" if indent else f"\\textbf{{{label}}}"
        ci = m.get("kappa_ci", [float("nan"), float("nan")])
        ci_txt = ("" if ci[0] != ci[0]
                  else f"\\,{{\\scriptsize [{ci[0]:.3f}, {ci[1]:.3f}]}}")
        return (f"{pre} & {m['utterances']} & {m['slots']:,} "
                f"& {m['agreement']*100:.1f}\\% & {m['kappa']:.3f}{ci_txt} \\\\")

    lines.append(row("Overall", o))
    lines.append(r"\addlinespace[2pt]\midrule\addlinespace[2pt]")
    for k in sorted(study.strata["subdialect"], key=order_key):
        lines.append(row(names.VARIETY.get(k, k), study.strata["subdialect"][k], indent=True))
    c = study.palestinian_contrast
    lines.append(r"\addlinespace[2pt]\midrule\addlinespace[2pt]")
    lines.append(row("Palestinian", c["palestinian"], indent=True))
    lines.append(row("Non-Palestinian", c["non_palestinian"], indent=True))
    lines += [
        r"\bottomrule", _close(True),
        (r"\caption{Inter-annotator agreement on the diacritization subset "
         f"({o['utterances']} utterances, 30 per variety; {o['slots']:,} diacritic "
         r"slots). The unit is the diacritic slot: one per Arabic letter, labelled with "
         r"the marks it carries. Words whose consonantal skeletons differ between the "
         r"two annotators are an orthographic divergence rather than a diacritic "
         f"disagreement and are excluded ({o['words_orthographically_divergent']} of "
         f"{o['words_compared'] + o['words_orthographically_divergent']:,} words). "
         r"Intervals are cluster bootstraps resampling utterances. The Palestinian "
         r"$-$ non-Palestinian difference in $\kappa$ is "
         f"{c['difference']:.3f}, 95\\% CI [{c['difference_ci'][0]:.3f}, "
         f"{c['difference_ci'][1]:.3f}].}}"),
        r"\label{tab:iaa}", r"\end{table}"]
    return "\n".join(lines)


#: Shape of a task's main results table. A task absent here prints every system
#: it scored, across both columns.
#:
#: A2P scores five systems and compares two. The cascades and the oracle answer
#: a different question and are the appendix table's subject; printing them in
#: both places gives the same number two homes, and two homes drift.
#: ``colsep`` and ``fill`` are measured, not guessed. Against a 219pt column:
#: diacritization is 214pt at 2pt, G2P is 213pt at 4pt once its gap figures
#: stack, and A2P is 126pt, far too short to fill without opening a hole
#: between its three columns.
MAIN_TABLE: dict[str, dict] = {
    "a2p": {"models": ("gemini-3.5-flash", "gpt-audio"), "wide": False,
            "fill": False},
    "diacritization": {"wide": False, "colsep": "2pt"},
    "g2p": {"wide": False, "colsep": "4pt", "stack_gap": True},
}


def write_all(evals: Sequence[Evaluation], out_dir: Path,
              task: str = "asr") -> list[Path]:
    """Write the tables the paper uses."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def put(name: str, body: str) -> None:
        p = out_dir / name
        p.write_text(body + "\n", encoding="utf-8")
        written.append(p)

    # The diacritization table prints one figure. Its second configuration is
    # the case-tanwin ablation of §3.1, not a fold of notation, so a grey gap
    # there would not mean what the same grey gap means on the ASR and A2P
    # tables. The ablation is reported in the prose instead.
    shape = MAIN_TABLE.get(task, {})
    put(f"{task}_results_table.tex",
        results_table(evals, label=f"tab:{task}-results", task=task,
                      tier="paper" if task == "diacritization" else None,
                      models=shape.get("models"), wide=shape.get("wide", True),
                      colsep=shape.get("colsep", "4pt"),
                      fill=shape.get("fill"),
                      stack_gap=shape.get("stack_gap", False)))
    put(f"{task}_models_table.tex", models_table(evals, task=task))

    # The systems the main table leaves out still have to be printed somewhere,
    # and the appendix table is where they are the subject rather than extra
    # columns.
    paths = names.CASCADE_PATHS.get(task)
    if paths and any(k in {e.model_key for e in evals} for _, _, k in paths):
        put(f"{task}_cascade_appendix.tex", cascade_table(evals, paths, task=task))

    two_source = sorted({v for e in evals for v, cells in e.by["subdialect_source"].items()
                         if len(cells) == 2}, key=order_key)
    for variety in two_source:
        slug = variety.split("/")[-1].lower()
        put(f"{task}_source_contrast_{slug}.tex",
            source_contrast_table(evals, variety, label=f"tab:{task}-source-{slug}"))
    return written
