"""Display names and ordering for models and varieties.

Kept out of the evaluator so that renaming a system in a table cannot change a
number, and so that the same label is used in every table, figure and report.
Model keys are the prediction filenames, which are the runner's own identifiers.
"""
from __future__ import annotations

#: ``regime`` separates unmodified base checkpoints, prompt-only runs,
#: fine-tuned checkpoints, task-trained models and rule-based systems. Cascades
#: and oracles keep the prompted regime and name their non-comparable structure.
MODELS: dict[str, dict[str, str]] = {
    # ASR
    "seamless-m4t-v2-large": {
        "display": "Seamless-MSA", "short": "Sml-MSA",
        "regime": "Base zero-shot",
        "config": r"\texttt{tgt\_lang=arb}",
        "url": "https://huggingface.co/facebook/seamless-m4t-v2-large"},
    "seamless-m4t-v2-large-arz": {
        "display": "Seamless-Egyptian", "short": "Sml-EGY",
        "regime": "Base zero-shot",
        "config": r"\texttt{tgt\_lang=arz}",
        "url": "https://huggingface.co/facebook/seamless-m4t-v2-large"},
    "whisper-large-v3": {
        "display": "Whisper v3", "short": "Whisper v3",
        "regime": "Base zero-shot", "config": "---",
        "url": "https://huggingface.co/openai/whisper-large-v3"},
    "hebarab-whisper-levantine": {
        "display": "Whisper-Levantine", "short": "Whisper-Lev.",
        "regime": "Fine-tuned", "config": "---",
        "url": "https://huggingface.co/HebArabNlpProject/WhisperLevantine"},
    "mms-1b-all": {
        "display": "MMS-1B", "short": "MMS-1b", "regime": "Fine-tuned",
        "config": r"\texttt{ara} adapter",
        "url": "https://huggingface.co/facebook/mms-1b-all"},
    "BYZÖ-whisper": {
        "display": "BYZÖ-Whisper", "short": "BYZÖ-W.",
        "regime": "Fine-tuned", "config": "---",
        "url": "https://huggingface.co/Yusser/whisper-dialect-Mauritania_Algeria_"
               "Morocco_Yemen_UAE_Egypt_Palestine_Jordan-ft-first-stage"},
    "cohere-transcribe-arabic": {
        "display": "Cohere Transcribe", "short": "Cohere",
        "regime": "Fine-tuned",
        "config": r"\texttt{language=ar}",
        "url": "https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026"},
    "gemini-asr": {
        "display": "Gemini 3.5 Flash", "short": "Gemini",
        "regime": "Prompted zero-shot", "config": "---",
        "url": "https://ai.google.dev/gemini-api/docs/audio"},
    "gpt-audio-asr": {
        "display": "GPT-Audio", "short": "GPT Audio",
        "regime": "Prompted zero-shot", "config": "---",
        "url": "https://platform.openai.com/docs/guides/audio"},
    # Diacritization
    "catt": {"display": "CATT", "short": "CATT", "regime": "Fine-tuned",
             "config": "---",
             "url": "https://github.com/abjadai/catt"},
    "mishkal": {"display": "Mishkal", "short": "Mishkal",
                "regime": "Rule-based", "config": "---",
                "url": "https://github.com/linuxscout/mishkal"},
    "shakkala": {"display": "Shakkala", "short": "Shakkala",
                 "regime": "Task-trained", "config": "---",
                 "url": "https://github.com/Barqawiz/Shakkala"},
    "levanti": {"display": "Levanti", "short": "Levanti",
                "regime": "Fine-tuned", "config": "---",
                "url": "https://huggingface.co/guymorlan/levanti_arabic2diacritics"},
    "gemini-3.5-flash": {
        "display": "Gemini 3.5 Flash", "short": "Gemini",
        "regime": "Prompted zero-shot", "config": "prompted",
        "url": "https://ai.google.dev/gemini-api/docs/models"},
    "gpt-5.5": {"display": "GPT-5.5", "short": "GPT-5.5",
                "regime": "Prompted zero-shot", "config": "prompted",
                "url": "https://platform.openai.com/docs/models"},
    "claude-opus-4-8": {
        "display": "Claude Opus 4.8", "short": "Opus 4.8",
        "regime": "Prompted zero-shot", "config": "prompted",
        "url": "https://docs.claude.com/en/docs/about-claude/models"},
    # G2P
    "charsiu-byt5": {
        "display": "CharsiuG2P", "short": "Charsiu",
        "regime": "Fine-tuned",
        "config": r"multilingual byT5 small",
        "url": "https://huggingface.co/charsiu/g2p_multilingual_byT5_small_100"},
    "jais-v2-8b": {
        "display": "JAIS v2 8B", "short": "JAIS-8B",
        "regime": "Prompted zero-shot", "config": "prompted",
        "url": "https://huggingface.co/inceptionai/Jais-2-8B-Chat"},
    # A2P
    "gpt-audio": {
        "display": "GPT-Audio", "short": "GPT-Audio",
        "regime": "Prompted zero-shot", "config": "---",
        "url": "https://platform.openai.com/docs/guides/audio"},
    "cascade-gemini": {
        "display": "Cascade: Gemini ASR", "short": "Casc. Gemini",
        "regime": "Prompted zero-shot cascade",
        "config": "Gemini G2P module"},
    "cascade-gpt": {
        "display": "Cascade: GPT-Audio ASR", "short": "Casc. GPT",
        "regime": "Prompted zero-shot cascade",
        "config": "Gemini G2P module"},
    "cascade-oracle": {
        "display": "Cascade: gold text", "short": "Oracle",
        "regime": "Prompted zero-shot oracle",
        "config": "Gemini G2P module"},
}

#: Per-task overrides, merged over :data:`MODELS`.
#:
#: A prediction filename is the runner's identifier, and one runner can serve two
#: tasks. ``gemini-3.5-flash`` is the prompted text model on the diacritization
#: table and the direct audio system on the A2P one, where the whole point of the
#: table is direct against cascade. The name has to say which; the key cannot.
TASK_MODELS: dict[str, dict[str, dict[str, str]]] = {
    "a2p": {
        "gemini-3.5-flash": {"display": "Direct: Gemini 3.5 Flash", "config": "---"},
        "gpt-audio": {"display": "Direct: GPT-Audio"},
    },
}

#: Systems whose row is not what the rest of the table is, and the sentence that
#: says why. A footnoted system keeps its place: the figure is real and a reader
#: looking for it should find it. What the mark withholds is the comparison.
#:
#: A key is ``task/model`` where only one task is affected, or a bare model where
#: every table it appears in needs the note.
FOOTNOTES: dict[str, str] = {
    # Scoped per task, because the sentence differs. Levanti seeded both
    # pronunciation tiers of the expansion (§2.3: "an automatic first pass over
    # both tiers"). On the diacritized tier 161 released rows are still
    # byte-identical to its silver; on the phonetic tier only 3 are, so the
    # G2P sentence claims the seeding, not identity.
    "diacritization/levanti":
        "Levanti produced the silver diacritization labels that the 780 "
        "expansion utterances were annotated from, so its score on those "
        "is not an independent measurement.",
    "g2p/levanti":
        "Levanti produced the first-pass phonetic transcription that the 780 "
        "expansion utterances were corrected from, so its score on those "
        "is not an independent measurement.",
    "a2p/cascade-oracle":
        "Cascade: gold text is not an audio system. It passes the gold "
        "orthographic transcription through the same G2P module, and is the "
        "floor the two cascades are measured against rather than a system "
        "competing with them.",
}

#: Superscripts, handed out in table order to the footnoted systems present. A
#: table with two of them needs two symbols, and the caption pairs each with its
#: own sentence.
MARKS_TEX = (r"$^{*}$", r"$^{\dagger}$", r"$^{\ddagger}$")

#: Table column order. Models absent from a run are dropped; models absent from
#: this list are appended alphabetically, so a new system still appears.
#:
#: One list serves every task, and a key can be shared: ``gemini-3.5-flash`` is
#: the prompted text model on the diacritization table and the direct audio
#: system on the A2P one. The diacritization entries run the prompted LLMs and
#: then the dedicated systems, as Table 4 does; the A2P entries run direct
#: systems first and then the cascades, which is the order §3.4 argues in. G2P
#: reuses the three prompted LLM keys and ``levanti``, and adds its own two, so
#: its columns come out prompted-LLMs-then-dedicated the way §3.2 reads.
MODEL_ORDER = [
    "seamless-m4t-v2-large", "seamless-m4t-v2-large-arz", "whisper-large-v3",
    "hebarab-whisper-levantine", "mms-1b-all", "BYZÖ-whisper",
    "cohere-transcribe-arabic", "gemini-asr", "gpt-audio-asr",
    "gemini-3.5-flash", "claude-opus-4-8", "gpt-5.5",
    "shakkala", "mishkal", "catt", "jais-v2-8b", "levanti", "charsiu-byt5",
    "gpt-audio", "cascade-gemini", "cascade-gpt", "cascade-oracle",
]

#: Rows of the A2P appendix table, as (audio system, path, model key).
#:
#: That table reads down one audio system rather than across systems, because
#: the comparison it makes is direct against cascade within a single model, so
#: the audio system is a row group and the varieties are the columns. The
#: labels are written here rather than taken from :data:`MODELS` because the
#: column names the audio, not the system: the oracle's audio is a gold
#: transcript, and the direct rows drop the ``Direct:`` prefix the main A2P
#: table needs to tell them from the cascades.
CASCADE_PATHS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "a2p": (
        ("Gemini 3.5 Flash", "direct A2P", "gemini-3.5-flash"),
        ("Gemini 3.5 Flash", r"ASR\,$\rightarrow$\,G2P cascade", "cascade-gemini"),
        ("GPT-Audio", "direct A2P", "gpt-audio"),
        ("GPT-Audio", r"ASR\,$\rightarrow$\,G2P cascade", "cascade-gpt"),
        ("Gold transcript", "G2P oracle", "cascade-oracle"),
    ),
}

VARIETY = {
    "Palestinian/Urban": "Pal./Urban",
    "Palestinian/Rural": "Pal./Rural",
    "Jordanian": "Jordanian",
    "Lebanese": "Lebanese",
    "Syrian": "Syrian",
}

SOURCE = {"casablanca": "Casablanca", "langmedia": "LangMedia"}

#: Task names as the captions print them.
TASK = {"asr": "ASR", "diacritization": "Diacritization", "g2p": "G2P", "a2p": "A2P"}

#: Plain-text metric names, for the console and for JSON.
METRIC = {"wer": "WER", "cer": "CER", "der": "DER", "cer_capped": "CER"}

#: The same names set in LaTeX. Kept apart so a table's typography cannot leak
#: into a terminal, and a terminal's plain text cannot leak into a caption.
METRIC_TEX = {**METRIC}


def info(model_key: str, task: str | None = None) -> dict[str, str]:
    """Everything known about a system, with the task's overrides applied."""
    return {**MODELS.get(model_key, {}),
            **TASK_MODELS.get(task or "", {}).get(model_key, {})}


def display(model_key: str, style: str = "display", task: str | None = None) -> str:
    return info(model_key, task).get(style, model_key)


def footnote(model_key: str, task: str | None = None) -> str:
    """This system's caption sentence for this task, or ""."""
    return FOOTNOTES.get(f"{task}/{model_key}", FOOTNOTES.get(model_key, ""))


def marks(model_keys, task: str | None = None) -> dict[str, str]:
    """model_key -> superscript, for the footnoted systems among these.

    Assigned in table order, so the symbols read left to right and a table with
    no footnoted system gets none.
    """
    present = [m for m in order(model_keys) if footnote(m, task)]
    if len(present) > len(MARKS_TEX):
        # Silently dropping one would print a system with no mark and no
        # sentence, which is the failure the footnote exists to prevent.
        raise ValueError(f"{len(present)} systems need a footnote and only "
                         f"{len(MARKS_TEX)} symbols are defined: {present}")
    return dict(zip(present, MARKS_TEX, strict=False))


def caption_notes(model_keys, task: str | None = None) -> str:
    """The mark-and-sentence pairs to append to a caption, or "".

    Built from the same assignment as :func:`marks`, so a symbol is never
    printed without the sentence that explains it.
    """
    return " ".join(f"{mark} {footnote(m, task)}"
                    for m, mark in marks(model_keys, task).items())


def order(model_keys):
    known = [m for m in MODEL_ORDER if m in set(model_keys)]
    rest = sorted(set(model_keys) - set(known))
    return known + rest
