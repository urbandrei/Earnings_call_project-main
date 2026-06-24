"""Prompt drafts for constrained LLM feature extraction (T6.1).

The prompt is part of the cache key (T6.2 caches by ``prompt+model`` version), so
``PROMPT_VERSION`` MUST be bumped on any wording change — otherwise stale cached
extractions would be silently reused under new wording. The embedded anchors are a
condensed mirror of ``docs/llm_feature_rubric.md`` (the rubric is the source of
truth the κ-audit scores against; this is the operational copy the model reads).

Drafts only — finalized after the T6.1 human reading of 20 calls (HANDOFF).
"""

from __future__ import annotations

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "You are a meticulous financial-analyst annotator. You read one section of an "
    "earnings-call transcript and rate it on a fixed rubric. Rules: (1) rate ONLY from "
    "the text shown — never use outside knowledge of the company or its later stock move; "
    "(2) when a field does not apply or there is no evidence, use the lowest value "
    "('none' or 0); (3) quote the exact span(s) that justify your ratings in 'evidence'. "
    "Output must match the required JSON schema exactly."
)

# Condensed anchors per field. Q&A-only fields are dropped from the prepared-remarks prompt.
_ANCHORS = {
    "guidance_direction": (
        "guidance_direction — direction of forward financial guidance vs. the prior outlook: "
        "'raise' (guidance/outlook increased), 'lower' (cut), 'maintain' (reaffirmed/unchanged), "
        "'none' (no forward guidance discussed)."
    ),
    "hedging_intensity": (
        "hedging_intensity (0-4) — density of hedging/uncertainty language (e.g. 'we believe', "
        "'roughly', 'it depends', 'hard to say'): 0 none, 1 occasional, 2 moderate, 3 heavy, "
        "4 pervasive non-committal language."
    ),
    "qa_evasiveness": (
        "qa_evasiveness (0-4) — how much answers dodge the question actually asked: 0 direct and "
        "complete, 1 mostly direct, 2 partial/deflected, 3 largely non-answers, 4 fully evasive."
    ),
    "surprise_mentions": (
        "surprise_mentions (integer count, 0-20) — number of explicit mentions of something being "
        "a surprise / unexpected / better- or worse-than-expected."
    ),
    "analyst_tone": (
        "analyst_tone (0-4) — aggregate stance of the analysts asking questions: "
        "0 hostile/critical, 1 skeptical, 2 neutral, 3 positive, 4 enthusiastic."
    ),
}

# Imported lazily-safe: kept here to avoid a hard schema import at module load in CI paths.
_SECTION_FIELDS = {
    "prepared_remarks": ("guidance_direction", "hedging_intensity", "surprise_mentions"),
    "qa": (
        "guidance_direction",
        "hedging_intensity",
        "qa_evasiveness",
        "surprise_mentions",
        "analyst_tone",
    ),
}

_SECTION_LABEL = {
    "prepared_remarks": "PREPARED REMARKS (management's scripted statement)",
    "qa": "Q&A (analyst questions and management answers)",
}


def build_user_prompt(section: str, text: str) -> str:
    """User prompt for one section: rubric anchors for the applicable fields + the text.

    For prepared remarks the Q&A-only fields (evasiveness, analyst tone) are omitted from
    the anchors and the model is told to set them to 0 (the schema still requires them).
    """
    if section not in _SECTION_LABEL:
        raise ValueError(f"unknown section {section!r}")
    fields = _SECTION_FIELDS[section]
    anchors = "\n".join(f"- {_ANCHORS[f]}" for f in fields)
    na_note = ""
    if section == "prepared_remarks":
        na_note = (
            "\nThis is the prepared-remarks section, so set qa_evasiveness and analyst_tone to 0 "
            "(not applicable here).\n"
        )
    return (
        f"Section: {_SECTION_LABEL[section]}\n\n"
        f"Rate the following fields:\n{anchors}\n{na_note}\n"
        f"--- SECTION TEXT START ---\n{text}\n--- SECTION TEXT END ---"
    )
