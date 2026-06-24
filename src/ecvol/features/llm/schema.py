"""Per-section LLM structured-feature schema (T6.1).

One extraction unit = one ``(call, section)`` pair. The LLM fills a fixed
``SectionFeatures`` shape; constrained decoding (T6.2, Outlines) guarantees the
shape, so 100% of outputs are schema-valid by construction — the *content* gate
is the human κ-audit (T6.2), not validity.

Field set is the one pre-registered in TASKS.md T6.1 / DESIGN.md §6 (Stage 5):
guidance direction, hedging intensity, Q&A evasiveness, surprise mentions,
analyst tone — plus a free-text ``evidence`` span for auditability. Written
rubrics for every field live in ``docs/llm_feature_rubric.md`` (the rubric is the
contract the κ-audit scores against).

Section applicability (``SECTION_FIELDS``): evasiveness and analyst-tone are
Q&A-only (no analyst speaks in prepared remarks); in the prepared-remarks section
the extractor sets them to the N/A floor (0) and the audit excludes those cells.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# The two transcript sections produced by T3.1 sectioning (features/text/sections.py).
SECTIONS = ("prepared_remarks", "qa")

GuidanceDirection = Literal["raise", "maintain", "lower", "none"]


class SectionFeatures(BaseModel):
    """LLM-extracted semantic features for one transcript section.

    Ordinal 0-4 fields share a common anchor convention (0 = none/absent,
    4 = extreme); see the rubric for per-field anchors. ``extra="forbid"`` so a
    hallucinated key fails loudly rather than being silently dropped.
    """

    model_config = ConfigDict(extra="forbid")

    guidance_direction: GuidanceDirection = Field(
        description="Direction of forward guidance vs. prior: raise/maintain/lower/none."
    )
    hedging_intensity: int = Field(
        ge=0, le=4, description="Density of hedging / uncertainty language (0 none .. 4 extreme)."
    )
    qa_evasiveness: int = Field(
        ge=0,
        le=4,
        description="How much answers dodge the question asked (0 direct .. 4 fully evasive). "
        "Q&A-only; 0 in prepared remarks (N/A).",
    )
    surprise_mentions: int = Field(
        ge=0,
        le=20,
        description="Count of explicit surprise / unexpected / better-or-worse-than-expected "
        "mentions (capped at 20).",
    )
    analyst_tone: int = Field(
        ge=0,
        le=4,
        description="Aggregate analyst stance toward the company (0 hostile .. 2 neutral .. "
        "4 enthusiastic). Q&A-only; 0 in prepared remarks (N/A).",
    )
    evidence: str = Field(
        default="",
        max_length=2000,
        description="Verbatim quoted span(s) from the section justifying the ratings.",
    )


# Field groupings drive the κ-audit (T6.2): categorical via Cohen's κ, ordinals via
# linearly-weighted κ, the count binarized to present/absent. ``evidence`` is not scored.
CATEGORICAL_FIELDS = ("guidance_direction",)
ORDINAL_FIELDS = ("hedging_intensity", "qa_evasiveness", "analyst_tone")
COUNT_FIELDS = ("surprise_mentions",)
LABEL_FIELDS = CATEGORICAL_FIELDS + ORDINAL_FIELDS + COUNT_FIELDS

# Which fields are applicable (audited / prompted for) per section. Q&A-only fields are
# set to the N/A floor in prepared remarks and excluded from κ there.
SECTION_FIELDS = {
    "prepared_remarks": ("guidance_direction", "hedging_intensity", "surprise_mentions"),
    "qa": LABEL_FIELDS,
}


def applicable_fields(section: str) -> tuple[str, ...]:
    """Audited/labeled fields for a section (raises on an unknown section)."""
    if section not in SECTION_FIELDS:
        raise ValueError(f"unknown section {section!r}; expected one of {SECTIONS}")
    return SECTION_FIELDS[section]
