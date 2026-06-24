# LLM structured-feature rubric (T6.1)

**Status: DRAFT v1 — pending the T6.1 human reading of 20 calls + schema sign-off.**
This is the contract the T6.2 human κ-audit scores model output against. Every field
has a written rubric with anchors so two raters (and the model) apply it the same way.

- **Extraction unit:** one `(call, section)` pair. Sections come from T3.1 sectioning
  (`prepared_remarks`, `qa`).
- **Schema:** `src/ecvol/features/llm/schema.py` (`SectionFeatures`).
- **Prompt:** `src/ecvol/features/llm/prompts.py` (`PROMPT_VERSION` = `v1`).
- **Cardinal rule (no leakage):** rate **only** from the section text shown. Never use
  outside knowledge of the company, the quarter's results, or the later stock move.
- **Applicability:** `qa_evasiveness` and `analyst_tone` are **Q&A-only**. In
  `prepared_remarks` set them to `0` (N/A); the audit excludes those cells.

---

## Fields

### `guidance_direction` — categorical {raise, maintain, lower, none}
Direction of *forward financial guidance* (revenue/EPS/margin outlook, next-quarter or
full-year), relative to the company's prior outlook.
- **raise** — guidance/outlook explicitly increased, raised, or described as above prior.
- **lower** — guidance cut, reduced, or withdrawn downward.
- **maintain** — guidance reaffirmed, reiterated, or left unchanged.
- **none** — no forward guidance discussed in this section.
- *Edge cases:* mixed (raise one metric, cut another) → judge the headline metric the
  speaker emphasizes; if genuinely balanced, `maintain`. Backward-looking results that are
  not forward guidance → `none`.

### `hedging_intensity` — ordinal 0–4
Density of hedging / uncertainty / non-committal language ("we believe", "roughly",
"approximately", "it depends", "hard to say", "we'll see", "no guarantees").
- **0** none — crisp, definite statements.
- **1** occasional — a few softeners.
- **2** moderate — hedging recurs but specifics still given.
- **3** heavy — most claims are qualified.
- **4** pervasive — answers are almost entirely non-committal.

### `qa_evasiveness` — ordinal 0–4 *(Q&A only)*
How much management's *answers dodge the question actually asked*.
- **0** direct — answers the question fully.
- **1** mostly direct — minor deflection.
- **2** partial — answers part, deflects part ("we don't break that out, but…").
- **3** largely non-answers — acknowledges then pivots away.
- **4** fully evasive — refuses / changes subject / pure boilerplate.
- *Scope:* aggregate over the whole Q&A section (one rating per section, not per question).

### `surprise_mentions` — count 0–20
Number of **explicit** mentions of something being a surprise / unexpected / better- or
worse-than-expected. Count distinct mentions, not repeated references to the same one.
Implicit surprise (a big beat stated without surprise language) does **not** count.

### `analyst_tone` — ordinal 0–4 *(Q&A only)*
Aggregate stance of the **analysts** asking questions (not management).
- **0** hostile/critical, **1** skeptical, **2** neutral, **3** positive, **4** enthusiastic.
- Judge from question framing ("congrats on a great quarter" → high; "why did margins
  collapse" → low). Neutral/factual questions → 2.

### `evidence` — free text
Verbatim quoted span(s) from the section that justify the ratings above. For auditability
only; not scored by κ.

---

## Acceptance protocol

- **T6.1 (schema applicability):** two human passes over **10 calls** agree the schema is
  applicable and every field has a usable rubric. Disagreements drive rubric edits before
  T6.2.
- **T6.2 (content gate):** human audit on **50 calls**; **κ > 0.6** on categorical fields
  (Cohen's κ for `guidance_direction`; linearly-weighted κ for the ordinals; `surprise`
  binarized present/absent) of **model vs. rubric labels** before any corpus-scale run.
- **Leakage guard:** the reading/audit samples are drawn from the **train split only**
  (`ecvol featurize llm-reading-pack`) so no val/test call informs the schema or taxonomy
  (mirrors TASKS.md TX1).

## Two-rater workflow

1. Run `ecvol featurize llm-reading-pack --dataset fincall --n 20 --seed 0`.
2. Read the rendered transcripts under `data/fincall/llm_reading/*.md`.
3. Fill the labeling sheet `data/coverage/fincall_llm_label_sheet.csv` (one row per
   `call_id × section`; only applicable field columns are present). A second rater fills a
   copy for the 10-call agreement subset.
4. Record agreement + any rubric edits in JOURNAL.md, then sign off so T6.2 extraction can
   be built against the frozen schema + `PROMPT_VERSION`.
