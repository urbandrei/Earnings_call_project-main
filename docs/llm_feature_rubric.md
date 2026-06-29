# LLM structured-feature rubric (T6.1)

**Status: v2 — pending schema sign-off.** v2 adds two **exploratory** fields
(`management_optimism`, `quantitative_specificity`) driven by rater-1 feedback + the
numeral-aware literature (DESIGN §3 R30/R31/R32), and narrows the corpus-scale κ-gate to a
pre-registered **confirmatory core**. See DECISIONS 2026-06-29.
This is the contract the T6.2 human κ-audit scores model output against. Every field
has a written rubric with anchors so two raters (and the model) apply it the same way.

**Field roles:**
- **Confirmatory core** (κ>0.6 gate blocks the corpus run): `guidance_direction`,
  `hedging_intensity`, `surprise_mentions`. Fixed pre-extraction from label variance + rater
  applicability feedback, *not* from any model κ score.
- **Reported (labeled, weak):** `qa_evasiveness`, `analyst_tone`. Rater 1 flagged these as
  low-variance here (management rarely evasive — if anything over-answers; analysts mostly
  agreeable except on big bad news). Scored + reported with their κ, but do **not** block.
- **Exploratory (v2, unlabeled):** `management_optimism`, `quantitative_specificity`.
  Extracted in the same OSC pass; no κ-gate until a rater labels them (then promotable per the
  DESIGN exploration policy).

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

### `management_optimism` — ordinal 0–4 *(v2 exploratory, both sections)*
How much management *oversells / self-promotes* versus giving a balanced account. Captures
the "trying to sound too good" axis rater 1 observed (distinct from evasiveness — companies
that over-answer to look strong score high here, low on evasiveness).
- **0** measured/balanced — acknowledges negatives and risks plainly.
- **1** mildly upbeat. **2** clearly positive. **3** strongly promotional.
- **4** relentlessly self-congratulatory — almost no acknowledgement of any negative.

### `quantitative_specificity` — ordinal 0–4 *(v2 exploratory, both sections)*
Density of *concrete quantitative disclosure* (specific figures, ranges, growth rates,
segment numbers) versus vague qualitative claims. Motivated by the numeral-aware volatility
literature (DESIGN §3: NAM/ECNum [R30], NumHTML [R31], GNAVol [R32]).
- **0** none/vague — no hard numbers. **1** a few numbers. **2** moderate.
- **3** heavily quantified. **4** dense hard figures throughout.

### `evidence` — free text
Verbatim quoted span(s) from the section that justify the ratings above. For auditability
only; not scored by κ.

---

## Acceptance protocol

- **T6.1 (schema applicability):** two human passes over **10 calls** agree the schema is
  applicable and every field has a usable rubric. Disagreements drive rubric edits before
  T6.2.
- **T6.2 (content gate):** human audit on **50 calls**; **κ > 0.6 on the confirmatory core**
  (`guidance_direction` Cohen's κ; `hedging_intensity` linearly-weighted κ; `surprise_mentions`
  binarized present/absent) of **model vs. rubric labels** before any corpus-scale run. The
  reported weak fields (`qa_evasiveness`, `analyst_tone`) and the exploratory v2 fields are
  scored/extracted but do not gate. A **borderline core result (κ≈0.45–0.6) re-blocks on a
  second rater** before any Stage-5/RQ3 claim (single-annotator κ can't disambiguate
  model-fault from task-ambiguity — DECISIONS 2026-06-29).
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
