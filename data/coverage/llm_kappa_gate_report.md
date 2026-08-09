# T6.2 κ-gate — result (2026-08-09)

**Model:** `bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M` @ HF rev `8911e8a4…`, llama.cpp engine,
YaRN-extended 65 536 context, greedy, `PROMPT_VERSION="v2"`.
**Audit set:** the frozen 50 train-only calls (seed 0) → 97 `(call, section)` rows, extracted
through the *same* engine that would run the corpus (audit-matches-corpus rule).
**Labels:** `data/coverage/fincall_llm_labels_rater1.csv` (single rater; IAA pending).

## Outcome: GATE FAILED — corpus extraction NOT started

| field | κ | n | role |
|---|---|---|---|
| guidance_direction | **0.165** | 97 | confirmatory |
| hedging_intensity | **−0.050** | 97 | confirmatory |
| surprise_mentions | **0.222** | 97 | confirmatory |
| qa_evasiveness | 0.108 | 50 | reported |
| analyst_tone | −0.040 | 50 | reported |

Gate is κ > 0.6 on the confirmatory core. All three are far below it; `hedging_intensity` is
*worse than chance*. Per the run's standing instruction, the corpus run was **not** started and
the GPU was left idle. This is nowhere near the "borderline 0.45–0.6" band that DECISIONS
2026-06-29 says re-blocks on rater 2 — it is a clear failure, and a second rater would not
change the conclusion.

## It is not a plumbing bug (checked before blaming the model)

- **Join:** 97 label keys, 97 matched, 0 unmatched. The merge is exact.
- **Section text:** spot-checked — real analyst prose, correctly assigned to `qa`
  vs `prepared_remarks`; the human's `NA` pattern (47 rows) matches the prepared-remarks count.
- **Schema validity:** 100% of rows decoded valid (constrained decoding), evidence populated on
  every row after the grammar fixes committed this session.

## What the disagreement actually looks like

Not a calibration offset — re-scoring with the model's ordinal shifted by ±1 or ±2 does not
rescue it:

| shift | hedging κ | raw agreement |
|---|---|---|
| +0 | −0.050 | 29.9% |
| −1 | −0.087 | 40.2% |
| −2 | +0.000 | 40.2% |

Per-field failure modes:

- **hedging_intensity** — human mass sits at 0–1 (85/97), model at 1–2 (95/97). The crosstab
  shows the model's rating is essentially *independent* of the human's: for human=0 the model
  says 1 or 2 (38/39), and for human=1 it also says 1 or 2 (45/46). No discriminative signal.
- **analyst_tone** — the model emits only {0, 2}, rating 47 of 50 Q&A sections exactly "2".
  Near-constant output, so κ ≈ 0 by construction. The human uses the full 1–4 range.
- **surprise_mentions** — model reports "absent" on 84.5% of rows vs the human's 62.9%, missing
  26 of 36 human-positive rows. Systematic under-detection.
- **guidance_direction** — the model defaults to `none` (47/97 vs the human's 28) and catches
  only 12 of the human's 41 `raise` rows. This is the strongest field, and it is still κ=0.165.

## Controls run to make the diagnosis actionable

*(These extract no corpus features and touch no canonical artifact — they exist so the decision
below is informed rather than a guess.)*

### Control 1 — is it prompt/rubric anchoring? **No.**

The frozen v2 system prompt was re-scored against a variant adding explicit decision rules
aimed at exactly the three observed failures (count hedge phrases and judge *density not raw
count*; classify guidance whenever any forward outlook appears and reserve `none` for its total
absence; use the full 0–4 analyst-tone range; count analyst surprise phrasing too). Same 97
sections, same model, same engine, in memory — nothing written to `data/`.

| field | frozen v2 | anchored variant | Δ |
|---|---|---|---|
| guidance_direction | 0.165 | 0.175 | +0.010 |
| hedging_intensity | −0.050 | −0.013 | +0.037 |
| surprise_mentions | 0.222 | 0.222 | 0.000 |
| analyst_tone | −0.040 | 0.012 | +0.052 |
| qa_evasiveness | 0.108 | 0.096 | −0.012 |

Every delta is negligible; nothing approaches 0.6. Sharpening the anchors does **not** rescue the
gate. (The run also reproduced the frozen-v2 κs to three decimals, confirming the pipeline is
deterministic.) So option 1 below — "just fix the prompt" — is unlikely to be sufficient alone.

### Control 2 — is it the 4-bit quantization?

`Q8_0` (near-lossless) re-run of the same audit sample through the same engine and policy.
It fits locally: 13.5 GB of 16.3 GB VRAM at 65 536 context, at 6.0 s/section (vs 4.63 for
Q4_K_M), so this is a control the project can keep using.

| field | Q4_K_M | Q8_0 | Δ |
|---|---|---|---|
| guidance_direction | 0.165 | 0.187 | +0.022 |
| hedging_intensity | −0.050 | −0.003 | +0.047 |
| surprise_mentions | 0.222 | 0.308 | +0.086 |
| qa_evasiveness | 0.108 | 0.180 | +0.072 |
| analyst_tone | −0.040 | −0.040 | 0.000 |

Q8_0 is nominally higher on four of five fields, but **a paired call-level bootstrap says every
one of these differences is indistinguishable from zero** (all 95% CIs span 0; see
"Uncertainty" below). The honest statement is therefore stronger than "quantization costs a
little": **quantization has no measurable effect on this gate at all**, and the Q4_K_M artifact
is not the thing to blame or discard.

### Control 3 — is it *example-based* calibration? Best lever so far, still far short.

Control 1 tested whether *describing* the scale better helps (it didn't). This tests whether
*showing worked examples* of the rater's own scale use helps — usually far more effective for
aligning ordinal judgments. 10 of the 50 audit calls became in-prompt exemplars (short excerpt +
rater-1 labels); κ is scored only on the **held-out 40 calls / 77 sections**, and the zero-shot
arm is re-scored on those same rows so the comparison is like-for-like.
Regenerate: `notebooks/llm_fewshot_calibration.py`.

| field | zero-shot v2 | few-shot | Δ |
|---|---|---|---|
| guidance_direction | 0.177 | 0.242 | +0.065 |
| hedging_intensity | −0.006 | 0.089 | +0.095 |
| surprise_mentions | 0.155 | 0.156 | +0.001 |
| analyst_tone | −0.036 | −0.020 | +0.016 |
| qa_evasiveness | 0.106 | 0.117 | +0.011 |

Worked examples move `hedging_intensity` from below-chance to slightly positive — a bigger gain
than instruction anchoring managed, and consistent with the "the rater's baseline is unstated"
reading, since exemplars communicate a baseline that prose cannot. But the ceiling reached is
**0.242**, and the two fields that fail by *non-discrimination* (`analyst_tone`,
`surprise_mentions`) barely move.

**Caveat that must travel with this number:** exemplars come from rater 1 and the gate scores
against rater 1, so few-shot partly fits this annotator's idiosyncrasy. That is legitimate for a
deployable configuration — the same exemplars would ship with the corpus run — but it makes the
resulting features a better measure of *this rater* rather than of the construct, which is the
opposite of what "auditable semantics" is supposed to buy.

### What the controls jointly rule out

No cheap model-side lever comes close. Prompt wording: nothing. Quantization: ~+0.02–0.09.
Worked examples: ~+0.07–0.10 on two fields, nothing on the rest. Even assuming these stacked
additively — which they will not — the best confirmatory field would reach roughly 0.3 against a
0.6 bar. Combined with the plumbing checks, that leaves a capability-vs-validity question:
either Qwen2.5-7B cannot reproduce this rater's judgments on this rubric, or the rubric/labels
are not reproducible in the first place.

Those two are distinguishable, and cheaply: **a second rater settles it.** If two humans agree
with each other far better than the model agrees with either, the model is the problem and scale
is the fix. If two humans disagree comparably, the schema is the problem and no model fixes it.
Note that the controls have already made the *pure* capability story less likely: two of the
five fields fail by emitting near-constant values, which is a symptom of an ill-posed question
more than of insufficient capability. See also the scale probe below.

## Uncertainty — how much of the above arithmetic is real?

Every κ here is estimated from 97 rows (50 for the Q&A-only fields), where the standard error is
around 0.1. Differences between point estimates at that size are not interpretable by eye, so
they are bootstrapped: 2000 resamples of **calls** (not rows — a call's two sections are
correlated), paired so the same resampled calls score every arm.
Regenerate: `notebooks/llm_kappa_uncertainty.py`.

**The gate result is robust.** Every confirmatory field's 95% upper bound sits far below 0.6:

| confirmatory field | κ (Q4_K_M) | 95% CI |
|---|---|---|
| guidance_direction | 0.165 | [0.064, **0.271**] |
| hedging_intensity | −0.050 | [−0.124, **0.033**] |
| surprise_mentions | 0.222 | [0.054, **0.378**] |

So the failure is not a small-sample accident — κ>0.6 can be ruled out at 95% confidence for all
three. This is the claim to make in the paper, rather than the bare point estimates.

**The control deltas are not.** Q8_0 − Q4_K_M, paired bootstrap: guidance +0.021 [−0.048,
+0.096]; hedging +0.045 [−0.028, +0.129]; surprise +0.090 [−0.037, +0.235]; evasiveness +0.068
[−0.071, +0.202]. **Every CI spans zero.** The same caution applies to the anchoring and few-shot
deltas above, which are of similar magnitude on similar sample sizes: they are reported because
they are informative about *direction of effort*, but none of them should be described as a
demonstrated improvement. Treat the earlier "best lever so far" language for few-shot as a
statement about point estimates only.

The practical consequence is uncomfortable but useful: **this audit sample is too small to
distinguish interventions of the size any of these levers produce.** Detecting a real +0.1 shift
with confidence would need substantially more than 50 calls. Anything that must be *chosen*
between on the basis of a ~0.05–0.1 κ difference cannot be chosen on this evidence — which is a
further argument for fixing the instrument rather than shopping for a model.

## Carried defect — the Colab/vLLM path is still exposed (fix before running it)

Two of the three bugs fixed this session live in the *schema*, not the engine, so they apply to
any Outlines-driven run too:

1. **Optional fields.** pydantic leaves defaulted fields out of `required`, so
   `evidence`, `management_optimism` and `quantitative_specificity` are all optional in the
   generated schema. Locally the model then skipped `evidence` on **every single section**, and a
   skipped exploratory field is filled by its default `0` — a *non-rating* stored as, and
   indistinguishable from, a real `0` rating. That is a silent-data-corruption bug, not a
   cosmetic one.
2. **Decode budget.** `MAX_NEW_TOKENS` was 640, too small for a full-length evidence span; it is
   now 1024 (shared by all engines, so this one is already fixed everywhere).

The `llamacpp` engine is protected by `grammar_json_schema()`, which forces
`required = all properties`. **`TransformersOutlinesEngine` and `VLLMEngine` still pass
`SectionFeatures` directly to `outlines.Generator` and are therefore still exposed.** The fix is
to hand Outlines the same all-required schema instead of the bare pydantic class. It was not made
here because there is no vLLM on this machine to verify it against, and shipping an unverified
change to the cloud path would be worse than a documented one. **Verify this on Colab before
trusting any vLLM-produced parquet** — check that `evidence` is non-empty across rows.

## Is this "just disagreement"? Decomposing the number

"κ ≈ 0.2" is hiding three different phenomena, which need three different fixes.

**(a) Anchor-baseline ambiguity — an instrument defect, not a model defect.** The rubric
(`docs/llm_feature_rubric.md`) defines hedging as `0` = "none — crisp, definite statements" and
`2` = "moderate — hedging recurs but specifics still given". At this project's extraction unit —
a whole section, median 25k characters — *"hedging recurs" is trivially true of nearly every
Q&A section*. The model rating **2** is arguably obeying the rubric as literally written, while
rater 1 applied an unstated "relative to a typical earnings call" baseline (85 of 97 sections
marked 0–1). Both readings are defensible, which means the instrument is underdetermined: the
rubric never says density *relative to what*, and at section length that omission decides the
rating. No increase in model scale fixes an ambiguous anchor.

**(b) Metric behaviour under skewed marginals — real, but a partial excuse at best.** For
`surprise_mentions` the raw agreement is **68.0%** while κ = 0.222, because prevalence is
lopsided (human 37.1% "present", model 15.5%); PABAK on the same table is **+0.361**. This is
the familiar κ-paradox regime. It should not be leaned on, though: the model still misses **26
of 36** human positives, a 72% false-negative rate on the positive class, which is a genuine
detection failure no metric choice repairs.

**(c) Genuine non-discrimination — not disagreement in any sense.** `analyst_tone`: the model
emits only {0, 2} and rates 47 of 50 Q&A sections exactly "2". A near-constant rater has no
discriminative power by construction. `hedging` is *statistically independent* of the human
label: when the human says 0 the model says "2" in 20/39 cases; when the human says 1 it says
"2" in 20/46 — the same distribution either way.

**The structural problem: the gate never measured its own ceiling.** κ against a single
unvalidated rater has an unknown maximum. Subjective ordinal annotation routinely sits at
human–human κ of 0.4–0.6; if rater 1's ceiling is in that band, a 0.6 model bar **was never
attainable by any model**, and "FAIL" would have been the outcome regardless. The 0.6 threshold
traces to the Landis–Koch "substantial" convention (DESIGN §6, §10) — a rule of thumb for
reliability studies, adopted here without ever estimating the annotator ceiling it implies.
This is why the second rater is diagnostic rather than cosmetic, and why the honest reporting
unit is model κ *relative to* human–human κ, not against an absolute bar.

## How this sits against the literature

The sub-literature this stage engages does not measure extraction validity at all:

- **ECC Analyzer** (R8, ICAIF '24) is the closest prior and the stated model for Stage 5. The
  July novelty scan verified in full text: "no typed schema, no evidence spans, **no human
  validation of extraction quality (no kappa anywhere)**" — while reporting a 27.7% MSE
  reduction from LLM-extracted semantics.
- **EvasionBench** (R35) reports κ=0.835, but DESIGN §13 already flags that this is *inter-LLM
  annotator* agreement, **not** a human audit — explicitly "never cite as κ-gate prior art".

So this is not a bar the field clears and we missed. We are the only ones measuring it, and the
measurement came back bad. That makes the result a **finding**, and it is the natural third leg
of the project's existing thesis: text signal is ticker identity (Phase 3), audio is inert
beyond past volatility (Phase 4), and LLM "semantic" features do not reproduce human judgment
(Phase 6) — while prior work consumes exactly such features as if they were validated
measurements. DESIGN §10 risk #5 pre-registered "LLM extraction quality poor / unreliable" with
this gate as the mitigation, so the risk register worked as designed.

**What this forecloses.** RQ3 asks whether *auditable* structured features beat opaque
embeddings, and DESIGN §6 sells Stage 5 as "explicit, **auditable** semantics" — auditability is
constitutive of the contribution, not decoration. A column that fails its audit may appear in a
results table as "LLM-rated hedging (validity κ=−0.05)", but it may not be called *hedging*.
Note also that Phases 2–5 found text, audio and fusion all inert beyond past-vol + identity, so
the expected *predictive* payoff of Stage 5 was low regardless; the validity result is the more
valuable output.

## Decision owed (user)

The gate blocks corpus scale until it passes. Ordered by what the controls actually support:

1. **Second rater (recommended first — it is the diagnostic, not just an IAA nicety).** ~50 calls
   of labeling settles whether this is a model problem or a schema problem, and every other option
   is a guess until it is answered. It was deferred to paper stage on the assumption the gate would
   pass; a κ≈0 result promotes it to the critical path. DECISIONS 2026-06-29 already says a
   borderline κ re-blocks on rater 2 — this is worse than borderline.
2. **Go bigger** — 32B-AWQ (or Llama-3.1-8B) on Colab, audited as its own quant. Justified *if*
   rater 2 shows humans agree with each other. Note the local box can now run this class of
   experiment for 8B-scale models directly.
3. **Revisit the rubric (T6.1 → `v3`)** — indicated if rater 2 shows humans *don't* agree, which
   would mean the anchors are underspecified. Note control 1 showed that better prompt *wording*
   alone does not move κ, so this means changing the rubric's definitions, not its phrasing.
4. **Narrow the confirmatory core** — only defensible on *label* grounds, recorded before
   re-extraction. Doing it because a model scored badly would be fitting the gate to the result,
   which the pre-registration in DECISIONS 2026-06-29 exists to prevent. Flagged only to name it
   as off-limits without a fresh pre-registration.

Nothing here is blocked on compute: the local machine can run an audit + gate for any 7–8B-class
model in ~10 minutes, and the full corpus in ~12 h once something passes.
