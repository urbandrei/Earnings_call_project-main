# Novelty & frontier scan — 2026-07-19

**Question:** is the project still good, unique, and unscooped; has anything surfaced (Dec 2024 –
Jul 2026) that we must compare against? **Method:** deep-research workflow — 5 search angles,
101 agents, 19 primary sources fetched, 93 claims extracted, top 25 adversarially verified by
3-vote panels (24 confirmed, 1 refuted). Every claim below traces to a fetched page, not model
memory. Companion file: `reference_audit.md` (existence audit of the current bibliography).

## Bottom line

**The project remains novel and defensible on all four contributions.** No verdict worse than
MUST-CITE OVERLAP. The lookahead-bias literature is now crowded-adjacent (four+ papers since
Dec 2025) but nobody occupies our exact spot; the window for stage claims is time-sensitive.

## Verdicts by contribution

### Q1 — Identity-controlled re-examination: **CLEAR**

"Same Company, Same Signal" (arXiv 2412.18029, Dec 2024) remains the sole prior: it shows
training-free identity/history baselines (PEV/STPEV) beat all transcript models and that
transcript representations predominantly encode ticker identity. No follow-up, rebuttal, or
independent replication running new identity-control experiments was found. Our modern-stack,
multimodal (audio included), econometric-baseline reproduction is first of its kind. Note:
2412.18029 still appears to be an unreviewed arXiv v1 — our reproduction arguably *matters
more* because of that.

**New must-cite bridging identity + lookahead:** Lopez-Lira, Tang & Zhu, "The Memorization
Problem" (arXiv 2504.14765, Apr 2025, rev. Dec 2025): GPT-4o deanonymizes entity-neutered
earnings-call transcripts (100% firm / 95.2% year / 92.1% quarter+year accuracy on Apple) and
recalls pre-cutoff economic series near-perfectly (S&P 500 MAPE 0.61% pre- vs 16.70%
post-cutoff). Independent evidence that identity leakage survives even masking — strengthens
our §3.1 threat model. No volatility prediction, so no scoop.

### Q2 — LLM structured auditable features → volatility: **MUST-CITE OVERLAP, not scooped**

- **ECC Analyzer** (arXiv 2404.18470, ICAIF '24) — still the closest prior. Verified in full
  text: its "Question Bank" extraction is retrieval-and-quiz consumed as *embeddings*; no typed
  schema, no evidence spans, no human validation of extraction quality (no kappa anywhere);
  evaluation is a temporal 8:2 split on the 572-call 2017 dataset, no identity controls —
  fully exposed to the identity critique. Our audit-gated, schema-constrained,
  identity-controlled version is intact.
- **KPI extraction** (arXiv 2605.03147, May 2026, ACL 2026 Industry) — open-weight LLMs
  (Llama-3.3-70B, Qwen3-30B, Gemma-3-27B) extract structured KPIs from earnings calls with
  human verification (Krippendorff α=0.429, precision 79.67%). Overlaps our *method* (open
  LLMs + structured extraction + human checks) but extracts objective KPI values, not
  qualitative ratings, and does **no market prediction**. Must-cite; caveat: its kappa is
  human-human agreement on verification, not an LLM-vs-gold gate like ours.
- **EvasionBench** (arXiv 2601.09142, Jan 2026) — 3-class managerial-evasion labels over
  S&P Capital IQ Q&A at scale, labeled by LLM consensus (its κ=0.835 is *inter-LLM-annotator*
  agreement — a claim that it constitutes a human audit was REFUTED in verification; do not
  cite it as prior art for our kappa gate). Text-only, classification-only, no market outcome,
  no temporal/company-disjoint splits. Must-cite as related work on auditable constructs.

**The combination schema-constrained extraction + human kappa go/no-go gate + volatility
prediction + identity controls remains unclaimed.**

### Q3 — Post-cutoff lookahead study: **MUST-CITE OVERLAP, not scooped — but move fast**

The lookahead line went from one paper (Dec 2025) to at least four by Jul 2026:

- **LAP / Detecting Lookahead Bias in LLM Forecasts** (arXiv 2512.23847, v2 Jun 2026; our R15)
  — a *diagnostic* (date-only recall queries); applications are news→returns and earnings
  calls→**capex**, not volatility; not a frozen-pipeline evaluation.
- **DatedGPT** (arXiv 2603.11838, Mar 2026; our R16) — *prevention*: twelve 1.3B models
  pretrained with annual cutoffs; validated on perplexity/NLP benchmarks only; no earnings
  calls, no volatility.
- **Look-Ahead-Bench** (arXiv 2601.13770, Jan 2026) — trading-workflow benchmark (alpha decay
  across regimes); finds significant bias in Llama-3.1-8B/70B (a family in our panel — useful
  ammunition); no transcripts/audio/volatility. Caveat: single-author, vendor-affiliated
  (Pitinf results are vendor-reported).
- **The Memorization Problem** (arXiv 2504.14765, above) + **MemGuard-Alpha** (arXiv
  2603.26797, Fed FEDS, Mar 2026) — memorization measurement/filtering via membership
  inference and cross-model cutoff disagreement.
- **Nuance found (softens our uniqueness wording):** an ECB press-conference paper (arXiv
  2508.13635, v4 May 2026) runs a post-cutoff out-of-sample validation of an LLM-agent
  disagreement measure against realized OIS *rates* volatility from Jan 2025 (n=30, corr
  0.43), explicitly to rule out contamination. Not earnings calls, not equity RV, not a frozen
  prediction pipeline — but "no paper has ever done post-cutoff validation of an LLM
  volatility measure" is now too strong. Say: **first frozen-pipeline post-cutoff evaluation
  in the earnings-call volatility literature.**

Nobody performs our exact stage (b): frozen earnings-call pipeline, equity realized
volatility, post-cutoff calls. Execute and post promptly — this corner is filling in.

**Tooling lead for Phase 7:** Chronologically Consistent LLMs (arXiv 2502.21206) and
DatedGPT-style models are candidate open point-in-time models worth evaluating as a
contamination-free comparison arm.

### Q4 — Benchmark + modern corpus: **CLEAR (moderate certainty)**

No new multimodal (audio-bearing) earnings-call dataset or leakage-proof benchmark surfaced
beyond FinCall-Surprise itself. DEC (from 2412.18029) is transcript-only; EvasionBench has no
market targets and class-balanced (not temporal/disjoint) splits; **MiMIC** (arXiv 2504.09257,
Apr 2025) is an Indian multimodal dataset whose authors *explicitly could not collect audio*
(transcripts + slides + fundamentals only) — if anything it documents the audio gap our corpus
fills. ecvol-bench and the 2024–2026 corpus stretch goal both still fill real gaps. Verdict is
an absence claim — treat as provisional and re-scan before submission.

### Q5 — Other new work to cite or compare

- **The Sound of Risk** (arXiv 2508.18653; our R12) — verified full text: claims text+acoustic
  features explain up to 43.8% OOS variance of 30-day RV on 1,795 calls, **directly opposing
  our "audio inert / text is identity" finding**. It reports no identity controls. This is our
  most important *empirical comparison target*: our controls predict its result would not
  survive an identity/shuffle analysis. Address head-on in discussion.
- Financial-LLM eval-rigor landscape (optional cites): five-biases position paper (arXiv
  2602.14233, Feb 2026), FLaME holistic financial-NLP benchmark (arXiv 2506.15846), LLM
  stock-forecasting survey (arXiv 2605.05211), decision-time leakage benchmark on OHLCV
  panels (arXiv 2605.23959, May 2026).

## Recommended actions

1. **Related work adds (must-cite):** 2504.14765, 2601.13770, 2601.09142, 2605.03147,
   2603.26797, 2508.13635; refresh R15 to its v2 (Jun 2026).
2. **Framing tweaks:** position the lookahead study inside a now-active literature (diagnostic
   LAP / preventive DatedGPT / benchmark Look-Ahead-Bench) as its missing empirical leg;
   scope the uniqueness claim to the earnings-call volatility literature; cite 2504.14765 as
   independent proof that identity leaks even under masking.
3. **Discussion section:** engage The Sound of Risk directly as the strongest recent positive
   claim our controls challenge.
4. **Phase 7:** consider a chronologically-consistent-LLM comparison arm (2502.21206).
5. **Timing:** stage (b)'s novelty is time-sensitive; keep the four-week plan.
6. **Re-scan before submission** (workshop proceedings and SSRN under-indexed; Jun–Jul 2026
   arXiv may lag).

## Caveats (verbatim concerns from the verification pass)

CLEAR verdicts rest on absence of evidence, not exhaustive coverage; several load-bearing
sources are unreviewed preprints (2412.18029 still v1; Look-Ahead-Bench vendor-affiliated);
EvasionBench's κ=0.835 must not be cited as a human-audit precedent (refuted 1–2 in
verification); the KPI paper's human-validation analogy survived only 2–1 and its kappa is
"fair" (0.39), measuring human-human agreement; FinCall-Surprise's own split/leakage
properties were treated as known and should be characterized precisely in the benchmark
section.
