# DESIGN.md — Earnings-Call Multimodal Volatility Prediction (Rework)

**Status:** Approved design, pre-implementation.
**Created:** 2026-06-12.
**Role of this document:** Source of truth for the project. Every implementation decision, experiment, and deviation traces back to (or amends) this document. Amendments are recorded in [DECISIONS.md](DECISIONS.md) — append-only, dated, with rationale; this file is never silently edited.

Working name for the package/benchmark: **`ecvol`** ("earnings-call volatility").

---

## Table of contents

1. [Motivation & research questions](#1-motivation--research-questions)
2. [Lessons from the legacy project](#2-lessons-from-the-legacy-project)
3. [Related work & known pitfalls](#3-related-work--known-pitfalls)
4. [Contributions & publication strategy](#4-contributions--publication-strategy)
5. [Data design](#5-data-design)
6. [Modeling ladder (Stages 0–6)](#6-modeling-ladder-stages-06)
7. [Evaluation protocol](#7-evaluation-protocol)
8. [System design](#8-system-design)
9. [Phase plan — summary; full tracker in TASKS.md](#9-phase-plan)
10. [Risk register](#10-risk-register)
11. [Decision log → DECISIONS.md](#11-decision-log)
12. [Reproducibility checklist](#12-reproducibility-checklist)
13. [References](#13-references)

---

## 1. Motivation & research questions

Earnings conference calls are one of the few recurring events where management speaks at length, semi-spontaneously, about a public company — in both *what* is said (transcript) and *how* it is said (audio). A line of research from Qin & Yang (2019) through HTML, MAEC, KeFVP, and ECC Analyzer claims that multimodal models over this data predict post-call stock volatility better than classical methods.

This project reworks an abandoned 2024 university study in that lineage, from the ground up, with three goals:

1. **Modernize** — replace 2019-era components (GloVe, Praat-only audio, hand-rolled hierarchical transformers) with current open-source models: instruction-tuned LLMs with constrained decoding, financial sentence embeddings, self-supervised speech representations, and audio emotion models.
2. **Open & reproduce** — use only openly licensed data and open-weight models, release everything needed to rebuild the study, and pin the environment. The financial-NLP literature has a documented reproducibility crisis (see §3.4); this project is designed against it.
3. **Evaluate honestly** — recent work shows much of the published leaderboard may be an artifact of ticker-identity memorization and weak baselines (see §3.1, §3.3). Every model here is measured against strong econometric baselines under identity-aware controls.

### Research questions

| ID | Question | Where answered |
|----|----------|----------------|
| **RQ1** | Does earnings-call *content* (text and/or audio) add predictive signal for post-call volatility beyond past-volatility persistence (HAR-RV, GARCH)? | Stages 0–2 vs. ladder; Δv target |
| **RQ2** | Is multimodal (audio + text) better than the best unimodal model, after controlling for past volatility? | Stage 4 ablations |
| **RQ3** | Do LLM-extracted *structured, auditable* features (guidance, hedging, evasiveness…) outperform opaque dense embeddings? | Stage 5 vs. Stage 2/4 |
| **RQ4** | Do any positive results survive (a) ticker-identity controls, (b) ticker-disjoint splits, (c) evaluation on post-LLM-knowledge-cutoff calls? | Control suite (§7.3) + Phase 7 |

A clean negative or mixed answer to RQ1/RQ4 is a publishable result (see §4); the study is designed so that no single outcome strands the work.

---

## 2. Lessons from the legacy project

The legacy project (read-only in `legacy/`; fully described in [OLDWORK.md](OLDWORK.md)) reproduced HTML and KeFVP on the EarningsCall and MAEC datasets and reported coming within +0.73% mean MSE of KeFVP — a single-seed result this rework neither defends nor chases. We inherit its concepts — the multimodal premise, the 3/7/15/30-day log-RV target family, the embedding-ladder methodology, and its data-validation instincts (now acceptance gates) — and reject its practices:

### Rejected (with reasons)

| Legacy practice | Why rejected | Replacement |
|---|---|---|
| Single seed (777), one run per config | No variance estimate; the +0.73% headline is statistically meaningless | ≥5 seeds, mean ± std, significance tests (§7.2) |
| 17-value α sweep chosen on val, then retrain on train+val | Selection-overfit machine; inflates results | Fixed multi-horizon heads or per-horizon models; honest validation, pre-registered comparisons |
| Per-sentence audio–text alignment (1 MP3 per transcript line) | The single most fragile pipeline component; alignment errors silently corrupt features | Section-level (prepared remarks vs. Q&A) and speaker-turn pooling; sentence alignment deferred to Stage 6+ |
| 12 copy-pasted notebooks, plain + `MAEC_` branch duplication everywhere | Any fix needs editing up to 6 places; variants drift silently | One Python package, one code path, datasets as config (§8) |
| No requirements.txt, no pinned versions | Unrebuildable 18 months later (proven: it is) | uv-managed lockfile, CI |
| yfinance/Alphavantage as primary price source, hardcoded `bad_tickers` lists | API drift; silent ticker drops | Stooq primary + Tiingo cross-check, explicit delisting reports (§5.2) |
| Raw data on a personal `D:` drive, artifacts gitignored individually | Data lost when the drive was wiped; ~200-line hand-written .gitignore | Checksummed manifests, documented acquisition scripts, simple glob ignores |
| No baselines beyond prior neural models | Can't tell whether *any* model beats past-vol persistence | Stage 0/1 econometric + identity baselines reported in every table |

---

## 3. Related work & known pitfalls

A condensed, annotated map. Full citations with URLs in [§13](#13-references).

### 3.0 The benchmark lineage (what we build on)

| Work | Year / venue | Approach | Notes |
|---|---|---|---|
| Qin & Yang, "What You Say and How You Say It Matters" [R1] | ACL 2019 | First text+audio volatility prediction; released the 572-call EarningsCall dataset | Origin of the targets and the field |
| MAEC [R2] | CIKM 2020 | 3,443-call aligned multimodal dataset (S&P 1500) | CC-BY-SA-4.0; 6× scale-up |
| HTML [R3] | WWW 2020 | Hierarchical transformer, multi-task across horizons | Legacy project's main architecture |
| VolTAGE [R4] | EMNLP 2020 | Graph convolution over stock relations + text/audio fusion | |
| DialogueGAT [R5] | EMNLP 2022 Findings | Graph attention over dialogue/speaker structure | |
| KeFVP [R6] | EMNLP 2023 Findings | Knowledge-enhanced pre-training + price-conditioned time-series module | The standard SOTA reference the legacy project chased |
| AMA-LSTM [R7] | NAACL 2024 Industry | Adversarial training to debias audio (gender) | See pitfall §3.5 |
| ECC Analyzer [R8] | ICAIF 2024 | LLM hierarchical summarization + RAG "question bank" + wav2vec2; reports 27.7% MSE reduction | Closest in spirit to our Stage 5; uses closed pipeline components |
| RiskLabs [R9] | 2024 (arXiv) | LLM fusion of calls + news + market time series | |
| ECHO-GL [R10] | AAAI 2024 | Heterogeneous graphs from call semantics (movement prediction) | |
| AT-FinGPT [R11] | Finance Research Letters 2025 | Audio-text LLM for risk prediction | |
| "The Sound of Risk" [R12] | 2024 (arXiv) | Physics-informed acoustics; paralinguistics explain up to 43.8% of 30-day RV variance | Motivates the audio ladder |
| FinAudio benchmark [R13] | 2025 (arXiv) | Audio-LLMs struggle on long financial audio | Caution for Stage 6 audio-LLM plans |

### 3.1 Pitfall: ticker-identity leakage — **the central threat to validity**

**"Same Company, Same Signal" [R14]** (arXiv 2412.18029) shows that transcript representations used across this literature predominantly encode *company identity*, not content: same-ticker transcripts cluster tightly, and **training-free baselines built on each company's past volatility patterns beat all transcript-based models** on their evaluation. Since volatility is strongly autocorrelated and company-specific, a model that memorizes "which company is this" inherits most of the apparent skill.

**Design response:** (a) the Δv / HAR-residual target variants, which subtract the company's own past volatility (§5.3); (b) a ticker-fixed-effect baseline in every table (Stage 1); (c) the identity-control suite — ticker-only model, same-ticker transcript shuffle, identity probes, ticker-disjoint splits (§7.3).

### 3.2 Pitfall: LLM lookahead bias

LLMs whose training data covers the evaluation period can "predict" outcomes they have effectively seen ([R15], arXiv 2512.23847; DatedGPT [R16]). Any Stage-5 result on 2019–2021 calls processed by a 2024-cutoff model is suspect by default.

**Design response:** (a) Phase 7 acquires a 2025–2026 post-knowledge-cutoff test set and re-evaluates the frozen pipeline; (b) a prompt-masking ablation (company names, tickers, dates removed) quantifies identity-based leakage in LLM features (§7.4).

### 3.3 Pitfall: weak baselines

Much of the literature compares only against prior neural models, omitting persistence, GARCH(1,1), and HAR-RV [R17] — which are very hard to beat, especially at longer horizons.

**Design response:** Stage 0 baselines computed first, reported in every results table, and used as the denominator of the headline metric (R²_OOS vs. persistence, §7.1). Gate: no deep-learning work begins until Result Table 1 (baselines) is committed.

### 3.4 Pitfall: reproducibility crisis

A 2025 survey of financial NLP [R18] found only ~14% of reported results reproduce exactly and ~59% reproduce worse than reported; causes include dead data links, API drift, and contamination.

**Design response:** §8 (lockfile, manifests with SHA-256, run artifacts keyed by config hash + git SHA), §12 checklist, and the `ecvol report` byte-identical regeneration test.

### 3.5 Pitfall: gender bias in audio features

Audio features encode speaker gender; female executives are underrepresented in these datasets, producing measurable MSE disparities ([R7], [R19], DocFin [R20] reports up to 30% bias reduction as a contribution).

**Design response:** a gender-confound analysis in Stage 3 (pitch-proxy correlation of audio features and errors, per-group error reporting). We analyze and report; adversarial debiasing à la AMA-LSTM is out of v1 scope.

### 3.6 Adjacent work intentionally out of scope (v1)

Agentic trading systems (MarketSenseAI 2.0 [R21], P1GPT [R22], AlphaAgents [R23]), time-series foundation models (Kronos [R24], FinCast [R25]), graph methods over stock universes (ECHO-GL), and RAG pipelines. These are scope-creep magnets; the Decision Log must record an explicit, dated case before any enters scope.

---

## 4. Contributions & publication strategy

### Planned contributions

1. **An open multimodal benchmark** ("ecvol-bench"): standardized volatility targets (level, Δ, HAR-residual), leakage-proof temporal + ticker-disjoint splits with embargo, SHA-256 data manifests, and an evaluation harness — built over FinCall-Surprise (primary) and MAEC (secondary), plus a fresh post-cutoff test set acquired via released scripts.
2. **Honest baselines everywhere**: persistence, EWMA, HAR-RV, GARCH(1,1), and ticker-fixed-effect models reported for every horizon, split, and target variant — the comparison most prior work omits.
3. **A modern open-model feature ladder on consumer hardware**: financial sentence embeddings, FinBERT, WavLM/emotion2vec+/eGeMAPS audio features, gated fusion, and LLM-extracted structured features via constrained decoding — all runnable on a single 16–24 GB GPU, fully open weights.
4. **A diagnostic suite** for this task family: identity probes, transcript-shuffle controls, ticker-disjoint evaluation, lookahead-bias tests on post-cutoff data, and gender-confound analysis.

### Pre-registered dual framing (decision gate after Phase 2/3)

The paper's spine is decided by the data, at a pre-registered gate, not retro-fitted to it:

- **Path A — "Positive showcase":** *adopted if* content-bearing models (Stage ≥2) beat the Stage-0/1 floor on the **Δv or HAR-residual target** with DM-significant improvements (p < 0.05) that survive the identity-control suite on at least 2 of 4 horizons. Story: "modern open LLM + audio models extract real incremental signal from earnings calls — here is an open, reproducible system."
- **Path B — "Rigorous re-examination":** *adopted otherwise.* Story: "we re-examined the premise of a decade of multimodal earnings-call prediction with modern models, strong baselines, and identity controls — here is the honest answer and an open benchmark for the field." [R14] demonstrates venues accept this.
- The gate is evaluated after Phase 3 (text ladder + early controls); the framing choice is recorded in the Decision Log. Confirmatory vs. exploratory analyses are labeled in §7.5 *now*, before any model result exists.

### Candidate venues (decide in Phase 8, results in hand)

- **ACM ICAIF** — natural fit either path; prior home of ECC Analyzer.
- **FinNLP workshop** (ACL/EMNLP co-located) — fastest, friendly to benchmark/negative results.
- **ACL/EMNLP Findings** — if the identity/lookahead diagnostic suite produces field-level insights (Path B strong form).
- **Finance Research Letters / applied finance journals** — if results skew econometric.
- arXiv preprint precedes any submission.

---

## 5. Data design

### 5.1 Datasets, in priority order

| # | Dataset | Size / span | Modalities | License | Role | Source |
|---|---|---|---|---|---|---|
| 1 | **FinCall-Surprise** [D1] | 2,688 calls, 2019–2021 | Full MP3 audio + JSON transcripts + slides | **Apache-2.0** | **Primary** development & main results | github.com/Tizzzzy/FinCall-Surprise (+ Google Drive for audio) |
| 2 | **MAEC** [D2] | ~3,443 calls, 2015–2018, S&P 1500 | Sentence-aligned audio features/clips + transcripts | CC-BY-SA-4.0 | Secondary: scale + literature comparability | github.com/Earnings-Call-Dataset/… |
| 3 | Legacy **EarningsCall** (Qin & Yang) [D3] | 572 calls, 2017, S&P 500 | Per-sentence MP3 + transcripts | Unclear (research release) | One comparability table only; **never redistributed** | github.com/GeminiLn/EarningsCall_Dataset |
| 4 | **Fresh 2025–2026 calls** (Phase 7) | target ≥200 calls, 2025-Q4–2026-Q1 | Audio + transcripts | N/A — scripts-not-data release | Post-LLM-cutoff holdout (lookahead experiment) | EarningsCall/EarningsCast API [D4]; company IR pages fallback |
| — | Earnings-21 / Earnings-22 [D5] | 39 h / 119 h | Audio + reference transcripts | CC-BY-SA-4.0 | Audio/ASR pipeline QC only (not a prediction dataset) | github.com/revdotcom/speech-datasets |
| — | SPGISpeech [D6] | 5,000 h | Audio + transcripts | Academic-only, no redistribution | **Skipped in v1** (license complicates the open story; domain adaptation is a Stage-6 luxury) | huggingface.co/datasets/kensho/spgispeech |

First implementation act of Phase 1: download, checksum, and locally mirror datasets 1–2 (link-rot is a top risk; retire it immediately).

### 5.2 Price data

- **Primary: Stooq** [D7] — free, stable, split/dividend-adjusted daily OHLCV; bulk CSV archives; accessible via pandas-datareader.
- **Cross-check: Tiingo** [D8] (free tier) on a 5% random ticker sample. **Acceptance gate:** adjusted-return correlation > 0.999 on the overlap; any ticker below 0.99 is investigated and documented.
- **Not used as primary:** yfinance (policy/data drift — the legacy `bad_tickers` failure mode), Alpha Vantage (rate limits), Polygon free tier (25 calls/day).
- All raw pulls cached to parquet with a date-stamped manifest entry (URL pattern, retrieval date, SHA-256). Delisted/unmatched tickers produce an explicit **coverage report** — never silently dropped.

### 5.3 Target definitions (exact)

Let `P_t` be the **adjusted close** on trading day `t`, with `t = 0` the last trading day **before** the call's information is public, defined by the **after-hours rule**: if the call begins at or after 16:00 ET (or timestamp is missing but the call is marked "after market close"), day `t = 1` is the next trading day; if before 09:30 ET, the call's own date is day 1; intraday calls (rare) are treated as after-hours and flagged. Daily simple return: `r_t = (P_t − P_{t−1}) / P_{t−1}`.

**Primary target — log realized volatility** over horizon τ ∈ {3, 7, 15, 30} *trading* days (Qin & Yang convention, for comparability):

```
v_post(τ) = ln( sqrt( (1/τ) · Σ_{t=1..τ} (r_t − r̄)² ) ),   r̄ = mean(r_1..r_τ)
```

**Pre-call volatility** `v_pre(τ)`: same formula over the τ trading days ending at day 0.

**Headline variants (identity-robust):**
- **Volatility change:** `Δv(τ) = v_post(τ) − v_pre(τ)` — subtracts the company's own level; a model must predict *how this call changes things*.
- **HAR residual:** `v_post(τ) − HAR_RV_forecast(τ)`, where the HAR-RV model [R17] (`RV_{t,t+τ} ~ β₀ + β_d·RV_daily + β_w·RV_weekly + β_m·RV_monthly`) is fit on training data only.

**Out of v1 scope** (exploratory backlog): directional movement, abnormal volume, implied-vol targets. *(Short-horizon/event-window RV and implied-vol are explored — not added to the headline targets — under TASKS.md TX2; DECISIONS.md 2026-06-14.)*

Edge rules (encode in `targets.py`, unit-tested): non-trading-day call dates roll forward; insufficient post-call history (τ days unavailable, e.g., delisting) → target is NaN and the (call, τ) row is excluded with a reason code; zero-variance windows → NaN (log of 0), excluded with reason code.

### 5.4 Split design (leakage-proof)

1. **Primary — temporal:** train < validation < test in strict calendar order by call timestamp, with a **30-trading-day embargo** between segments so no target window crosses a boundary (mandatory because of the 30-day target; most prior work ignores this). Approximate proportions 70/10/20 by call count, boundaries snapped to embargo-respecting dates.
2. **Secondary — ticker-disjoint:** companies in val/test never appear in train (grouped split by ticker, stratified by sector where metadata allows). The headline identity-control split.
3. **Tertiary — temporal × ticker-disjoint:** both constraints; hardest condition, reported as a robustness row.
4. **Information rule (codified, enforced):** every feature function receives `(call, as_of_timestamp)`; the harness fails any feature that reads price data after `as_of`. Past-vol covariates use only `t ≤ 0` data.
5. FinCall-Surprise spans COVID. Per-year breakdowns are mandatory in reporting; a sensitivity run excluding Feb–Apr 2020 is exploratory, not the main split.
6. Split assignments are **committed CSVs** generated once by `ecvol splits build` and never regenerated silently; leakage assertions run in `pytest` on every CI run.

### 5.5 Legal & release strategy

**Released:** all code/configs (Apache-2.0 or MIT), data manifests (URL + SHA-256 + license per file), derived numeric features (embeddings, eGeMAPS vectors, LLM-extracted JSON) for FinCall-Surprise (Apache-2.0-derived) and MAEC (released features inherit **CC-BY-SA-4.0** — stated explicitly), all result tables/artifacts, and acquisition scripts (incl. Phase 7 API scripts with a documented terms-of-use review).

**Never redistributed:** raw audio/transcripts from the legacy EarningsCall dataset (license unclear) or from the Phase-7 API/IR sources. The pattern is *scripts-not-data* for anything not clearly licensed (the SEC-EDGAR/becomingquant pattern; note: EDGAR 8-Ks contain press releases, only occasionally full transcripts — EDGAR serves as metadata source for earnings timestamps, not as a transcript source).

---

## 6. Modeling ladder (Stages 0–6)

**Invariant:** from Stage 2 up, every learned model includes past-vol covariates (`v_pre` at all horizons + the HAR-RV forecast) in its input — and is *also* reported with them ablated. This bakes the persistence control into every comparison instead of leaving it to a separate table.

**Heads policy:** small-data regime (10³–10⁴ samples) ⇒ frozen representations + light heads (ridge, shallow MLP, LightGBM) before any fine-tuning. Per-horizon models by default; a shared multi-task head is an ablation, not the headline (the legacy α-sweep is explicitly dead).

| Stage | Model | Hypothesis (falsifiable) | Stop/go criterion | Compute |
|---|---|---|---|---|
| **0** | Persistence (`v_pre`), EWMA, **HAR-RV** (`statsmodels` OLS), **GARCH(1,1)** (`arch`) | "What does zero call content achieve?" — establishes the floor | Sanity gate: HAR-RV ≥ persistence at τ=30 (stylized fact); if violated, debug targets before proceeding | CPU, minutes |
| **1** | LightGBM on tabular: past-vol features + trivial metadata (transcript length, Q&A turn count, sector, market-cap bucket) + **ticker fixed effect** | "How much of published 'multimodal' performance is identity + metadata?" — the in-house Same-Company-Same-Signal baseline | Always reported; no gate | CPU |
| **2** | Frozen text: BGE-large / GTE sentence embeddings (section-pooled), FinBERT sentiment aggregates per section, surface statistics → ridge / shallow MLP | "Does transcript *content* add signal beyond Stage 1?" (RQ1-text) | If no DM-significant gain over Stage 1 on Δv on any horizon → run §7.3 controls early and trigger the framing gate (§4) | GPU inference only |
| **3** | Frozen audio: **eGeMAPS first** (openSMILE, CPU-cheap, interpretable), then WavLM-Large pooled chunks + emotion2vec+ → same heads. Speaker-turn pooling via pyannote 3.1 (flag-gated). **No per-sentence alignment.** | "Does prosody add signal beyond text + past-vol?" (RQ1-audio) + gender-confound check | Full-corpus extraction ETA measured on a 50-call sample before committing the GPU-weeks | GPU, throughput-bound |
| **4** | Fusion: gated fusion and cross-attention heads over frozen modality embeddings; late-fusion stacking with the Stage-1 GBDT | "Is multimodal > best unimodal, or just noise ensembling?" (RQ2) | Main paper table regardless of outcome | GPU, light training |
| **5** | **LLM-extracted structured features:** Qwen2.5-7B-Instruct (4-bit) + Outlines-constrained JSON per call section — guidance direction, hedging/uncertainty intensity, Q&A evasiveness, surprise mentions, analyst tone — fed into the Stage-1 GBDT alongside covariates | "Do explicit, auditable semantics beat opaque embeddings?" (RQ3) — ECC Analyzer's idea with open weights and no RAG | Human-audit gate: κ > 0.6 vs. human labels on 50 calls for categorical fields *before* scaling to the corpus | 16–24 GB GPU, slow batch |
| **6** *(optional, cloud, gated)* | QLoRA fine-tune of Qwen2.5-7B on transcripts; audio-LLM scoring (Qwen2.5-Omni / Qwen2-Audio); per-sentence alignment revisit | "Does end-to-end adaptation beat frozen features in this small-data regime?" | Only if Stages 2–5 show DM-significant signal; requires a Decision-Log entry with budget | Cloud A100s |

Model/version pins (exact HF revisions recorded in configs at implementation time): `BAAI/bge-large-en-v1.5` (or current best financial-domain embedding on MTEB at Phase-3 start — pinned then), `ProsusAI/finbert`, `microsoft/wavlm-large`, `emotion2vec/emotion2vec_plus_large`, `openai/whisper-large-v3-turbo` (ASR/QC only), `pyannote/speaker-diarization-3.1`, `Qwen/Qwen2.5-7B-Instruct`.

**Exploration hooks (adopted from prior-team work; DECISIONS.md 2026-06-14, exploratory until they clear §7.3/§4):** Stage 5 has an open-weight, data-driven variant — QA-generation → train-only volatility-topic clustering → per-call topic-frequency features (TASKS.md **TX1**). Stage 6's audio-LLM scoring has a concrete recipe — Qwen2.5-Omni-7B, masked-mean-pool the Thinker last hidden state (3584-d), 4-bit NF4, QA-conditioned + task-aware prompt (TASKS.md T8.3).

---

## 7. Evaluation protocol

### 7.1 Metrics

- **MSE on log RV per horizon** — comparability with the entire literature.
- **R²_OOS vs. persistence** (Gu–Kelly–Xiu-style, [R26]): `1 − Σ(y − ŷ)² / Σ(y − ŷ_persistence)²` — the **headline metric**; not gameable by identity memorization the way raw MSE is.
- MAE; **cross-sectional Spearman rank correlation per calendar quarter** (what a practitioner ranking risk actually uses).
- All metrics reported on all three targets (level v, Δv, HAR residual) × all splits.

### 7.2 Statistical rigor

- **≥ 5 seeds** per learned model; report mean ± std. Seeds live in configs.
- **Diebold–Mariano tests** [R27] on per-call squared-error differentials vs. (a) HAR-RV and (b) the best preceding stage.
- **Cluster bootstrap CIs** (1,000 resamples), clustered by ticker and, separately, by calendar quarter — errors are not i.i.d. across either.
- Multiple-comparison awareness: confirmatory claims (§7.5) use Holm-corrected p-values across horizons.

### 7.3 Identity-control suite (the signature section)

1. **Ticker-only model:** learned ticker embedding → head, no content. If it matches a multimodal model on level-v, that model reads identity.
2. **Same-ticker transcript shuffle:** replace each test call's transcript with a *different call from the same ticker*; if performance is unchanged, the model reads identity, not content.
3. **Identity probe:** linear probe predicting ticker from frozen call embeddings; report accuracy (expected high per [R14]) and then show whether Δv-target results survive.
4. **Ticker-disjoint split** results side-by-side with temporal-split results, every table.

### 7.4 Lookahead-bias experiments

- **Post-cutoff evaluation (Phase 7):** the entire frozen pipeline (no retraining, no threshold changes after first look — rule stated here, in advance) evaluated on 2025–2026 calls. Significant degradation of Stage-5 LLM features relative to other stages ⇒ contamination evidence.
- **Masking ablation:** Stage-5 prompts with company names, tickers, and dates masked vs. unmasked; the gap estimates identity-mediated leakage.

### 7.5 Pre-registered analyses

**Confirmatory** (claims the paper can make, decided now): Stage-k vs. Stage-1 DM tests on Δv at each τ (Holm-corrected); Stage 4 vs. best unimodal on Δv; Stage 5 vs. Stage 2 on Δv; temporal vs. ticker-disjoint gap per stage.
**Exploratory** (reported as such): per-year/regime breakdowns, COVID-exclusion sensitivity, prepared-remarks-vs-Q&A ablation, call-length sensitivity, gender-confound analysis, level-v leaderboard comparisons, legacy-dataset comparability table.

### 7.6 Ablation grid (minimum for a credible paper)

modality {text, audio, both} × past-vol covariates {in, out} × split {temporal, disjoint, both-constraints} × target {v, Δv, HAR-resid} × horizon {3,7,15,30} — populated by Phase 5; LLM-feature and masking ablations added by Phase 6.

---

## 8. System design

### 8.1 Directory layout

```
.                            # project root
  CLAUDE.md                  # agent guide & working principles
  DESIGN.md                  # this document — source of truth
  TASKS.md                   # living task tracker (operational copy of §9)
  OLDWORK.md                 # legacy-project reference (outdated, read-only)
  legacy/                    # abandoned 2024 project: notebooks, Paper.pdf, papers/ (read-only)
  pyproject.toml             # uv-managed; lockfile committed
  configs/                   # one YAML per experiment; pydantic-validated
  src/ecvol/
    cli.py                   # Typer app: prices|targets|splits|featurize|train|evaluate|report
    config/                  # pydantic schemas for all configs
    data/                    # stooq.py, tiingo.py, fincall.py, maec.py, legacy.py,
                             # manifests.py, splits.py, targets.py, calendar.py
    features/
      text/                  # sections.py, embeddings.py, finbert.py, surface.py
      audio/                 # qc.py, egemaps.py, wavlm.py, emotion2vec.py, diarize.py
      llm/                   # schema.py, prompts.py, extract.py (Outlines), audit.py
    models/                  # baselines.py (persistence/EWMA/HAR/GARCH), gbdt.py,
                             # heads.py (ridge/MLP), fusion.py, ticker_only.py
    eval/                    # metrics.py, significance.py (DM, bootstrap),
                             # controls.py (shuffle/probe), report.py
  tests/                     # unit + leakage-assertion tests (run in CI)
  notebooks/                 # exploration and figures ONLY — no pipeline logic
  data/                      # gitignored payloads; manifests/ committed (JSON + SHA-256)
  artifacts/                 # gitignored run outputs: runs/<run_id>/{config,metrics,model}
```

### 8.2 Conventions

- **CLI contract:** each verb is idempotent and resumable; re-running with an identical config is a no-op (content-hash caching) or a bit-identical regeneration.
- **Configs:** YAML validated by pydantic schemas. Deliberately **not Hydra** — one researcher; debuggability beats composability. Every experiment = one config file under `configs/`, committed.
- **Tracking:** every run writes `artifacts/runs/<run_id>/` containing resolved config, config hash, git SHA, seed list, environment fingerprint, and metrics parquet. Deliberately **not DVC/wandb** for v1 — JSON manifests + parquet are sufficient and have no service dependency (wandb may be added later via Decision Log).
- **Manifests:** every external file gets `{path, source_url, retrieved_at, sha256, license}` in `data/manifests/*.json`, committed.
- **Feature caches:** parquet keyed by `(extractor_name, extractor_version, content_sha256)` — deterministic re-extraction is an acceptance test.
- **Testing:** pytest; leakage assertions (split overlap, embargo, as-of rule) are tests, not conventions. CI (GitHub Actions): ruff + pytest on every push.
- **Seeds:** in configs, never hardcoded. Multi-seed runs are first-class (`train --seeds 5`).

### 8.3 GPU memory budget (24 GB card; 16 GB noted)

| Component | Precision | Approx. VRAM | 16 GB viable? |
|---|---|---|---|
| BGE-large / GTE embedding inference | fp16 | ~2 GB | Yes |
| FinBERT inference | fp16 | ~1 GB | Yes |
| WavLM-Large inference (chunked ≤30 s) | fp16 | ~4–6 GB | Yes (smaller chunks) |
| emotion2vec+ (funasr) | fp16 | ~2–3 GB | Yes |
| Whisper-large-v3-turbo (QC/ASR) | fp16 | ~6 GB | Yes |
| pyannote 3.1 diarization | fp16 | ~3 GB | Yes |
| Qwen2.5-7B-Instruct, 4-bit + Outlines | int4 | ~7–9 GB (+KV cache; 8k ctx OK) | Yes, tight |
| Fusion-head training (frozen features) | fp32 | <2 GB | Yes |
| QLoRA 7B fine-tune (Stage 6) | int4 + bf16 adapters | ~16–20 GB | Cloud preferred |

Components run sequentially, never co-resident; eGeMAPS (openSMILE) is CPU-only and parallelizes across cores.

---

## 9. Phase plan

**The full task breakdown — every task with goal, end result, acceptance test, and subtasks — lives in [TASKS.md](TASKS.md)**, the living tracker and operational document for execution status. Task IDs (T0.1–T8.3) are stable identifiers shared between the two documents; adding or materially changing a task requires an entry in [DECISIONS.md](DECISIONS.md).

**Sequencing principles:** (1) eval harness + baselines before any deep learning; (2) data risk retired before model risk; (3) every phase ends with a committed, regenerable artifact (result table or report); (4) identity controls run *early* (Phase 3), not as a paper-writing afterthought.

| Phase | Title | Delivers | Key gate |
|---|---|---|---|
| 0 | Scaffolding (~3–5 days) | Git repo, installable `ecvol` package, configs, run tracking, CI | Bit-identical rerun of an identical config |
| 1 | Data foundation (~1–2 wks) | Mirrored datasets, prices, targets, ingestion, splits | ≥95% price-join rate; leakage assertions green |
| 2 | Eval harness + baselines (~1 wk) | Metrics/significance modules; **Result Table 1** (persistence / EWMA / HAR-RV / GARCH / ticker-FE) | HAR-RV ≥ persistence at τ=30 sanity check |
| 3 | Text ladder (~1–2 wks) | Sectioning, frozen text features, **Result Table 2**, early identity controls | §4 framing gate triggered by control outcomes |
| 4 | Audio ladder (~2 wks) | Audio QC, eGeMAPS, WavLM/emotion2vec+, **Result Table 3**, gender analysis | 50-call ETA measurement before full extraction |
| 5 | Fusion + ablations (~1 wk) | Fusion heads; **Result Table 4** (main table, full §7.6 grid) | All confirmatory comparisons Holm-corrected |
| 6 | LLM structured features (~2 wks) | Schema, constrained extraction, **Result Table 5**, masking ablation | Human-audit κ > 0.6 before corpus-scale runs |
| 7 | Post-cutoff lookahead study (~2 wks) | Fresh 2025–26 acquisition scripts; frozen-pipeline evaluation | No retraining after first look (pre-registered) |
| 8 | Paper + repro package (~2–3 wks) | REPRODUCE.md, feature release, manuscript, venue choice | Clean-machine reproduction of Table 1 |

---

## 10. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | FinCall-Surprise Drive links rot / quality below advertised | Medium | High | T1.1 runs first; immediate local mirror + checksums; MAEC promoted to primary as fallback |
| 2 | Stooq coverage gaps for 2019–2021 delisted tickers | Medium | Medium | Tiingo secondary; explicit coverage report; documented universe restriction over silent drops |
| 3 | Audio extraction too slow on one GPU (thousands of hours) | High | Medium | eGeMAPS-first (CPU); 50-call ETA measurement gate (T4.3) before committing; budgeted cloud burst option |
| 4 | **Nothing beats HAR-RV under controls (negative result)** | Medium-high | Low (by design) | Path-B framing pre-registered (§4); control suite + Δv targets + open benchmark carry the paper; [R14] precedent |
| 5 | LLM extraction quality poor / unreliable | Medium | Medium | Constrained decoding (shape) + 50-call human-audit gate κ>0.6 (content) before scaling; schema designed from manual reading |
| 6 | Fresh-data API terms change / forbid research use | Medium | Low | Phase 7 isolated — paper stands without it (masking ablation remains); IR-page fallback |
| 7 | Call timestamps missing → after-hours rule unapplicable | Medium | Medium | T1.4 investigates early; documented fallback (assume after-hours, flag) with sensitivity check |
| 8 | Scope creep (agents, graphs, fine-tuning, new targets) | High | Medium | §3.6 out-of-scope list; Stage 6 gated; Decision-Log entry required for any scope change |
| 9 | Reproducibility drift (the 14% problem) | Medium | High | Lockfile, manifests, config-hash run artifacts, byte-identical `report` regeneration in CI |
| 10 | Compute upgrade never materializes | Low | Low | Ladder is consumer-GPU-complete through Stage 5; Stage 6 is optional by design |

---

## 11. Decision log

The decision log lives in **[DECISIONS.md](DECISIONS.md)** (append-only; format `YYYY-MM-DD — decision — rationale — alternatives rejected`). Day-to-day work history is recorded in **[JOURNAL.md](JOURNAL.md)**. Any deviation from this document requires a DECISIONS.md entry before (or alongside) the change.

---

## 12. Reproducibility checklist

What ships with the paper (tracked against [R18]'s failure causes):

- [ ] uv lockfile + Python version pin; environment fingerprint in every run artifact.
- [ ] Data manifests: URL, retrieval date, SHA-256, license for every external file.
- [ ] Acquisition scripts for all data (including Phase 7), with ToS review notes.
- [ ] Committed split CSVs; leakage assertions in CI.
- [ ] One config file per experiment, committed; config hash + git SHA in every artifact.
- [ ] ≥5 seeds for every learned model; seeds in configs.
- [ ] `REPRODUCE.md`: one command per paper table; clean-machine verification performed.
- [ ] Released derived features (FinCall-Surprise: Apache-2.0; MAEC-derived: CC-BY-SA-4.0).
- [ ] Exact model revisions (HF commit hashes) pinned in configs.
- [ ] All paper numbers generated by `ecvol report` from artifacts — no hand-transcribed results.

---

## 13. References

### Papers — lineage & SOTA

- [R1] Qin & Yang, "What You Say and How You Say It Matters: Predicting Stock Volatility Using Verbal and Vocal Cues," ACL 2019. https://aclanthology.org/P19-1038/
- [R2] Li et al., "MAEC: A Multimodal Aligned Earnings Conference Call Dataset for Financial Risk Prediction," CIKM 2020. https://dl.acm.org/doi/10.1145/3340531.3412879
- [R3] Yang et al., "HTML: Hierarchical Transformer-based Multi-task Learning for Volatility Prediction," WWW 2020. https://github.com/YangLinyi/HTML-Hierarchical-Transformer-based-Multi-task-Learning-for-Volatility-Prediction
- [R4] Sawhney et al., "VolTAGE: Volatility Forecasting via Text-Audio Fusion with Graph Convolution Networks," EMNLP 2020. https://aclanthology.org/2020.emnlp-main.643/
- [R5] Sang & Bao, "DialogueGAT: A Graph Attention Network for Financial Risk Prediction by Modeling the Dialogues in Earnings Conference Calls," Findings of EMNLP 2022. https://aclanthology.org/2022.findings-emnlp.117/
- [R6] Niu et al., "KeFVP: Knowledge-enhanced Financial Volatility Prediction," Findings of EMNLP 2023. https://aclanthology.org/2023.findings-emnlp.770/ · code: https://github.com/hankniu01/KeFVP
- [R7] Wang et al., "AMA-LSTM: Pioneering Robust and Fair Financial Audio Analysis for Stock Volatility Prediction," NAACL 2024 Industry. https://arxiv.org/abs/2407.18324
- [R8] Cao et al., "ECC Analyzer: Extracting Trading Signal from Earnings Conference Calls using Large Language Models for Stock Volatility Prediction," ICAIF 2024. https://arxiv.org/abs/2404.18470
- [R9] Cao et al., "RiskLabs: Predicting Financial Risk Using Large Language Model based on Multimodal and Multi-Sources Data," 2024. https://arxiv.org/abs/2404.07452
- [R10] Liu et al., "ECHO-GL: Earnings Calls-Driven Heterogeneous Graph Learning for Stock Movement Prediction," AAAI 2024. https://ojs.aaai.org/index.php/AAAI/article/view/29305
- [R11] "AT-FinGPT: Financial risk prediction via an audio-text large language model," Finance Research Letters, 2025. https://www.sciencedirect.com/science/article/abs/pii/S1544612325002314
- [R12] Chen et al., "The Sound of Risk: A Multimodal Physics-Informed Acoustic Model for Forecasting Market Volatility," 2024. https://arxiv.org/abs/2508.18653
- [R13] Cao et al., "FinAudio: A Benchmark for Audio Large Language Models in Financial Applications," 2025. https://arxiv.org/abs/2503.20990

### Papers — pitfalls & methodology

- [R14] "Same Company, Same Signal: The Role of Identity in Earnings Call Transcripts," 2024. https://arxiv.org/abs/2412.18029 — *central threat-to-validity reference*
- [R15] Gao, Jiang & Yan, "A Test of Lookahead Bias in LLM Forecasts," 2024. https://arxiv.org/abs/2512.23847
- [R16] "DatedGPT: Preventing Lookahead Bias in Large Language Models with Time-Aware Pretraining," 2025. https://arxiv.org/abs/2603.11838
- [R17] Corsi, "A Simple Approximate Long-Memory Model of Realized Volatility" (HAR-RV), Journal of Financial Econometrics, 2009. https://doi.org/10.1093/jjfinec/nbp001
- [R18] "Language Modeling for the Future of Finance: A Survey into Metrics, Tasks, and Data Opportunities," 2025. https://arxiv.org/abs/2504.07274 — *14% exact-reproduction finding*
- [R19] "Emo-bias: A Large Scale Evaluation of Social Bias on Speech Emotion Recognition," 2024. https://arxiv.org/abs/2406.05065
- [R20] Mathur et al., "DocFin: Multimodal Financial Prediction and Bias Mitigation using Semi-structured Documents," Findings of EMNLP 2022. https://aclanthology.org/2022.findings-emnlp.139/
- [R26] Gu, Kelly & Xiu, "Empirical Asset Pricing via Machine Learning," Review of Financial Studies, 2020 (R²_OOS convention). https://doi.org/10.1093/rfs/hhaa009
- [R27] Diebold & Mariano, "Comparing Predictive Accuracy," Journal of Business & Economic Statistics, 1995. https://doi.org/10.1080/07350015.1995.10524599

### Papers — adjacent (out of v1 scope, §3.6)

- [R21] Fatouros et al., "MarketSenseAI 2.0: Enhancing Stock Analysis through LLM Agents," 2025. https://arxiv.org/abs/2502.00415
- [R22] Lu et al., "P1GPT: A Multi-Agent LLM Workflow Module for Multi-Modal Financial Information Analysis," 2025. https://arxiv.org/abs/2510.23032
- [R23] "AlphaAgents: Large Language Model based Multi-Agents for Equity Portfolio Constructions," 2025. https://arxiv.org/abs/2508.11152
- [R24] "Kronos: A Foundation Model for the Language of Financial Markets," 2025. https://arxiv.org/abs/2508.02739
- [R25] Zhu et al., "FinCast: A Foundation Model for Financial Time-Series Forecasting," CIKM 2025. https://arxiv.org/abs/2508.19609

### Papers — additional ECC-volatility methods (surfaced by the Undermind review; added 2026-06-14, DECISIONS.md)

- [R28] Ye, Qin & Xu, "Financial Risk Prediction with Multi-Round Q&A Attention Network," IJCAI 2020 (dialogue/Q&A-structure modeling). https://www.ijcai.org/proceedings/2020/631
- [R29] Sawhney et al., "Multimodal Multi-Task Financial Risk Forecasting," ACM Multimedia 2020 (joint volatility + price-movement). https://dl.acm.org/doi/10.1145/3394171.3413752
- [R30] Chen et al., "Distilling Numeral Information for Volatility Forecasting" (NAM / ECNum), CIKM 2021 (numeral-aware text). https://doi.org/10.1145/3459637.3482089
- [R31] Yang et al., "NumHTML: Numeric-Oriented Hierarchical Transformer Model for Multi-task Financial Forecasting," AAAI 2022. https://arxiv.org/abs/2201.01770
- [R32] Shi et al., "Enhancing Volatility Forecasting in Financial Markets: A General Numeral Attachment Dataset for Understanding Earnings Calls" (GNAVol), IJCNLP-AACL 2023. https://aclanthology.org/2023.ijcnlp-short.5/
- *Deferred:* DeFVP (ICME 2024, differentiable sentence-selection for volatility) — surfaced by the review but no canonical URL verified as of 2026-06-14; add when confirmed.

### Datasets

- [D1] FinCall-Surprise (Apache-2.0, 2,688 calls 2019–2021, audio+transcripts+slides). Paper: https://arxiv.org/abs/2510.03965 · Data: https://github.com/Tizzzzy/FinCall-Surprise
- [D2] MAEC (CC-BY-SA-4.0). https://github.com/Earnings-Call-Dataset/MAEC-A-Multimodal-Aligned-Earnings-Conference-Call-Dataset-for-Financial-Risk-Prediction
- [D3] EarningsCall dataset (Qin & Yang 2019). https://github.com/GeminiLn/EarningsCall_Dataset
- [D4] EarningsCall / EarningsCast API (~9k companies, 55k+ historical calls). https://earningscall.biz/
- [D5] Earnings-21 / Earnings-22 (Rev.com, CC-BY-SA-4.0). https://github.com/revdotcom/speech-datasets · HF: https://huggingface.co/datasets/Revai/earnings21, https://huggingface.co/datasets/Revai/earnings22
- [D6] SPGISpeech (Kensho, academic-only). https://huggingface.co/datasets/kensho/spgispeech · paper: https://arxiv.org/abs/2104.02014
- [D7] Stooq historical price database. https://stooq.com/db/h/
- [D8] Tiingo. https://www.tiingo.com/

### Models & tools (pin exact revisions in configs at use time)

- Qwen2.5-7B-Instruct (Apache-2.0): https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- BGE embeddings (MIT): https://huggingface.co/BAAI/bge-large-en-v1.5
- FinBERT: https://huggingface.co/ProsusAI/finbert
- WavLM-Large (MIT): https://huggingface.co/microsoft/wavlm-large
- emotion2vec+ : https://huggingface.co/emotion2vec/emotion2vec_plus_large · https://github.com/ddlBoJack/emotion2vec
- Whisper-large-v3-turbo (MIT): https://huggingface.co/openai/whisper-large-v3-turbo
- pyannote speaker-diarization-3.1: https://huggingface.co/pyannote/speaker-diarization-3.1
- openSMILE / eGeMAPS: https://github.com/audeering/opensmile-python
- Outlines (constrained decoding): https://github.com/dottxt-ai/outlines
- arch (GARCH): https://github.com/bashtage/arch · statsmodels: https://www.statsmodels.org/
- LightGBM: https://github.com/microsoft/LightGBM
- uv: https://github.com/astral-sh/uv · Typer: https://typer.tiangolo.com/ · pydantic: https://docs.pydantic.dev/
- exchange_calendars: https://github.com/gerrymanoim/exchange_calendars

---

*End of DESIGN.md. Amendments via DECISIONS.md only.*
