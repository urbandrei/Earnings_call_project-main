# TASKS.md — Living Task Tracker

**This is the operational copy of the phase plan in [DESIGN.md](DESIGN.md) §9.** Design rationale, hypotheses, and protocols live there; execution state lives here.

## How to use this file

- **Statuses:** `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked. Update the task status line and subtask checkboxes **as part of finishing the work**, not after.
- **Definition of done:** a task is `[x]` only when its **acceptance test** passes. No exceptions; if the test is wrong, fix the test via a [DECISIONS.md](DECISIONS.md) entry first.
- **Notes:** record completion date, deviations, gotchas discovered, and links to run artifacts in the task's Notes line. The narrative of *how* the work went belongs in [JOURNAL.md](JOURNAL.md); link the journal entry from Notes when useful.
- **Task IDs are stable.** Never renumber. Adding, removing, or materially changing a task requires a dated entry in [DECISIONS.md](DECISIONS.md).
- **Sequencing:** phases are ordered to retire risk early (data before models, baselines before deep learning, controls before audio investment). Within a phase, tasks may interleave unless a dependency is noted.

---

## Phase 0 — Scaffolding (~3–5 days)

### T0.1 Package skeleton — `[~]`
- **Goal:** installable `ecvol` package with version control and CI.
- **End result:** git repo initialized at project root; `pip install -e .` (via uv) works; `ecvol --help` lists all verbs as stubs.
- **Acceptance test:** GitHub Actions green on ruff + pytest (empty suite passes); fresh-machine install from lockfile documented and tested once; `git log` shows small, focused commits from the very first one.
- **Subtasks:**
  - [ ] `git init` at project root; `.gitignore` (glob patterns: `data/` payloads, `artifacts/`, caches — **not** hand-listed files; decide handling of `legacy/` bulk binaries: ignore or track-without-LFS, document choice)
  - [ ] Initial commit: the six root .md files (CLAUDE, DESIGN, TASKS, DECISIONS, JOURNAL, OLDWORK)
  - [ ] `pyproject.toml` + uv lockfile
  - [ ] `src/ecvol/` layout per DESIGN.md §8.1
  - [ ] Typer CLI stub (`prices|targets|splits|featurize|train|evaluate|report`)
  - [ ] pre-commit (ruff format + lint)
  - [ ] CI workflow (GitHub Actions)
- **Notes:** —

### T0.2 Config system — `[ ]`
- **Goal:** every experiment definable as a validated YAML.
- **End result:** pydantic schemas for data/feature/model/eval configs; loader with clear errors.
- **Acceptance test:** invalid config (bad horizon, missing seed list) fails with an actionable message; round-trip load→resolve→dump is stable.
- **Subtasks:**
  - [ ] Schema modules under `src/ecvol/config/`
  - [ ] `configs/example.yaml`
  - [ ] Config-hash function (canonicalized JSON → SHA-256)
- **Notes:** —

### T0.3 Run tracking & manifests — `[ ]`
- **Goal:** every run and every data file is traceable.
- **End result:** run-artifact writer (config hash, git SHA, seeds, env fingerprint, metrics parquet); manifest module with checksum verify command.
- **Acceptance test:** rerunning an identical CPU-only config reproduces metrics **bit-identically**; `ecvol data verify` detects a deliberately corrupted file.
- **Subtasks:**
  - [ ] `manifests.py` (path, source URL, retrieved_at, SHA-256, license)
  - [ ] Run-ID scheme + `artifacts/runs/<id>/` writer
  - [ ] Env fingerprint capture (lockfile hash, CUDA/driver versions)
- **Notes:** —

---

## Phase 1 — Data foundation (~1–2 weeks) — *highest-risk-first*

### T1.1 Dataset acquisition & mirroring — `[ ]` *(do first — retires link-rot risk)*
- **Goal:** local, checksummed mirrors of FinCall-Surprise and MAEC.
- **End result:** `data/` mirrors + committed manifests; download scripts in-repo.
- **Acceptance test:** manifest verify passes; counts match published sizes (FinCall-Surprise ≈ 2,688 calls; MAEC ≈ 3,443) or the gap is documented; audio files spot-decoded (random 50) without error.
- **Subtasks:**
  - [ ] FinCall-Surprise GitHub + Google Drive fetcher
  - [ ] MAEC fetcher
  - [ ] Checksum + license fields in manifests
  - [ ] Storage estimate & location decision (external disk OK — manifests make it portable)
- **Notes:** —

### T1.2 Price ingestion (Stooq + Tiingo cross-check) — `[ ]`
- **Goal:** reliable adjusted daily OHLCV for the combined ticker universe.
- **End result:** `ecvol prices pull` → per-ticker parquet + manifest; coverage report (matched/missing/delisted with reasons).
- **Acceptance test:** ≥98% ticker coverage for the FinCall-Surprise universe (or documented shortfall); Tiingo cross-check correlation >0.999 on a 5% sample; zero silent drops (every excluded ticker has a reason code).
- **Subtasks:**
  - [ ] Ticker normalization (share classes, renames)
  - [ ] Stooq fetcher with caching
  - [ ] Tiingo sampler
  - [ ] Trading-calendar module (`exchange_calendars`)
  - [ ] Coverage report generator
- **Notes:** —

### T1.3 Target computation — `[ ]`
- **Goal:** exact, tested implementation of DESIGN.md §5.3.
- **End result:** `ecvol targets build` → one row per (call, τ) with `v_pre`, `v_post`, `Δv`, HAR inputs, reason codes for exclusions.
- **Acceptance test:** unit tests on synthetic price series with analytically known RV; the after-hours date rule tested explicitly (16:00 ET boundary cases); 3 calls hand-verified end-to-end against manual spreadsheet computation.
- **Subtasks:**
  - [ ] `targets.py` with `(call, as_of_timestamp)` signature
  - [ ] Edge-rule handling (non-trading days, insufficient history, zero variance)
  - [ ] HAR input features
  - [ ] Exclusion accounting with reason codes
- **Notes:** legacy notebooks 3/3b contain the old target logic — useful as a cross-check, do not port (see OLDWORK.md §7).

### T1.4 FinCall-Surprise ingestion — `[ ]`
- **Goal:** normalized call records on the common schema.
- **End result:** `(call_id, ticker, utc_timestamp, transcript_json, audio_path, speaker_metadata, source)` records for the full set.
- **Acceptance test:** 100% of calls parse or are excluded with reason codes; audio-duration distribution report generated; **≥95% join rate to price data + targets**.
- **Subtasks:**
  - [ ] Transcript JSON parser
  - [ ] Timestamp extraction/validation (after-hours rule needs call time — investigate availability; documented fallback: assume after-hours, flag it)
  - [ ] Ticker resolution
  - [ ] Join audit
- **Notes:** —

### T1.5 MAEC ingestion — `[ ]`
- **Goal:** same contract as T1.4 for MAEC.
- **End result:** MAEC on the common schema; documented gaps (audio availability, year coverage).
- **Acceptance test:** same gates as T1.4 (≥95% price join); discrepancies vs. published MAEC stats documented.
- **Subtasks:**
  - [ ] Folder-format parser (`YYYYMMDD_TICKER`)
  - [ ] Audio-feature vs. raw-audio availability audit
  - [ ] Schema mapping
- **Notes:** —

### T1.6 Split builder — `[ ]`
- **Goal:** committed, leakage-proof splits per DESIGN.md §5.4.
- **End result:** `ecvol splits build` → CSVs for temporal (30-trading-day embargo), ticker-disjoint, and combined splits, per dataset.
- **Acceptance test:** pytest leakage assertions: no target-window overlap across temporal boundaries; zero ticker intersection in disjoint splits; embargo verified against the trading calendar; split CSVs committed and stable across reruns.
- **Subtasks:**
  - [ ] `splits.py` with embargo logic
  - [ ] Grouped/stratified ticker split
  - [ ] Leakage assertion test module (runs in CI forever after)
- **Notes:** —

---

## Phase 2 — Eval harness + econometric baselines (~1 week) — *the floor*

### T2.1 Metrics & significance module — `[ ]`
- **Goal:** DESIGN.md §7.1–7.2 implemented and validated.
- **End result:** `eval/metrics.py` (MSE, MAE, R²_OOS, quarterly Spearman) and `eval/significance.py` (DM test, cluster bootstrap, Holm correction).
- **Acceptance test:** DM test validated against a published worked example / statsmodels reference; bootstrap CIs validated on synthetic data with known sampling distribution.
- **Subtasks:**
  - [ ] Metric functions over the (call, τ) prediction frame
  - [ ] Clustering keys (ticker, quarter)
  - [ ] Significance API consumed by `report`
- **Notes:** —

### T2.2 Stage-0/1 baselines → Result Table 1 — `[ ]`
- **Goal:** the honest floor, committed.
- **End result:** persistence, EWMA, HAR-RV, GARCH(1,1), ticker-FE LightGBM evaluated on every (dataset × split × target × τ); **Result Table 1** artifact.
- **Acceptance test:** sanity gate — HAR-RV beats persistence at τ=30 on the temporal split (stylized fact; if violated, **halt and debug targets**); GARCH fits converge for >95% of series or documented fallback.
- **Subtasks:**
  - [ ] `models/baselines.py`
  - [ ] `models/gbdt.py` with ticker fixed effect
  - [ ] Multi-seed runner (GBDT)
  - [ ] First end-to-end `ecvol evaluate` run
- **Notes:** this table alone tests the field's premise on open data, and feeds the §4 framing gate.

### T2.3 Reporting — `[ ]`
- **Goal:** all paper tables regenerable by one command.
- **End result:** `ecvol report` renders LaTeX + Markdown tables from run artifacts.
- **Acceptance test:** byte-identical regeneration from unchanged artifacts (CI check); Result Table 1 renders in both formats.
- **Subtasks:**
  - [ ] `eval/report.py`
  - [ ] Table specs as data
  - [ ] Figure stubs for notebooks
- **Notes:** —

---

## Phase 3 — Text ladder + early identity controls (~1–2 weeks)

### T3.1 Transcript normalization — `[ ]`
- **Goal:** robust sectioning and speaker structure without per-sentence alignment.
- **End result:** per-call structure: prepared remarks vs. Q&A; speaker turns with roles (operator/management/analyst) where derivable.
- **Acceptance test:** section-detection precision >90% on a 30-call hand-checked sample; speaker-role tagging audited on the same sample.
- **Subtasks:**
  - [ ] `features/text/sections.py` (heuristics + format-specific parsers)
  - [ ] Audit notebook
- **Notes:** —

### T3.2 Frozen text features — `[ ]`
- **Goal:** Stage-2 representations, cached.
- **End result:** BGE/GTE section-pooled embeddings; FinBERT sentiment aggregates (per section, per speaker role); surface stats (length, Q&A turn counts, numeric density). Parquet caches keyed by content hash.
- **Acceptance test:** deterministic re-extraction (cache hit = bit-identical); GPU throughput benchmarked and full-corpus ETA recorded.
- **Subtasks:**
  - [ ] `embeddings.py`, `finbert.py`, `surface.py`
  - [ ] Pooling strategies (mean, section-weighted) as config options
- **Notes:** —

### T3.3 Stage-2 results → Result Table 2 — `[ ]`
- **Goal:** first content-bearing models, honestly evaluated.
- **End result:** ridge + shallow-MLP heads, 5 seeds, with and without past-vol covariates; **Result Table 2** with DM tests vs. HAR-RV and vs. Stage 1.
- **Acceptance test:** every cell carries mean ± std over seeds; report regenerates; confirmatory comparisons labeled per DESIGN.md §7.5.
- **Subtasks:**
  - [ ] `models/heads.py`
  - [ ] Multi-seed orchestration
  - [ ] Ablation configs
- **Notes:** —

### T3.4 Identity-control suite (text) — `[ ]` *(run NOW, not at paper time)*
- **Goal:** know what the text models are actually reading before investing in audio.
- **End result:** ticker-only model, same-ticker transcript shuffle, identity linear probe; control table committed.
- **Acceptance test:** all three controls produce numbers for Stage 2 on both temporal and disjoint splits; outcome triggers the DESIGN.md §4 framing-gate review (DECISIONS.md entry required either way).
- **Subtasks:**
  - [ ] `models/ticker_only.py`
  - [ ] `eval/controls.py` (shuffle, probe)
  - [ ] Framing-gate review write-up
- **Notes:** —

---

## Phase 4 — Audio ladder (~2 weeks, throughput-bound)

### T4.1 Audio QC — `[ ]`
- **Goal:** know the corpus before burning GPU-weeks.
- **End result:** QC report for 100% of audio (duration, sample rate, clipping, silence ratio, decode errors); 16 kHz mono resampled store.
- **Acceptance test:** pipeline validated on Earnings-21 samples (known-good reference); corrupt files flagged with reason codes, not dropped silently.
- **Subtasks:**
  - [ ] `features/audio/qc.py`
  - [ ] ffmpeg-based resampler
  - [ ] QC report artifact
- **Notes:** —

### T4.2 eGeMAPS extraction (CPU, first) — `[ ]`
- **Goal:** cheap interpretable paralinguistics for the whole corpus.
- **End result:** openSMILE eGeMAPS functionals per call (and per speaker turn where diarization available), cached.
- **Acceptance test:** deterministic; full corpus completes on CPU (parallelized); feature distributions sanity-checked against published eGeMAPS ranges.
- **Subtasks:**
  - [ ] `egemaps.py`
  - [ ] Multiprocessing harness
  - [ ] Distribution report
- **Notes:** —

### T4.3 Neural audio representations — `[ ]`
- **Goal:** WavLM-Large + emotion2vec+ embeddings, chunked for consumer VRAM.
- **End result:** pooled per-call (and per-turn) embeddings, cached; pyannote diarization behind a config flag.
- **Acceptance test:** **ETA measured on a 50-call sample before full run; full-corpus plan (local vs. cloud burst) recorded in DECISIONS.md**; extraction idempotent/restartable mid-corpus.
- **Subtasks:**
  - [ ] Chunking strategy (≤30 s windows, documented overlap)
  - [ ] `wavlm.py`, `emotion2vec.py`, `diarize.py`
  - [ ] Resume logic
- **Notes:** —

### T4.4 Stage-3 results + gender-confound analysis → Result Table 3 — `[ ]`
- **Goal:** audio's honest contribution, plus the DESIGN.md §3.5 analysis.
- **End result:** audio-only and audio+covariate heads (5 seeds); **Result Table 3**; gender analysis (F0-based speaker-gender proxy → feature/error correlations, per-group error rates).
- **Acceptance test:** DM tests vs. Stage 1 and Stage 2; gender analysis covers ≥90% of calls with a dominant-speaker proxy; limitations paragraph drafted.
- **Subtasks:**
  - [ ] Head configs
  - [ ] Gender-proxy construction
  - [ ] Per-group reporting in `report.py`
- **Notes:** —

---

## Phase 5 — Fusion + full ablation grid (~1 week)

### T5.1 Fusion models — `[ ]`
- **Goal:** Stage-4 multimodal heads.
- **End result:** gated fusion and cross-attention heads over frozen modality embeddings; late-fusion stacking with Stage-1 GBDT; 5 seeds each.
- **Acceptance test:** fusion params <5M (small-data discipline); training fits in <2 GB VRAM; results reproducible from configs.
- **Subtasks:**
  - [ ] `models/fusion.py`
  - [ ] Stacking harness
  - [ ] Hyperparameter ranges fixed in configs (no post-hoc sweeps beyond pre-registered grid)
- **Notes:** —

### T5.2 Full ablation grid → Result Table 4 (main table) — `[ ]`
- **Goal:** the DESIGN.md §7.6 grid, populated.
- **End result:** **Result Table 4** — modality × covariates × split × target × horizon, with significance annotations; per-year breakdown appendix table.
- **Acceptance test:** every confirmatory comparison from §7.5 has a Holm-corrected p-value; `ecvol report` regenerates the whole grid from artifacts.
- **Subtasks:**
  - [ ] Grid runner (config templating)
  - [ ] Compute-budget check
  - [ ] Appendix tables
- **Notes:** —

---

## Phase 6 — LLM structured features (~2 weeks)

### T6.1 Feature schema design — `[ ]`
- **Goal:** an auditable semantic feature set, grounded in actual calls.
- **End result:** pydantic JSON schema (per-section): guidance direction {raise/maintain/lower/none}, hedging intensity (0–4), Q&A evasiveness (0–4), surprise mentions, analyst-tone (0–4), plus free-text evidence spans for auditability. Designed from manual reading of 20 calls.
- **Acceptance test:** two human passes over 10 calls agree on the schema's applicability; every field has a written rubric.
- **Subtasks:**
  - [ ] Manual reading notes (20 calls)
  - [ ] Schema + rubric doc
  - [ ] Prompt drafts (`features/llm/prompts.py`)
- **Notes:** —

### T6.2 Constrained extraction + human-audit gate — `[ ]`
- **Goal:** reliable corpus-scale extraction on consumer GPU.
- **End result:** Qwen2.5-7B-Instruct (4-bit) + Outlines pipeline; vLLM if VRAM allows, llama.cpp fallback; extracted features for the full corpus, cached with prompt+model version keys.
- **Acceptance test:** 100% schema-valid outputs (constrained decoding guarantees shape; the gate is on content): **human audit on 50 calls, κ > 0.6 on categorical fields vs. rubric labels — scaling to corpus is blocked until passed**; throughput ETA recorded.
- **Subtasks:**
  - [ ] `extract.py` (Outlines, batched, resumable)
  - [ ] Audit tooling (`audit.py`) + labeling sheet
  - [ ] Batch runner with resume
- **Notes:** —

### T6.3 Stage-5 results + masking ablation → Result Table 5 — `[ ]`
- **Goal:** RQ3 answered; lookahead leakage estimated.
- **End result:** LLM features → Stage-1 GBDT (with covariates), 5 seeds; masked-prompt (names/tickers/dates removed) variant; **Result Table 5**.
- **Acceptance test:** DM tests vs. Stage 2 and Stage 4; masked-vs-unmasked gap reported with CI.
- **Subtasks:**
  - [ ] Feature join
  - [ ] Masking transform
  - [ ] Result configs
- **Notes:** —

---

## Phase 7 — Post-cutoff data + lookahead study (~2 weeks, calendar-dependent)

### T7.1 Fresh acquisition pipeline (scripts-not-data) — `[ ]`
- **Goal:** ≥200 calls from 2025-Q4 / 2026-Q1 with audio + transcript + price joins.
- **End result:** acquisition scripts (EarningsCall/EarningsCast API primary; company-IR-page fetcher fallback) + terms-of-use review note; local-only data with manifests.
- **Acceptance test:** ≥200 calls pass the same ingestion gates as T1.4 (≥95% price join); ToS review written **before** any bulk pull; zero raw data committed.
- **Subtasks:**
  - [ ] API client + key handling (.env)
  - [ ] IR-page fallback fetcher
  - [ ] Ingestion onto the common schema
  - [ ] Universe selection rule (e.g., S&P 500 members, pre-registered)
- **Notes:** —

### T7.2 Frozen-pipeline post-cutoff evaluation — `[ ]`
- **Goal:** the lookahead-bias experiment (DESIGN.md §7.4).
- **End result:** all stages (0–5) evaluated on the post-cutoff set **with zero retraining or threshold changes after first look** (rule pre-registered in DESIGN.md); lookahead table comparing in-cutoff vs. post-cutoff degradation per stage.
- **Acceptance test:** evaluation run from frozen artifacts only (CI-verifiable: no training code touched); table regenerates.
- **Subtasks:**
  - [ ] Frozen-eval runner
  - [ ] Degradation metrics with bootstrap CIs
  - [ ] Write-up
- **Notes:** —

---

## Phase 8 — Paper + reproducibility package (~2–3 weeks)

### T8.1 Reproducibility package — `[ ]`
- **Goal:** anyone can rebuild every table.
- **End result:** `REPRODUCE.md` (one command per table), released derived-feature archives (FinCall-Surprise: Apache-2.0; MAEC features: CC-BY-SA-4.0), license audit note, environment lockfile verification on a clean machine.
- **Acceptance test:** clean-machine dry run reproduces Result Table 1 exactly and one GPU table within seed-variance bounds.
- **Subtasks:**
  - [ ] `REPRODUCE.md`
  - [ ] Feature-archive packaging + upload
  - [ ] License audit
  - [ ] Clean-machine verification
- **Notes:** —

### T8.2 Paper — `[ ]`
- **Goal:** the manuscript, framing per the DESIGN.md §4 gate decision.
- **End result:** draft with all tables generated by `ecvol report`; figures from `notebooks/`; venue selected; arXiv preprint.
- **Acceptance test:** every number in the paper traces to a run artifact; co-author/advisor review pass.
- **Subtasks:**
  - [ ] Draft (framing per gate decision)
  - [ ] Figures notebooks
  - [ ] Venue selection memo
  - [ ] arXiv submission
- **Notes:** —

### T8.3 (Conditional) Stage-6 cloud experiments — `[ ]`
- **Goal:** QLoRA fine-tuning / audio-LLM experiments, only if the DESIGN.md §6 Stage-6 gate passed.
- **End result:** either the experiments + results table, or an explicit skip recorded as future work.
- **Acceptance test:** a DECISIONS.md entry exists with budget and hypothesis before any cloud spend; or the skip is documented.
- **Subtasks:**
  - [ ] Gate evaluation + DECISIONS.md entry
  - [ ] (If go) cloud setup, QLoRA runs, audio-LLM runs
- **Notes:** —
