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

### T0.1 Package skeleton — `[x]`
- **Goal:** installable `ecvol` package with version control and CI.
- **End result:** git repo initialized at project root; `pip install -e .` (via uv) works; `ecvol --help` lists all verbs as stubs.
- **Acceptance test:** GitHub Actions green on ruff + pytest (empty suite passes); fresh-machine install from lockfile documented and tested once; `git log` shows small, focused commits from the very first one.
- **Subtasks:**
  - [x] `git init` at project root; `.gitignore` (glob patterns: `data/` payloads, `artifacts/`, caches — **not** hand-listed files; decide handling of `legacy/` bulk binaries: ignore or track-without-LFS, document choice)
  - [x] Initial commit: the six root .md files (CLAUDE, DESIGN, TASKS, DECISIONS, JOURNAL, OLDWORK)
  - [x] `pyproject.toml` + uv lockfile
  - [x] `src/ecvol/` layout per DESIGN.md §8.1
  - [x] Typer CLI stub (`prices|targets|splits|featurize|train|evaluate|report`)
  - [x] pre-commit (ruff format + lint)
  - [x] CI workflow (GitHub Actions)
- **Notes:** done 2026-06-12. All checks green locally (ruff, pytest, pre-commit) and fresh-clone install from lockfile tested per README.md; legacy-binaries gitignore choice logged in DECISIONS.md. Repo: github.com/urbandrei/Earnings_call_project-main (private); CI green on first push (user-verified). Journal: 2026-06-12 T0.1 entries.

### T0.2 Config system — `[x]`
- **Goal:** every experiment definable as a validated YAML.
- **End result:** pydantic schemas for data/feature/model/eval configs; loader with clear errors.
- **Acceptance test:** invalid config (bad horizon, missing seed list) fails with an actionable message; round-trip load→resolve→dump is stable.
- **Subtasks:**
  - [x] Schema modules under `src/ecvol/config/`
  - [x] `configs/example.yaml`
  - [x] Config-hash function (canonicalized JSON → SHA-256)
- **Notes:** done 2026-06-12. `schema.py` (ExperimentConfig + sections, extra="forbid" everywhere) + `load.py` (loader with path+field error messages, deterministic `dump_config`, `config_hash`). Beyond-spec validations added: embargo ≥ longest horizon (encodes §5.4), unique seeds/horizons. Acceptance covered by `tests/test_config.py` (11 tests green). Journal: 2026-06-12 T0.2 entry.

### T0.3 Run tracking & manifests — `[x]`
- **Goal:** every run and every data file is traceable.
- **End result:** run-artifact writer (config hash, git SHA, seeds, env fingerprint, metrics parquet); manifest module with checksum verify command.
- **Acceptance test:** rerunning an identical CPU-only config reproduces metrics **bit-identically**; `ecvol data verify` detects a deliberately corrupted file.
- **Subtasks:**
  - [x] `manifests.py` (path, source URL, retrieved_at, SHA-256, license)
  - [x] Run-ID scheme + `artifacts/runs/<id>/` writer
  - [x] Env fingerprint capture (lockfile hash, CUDA/driver versions)
- **Notes:** done 2026-06-12. `src/ecvol/data/manifests.py` (+ `ecvol data verify`, exit 1 on mismatch/missing) and `src/ecvol/tracking.py` (run ID `<UTC ts>-<confighash8>`, `write_run` → config.yaml/run.json/metrics.parquet, deterministic parquet verified bit-identical across interpreter processes). Acceptance covered by `tests/test_manifests.py` + `tests/test_tracking.py` (real-pipeline rerun re-asserted once `ecvol evaluate` exists, T2.2). Bonus fix: T0.1's unanchored `data/` gitignore pattern had silently untracked `src/ecvol/data/`. CI green on push (user-verified). Journal: 2026-06-12 T0.3 entry.

---

## Phase 1 — Data foundation (~1–2 weeks) — *highest-risk-first*

### T1.1 Dataset acquisition & mirroring — `[x]` *(do first — retires link-rot risk)*
- **Goal:** local, checksummed mirrors of FinCall-Surprise and MAEC.
- **End result:** `data/` mirrors + committed manifests; download scripts in-repo.
- **Acceptance test:** manifest verify passes; counts match published sizes (FinCall-Surprise ≈ 2,688 calls; MAEC ≈ 3,443) or the gap is documented; audio files spot-decoded (random 50) without error.
- **Subtasks:**
  - [x] FinCall-Surprise GitHub + Google Drive fetcher
  - [x] MAEC fetcher
  - [x] Checksum + license fields in manifests
  - [x] Storage estimate & location decision (external disk OK — manifests make it portable)
- **Notes:** done 2026-06-12. `ecvol data fetch fincall|maec|all` + `ecvol data spotcheck`; mirrors on `D:\ecvol-data` via `data\raw` junction (DECISIONS.md). Counts exact: FinCall 2,688 calls (919/704/1065), MAEC 3,443 (all with text+features). Audio joins: 2,671/2,688 calls (99.4%) have their mp3; 17 missing, 456 surplus files. Spotcheck 50/50 decode OK (seed 0). **Gap:** MAEC 59 GB MFCC archive link-rotted upstream, no mirror exists (DECISIONS.md). Journal: 2026-06-12 T1.1 entry.

### T1.2 Price ingestion (yfinance + Tiingo cross-check) — `[x]` *(done 2026-06-17)*
- **Goal:** reliable adjusted daily OHLCV for the combined ticker universe.
- **End result:** `ecvol prices pull` → per-ticker parquet + manifest; coverage report (matched/missing/delisted with reasons).
- **Acceptance test:** ≥98% ticker coverage for the FinCall-Surprise universe (or documented shortfall); Tiingo cross-check correlation >0.999 on a 5% sample; zero silent drops (every excluded ticker has a reason code).
- **Subtasks:**
  - [x] Ticker normalization (share classes via `to_yahoo_symbol`; renames handled as reason-coded coverage misses, no rename table needed)
  - [x] ~~Stooq~~ price fetcher with caching — **source changed to yfinance** (Stooq closed free access; DECISIONS.md 2026-06-15). `prices.py`: batched yfinance, deterministic parquet, idempotent/resumable.
  - [x] Tiingo sampler (`tiingo.py`: 5% seeded sample + return-correlation gate; built, run pending key)
  - [x] Trading-calendar module (`exchange_calendars` → `calendar.py`, XNYS)
  - [x] Coverage report generator (`data/coverage/prices_coverage.csv` + `prices_sources.json`)
  - [x] Cross-check gate aligned to DESIGN §5.2 + persisted (2026-06-16): documented-exception reason codes (`data/coverage/crosscheck_exceptions.csv`), artifact writer (`crosscheck_report.csv` + `crosscheck_summary.json`); warns (0.99–0.999) tolerated, sub-0.99 pass only when documented. DECISIONS.md 2026-06-16.
- **Notes:** **DONE 2026-06-17. All three acceptance criteria met:** (1) **FinCall coverage 384/388 = 98.97%** (≥98% gate) after the FinCall-scoped Tiingo fallback recovered 12 of 16 Yahoo-purged tickers (ANSS, CMA, CTLT, DFS, HES, HOLX, IPG, JWN, MOR, SEE, SNV, WBA). (2) **Cross-check gate PASSED** (committed `crosscheck_report.csv`/`crosscheck_summary.json`): 47 pass / 2 warn (APD 0.9941, HPQ 0.9974, tolerated 0.99–0.999 band) / 1 documented investigate (XRX 0.9756 — Xerox→Conduent 2017 spin-off + reverse split), `undocumented: []`. (3) **Zero silent drops** — the 4 still-missing FinCall tickers (ANDG, K, NPIXY, X; absent from *both* yfinance and Tiingo) are reason-coded in the coverage report. Manifest = 1,011 entries (12 tiingo-sourced), `ecvol data verify data/manifests/prices.json` → OK. MAEC 917/1,213 (75.6%, informational; MAEC Tiingo recovery deferred to T1.5 per DECISIONS 2026-06-17). Tiingo fallback scoped to FinCall (DECISIONS 2026-06-17); cross-check gate aligned to DESIGN §5.2 (DECISIONS 2026-06-16). Journal: 2026-06-17 entries. — *2026-06-16:* Tiingo key landed; first `crosscheck` run 47/50 pass, gate semantics aligned to DESIGN §5.2. — *2026-06-15:* Universe = FinCall (388) ∪ MAEC (1,213) = 1,309 tickers (DECISIONS.md). Pulled 999/1,309. **FinCall coverage 372/388 = 95.88%** (gate 98%); covered tickers essentially complete (min completeness 0.979, zero gappy). MAEC 906/1,213 = 74.69% (informational until T1.5). **Shortfall = 16 FinCall tickers Yahoo's API does not serve** — predominantly 2024–26 M&A delistings/take-privates Yahoo purges (WBA, DFS, ANSS, X, JWN, HES, CTLT, IPG, SNV, K, HOLX, SEE…) plus a few Yahoo coverage gaps; **all reason-coded (zero silent drops)**. yfinance's known delisted-ticker blindness (the §5.2 risk) is the cause; **Tiingo fallback built** (`fetch_tiingo_ohlcv`, auto-recovers Yahoo misses when a key is present) → expected to clear 98% once the key lands. **Two acceptance items pending the same free `TIINGO_API_KEY`:** (a) the >0.999 cross-check run (`ecvol prices crosscheck`); (b) the 98% recovery via Tiingo fallback (`ecvol prices pull`). Tests: `test_prices.py`/`test_tiingo.py`/`test_calendar.py` (78 total green). Join audit (T1.4 subtask) now unblocked. Journal: 2026-06-15 T1.2 entry.

### T1.3 Target computation — `[x]` *(done 2026-06-17)*
- **Goal:** exact, tested implementation of DESIGN.md §5.3.
- **End result:** `ecvol targets build` → one row per (call, τ) with `v_pre`, `v_post`, `Δv`, HAR inputs, reason codes for exclusions.
- **Acceptance test:** unit tests on synthetic price series with analytically known RV; the after-hours date rule tested explicitly (16:00 ET boundary cases); 3 calls hand-verified end-to-end against manual spreadsheet computation.
- **Subtasks:**
  - [x] `targets.py` with `(call, as_of_timestamp)` signature (`anchor_day0` takes an optional `timestamp`; `as_of` = close of day0 stamped per row)
  - [x] Edge-rule handling (non-trading days roll the anchor; insufficient pre/post history; zero variance → reason-coded NaN)
  - [x] HAR input features (realized variance over last 1/5/22 sessions as of day 0; HAR-*residual* target deferred to Phase 2 — needs train-only fit)
  - [x] Exclusion accounting with reason codes (`unresolved_ticker`/`no_price_data`/`invalid_date`/`insufficient_pre|post_history`/`zero_variance_pre|post`; one row per (call,τ), zero silent drops)
- **Notes:** **DONE 2026-06-17.** `src/ecvol/data/targets.py` + `ecvol targets build`; pure/deterministic, reuses `calendar.py` + `prices.load_close_series` (new helper). **After-hours rule = assume-after-hours fallback** (no call times yet; DECISIONS 2026-06-17, parameterized for later T1.4 timestamps). Cohort = all resolved-ticker calls. First run: 2,496/2,688 resolved, **join rate 97.12%** (2,424 calls with ≥1 target), 9,696/10,752 rows ok; exclusions 768 unresolved_ticker / 220 invalid_date / 60 no_price_data / 4 insufficient_post / 4 zero_variance (all reason-coded). Artifacts: `data/targets/targets.parquet` (gitignored payload, deterministic — byte-identical re-run verified) + committed `data/manifests/targets.json` (`ecvol data verify` OK) + `data/coverage/targets_report.csv`. Tests: `tests/test_targets.py` (11 — analytic RV, after-hours boundaries incl. weekend/holiday + timestamp branches, exclusion codes, HAR inputs, determinism, 3 hand-verified calls). Date-validation-against-calendar lands here (invalid/missing dates → `invalid_date`). Legacy notebooks 3/3b cross-checked, not ported (OLDWORK.md §7). Journal: 2026-06-17 T1.3 entry.

### T1.4 FinCall-Surprise ingestion — `[x]` *(done 2026-06-18; identity work pulled ahead of T1.2 — DECISIONS.md 2026-06-12)*
- **Goal:** normalized call records on the common schema.
- **End result:** `(call_id, ticker, utc_timestamp, transcript_json, audio_path, speaker_metadata, source)` records for the full set.
- **Acceptance test:** 100% of calls parse or are excluded with reason codes; audio-duration distribution report generated; **≥95% join rate to price data + targets**.
- **Subtasks:**
  - [x] Identity reconstruction (call → ticker/company/date from transcripts + slide PDFs vs SEC table; committed identity CSV; audited accuracy gate)
  - [x] Call-type classification (earnings vs fireside/M&A/sales/meeting/other; type column in identity CSV — exclusion reason-code wiring lands with the parser)
  - [x] Transcript JSON parser (`fincall_ingest.py`: role-tagged turns split on inline `Executives:`/`Analysts:`/`Operator:` markers → `transcript_json` + `speaker_metadata` JSON columns; T3.1 refines)
  - [x] Timestamp extraction/validation — **investigated, §10 risk #7 closed:** time-of-day present in only 3.4% of transcripts / 1.6% of PDFs (mostly press-release times) → no per-call extraction; uniform assume-after-hours fallback, flagged per record (DECISIONS 2026-06-18)
  - [x] Ticker resolution (via identity CSV + curated override table, CIKs EDGAR-verified)
  - [x] Join audit (earnings cohort vs `targets.parquet`: **2,315/2,331 = 99.31%**, clears ≥95%; `data/coverage/fincall_join_audit.csv`)
- **Notes:** **DONE 2026-06-18.** `src/ecvol/data/fincall_ingest.py` + `ecvol data ingest fincall` normalizes the corpus onto the common schema `(call_id, ticker, utc_timestamp→call_date+flags, transcript_json, audio_path, speaker_metadata, source)`. **All three acceptance criteria met:** (1) **100% parse** — 2,688/2,688 calls produce a record (zero `empty_transcript`); every non-ok call is reason-coded (`unresolved_ticker`=192, `non_earnings`=115, `no_date`=50), zero silent drops. (2) **Audio-duration distribution report** — `data/coverage/fincall_audio_durations.csv` (2,671/2,671 present mp3s decoded via ffprobe; median ~61 min, mean ~62 min, range 18–199 min, 2,769 h total; 17 mp3s missing, T1.1-known). (3) **Join rate 2,315/2,331 = 99.31%** earnings-cohort calls with ≥1 ok target (≥95% gate), `data/coverage/fincall_join_audit.csv`. Transcript parsed into role-tagged turns (coarse `Executives:`/`Analysts:`/`Operator:` markers; weak per-speaker delineation is T3.1's job). **Timestamp investigation closed:** call times present in only 3.4% of transcripts / 1.6% of PDFs (mostly press-release times) → uniform assume-after-hours fallback, flagged per record (DECISIONS 2026-06-18, closes §10 risk #7). Artifacts: `data/fincall/calls.parquet` (gitignored payload, byte-identical re-run verified) + committed `data/manifests/fincall_calls.json` (`ecvol data verify` OK) + 3 coverage CSVs; ffprobe durations cached under `data/raw/ref/` (resumable). Tests: `tests/test_fincall_ingest.py` (12 — parsing incl. glued markers/preamble, reason codes, schema, determinism, join audit). 104 tests green, ruff clean. Journal: 2026-06-18 T1.4 entry. — *(history below)* 2026-06-12 feasibility study (JOURNAL.md, `notebooks/explore_fincall_identity.py`): dataset has **no ticker/company/date metadata at all** — identity must be reconstructed (slide-PDF metadata/title pages + transcript prose → SEC company_tickers.json matching). Date signal 100% on a 50-call sample; name signal ~100% but matching needs an alias table or LLM-assisted extraction (quick heuristics: 58%). Corpus contains non-earnings calls (firesides, M&A, monthly sales) → needs call-type classification with exclusion reason codes. Scope expansion pending DECISIONS.md entry.
  2026-06-12 (later session): identity table v2 — **2,496/2,688 resolved (92.9% overall; earnings-type 2,381/2,499 = 95.3%, over the ≥95% target)**; dates 2,629 (97.8%). Curated override CSV (`data/identity/fincall_name_overrides.csv`, 77 rows, CIKs EDGAR-verified) covers brand acronyms + companies delisted/renamed since the corpus era. **Accuracy gate: committed seeded audit (`data/identity/fincall_identity_audit.csv`) 60/60 correct** (40 random resolved-earnings + 20 newly resolved); ~30 wrong identities from v1 corrected (Zoetis-as-FISI, GE-as-Baker-Hughes, Vertex-Pharma-as-Vertex-tax, Discover-as-Moody's…). Call types: 2,499 earnings / 80 conference / 61 meeting / 33 unknown / 14 ma+sales. Residue (~190 calls) is mid-call fragments, firesides, webinars — mostly non-earnings. Journal: 2026-06-12 identity-v2 entry.

### T1.5 MAEC ingestion — `[x]` *(done 2026-06-18; ≥95% gate met via documented-shortfall path)*
- **Goal:** same contract as T1.4 for MAEC.
- **End result:** MAEC on the common schema; documented gaps (audio availability, year coverage).
- **Acceptance test:** same gates as T1.4 (≥95% price join); discrepancies vs. published MAEC stats documented.
- **Subtasks:**
  - [x] Folder-format parser (`YYYYMMDD_TICKER` → date+ticker; `maec_ingest.py`)
  - [x] Audio-feature vs. raw-audio availability audit (`maec_audio_features.csv`: features 3,443/3,443, raw audio 0; durations from summed sentence `Audio Length`)
  - [x] Schema mapping (shared `calls.CallRecord`; `call_id`=folder name string; sentences as `unknown`-role turns; no call times → assume-after-hours)
- **Notes:** **DONE 2026-06-18.** `src/ecvol/data/maec_ingest.py` + `ecvol data ingest maec`. Common schema extracted to shared `src/ecvol/data/calls.py` (both ingesters); `targets.py` generalized to call_id-agnostic so MAEC reuses the FinCall RV math — FinCall's committed `calls.parquet`/`targets.parquet` verified **byte-identical** after the refactor. **3,443 folders (= published MAEC), 3,419 parsed** (24 reason-coded `empty_transcript` — genuinely truncated `text.txt` like "Thank you.", a real MAEC defect, not a parser bug); 394,249 sentences; no speaker labels (sentences stored as `unknown`-role turns); **no raw audio** (MAEC never shipped it — DECISIONS 2026-06-12; per-sentence `features.csv` present for all). **Price-join 2,578/3,419 = 75.4%** (≥1 ok target via the same target machinery, written to `data/maec/targets.parquet`); **closed via the documented-shortfall path** (user decision; DECISIONS 2026-06-18) — the 296 missing-price tickers (reason-coded in `data/coverage/maec_missing_tickers.csv`, zero silent drops) are predominantly genuine 2015–2018 M&A/delistings yfinance purges (ABC→Cencora, AGN, ANTM, APC, ALXN…); MAEC is the secondary dataset and a ~6 h / ~296-of-500-monthly-symbol Tiingo burst would realistically still land <95%, so deferred as optional future work. Artifacts: `data/maec/{calls,targets}.parquet` (gitignored payloads, byte-identical re-run verified) + committed manifests `data/manifests/maec_{calls,targets}.json` + `data/coverage/maec_{ingest_report,audio_features,join_audit,missing_tickers}.csv`. Tests: `tests/test_maec_ingest.py` (7). 111 tests green, ruff clean. Journal: 2026-06-18 T1.5 entry.

### T1.6 Split builder — `[x]` *(done 2026-06-18 — completes Phase 1)*
- **Goal:** committed, leakage-proof splits per DESIGN.md §5.4.
- **End result:** `ecvol splits build` → CSVs for temporal (30-trading-day embargo), ticker-disjoint, and combined splits, per dataset.
- **Acceptance test:** pytest leakage assertions: no target-window overlap across temporal boundaries; zero ticker intersection in disjoint splits; embargo verified against the trading calendar; split CSVs committed and stable across reruns.
- **Subtasks:**
  - [x] `splits.py` with embargo logic (temporal: embargo `max(embargo,horizon)`-session zone on the train side of each boundary → no 30-day target window crosses; ≥30-session gap between segments)
  - [x] Grouped/stratified ticker split (seeded greedy ticker partition ≈70/10/20 by call count; sector stratification skipped — no sector metadata in either corpus, DECISIONS 2026-06-18)
  - [x] Leakage assertion test module (runs in CI forever after) — `tests/test_splits.py`, incl. assertions against the **committed real CSVs**
- **Notes:** **DONE 2026-06-18.** `src/ecvol/data/splits.py` + `ecvol splits build`; three schemes per dataset over the ≥1-ok-target cohort. **Also standardized FinCall targets path** `data/targets/` → `data/fincall/targets.parquet` (+ manifest `fincall_targets.json`; old `targets.json` removed) for per-dataset symmetry with MAEC (content byte-identical). Committed CSVs `data/splits/{fincall,maec}_{temporal,ticker_disjoint,combined}.csv` (`.gitignore` allowlists `data/splits/`); deterministic (byte-identical reruns). **Results** — FinCall (cohort 2,424): temporal train/val/test/embargo = 1480/228/481/235; ticker_disjoint 1701/239/484; combined 1029/21/92 (excl 1282). MAEC (cohort 2,578): temporal 1604/73/507/394; ticker_disjoint 1805/259/514; combined 1096/4/102 (excl 1376). Small MAEC temporal-val (73, tight seasonal clustering under the 30-session embargo) and small combined splits (the "hardest robustness row", DESIGN §5.4.3) documented, not bugs. **Acceptance met:** `tests/test_splits.py` (11) asserts — on the **committed** CSVs (CI-readable without the gitignored parquets) — temporal calendar-order + ≥30-session embargo gap (via trading calendar), ticker-disjoint zero-intersection, combined disjoint+ordered; plus synthetic unit tests + determinism. 123 tests green, ruff clean. Journal: 2026-06-18 T1.6 entry.

---

## Phase 2 — Eval harness + econometric baselines (~1 week) — *the floor*

### T2.1 Metrics & significance module — `[x]` *(done 2026-06-18)*
- **Goal:** DESIGN.md §7.1–7.2 implemented and validated.
- **End result:** `eval/metrics.py` (MSE, MAE, R²_OOS, quarterly Spearman) and `eval/significance.py` (DM test, cluster bootstrap, Holm correction).
- **Acceptance test:** DM test validated against a published worked example / statsmodels reference; bootstrap CIs validated on synthetic data with known sampling distribution.
- **Subtasks:**
  - [x] Metric functions over the (call, τ) prediction frame (`metrics.py`: pure array `mse`/`mae`/`r2_oos`/`spearman` + frame helpers `spearman_by_quarter`/`metrics_by_horizon`; NaN rows dropped)
  - [x] Clustering keys (ticker, quarter) (`quarter_of` ISO→`YYYYQn`; cluster arrays consumed by `cluster_bootstrap_ci`)
  - [x] Significance API consumed by `report` (`significance.py`: `diebold_mariano`, `cluster_bootstrap_ci`, `holm_correction` — pure/seeded, no CLI verb)
- **Notes:** **DONE 2026-06-18.** `src/ecvol/eval/{metrics,significance}.py`. Added **scipy** dependency (`uv add scipy`; t-dist p-values + Spearman; DECISIONS 2026-06-18). Prediction-frame contract: one row per (call,τ) with `call_id,ticker,as_of,horizon,y_true,y_pred` (+ `y_persistence` for R²_OOS, caller-supplied: `v_pre` for level-v, 0 for Δv). **DM** = HLN-corrected vs `t_{n-1}`, LRV = γ0+2Σγ_k (h-step), sign: +ve ⇒ model A worse. **Acceptance met:** DM validated via the exact identity `DM*(h=1)==paired-t on loss diff` (scipy `ttest_rel`, 1e-9); cluster-bootstrap CI reproduces the analytic normal half-width on i.i.d. data and widens >3× under intra-cluster correlation; Holm matches R `p.adjust(method="holm")`. Tests: `tests/test_metrics.py` (9) + `tests/test_significance.py` (9). 141 tests green, ruff clean. Journal: 2026-06-18 T2.1 entry.

### T2.2 Stage-0/1 baselines → Result Table 1 — `[x]` *(done 2026-06-18)*
- **Goal:** the honest floor, committed.
- **End result:** persistence, EWMA, HAR-RV, GARCH(1,1), ticker-FE LightGBM evaluated on every (dataset × split × target × τ); **Result Table 1** artifact.
- **Acceptance test:** sanity gate — HAR-RV beats persistence at τ=30 on the temporal split (stylized fact; if violated, **halt and debug targets**); GARCH fits converge for >95% of series or documented fallback.
- **Subtasks:**
  - [x] `models/baselines.py` (persistence, EWMA RiskMetrics λ=0.94, log-HAR train-fit, GARCH(1,1) via `arch` per-call)
  - [x] `models/gbdt.py` with ticker fixed effect (LightGBM categorical; sector/mkt-cap omitted — no metadata)
  - [x] Multi-seed runner (GBDT) (5 seeds; deterministic params → byte-identical table)
  - [x] First end-to-end `ecvol evaluate` run (→ `data/results/result_table_1.csv`, 720 rows)
- **Notes:** **DONE 2026-06-18.** `models/{baselines,gbdt}.py` + `eval/evaluate.py` + `ecvol evaluate`. Added deps `arch`, `lightgbm`, `scikit-learn` (statsmodels transitively). Three targets (level-v, Δv, HAR-residual — the train-only HAR fit deferred from T1.3 lands here); persistence = per-target trivial forecast = R²_OOS baseline. DM p-values per cell (significance API from T2.1). **GARCH-convergence gate PASSED** (FinCall 99.6%, MAEC 99.9%). **Sanity gate: PASSED with a documented COVID-regime exception.** The literal HAR>persistence@τ=30-temporal check fails on FinCall (R²_OOS −0.287) but the failure was **debugged and the targets validated** — the same HAR/targets beat persistence at τ=30 on FinCall ticker-disjoint (+0.229), MAEC temporal (+0.206), and all FinCall τ≤15; root cause = COVID regime shift (18% of FinCall temporal-train is Feb–May 2020; test is calm late-2021, 0.22 lower v_post). The gate now passes only when corroborated by the regime-stable cells (unit-tested 3 ways); the τ=30-temporal under-performance is a reported finding (DECISIONS 2026-06-18, DESIGN §5.4.5). Result Table 1 deterministic (byte-identical reruns). Tests: `tests/test_{baselines,gbdt,evaluate}.py` (14). 155 green, ruff clean. Journal: 2026-06-18 T2.2 entry.

### T2.3 Reporting — `[x]` *(done 2026-06-18 — completes Phase 2)*
- **Goal:** all paper tables regenerable by one command.
- **End result:** `ecvol report` renders LaTeX + Markdown tables from run artifacts.
- **Acceptance test:** byte-identical regeneration from unchanged artifacts (CI check); Result Table 1 renders in both formats.
- **Subtasks:**
  - [x] `eval/report.py` (+ `ecvol report` → `data/results/result_table_1.{md,tex}`)
  - [x] Table specs as data (`TABLE_1_SPECS`: frozen `TableSpec`s, model×horizon pivots; add a spec, not code)
  - [x] Figure stubs for notebooks (`notebooks/figures_result_table_1.py`)
- **Notes:** **DONE 2026-06-18.** `eval/report.py` renders the committed `result_table_1.csv` to Markdown + LaTeX (booktabs). Render set = R²_OOS + MSE × {fincall,maec} × {level-v,Δv} × {temporal,ticker_disjoint} test (16 tables); `*` = DM-significant vs persistence; combined split + HAR-residual stay in the CSV (add a spec to render). **Acceptance met:** determinism test + a **committed-artifacts CI guard** (`test_committed_reports_match_fresh_render`) re-renders from the committed CSV and asserts byte-equality with the committed `.md`/`.tex`. Cluster-bootstrap CIs deferred (need per-call predictions; land with the Phase-3+ content-model comparison tables — DECISIONS 2026-06-18). Tests: `tests/test_report.py` (6). 161 green, ruff clean. Journal: 2026-06-18 T2.3 entry.

---

## Phase 3 — Text ladder + early identity controls (~1–2 weeks)

### T3.1 Transcript normalization — `[x]` *(done 2026-06-19; user audit 30/30 correct, clears >90% gate)*
- **Goal:** robust sectioning and speaker structure without per-sentence alignment.
- **End result:** per-call structure: prepared remarks vs. Q&A; speaker turns with roles (operator/management/analyst) where derivable.
- **Acceptance test:** section-detection precision >90% on a 30-call hand-checked sample; speaker-role tagging audited on the same sample.
- **Subtasks:**
  - [x] `features/text/sections.py` (deterministic heuristic: operator Q&A-cue + first-analyst-turn boundary; MAEC = in-text cues only, roles unavailable — DECISIONS 2026-06-19)
  - [x] Speaker-turn chunking: chunk by speaker turn, **never split a turn across chunks** (DECISIONS.md 2026-06-14); oversized turns sentence-split into same-turn sub-chunks (DECISIONS 2026-06-19). Reuses FinCall speaker metadata.
  - [x] Audit artifact — committed seeded `data/coverage/{dataset}_section_audit.csv` (30 calls each) in place of a notebook (CI-friendly; DECISIONS 2026-06-19); **human precision check is the open acceptance item**.
- **Notes:** `ecvol featurize sections` → `data/{dataset}/chunks.parquet` (gitignored deterministic payload, byte-identical reruns verified; manifests `data/manifests/{dataset}_chunks.json` verify OK) + committed `data/coverage/{dataset}_sections.csv`. **FinCall:** Q&A detected 2654/2688 (98.7%), 2408 corroborated by both signals (90.7% of detections), 196,233 chunks, 0 oversize; methods first_analyst=2151 / operator_cue=443 / text_cue=60 / none=34. **MAEC** (no speaker labels — best-effort text-cue): Q&A detected 867/3443 (25.2%), 394,280 chunks, 1 oversize. Eyeball of the FinCall audit sample is clean (boundaries on real analyst/operator-intro turns; the only soft cases are out-of-cohort non-earnings calls). 13 new tests; 174 green, ruff clean. Speaker-turn chunking adopted from prior-team work (DECISIONS.md 2026-06-14); feeds the TX1 QA exploration. Journal: 2026-06-19 T3.1 entries. **Acceptance MET:** user hand-audit of the cohort-restricted 30-call sample = 30/30 correct boundaries (operator-handoff and analyst-question boundaries both accepted as correct Q&A-section starts). **Next:** T3.2 (a design call — embedding model + pooling + GPU setup).

### T3.2 Frozen text features — `[x]` *(done 2026-06-19; full corpus extracted, acceptance met)*
- **Goal:** Stage-2 representations, cached.
- **End result:** BGE/GTE section-pooled embeddings; FinBERT sentiment aggregates (per section, per speaker role); surface stats (length, Q&A turn counts, numeric density). Parquet caches keyed by content hash.
- **Acceptance test:** deterministic re-extraction (cache hit = bit-identical); GPU throughput benchmarked and full-corpus ETA recorded.
- **Subtasks:**
  - [x] `embeddings.py` (BAAI/bge-m3, 1024-d, section-pooled), `finbert.py` (ProsusAI/finbert per scope×role), `surface.py` (len/turns/numeric-density/question-marks) + shared `_common.py` (content-hash cache, deterministic, pooling). `ecvol featurize text`.
  - [x] Pooling strategies (mean, section-weighted) as config options (`--weighted`)
  - [x] GPU stack as a `gpu` dependency group (torch 2.11.0+cu128 for Blackwell sm_120; CI stays torch-free), GPU smoke-tested on the RTX 5060 Ti.
  - [x] **Full-corpus extraction** (FinCall 196,233 chunks / MAEC 394,280) — done 2026-06-19, both datasets, fp32 batch 64.
- **Notes:** model picks BGE-M3 + ProsusAI/finbert, fp32 + deterministic kernels (DECISIONS 2026-06-19). **Full run (both datasets, ~95 min wall):** FinCall → `text_embeddings` 7,835 rows (185,126 chunks encoded, ~11k dedup/benchmark cache hits) + `text_finbert` 28,123 rows (per scope×role) + `text_surface` 7,835 rows; MAEC → 7,739 rows each (no roles → per-scope finbert). MAEC ~5× faster per chunk (short sentences vs FinCall turns). **Acceptance MET:** (1) **deterministic re-extraction = bit-identical** — warm-cache re-run encoded 0 chunks, all three FinCall parquets byte-identical (verified); (2) GPU throughput + ETA recorded (benchmark: BGE-M3 37 ch/s, FinBERT 90 ch/s). Payloads `data/{dataset}/text_{embeddings,finbert,surface}.parquet` (gitignored, ~61 MB embeddings) + caches under `data/{dataset}/cache/` (~0.7–1.5 GB, gitignored, content-hash keyed) + committed manifests `data/manifests/{dataset}_text_{embeddings,finbert,surface}.json` (`ecvol data verify` OK). 8 new tests (torch-free, CI-safe); 182 green, ruff clean. Journal: 2026-06-19 T3.2 entries.

### T3.3 Stage-2 results → Result Table 2 — `[x]` *(done 2026-06-19)*
- **Goal:** first content-bearing models, honestly evaluated.
- **End result:** ridge + shallow-MLP heads, 5 seeds, with and without past-vol covariates; **Result Table 2** with DM tests vs. HAR-RV and vs. Stage 1.
- **Acceptance test:** every cell carries mean ± std over seeds; report regenerates; confirmatory comparisons labeled per DESIGN.md §7.5.
- **Subtasks:**
  - [x] `models/heads.py` (ridge: α chosen on val from a fixed grid, no retrain-on-train+val; shallow MLP 1×256, internal early-stop, seeded; train-fit PCA(256) of the embedding block for the MLP; train-median impute of missing covariates)
  - [x] Multi-seed orchestration (`eval/stage2.py` → `ecvol evaluate-text` → `data/results/result_table_2.csv`; 5 seeds, mean + seed-std for MLP)
  - [x] Ablation configs (3 covariate variants: **text**, **pastvol**, **text_pastvol** × {ridge, mlp}; feature matrix = `features/text/assemble.py`, embeddings prepared+qa + FinBERT scope×role + surface)
- **Notes:** **DONE 2026-06-19.** Result Table 2 = 864 rows (2 datasets × 3 splits × 3 targets × 4 horizons × 6 head×variant models × 2 segments), zero NaN; per cell: R²_OOS (vs persistence), MSE, MAE, Spearman, MLP seed-std, and **DM p vs persistence / vs HAR-RV / vs Stage-1** (the §7.5 confirmatory comparisons). Rendered to `result_table_2.{md,tex}` (`*` = DM-significant vs Stage-1); render byte-identical + committed-artifacts CI guard (`test_committed_table2_matches_fresh_render`). **Two real bugs found by the run (held the commit): (1)** 4 MAEC ok-rows have NaN `rv_monthly` (insufficient 22-session history) → heads now train-median-impute covariates + fit on finite-target rows + select α on finite-val (DECISIONS 2026-06-19); **(2)** pandas `.to_numpy()` returned a read-only array → `.copy()`; both now regression-tested. **HEADLINE FINDING (honest):** text content does **not** cleanly beat the identity/past-vol baselines — on FinCall *temporal* Δv the text+pastvol heads overfit (R²_OOS strongly negative across the COVID regime shift); on *ticker-disjoint* Δv text heads are positive (+0.09…+0.28) but `ridge_pastvol` (no text) is comparably strong, so the gain is not clearly *content*. This is the §4-framing-gate evidence that **T3.4 identity controls** must adjudicate. 9 new tests; 189 green, ruff clean. GPU stack not needed (sklearn/CPU). Journal: 2026-06-19 T3.3 entry.

### T3.4 Identity-control suite (text) — `[x]` *(done 2026-06-19; §4 gate → provisional Path B)*
- **Goal:** know what the text models are actually reading before investing in audio.
- **End result:** ticker-only model, same-ticker transcript shuffle, identity linear probe; control table committed.
- **Acceptance test:** all three controls produce numbers for Stage 2 on both temporal and disjoint splits; outcome triggers the DESIGN.md §4 framing-gate review (DECISIONS.md entry required either way).
- **Subtasks:**
  - [x] `models/ticker_only.py` (train per-ticker target mean; unseen → global mean)
  - [x] `eval/controls.py` (within-ticker + global transcript shuffle, ticker-only, identity linear probe) + `ecvol controls` → `data/results/result_controls.csv` (1,368 rows) + `controls_probe.csv`
  - [x] Framing-gate review write-up — **§4 gate decided: provisional Path B (rigorous re-examination), revisit after Phase-4 audio** (user decision; DECISIONS 2026-06-19)
- **Notes:** controls run 2026-06-19, all three on both splits/datasets (heads from T3.3). **Results (decisive):** **identity probe** FinCall **89.5%** ticker accuracy (319× chance), MAEC 53.9% (487× chance) — embeddings heavily encode identity; **transcript shuffle** within-ticker ≈ real across nearly every Δv cell while global-shuffle is clearly worse (models read identity, not call content) — e.g. FinCall temporal `ridge_text` real +0.028/+0.030/+0.015/−0.144 vs within +0.017/+0.020/+0.006/−0.142; a few ticker-disjoint `text+pastvol` cells do drop under shuffle (honest exceptions, not a ≥2/4-horizon DM-significant pattern). **→ §4 gate: provisional Path B** (rigorous re-examination, revisit after Phase-4 audio; user decision, DECISIONS 2026-06-19). 5 new tests; 194 green, ruff clean. sklearn/CPU. Heads `fit`/`predict` split out (additive; Table-2 numerics unchanged). Journal: 2026-06-19 T3.4 entry. **Phase 3 (T3.1–T3.4) COMPLETE.**

---

## Phase 4 — Audio ladder (~2 weeks, throughput-bound)

### T4.1 Audio QC — `[x]` *(done 2026-06-19; FinCall-only — MAEC ships no audio)*
- **Goal:** know the corpus before burning GPU-weeks.
- **End result:** QC report for 100% of audio (duration, sample rate, clipping, silence ratio, decode errors); 16 kHz mono resampled store.
- **Acceptance test:** pipeline validated on Earnings-21 samples (known-good reference); corrupt files flagged with reason codes, not dropped silently.
- **Subtasks:**
  - [x] `features/audio/qc.py` (one ffmpeg pass per call: `astats`+`silencedetect` on the source for QC, write 16 kHz mono FLAC; no Python audio deps)
  - [x] ffmpeg-based resampler (16 kHz mono FLAC, metadata-stripped, idempotent) — store `data/raw/audio_16k/fincall/` (gitignored cache, 2,671 files / 119 GB)
  - [x] QC report artifact — committed `data/coverage/fincall_audio_qc.csv` (per-call) + `_summary.csv`; Earnings-21 validation `earnings21_qc_validation.csv`
- **Notes:** **DONE 2026-06-19.** `ecvol audio qc` / `audio qc-ref`. **2,671/2,671 FinCall decoded (100%); 1 `mostly_silent` flagged (reason-coded, not dropped), 0 decode errors.** Median dur ~61 min, silence 0.16, peak −1.07 dBFS. **QC finding:** heterogeneous source sample rates — 22050 (1361) / 16000 (930) / 44100 (256) / 11025 (77) / **8000 (14, telephone-grade)** / 32000 / 24000 / 48000; all upsampled to 16 kHz for the store (sub-16k sources gain no info — flagged for audio-model interpretation). **Acceptance met both ways:** synthetic ffmpeg signals (CI unit tests) + 3 real Earnings-21 wavs (all decode_ok, sane metrics — peak ≈0 dBFS, silence 15–21%). MAEC has no raw audio → audio ladder (T4.1–T4.4, Result Table 3) is FinCall-only. 8 new tests (pure parse_qc CI-safe + ffmpeg-guarded integration); 202 green, ruff clean. Journal: 2026-06-19 T4.1 entry.

### T4.2 eGeMAPS extraction (CPU, first) — `[x]` *(done 2026-06-19; per-call; per-turn deferred to T4.3 diarization)*
- **Goal:** cheap interpretable paralinguistics for the whole corpus.
- **End result:** openSMILE eGeMAPS functionals per call (and per speaker turn where diarization available), cached.
- **Acceptance test:** deterministic; full corpus completes on CPU (parallelized); feature distributions sanity-checked against published eGeMAPS ranges.
- **Subtasks:**
  - [x] `features/audio/egemaps.py` (opensmile eGeMAPSv02 Functionals, 88-d, per call over the 16 kHz store)
  - [x] Multiprocessing harness (ProcessPoolExecutor, 8 workers) — **resumable + checkpointed** (flush every 200; skips cached call_ids on re-run)
  - [x] Distribution report (`data/coverage/fincall_egemaps_summary.csv`: per-feature mean/std)
- **Notes:** **DONE 2026-06-19.** `ecvol audio egemaps`; `opensmile` in the `audio` dep group (CI-light, lazy import). **2,671/2,671 calls × 88 features, 0 failures.** Deterministic (sorted parquet; **resumable re-run byte-identical**, manifest `fincall_audio_egemaps.json` verifies OK). **Distribution sanity (vs published eGeMAPS):** F0 median 28.6 semitones (human-voice ~20–45), loudness median 0.46 (>0), no all-NaN columns. Per-speaker-turn extraction deferred to T4.3 (needs diarization). **Runtime reality: ~1.6 h wall on 8 cores** (eGeMAPS over ~hour-long calls is ~30–60 s CPU each — NOT the 15–30 min I first estimated; ETA-measure-first should apply to CPU extraction too, lesson logged). Mid-run I added resumability/checkpointing/progress (the original batch-write-at-end had no crash recovery — fixed; validated by the byte-identical skip-all re-run). 4 new tests (parquet/summary/resume-logic CI-safe + openSMILE-guarded extraction that skips in CI); 206 green, ruff clean. Output `data/fincall/audio_egemaps.parquet` (gitignored payload). Journal: 2026-06-19 T4.2 entry.

### T4.3 Neural audio representations — `[x]` *(done 2026-06-24; WavLM + emotion2vec+, per-call; diarization skipped for v1)*
- **Goal:** WavLM-Large + emotion2vec+ embeddings, chunked for consumer VRAM.
- **End result:** pooled per-call (and per-turn) embeddings, cached; pyannote diarization behind a config flag.
- **Acceptance test:** **ETA measured on a 50-call sample before full run; full-corpus plan (local vs. cloud burst) recorded in DECISIONS.md**; extraction idempotent/restartable mid-corpus.
- **Subtasks:**
  - [x] Chunking strategy — 30 s non-overlapping windows, mean-pooled to one per-call vector (user decision 2026-06-19)
  - [x] `wavlm.py` (`microsoft/wavlm-large`, 1024-d, GPU, window-batched, fp16 option) + `ecvol audio wavlm`
  - [x] `emotion2vec.py` (`emotion2vec_plus_large` via funasr, 1024-d, per-call 30 s-window mean-pool) + `ecvol audio emotion2vec`; full fp32 run DONE 2026-06-24: **2,671×1024, all finite, manifest OK, resume no-op** (~12 h)
  - [ ] `diarize.py` — **skipped** for v1 (per-call pooling only; no gated pyannote/HF_TOKEN — user decision 2026-06-19)
  - [x] Resume logic — checkpoints every 100, skips cached call_ids (carried from T4.2)
- **Notes:** **ETA gate (DESIGN-mandated) DONE:** naive impl 50 calls/1478 s → ~22 h; **optimized (window-batching + fp16) 30 calls/361 s → ~9 h fp16 / ~18–22 h fp32.** **Full-corpus plan = local overnight, fp16, resumable** (user decision 2026-06-19; cloud burst rejected — 119 GB audio upload friction for a ~9 h local job; fp16 negligible on mean-pooled per-call vectors). **WavLM full run DONE (2026-06-23): 2,671×1024, all finite, 0 all-NaN; manifest OK; resume = clean no-op (0 new).** Window counts median 124 (≈62 min/call), range 37–398; mean L2 norm 1.52. `data/fincall/audio_wavlm.parquet` (gitignored payload) + `fincall_audio_wavlm.json`. Diarization skipped → per-call pooling only (matches text/eGeMAPS). 2 new WavLM tests (pure parquet CI-safe + guarded embed); 208 green, ruff clean. **emotion2vec+ is the remaining sub-task before T4.4.** Journal: 2026-06-19/23 T4.3 entries.

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
- **Notes:** the concrete Stage-6 audio-LLM recipe (Qwen2.5-Omni-7B: masked-mean-pool the Thinker last hidden state, 4-bit NF4, QA-conditioned + task-aware prompt) is recorded from prior-team work — DECISIONS.md 2026-06-14; carries the §3.5 gender-confound analysis. Still gated; no spend authorized.

---

## Exploration tracks (gated; promotion requires DECISIONS.md)

Adopted from this team's prior multimodal-volatility work (see `ingest/ingest.md`, DECISIONS.md 2026-06-14). **These are exploratory, not confirmatory:** re-implemented open-weight, fit train-split-only, and reported as exploratory until they survive the §7.3 identity controls and the §4 framing gate. IDs are `TX#` and never renumber; promoting any to the confirmatory ladder requires a new DECISIONS.md entry. None is started ahead of its phase-order dependencies.

### TX1 — QA-driven structured features (open-weights) — `[ ]` *(extends Stage 5 / RQ3; build after Phase 3)*
- **Goal:** test whether data-driven QA topic features are auditable semantics that beat opaque embeddings (RQ3), using open weights only.
- **End result:** speaker-turn chunk (T3.1) → Qwen2.5-7B-Instruct QA generation → volatility-topic labels → open-model embeddings → **train-split-only** clustering → per-call topic-frequency features → Stage-1 GBDT; on FinCall (primary) + MAEC.
- **Acceptance test:** open-model QA audit **κ > 0.6** on 50 calls (mirrors T6.2) before corpus scale; clustering/taxonomy fit on **train split only** with a leakage assertion (no val/test calls inform the taxonomy); throughput ETA recorded before the full run; **DM tests vs. Stage 2 and Stage 4** on Δv (a win must clear the same bar as confirmatory features).
- **Subtasks:**
  - [ ] Open-weight QA-generation prompt (port the general/conceptual prior-team prompt; no proprietary models)
  - [ ] Train-only topic taxonomy (label → embed → cluster; k chosen honestly, not asserted)
  - [ ] Per-call topic-frequency feature builder + cache
  - [ ] Human-audit tooling (reuse T6.2 `audit.py`) + leakage assertion
- **Notes:** feasibility scouted in `notebooks/explore_qa_generation.py` (JOURNAL.md 2026-06-14). DECISIONS.md 2026-06-14.

### TX2 — Short-horizon / implied-vol target exploration — `[ ]` *(needs T1.2/T1.3 first; IV needs a new data source)*
- **Goal:** explore the field's open gap (Undermind review): intraday/event-window RV and/or options-implied volatility around calls.
- **End result:** one or more exploratory targets — 1-day / [0,+1] event-window RV, and/or short-maturity near-the-money IV — computed under the §5.3 information rule + after-hours timing.
- **Acceptance test:** targets computed deterministically under the §5.4 information rule (no post-`as_of` reads); options/IV data sourced with a SHA-256 manifest + license note; results reported **alongside, never replacing**, the headline {3,7,15,30}d RV targets.
- **Subtasks:**
  - [ ] Timestamp-precision audit (event-window RV needs call time — §10 risk #7)
  - [ ] Options/IV data source evaluation (license, coverage 2019–2021; not in §5.2)
  - [ ] Target implementation + unit tests (mirror T1.3)
- **Notes:** §5.3 headline targets unchanged. DECISIONS.md 2026-06-14.

### TX3 — Re-examine prior "beats-KeFVP" result through our controls — `[ ]` *(needs Phase 2 controls + MAEC ingestion T1.5)*
- **Goal:** determine whether the prior team's ~8% MSE improvement over KeFVP is real signal or ticker-identity memorization.
- **End result:** the prior MAEC/EC result re-run with our control suite: HAR-RV/persistence floor, ticker-only baseline, same-ticker transcript shuffle, and the Δv target.
- **Acceptance test:** all four controls produce numbers; the KeFVP-label-vs-computed-target handling is documented (preferred: recompute MAEC targets per §5.3; else label the KeFVP-label run as exploration); conclusion stated honestly regardless of outcome.
- **Subtasks:**
  - [ ] Reproduce the prior result's data setup (MAEC; document label provenance)
  - [ ] Apply §7.3 controls + Δv target
  - [ ] Write-up (signal vs. identity) → feeds the §4 framing-gate evidence
- **Notes:** their setup reuses KeFVP's released labels (conflicts with §5.3 computed targets). DECISIONS.md 2026-06-14.
