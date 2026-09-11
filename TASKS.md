# TASKS.md — Living Task Tracker

**This is the operational copy of the phase plan in [DESIGN.md](DESIGN.md) §9.** Design rationale, hypotheses, and protocols live there; execution state lives here.

## How to use this file

- **Statuses:** `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked · `[-]` dropped/deferred (with a DECISIONS.md entry). Update the task status line and subtask checkboxes **as part of finishing the work**, not after.
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

### T4.4 Stage-3 results + gender-confound analysis → Result Table 3 — `[x]` *(done 2026-06-24 — completes Phase 4)*
- **Goal:** audio's honest contribution, plus the DESIGN.md §3.5 analysis.
- **End result:** audio-only and audio+covariate heads (5 seeds); **Result Table 3**; gender analysis (F0-based speaker-gender proxy → feature/error correlations, per-group error rates).
- **Acceptance test:** DM tests vs. Stage 1 and Stage 2; gender analysis covers ≥90% of calls with a dominant-speaker proxy; limitations paragraph drafted.
- **Subtasks:**
  - [x] Head configs (`eval/stage3.py` + `features/audio/assemble.py`: ridge+MLP on eGeMAPS / WavLM / emotion2vec+ / WavLM+eGeMAPS / WavLM+text fusion × {audio, audio+pastvol}, 5 seeds; reuses the T3.3 harness) → `ecvol evaluate-audio` → `result_table_3.csv` (1,296 rows, DM vs persistence/HAR/Stage-1/Stage-2)
  - [x] Gender-proxy construction (`eval/audio_eval.py`: eGeMAPS-F0 pitch proxy, 100% coverage; per-group test MSE + F0↔error/prediction correlations → `audio_gender.csv`) + identity probe (WavLM 76%/emotion2vec 29%) + same-ticker audio shuffle (`audio_shuffle.csv`)
  - [x] Per-group reporting in `report.py` (`write_reports3` → `result_table_3.{md,tex}`, `*` = DM-sig vs Stage-1; byte-identical render)
- **Notes:** **DONE 2026-06-24 — Phase 4 COMPLETE.** **Acceptance met:** DM vs Stage-1 & Stage-2 in every cell; gender F0 proxy covers **100%** (≥90% gate); limitations drafted (DECISIONS + here). **Findings (RQ1-audio, decisive for the framing gate):** (1) **identity probe** WavLM **76.3%** ticker accuracy (271× chance), emotion2vec 29.4% (105×) — audio encodes identity. (2) **Δv vs Stage-1:** the only DM-significant wins are the **+past-vol** variants on **ticker-disjoint** (e.g. `ridge_wavlm_audio_pastvol` τ3 +0.337/τ7 +0.171/τ30 +0.212); **nothing** beats the floor on temporal. (3) **Audio shuffle (the clincher):** those ticker-disjoint Δv gains are ~unchanged even under **global** audio shuffle (τ3 0.337→0.318, τ30 0.212→0.100) → **the WavLM features are inert; past-vol carries the signal.** (4) **Gender:** per-group MSE near-identical (low 0.241 / high 0.242), F0↔err² corr 0.03 — no large disparity. → **§4 Path A criterion NOT met.** **Framing gate (user decision 2026-06-24): Path B kept PROVISIONAL, revisit after Phase-5 fusion + Phase-6 LLM** (DECISIONS 2026-06-24). 6 new tests; ruff clean. FinCall-only (MAEC no audio). Journal: 2026-06-24 T4.4 entry.

---

## Phase 5 — Fusion + full ablation grid (~1 week)

### T5.1 Fusion models — `[x]` *(done 2026-06-24)*
- **Goal:** Stage-4 multimodal heads.
- **End result:** gated fusion and cross-attention heads over frozen modality embeddings; late-fusion stacking with Stage-1 GBDT; 5 seeds each.
- **Acceptance test:** fusion params <5M (small-data discipline); training fits in <2 GB VRAM; results reproducible from configs.
- **Subtasks:**
  - [x] `models/fusion.py` — **gated fusion** (per-modality train-fit PCA(64) + L2-norm "gate" → concat → shallow MLP) + **late-fusion stacking** (meta-ridge over [text, audio, Stage-1 GBDT] base preds, fit on val). sklearn/CPU, trivially <5M params / <2 GB.
  - [x] Stacking harness (`stack_fit_predict`; bases fit on train, meta on val, applied to test — no train-pred leakage)
  - [x] Hyperparameter ranges fixed (reuses heads.py MLP/ridge configs; no post-hoc sweeps) — `eval/stage4.py` + `ecvol evaluate-fusion`
- **Notes:** **DONE 2026-06-24.** All modalities (text BGE + WavLM + emotion2vec+ + eGeMAPS) + past-vol; covariates {in, out}; DM vs persistence/Stage-1/Stage-2/Stage-3. **Cross-attention omitted** (one pooled vector per modality → 2–3-token attention ≈ gated fusion; DECISIONS 2026-06-24). **`result_table_4_fusion.csv` = 288 rows, 0 NaN.** **RQ2 finding: multimodal is NOT > best unimodal** — only 1/48 Δv-test cells beats all of Stage-1/2/3 DM-significantly (gated τ7 r2 +0.019, negligible); best fusion (stack ticker-disjoint τ3 r2 +0.436) is **not** DM-sig vs Stage-1 (p=0.13). Fusion ≈ best unimodal ≈ past-vol → consistent with provisional Path B. sklearn/CPU (CI-torch-free). 5 new tests; ruff clean. FinCall-only. Journal: 2026-06-24 T5.1 entry.

### T5.2 Full ablation grid → Result Table 4 (main table) — `[x]` *(done 2026-06-24 — completes Phase 5)*
- **Goal:** the DESIGN.md §7.6 grid, populated.
- **End result:** **Result Table 4** — modality × covariates × split × target × horizon, with significance annotations; per-year breakdown appendix table.
- **Acceptance test:** every confirmatory comparison from §7.5 has a Holm-corrected p-value; `ecvol report` regenerates the whole grid from artifacts.
- **Subtasks:**
  - [x] Grid runner — `eval/grid.py` + `ecvol grid`: consolidates the **pre-registered canonical model per stage** (persistence/HAR/GBDT/ridge-text/ridge-audio/fusion-stack — all **ridge/structural, no MLP**) → `result_table_4.csv` (720 rows)
  - [x] Holm correction — confirmatory Stage-k-vs-Stage-1 DM on Δv, across 4 horizons per (stage, split); `holm_p_vs_stage1` column
  - [x] Appendix tables — `report.py write_reports4` → `result_table_4.{md,tex}` (committed-artifact CI guard) + per-year breakdown `result_table_4_peryear.csv`
- **Notes:** **DONE 2026-06-24 — Phase 5 COMPLETE.** Acceptance met: Holm-corrected Δv p-values; byte-identical `ecvol report` regen (CI-guarded) + byte-identical grid rebuild. **Canonical models are ridge/structural by design — the main table deliberately avoids the seed-unreliable MLP heads flagged below.** Findings: 15 Δv-test cells Holm-sig vs Stage-1, but the 7 favorable ones are all on the **ticker-disjoint** split (identity/past-vol-confounded per T3.4/T4.4 shuffle); on the regime-honest **temporal** split text/fusion are Holm-sig *worse* than Stage-1. Per-year: FinCall temporal test = entirely 2021; there every content model underperforms persistence (level-v R²_OOS HAR −0.29 / GBDT −0.82 / text −1.09 / audio −1.66 / fusion −1.02). Reinforces provisional Path B. 3 new tests; ruff clean. Journal: 2026-06-24 T5.2 entry.
- **Notes (future-work):** **Future-work (from 2026-06-24 validation/gut-check):** the **shallow MLP heads are unreliable** — across-seed `seed_std` reaches 1.07 mean / 3.36 max R² units (`mlp_egemaps_audio_pastvol`, Table 3), so the catastrophic negative R²s (−26 Table 2, −77 Table 3) are seed-driven divergence, not signal (features are StandardScaler'd; ridge/structural heads are deterministic, GBDT MSE-std ~0.005). When building the unified grid, **report the MLP as median-over-seeds (or IQR), retune (higher `MLP_ALPHA`/lower capacity), or demote it to a robustness check** so the main table isn't read off divergent seeds. Also note the **"past-vol" covariate block (`stage2.py` `PASTVOL = [v_pre, rv_daily, rv_weekly, rv_monthly]`, standardized, val-selected α) is NOT the same estimator as Table-1 HAR** (untuned OLS on the 3 RV terms) — it tanks to −2.0 on temporal τ=15 where HAR is +0.21 (regime-robustness gap; they agree on ticker-disjoint). Keep them labeled distinctly in the grid.

---

## Phase 6 — LLM structured features — **DROPPED 2026-08-23** (DECISIONS 2026-08-23 §11)

> **Phase 6 is closed and replaced by Phase 6R (reproduction & audit study).** RQ3 is withdrawn as a
> pre-registered research question. The κ-gate failure (2026-08-09) stands as a *reported* negative
> finding — the measurement is retained, the corpus run and RQ3's predictive arm are not. The frozen
> v2 schema, rubric, extraction code, and audit tooling stay in-tree as released benchmark artifacts.
> **Rater-2 collection is CANCELLED** (see HANDOFF).

### T6.1 Feature schema design — `[x]` *(v2 schema signed off 2026-06-29)*
- **Goal:** an auditable semantic feature set, grounded in actual calls.
- **End result:** pydantic JSON schema (per-section): guidance direction {raise/maintain/lower/none}, hedging intensity (0–4), Q&A evasiveness (0–4), surprise mentions, analyst-tone (0–4), plus free-text evidence spans for auditability. Designed from manual reading of 20 calls.
- **Acceptance test:** two human passes over 10 calls agree on the schema's applicability; every field has a written rubric.
- **Subtasks:**
  - [x] Manual reading notes — *rater 1 read + labeled the 50-call audit sample (`ingest/Ratings_1.xlsx`); qualitative feedback (relayed by user) drove the v2 schema revision*
  - [x] Schema + rubric doc — `features/llm/schema.py` (`SectionFeatures`, **v2**) + `docs/llm_feature_rubric.md` (per-field anchors, applicability, field roles, κ protocol)
  - [x] Prompt drafts (`features/llm/prompts.py`) — section-aware, `PROMPT_VERSION="v2"`
- **Notes:** 2026-06-24 — v1 scaffolding committed (engineering); field set is the pre-registered one. **Blocked:** acceptance test is human (two-pass agreement over 10 calls) → blank labeling sheet `data/coverage/fincall_llm_label_sheet.csv` (one row per call×section, leakage-safe train-only sample). User reads + labels + signs off (or requests rubric edits) before T6.2 extraction is built against the frozen schema+prompt. DECISIONS 2026-06-24.
  - 2026-06-29 — **schema → v2 (superset)** per rater-1 feedback + numeral-aware literature (DESIGN §3 R30/R31/R32). Added two **exploratory** fields (`management_optimism`, `quantitative_specificity`, both sections, unlabeled → no κ-gate); **confirmatory core** for the gate = `guidance_direction`/`hedging_intensity`/`surprise_mentions` (fixed pre-extraction on label variance, not model κ); weak labeled fields (`qa_evasiveness`, `analyst_tone`) reported-not-gated. `PROMPT_VERSION` v1→v2. 29 LLM tests green. DECISIONS 2026-06-29.
  - 2026-06-29 — **v2 SIGNED OFF (user).** Schema + rubric + `PROMPT_VERSION="v2"` frozen; OSC extraction may build against it. Acceptance met via **single careful rater + user sign-off** (the literal two-pass-IAA test deferred to pre-publication per DECISIONS 2026-06-29 — same single-rater stance as the κ-gate). T6.2 corpus extraction is now unblocked on the schema side (remaining blockers: OSC access + >32k-token context policy, HANDOFF).

### T6.2 Constrained extraction + human-audit gate — `[-]` *(**DROPPED 2026-08-23**; local route works, κ-gate FAILED and is retained as a reported finding — DECISIONS 2026-08-23 §11)*
- **Goal:** reliable corpus-scale extraction on consumer GPU.
- **End result:** Qwen2.5-7B-Instruct (4-bit) + Outlines pipeline; vLLM if VRAM allows, llama.cpp fallback; extracted features for the full corpus, cached with prompt+model version keys.
- **Acceptance test:** 100% schema-valid outputs (constrained decoding guarantees shape; the gate is on content): **human audit on 50 calls, κ > 0.6 on categorical fields vs. rubric labels — scaling to corpus is blocked until passed**; throughput ETA recorded.
- **Subtasks:**
  - [x] `extract.py` (Outlines 1.3 constrained decoding, resumable, deterministic, model-suffixed parquet) — engine-agnostic: `transformers`+bitsandbytes-4bit (Windows local) / `vllm` (OSC Linux)
  - [x] Audit tooling (`audit.py` — per-field κ + gate + model-vs-model matrix; `ecvol llm-kappa`) + 50-call labeling sheet (`featurize llm-audit-sample`, train-only)
  - [x] Batch runner with resume (`ecvol featurize llm`) + ETA probe (`featurize llm-eta`)
  - [x] Rater-workbook ingest (`ecvol featurize llm-ingest-ratings` → `features/llm/ratings.py`): stdlib xlsx parse → canonical label CSV, validated vs. the frozen sample + schema ranges; 6 tests
- **Notes:** 2026-06-24 — engineering done + gated (242 tests). Plan: multi-model panel on **OSC** (exploration: does scale → signal?) if local ETA >20h; `cloud/osc/` package + DECISIONS spend entry. **Pending:** (1) local ETA probe result, (2) frozen schema (T6.1 sign-off), (3) per-model **κ>0.6** human audit before corpus scale. See DECISIONS 2026-06-24, HANDOFF.
  - 2026-06-29 — **rater 1 labels ingested** → `data/coverage/fincall_llm_labels_rater1.csv` (97 rows / 50 calls, exact key match, all in range). Per DECISIONS 2026-06-29 the **κ-gate runs on this single rater for the OSC go/no-go**; second rater's IAA deferred to pre-publication (not a compute blocker). **A borderline result (κ≈0.45–0.6) re-blocks on rater 2** before any Stage-5/RQ3/Path-B claim. Any κ reported = "single-annotator; IAA pending". 248 tests green.
  - 2026-06-29 — **>32k context policy resolved: YaRN-extend to 65536** (DECISIONS 2026-06-29). `extract.yarn_rope_scaling` + `VLLMEngine` `hf_overrides` + CLI `--yarn`; `cloud/osc/slurm/extract.sbatch` defaults `MAX_MODEL_LEN=65536`/`YARN=1`. **OSC package now turnkey** — only the operational human steps remain (OSC allocation/account code; build sif; stage weights; smoke `--limit` job; submit panel). See `cloud/osc/README.md` + HANDOFF. 254 tests green.
  - 2026-07-04 — **compute REROUTED OSC → Colab Pro+** (OSC access + funds lapsed; DECISIONS 2026-07-04, superseding the 2026-06-24 $1000 OSC spend — void/unused). Same `--engine vllm` + Outlines + YaRN engine, on a Colab GPU. Only new code: `featurize llm --audit-sample` (extract the 50 κ-gate calls through the same engine as the corpus; mutually exclusive with `--limit`; 2 tests). New ops package `cloud/colab/` (README + RUNBOOK + `setup.sh` + `run.ipynb`) mirrors `cloud/osc/`; Drive = persistence/resume store, GPU class not guaranteed (A100-40GB fits 7B/8B-fp16 @65k; 32B→AWQ; 72B dropped). No paid-API budget; flat Pro+, no deadline. `cloud/osc/` retained (reference). **Remaining: operational human steps on Colab** (repo+2 parquets on Drive; run the notebook; smoke → κ-gate → corpus). See `cloud/colab/README.md` + HANDOFF.
  - 2026-08-09 — **local route BUILT and WORKING; κ-gate FAILED → corpus NOT started.** The 2026-06-24 "local is infeasible" finding was engine-specific, not hardware-specific: a new `--engine llamacpp` (`llama-server` + HTTP JSON-schema decoding, same frozen v2 schema/prompt and same YaRN-65536 policy) runs Qwen2.5-7B **Q4_K_M** whole-section on the 16 GB card at **4.63 s/section** → FinCall ~6.6 h, +MAEC ~12 h total. Three real defects fixed en route: (1) pydantic omits defaulted fields from `required`, so the grammar closed early — `evidence` was empty on *every* section and a skipped exploratory field would have become a silent 0 *rating*; (2) an unbounded evidence string ran past the decode cap and emitted unparseable JSON; (3) the CLI died with UnicodeEncodeError the moment its output was redirected to a log (cp1252 vs "κ"). Defects (1)–(2) are **latent in the Outlines/vLLM path too** — same pydantic schema, same optional-field grammar — so the Colab route needs them before it runs. **Gate result on the 97 audit rows: guidance_direction κ=0.165, hedging_intensity κ=−0.050, surprise_mentions κ=0.222 (confirmatory core; all ≪0.6) → FAIL.** Not a plumbing bug (97/97 label join, section text spot-checked, 100% schema-valid) and not a calibration offset (shifting the ordinal by ±1/±2 does not rescue κ); the model is near-constant on `analyst_tone` (47/50 rated "2") and independent of the human on hedging. Full diagnosis + the decision options: `data/coverage/llm_kappa_gate_report.md`. **Corpus scale stays blocked** (DECISIONS 2026-06-29 rule) — this is far below the 0.45–0.6 borderline band, so rater 2 would not change the go/no-go, though it is now *diagnostic* (is the schema the problem?). See DECISIONS 2026-08-09, HANDOFF.

### T6.3 Stage-5 results + masking ablation → Result Table 5 — `[-]` *(**DROPPED 2026-08-23** with RQ3 — DECISIONS 2026-08-23 §11; no Result Table 5 will exist)*
- **Goal:** RQ3 answered; lookahead leakage estimated.
- **End result:** LLM features → Stage-1 GBDT (with covariates), 5 seeds; masked-prompt (names/tickers/dates removed) variant; **Result Table 5**.
- **Acceptance test:** DM tests vs. Stage 2 and Stage 4; masked-vs-unmasked gap reported with CI.
- **Subtasks:**
  - [ ] Feature join
  - [ ] Masking transform
  - [ ] Result configs
- **Notes:** —

---

## Phase 6R — Reproduction & audit study (~3–4 weeks) — *replaces Phase 6; DECISIONS 2026-08-23*

### T6R.1 Benchmark substrate audit → Result Table 5R — `[x]` *(done 2026-09-03; `ecvol audit substrate`, run 20260902T054152Z-audit-substrate-48b9432a)*
- **Goal:** document what the evaluation substrate this literature shares actually contains. Requires reproducing no model.
- **End result:** an `ecvol audit substrate` command + released report covering, for EC / MAEC-15 / MAEC-16: ticker overlap between train and test, embargo gap at each split boundary, sentinel/degenerate label cells, and label-schema hazards; plus a diff of the literature's shipped labels against our §5.3 targets.
- **Acceptance test:** every number regenerates from the command; each claim carries the file and row/cell counts it was computed from; the "no paper reports a ticker-disjoint condition" claim is stated as a literature-coverage claim, not a data claim.
- **Subtasks:**
  - [x] Fetch + manifest the canonical label/split files — `D:\ecvol-data\raw\ref\lineage\{KeFVP,VolTAGE,HTML}` pinned to commits 95993892 / 597d69d8 / 9d717972, 44 files, `lineage_manifest.json` (SHA-256)
  - [x] Overlap + embargo metrics per benchmark + our splits as contrast rows (`src/ecvol/eval/substrate.py`)
  - [x] Label-defect scan — 137/16,800 single-day zeros (train 94 / val 9 / test 34; averaged series 0); `future_label_{3,7,15,30}` binary in all 6 KeFVP MAEC price files (154/154 etc.)
  - [x] Label-vs-our-targets diff — KeFVP `maec{15,16}_test_avg_val.csv` × our calendar-day `v_post`: joins 113–114/154 and 216/280 at τ≥7 (74–77%; τ=3 only 30/154, 106/280 because our τ=3 calendar windows exclude ≤1-session calls), corr 0.67–0.88 (MAEC-15) / 0.51–0.82 (MAEC-16), τ=3 0.71 / 0.25. The 08-23 "99.4% join" was on the price-label file's ids, not on priced targets.
  - [x] Report + release packaging — `results/result_table_5r.csv` + `_labels.csv` (manifested run), `paper/tables/substrate.tex`, §3.1 text, A1 pointer
- **Done (2026-09-03):** every 2026-08-23 number regenerates exactly (EC 80.4% / MAEC-15 44.2% / MAEC-16 55.7% overlap; 0-day embargos except MAEC-16 va→te 3 d; byte-identical VolTAGE/KeFVP files by SHA). Ticker-disjoint rows carry no embargo metric (not temporally ordered). **Open:** the sentinel-zero interpretation (zero-variance day vs missing) is still unsettled — the paper states the count only.
- **Notes:** findings already established 2026-08-23 (JOURNAL; `docs/advisor_redirection_2026-08.md` §5) — **VolTAGE and KeFVP ship byte-identical split files**; EC 80.4% / MAEC-15 44.2% / MAEC-16 55.7% test-ticker-in-train; **0-day embargo** on every boundary against τ≤30 targets; 137 zeros / 16,800 cells in the single-day series (headline Avg_Series clean); KeFVP MAEC files interleave binary labels at exactly τ ∈ {3,7,15,30} among 26 price columns (154/154 rows). **This task formalises and regenerates them, it does not re-derive them.** Open: settle the sentinel-zero interpretation (zero-variance day vs missing) before it enters a paper claim.

### T6R.2 Reproduction of prior models under our controls → Result Table 6R — `[x]` *(closed 2026-09-03 with reason codes: HTML + SCSS reproduced under both conditions; KeFVP / DialogueGAT / Sawhney reason-coded — see DECISIONS 2026-09-03 (closing))*
- **Goal:** demonstrate the benchmark's value by re-evaluating released models under embargoed, ticker-disjoint conditions.
- **End result:** HTML, Same-Company-Same-Signal, KeFVP, DialogueGAT, and Sawhney (ACM MM 2020) run **each on its own original dataset** under our control suite; published-split vs leakage-proof-split deltas reported per model.
- **Acceptance test:** for every model, either a result under both split conditions, or a documented reason code for why it could not run; the published-split numbers are reproduced within a stated tolerance before any controlled number is claimed; no model is evaluated on a dataset it was not originally trained on without that being labelled a transfer test.
- **Subtasks:**
  - [x] HTML as an `ecvol` head over our embeddings — `src/ecvol/models/html_head.py` + `eval/reproduce.py` + `ecvol reproduce html` (DECISIONS 2026-09-03 (later)); EC ingested (558/572 ok, 550 priced; published split 385/54/109). **Run 20260902T153103Z-reproduce-html-1ac22d6e:** published split + published labels MSE 0.769/0.436/0.285/0.223 (paper 1.175/0.372/0.153/0.133; persistence on the shipped labels 1.485/0.570/0.340/0.202 — the published τ=15/30 values sit *below* the persistence floor and are not reproduced); our labels 0.739/0.395/0.277/0.218; embargoed temporal 0.745/0.418/0.273/0.231; **ticker-disjoint 0.808/0.486/0.404/0.296 (+9/+23/+46/+36% vs published split)**; R² vs persistence < 0 at τ=30 everywhere. Paper §6 + `tables/repro.tex`.
  - *Re-run 2026-09-09 (T6R.3/H4, run 20260909T223252Z):* with the single-day auxiliary label the published-label rows are unchanged; ours 0.760/0.420/0.285/0.214, temporal 0.778/0.448/0.278/0.265, ticker-disjoint 0.773/0.503/0.394/0.291 → ticker-disjoint penalty **+2/+20/+38/+36%** (was +9/+23/+46/+36%), R² at τ=15 0.20→0.06; paper updated.
  - [x] Same-Company-Same-Signal — `ecvol reproduce scss` on DEC (run 20260902T153620Z): PEV/STPEV reproduce to 3 decimals (2023 STPEV 0.240/0.253/0.227/0.246 vs 0.239/0.253/0.227/0.246); ticker-disjoint → reason code `undefined_without_same_ticker_history`
  - [-] KeFVP — **reason-coded, not ported** (2026-09-03): requirements.txt uninstallable (spacy==.5.3, pickle), torch 1.12 predates Blackwell, MAEC KePt embeddings unavailable upstream; multi-day port. `code_availability.csv` verdict `partial`.
  - [-] DialogueGAT — **reason-coded, not ported**: τ ≤ 15 only and the dialogue corpus must be rebuilt (verdict `partial`)
  - [-] Sawhney ACM MM 2020 — **reason-coded, not ported**: TF 2.1 / Keras 2.3.1 pins, last push 2020 (verdict `partial`)
  - [x] Code-availability audit table — `ecvol audit code` → `results/code_availability.csv` (run 20260902T173314Z): 17 papers, **2 runnable / 6 partial / 9 none**, live GitHub probe (status, code files, last push, licence, probe date) + curated reasons; `paper/tables/code_audit.tex`
- **Notes:** DECISIONS 2026-08-23 §5. **Confound rule: each model runs on its own dataset**, else "the model fails" is indistinguishable from "the model does not transfer." Only 5 of ~20 papers released runnable code; the audit of the other 15 (ECC Analyzer closed/GPT-4, Sound of Risk figures-only, DeFVP repo 0 KB, ECHO-GL "cannot run", NumHTML no URL, GNA-Vol 404, AMA-LSTM stub, AT-FinGPT paywalled) is itself a deliverable. Consider adopting **FinTrust** (`yingpengma/FinTrust`, ACL 2023, by the HTML author) as a complementary perturbation-control axis.

### T6R.3 Faithful-reproduction sprint (five papers) — `[~]` *(opened 2026-09-09; policy DECISIONS 2026-09-09; recon in JOURNAL 2026-09-09)*
- **Goal:** each of the five slate papers running on its own data with the authors' code where it exists, the published number matched or the gap explained, *before* our controls are added (next sprint).
- **End result:** one `ecvol reproduce <model>` command per paper (manifested), a per-model fidelity ledger (what is verbatim / patched / substituted / reason-coded), and a Result Table 6R-F (faithful) beside 6R.
- **Acceptance test:** for every paper, either (a) a run of the authors' code (forward-port patches listed) on the authors' data reaching a stated tolerance of the published cell, or (b) a harness port with every substitution labelled and the published cell shown beside it, or (c) a reason code with the missing artefact named; no cell is called "reproduced" under (b) or (c).
- **Subtasks (threads; order = certainty):**
  - **HTML** (authors' `RTransformer` verbatim, driver rewritten — the shipped driver decorrelates X and y by three independent unseeded `train_test_split` calls):
    - [x] H1 BERT-WWM-Large sentence embeddings (layer −2 mean, `TextSequence.txt` lines) for 572 EC calls → `data/ec/cache/html_bert_wwm_large_l2_sentences.parquet` (89,722 sentences; 2026-09-09)
    - [x] H2 (2026-09-09, run 20260909T080339Z) text-only faithful run — **code 0.660/0.336/0.253/0.160, paper 0.972/0.439/0.406/0.248, published_split 1.23/1.12/1.06/1.48 vs published 1.175/0.372/0.153/0.133**; the published recipe underfits at 10 epochs (raw log-vol targets, lr 2e-5) and never beats persistence at τ≥15 — three conditions: `code` (random 70/10/20, dropout 0 as shipped, min-over-epochs val), `paper` (chronological 7:1:2, dropout 0.5), `published_split` (VolTAGE split3) — vs 1.175/0.372/0.153/0.133
    - [x] H3 (2026-09-09, run 20260909T174611Z) 27 Praat sentence features (`data/ec/cache/praat27_sentences.parquet`, 89,722 clips = 89,722 lines, exact positional alignment) → text+audio: **code 0.658/0.347/0.244/0.165 vs published 0.845/0.349/0.251/0.158 (reproduces at τ≥7)**; paper 0.876/0.520/0.291/0.239; published_split diverges — the multimodal cell reproduces only under the code's random split
    - [x] H4 (2026-09-09) our port's auxiliary label = ln|r| on session +τ (verified = the shipped `future_Single_τ` to 3 decimals); `ecvol reproduce html` to be re-run for the `ours` rows
  - **KeFVP** (authors' `final_series_infer.py`, path/Windows patches; audio branch is commented out upstream, so text+price only is the faithful setting):
    - [!] K1 obtain the released EC KePt embedding pickle (Drive `1F83bjiJKEpq_MYrc0lzQb9rOLgooz-5E`): `gdown`, `curl` and browser-cookie routes all refused; the file opens in the user's browser → **user download, HANDOFF 2026-09-09**
    - [ ] K2 EC faithful run, 10 repeats × 4 τ → vs 0.610/0.291/0.183/0.114 (mean of 10)
    - [x] K3 (2026-09-11, run 20260911T044848Z) MAEC-15/16 with **regenerated** `raw_bert_base_uncased` embeddings (the authors' generator; six import-level release defects patched, see ledger) — **MAEC-15 0.419/0.185/0.123/0.086 vs 0.418/0.187/0.122/0.087; MAEC-16 0.442/0.288/0.363/0.188 vs 0.445/0.279/0.303/0.177**; 7/8 cells within one published std, seed spreads match via the authors' `generatePtmEmbeddings.py` (upstream never released these) → vs 0.418/0.187/0.122/0.087 and 0.445/0.279/0.303/0.177 (Table 2 MAEC-16 row is internally inconsistent — noted)
    - [ ] K4 `[colab]` KePt adaptive pre-training (BERT, 60 epochs, ~5 h GPU; needs the `kept_dataset` Drive pickles) — not run locally
  - **SCSS** (authors' TSLib-style `run.py`):
    - [x] S1 restore the clone (filenames with `|` cannot exist on Windows → sparse checkout minus `earnings_results/`, index dumped to `raw/ref/scss/earnings_results_index.txt`, 304 published MSEs in the names)
    - [x] S2 TSMixer, 76 runs on DEC → **median |ours − published| 1.6e-5, max 4.2e-3, 70/76 within 1e-3** (`result_table_6r_scss_tsmixer.csv`, run 20260909T054744Z)
    - [x] S3 TMLP on the OpenAI embeddings (Drive `Embeddings/openai/`: `DEC.npz`, `DEC2RandomTicker.npz` ≙ script's `DECRandomTicker`, `DECRandomAll.npz`; 1800 × 3072) — **228 cells, median |Δ| 0.014, window means within ~0.01; real ≈ ticker-shuffled < all-shuffled reproduces** (run 20260909T065848Z)
    - [ ] S4 Aug_PEV / Aug_STPEV on EC + MAEC from the shipped `dataset/EC|MAEC/*.csv` (notebook formulas re-implemented) → vs 0.367/0.296 (EC), 0.283/0.225 (MAEC15), 0.229/0.247 (MAEC16)
  - **DialogueGAT** (faithful impossible: training pickle unreleased, corpus = authors' private SeekingAlpha re-scrape with real speaker names, labels need CRSP, DGL has no wheels for this stack):
    - [x] D1 reason code confirmed with the missing artefacts named (`data/data_swd.pkl`, ~3,400 HTMLs, CRSP)
    - [x] D2 (2026-09-09, run 20260909T090023Z) harness port (`ecvol reproduce dialoguegat`; needs `torch_geometric`, to be added to the gpu group once the jobs release the venv) — **2019 0.694/0.320/0.206/0.134, 2020 0.621/0.350/0.392/0.270 (all epoch-0 stops), 2021 0.794/0.344/0.182/0.082**; beats persistence in 9/12 cells, loses across the 2020 regime boundary: turn-graph GAT (PyG) over BGE-M3 turn embeddings + v_past, chain + speaker edges, on FinCall (named speakers) and MAEC person-label turns (anonymous ⇒ the paper's "w/o speaker" ablation); published 2015/2016/2017 cells shown as reference only
  - **Sawhney 2020** (faithful impossible as shipped: no `stock_data.csv`, no price series, no lexicons, TF 2.1/Keras 2.3.1/TFA 0.8.3; the released code has no multi-task loss and tunes the ensemble on the test set):
    - [x] W1 (2026-09-09, run 20260909T055052Z) PyTorch port of the text BiLSTM — **finance SVR 0.712/0.370/0.218/0.154 vs published ensemble 0.601/0.308/0.181/0.119** (persistence 1.491/0.563/0.345/0.205); the 2-epoch text BiLSTM sits above persistence (2.14/1.58/1.38/1.40, five seeds) so the ensemble collapses onto finance (α=0 whether tuned on val or on test); 547 EC calls carry their labels (Mittens-retrofitted GloVe sentence means; plain GloVe first, Mittens if time) + SVR financial branch + (α,β) ensemble, their 60/20/20 split and mean-subtracted labels from our EC prices → vs 0.601/0.308/0.181/0.119 (second-hand, KeFVP Table 2)
    - [-] W2 Mittens retrofit — **not run, reason-coded (2026-09-09):** the text branch never leaves the untrained regime under the authors' 2-epoch protocol (W1), so retrofitting its word vectors cannot change the ensemble (α = 0 in every cell); revisit only with a labelled longer-training variant
    - [x] W3 (2026-09-09, run 20260909T071241Z) audio branch (27 Praat features/sentence, `features/audio/praat.py`, shared with H3) + aligned cross-attention model: audio alone untrained after their 1 epoch (MSE ≈ 5–6); **3-way ensemble 0.639/0.298/0.175/0.102 test-tuned (α=0, β≈0.1) vs published 0.601/0.308/0.181/0.119** — the published cell is reachable with an untrained audio branch acting as an intercept on the finance SVR
- **Fidelity ledger:** `docs/faithful_ledger.md` (one section per paper: verbatim / patched / substituted / reason-coded, with file:line of every patch).
- **Notes:** third-party code stays under `D:\ecvol-data\raw\ref\repos\` (pinned in `data/manifests/repos.json`); patches are applied at run time into a scratch build dir, never committed.

---

## Phase 9 — Reframe work (benchmark-first) — *DECISIONS 2026-08-23*

### T9.1 Dual-convention targets (calendar + trading day) — `[x]` *(done 2026-09-02; DECISIONS 2026-09-01 fixes the window rule; Table 1 carries both conventions, trading rows byte-identical)*
- **Goal:** make our targets comparable to the published leaderboard without discarding the validated trading-day set.
- **End result:** `ecvol targets build` emits both conventions; every result table carries the convention as a column; the delta between them is reported.
- **Acceptance test:** calendar-day targets unit-tested against hand-computed values exactly as the trading-day set was (T1.3); both sets regenerate deterministically; DESIGN §5.3 amended.
- **Notes:** DECISIONS 2026-08-23 §3. Qin & Yang used **calendar** days (verified in the source PDF); DESIGN §5.3 misattributed the convention as trading days. ~30 vs ~21 sessions at τ=30. **Blocks any comparability claim in T6R.2.**
- **Done (2026-09-02):** `targets.py` computes both conventions (`convention=` kwarg; calendar window = sessions dated within τ calendar days; population variance over the n sessions present; `<2` sessions → `short_window_{pre,post}`); `ecvol targets build` and `ecvol data ingest maec` write `targets.parquet` (trading, **values byte-identical on the original 15 columns**, both datasets) + `targets_calendar.parquet`, each with `convention`/`n_pre`/`n_post`; `ecvol targets compare` → `data/coverage/targets_convention_delta.csv`; `ecvol evaluate` evaluates both conventions (GARCH over the row's `n_post`, HAR refit per convention) → Result Table 1 has a `convention` column, calendar rows appended, **trading rows identical to the committed table** (run `20260902T040159Z-evaluate-cbe08bbd`, provenance `run`); `ecvol report` renders the calendar views after the trading ones. Tests: hand-counted windows around the 2021 MLK weekend, closed-form calendar τ=7 values (5 pre / 4 post sessions), short/missing windows, both-file determinism, trading-rows-unchanged regression. Splits rebuilt → committed CSVs unchanged. DESIGN §5.3 amended. **Findings:** τ=3 calendar → 48.8% of FinCall calls (any Thu/Fri call) have ≤1 post session; corr(v_post) between conventions 0.75/0.93/0.96/0.94 at τ=3/7/15/30; persistence MSE ~doubles at τ=3 under calendar; the COVID-cell gate failure (HAR τ=30 FinCall temporal −0.287) becomes +0.017 under calendar. **Open (design call):** Stages 2–4 under calendar days — not done (doubles head compute; MLP seed instability); Tables 2–4 gain the `convention` column when re-run for the T9.3 backfill→run upgrade.

### T9.2 Call-timestamp retrofit — `[x]` *(done 2026-09-03; EDGAR 8-K anchor with tiers; measured-target variant + sensitivity; re-baselining = open user decision, HANDOFF)*
- **Goal:** replace the uniform assume-after-hours fallback with measured call datetimes.
- **End result:** per-call datetime + timezone with a provenance tier, joined into the target pipeline; a sensitivity run comparing the fallback against measured timings.
- **Acceptance test:** coverage reported by provenance tier; the after-hours sensitivity check finally runs (DESIGN §10 risk #7); zero post-`as_of` reads.
- **Subtasks:**
  - [x] Check SCSS `beforeAfterMarket` coverage against our corpus first — 323 exact FinCall matches (13%); 80.5% BeforeMarket; kept as the `dec_flag` tier only
  - [x] EDGAR 8-K Item 2.02 retrofit — 50-call pilot (48/50, 43/48 agree with DEC), then bulk: `ecvol timing build {fincall,maec,earnings25}` → `coverage/{dataset}_timing.csv`
- **Done (2026-09-03, DECISIONS 2026-09-03):** `src/ecvol/data/timing.py` + `ecvol timing build|targets|sensitivity` (6 tests). Anchor = release time (8-K acceptance), tiers `edgar_8k` › `earnings25_metadata` › `dec_flag` › `assumed_after_hours`. **Coverage by tier:** FinCall 2182/2443 EDGAR (89.3%; 5 DEC; 254 fallback; 2 invalid dates), MAEC 2362/3443 (68.6%), Earnings25 475/476 (99.8%). **Session rule:** FinCall before_open 1295 / after_close 554 / intraday 46 / after_call_date 292 (the last likely identity-date errors — 32% of FinCall dates come from slide-PDF stamps); Earnings25 before_open 341 / after_close 123 / intraday 11 (vs 137 intraday *call starts* — releases precede calls). **Sensitivity (`results/timing_sensitivity.csv`, run `20260902T053423Z-timing-sensitivity-a9f55c8a`):** as_of moves for 53.3% FinCall / 58.6% MAEC / 27.6% Earnings25 calls; persistence MSE (level-v, test) ×1.01–1.08 FinCall, ×1.06–1.26 MAEC, Earnings25 τ=3 ×0.93; HAR R²_OOS mean Δ +0.06 FinCall disjoint, −0.03 FinCall temporal, +0.06 MAEC temporal, +0.08 Earnings25. Zero post-`as_of` reads: measured targets go through the same `anchor_day0`/window code as the primary set. **Open (user):** re-baseline the primary targets/splits/Tables 1–4 on the measured anchor, or keep the variant as a released sensitivity — HANDOFF 2026-09-03.
- **Notes:** DECISIONS 2026-08-23 §8. No call time-of-day exists anywhere today (only 3.4% of transcripts mention a clock time); 32.4% of recovered *dates* come from slide-PDF creation stamps.

### T9.3 Reproducibility-debt closure — `[x]` *(2026-08-26; per-command run manifests, CI-guarded; Table 1 re-run live → byte-identical)*
- **Goal:** make DESIGN §8.2/§12 true rather than aspirational, since the artifact is now the contribution.
- **End result:** every result-producing run writes `artifacts/runs/<run_id>/`; one committed config per experiment under `configs/`; `ecvol report` regenerates every paper table byte-identically from artifacts.
- **Acceptance test:** *(amended 2026-08-26 — DECISIONS 2026-08-26 (later) §2)* every committed result CSV matches a committed run manifest byte-for-byte and the md/tex tables re-render byte-identically, both asserted in CI (`tests/test_runs.py`, `tests/test_report.py`); every result-producing command has a loadable committed config; clean-machine regeneration of Result Table 1 is executed and recorded under T8.1.
- **Subtasks:**
  - [x] Determine whether `ecvol report` currently sources `artifacts/` or `data/results/` — **`data/results/` (2026-08-26):** the `evaluate*` commands write `data/results/*.csv` directly (git-tracked); `report` renders from them; `tracking.write_run` (T0.3) is never called by any command and `artifacts/runs/` does not exist
  - [x] Backfill run artifacts + per-experiment configs — `configs/{evaluate,evaluate-text,controls,evaluate-audio,evaluate-fusion,grid}.yaml` (`CommandConfig`); `ecvol runs backfill` manifested all 11 result CSVs (`provenance: backfill`, commit e223815); `ecvol evaluate` then re-run live → `result_table_1.csv` **byte-identical** to the committed file, manifest `20260826T190559Z-evaluate-cbe08bbd` (`provenance: run`, same config hash as the backfill)
  - [x] CI assertion — `tests/test_runs.py::test_committed_results_have_matching_manifests` (+ config loadability, + the existing md/tex re-render guard)
- **Notes:** DECISIONS 2026-08-23 §9; design + acceptance amendment DECISIONS 2026-08-26 (later). Before: `artifacts/` held 8 diagnostic files and no run payload, `configs/` only `example.yaml`, `tracking.write_run` never called. After: `ecvol report` renders only provenance-verified CSVs; `ecvol runs verify` checks all eleven. **Still `backfill`:** Tables 2/3/4, controls, audio, fusion — re-run their commands (GPU features are cached) to upgrade them to `provenance: run`; the MLP-head seed instability (T5.2 caveat) may make some of those *not* byte-identical, which would itself be a finding to record. Curated `paper/tables/*.tex` remain hand-copied cells → T8.1.

### T9.4 Earnings25 bitrate-stratified audio ladder re-run — `[x]` *(done 2026-09-03; run 20260902T124516Z-evaluate-audio-earnings25-6b0f8941)* *(re-scoped 2026-09-02 — DECISIONS 2026-09-02 §1: Earnings25 audio is 64/24/16 kbps, not clean)*
- **Goal:** test whether "audio is inert" depends on audio quality, using Earnings25's two bitrate strata (64 kbps: 281 calls; ≤24 kbps: 233 calls) within one quarter. *(Original goal — "audio that is not a 40 kbps transcode" — is unattainable: no clean corpus exists; DECISIONS 2026-09-02 §1.)*
- **End result:** the frozen Stage-3 audio ladder (eGeMAPS + WavLM + emotion2vec+) re-run on Earnings25 audio with the identity controls attached.
- **Acceptance test:** same extractors, same heads, same controls as T4.4; real-vs-shuffle reported **per bitrate stratum and pooled**; any divergence between strata, or from the FinCall result, stated with the bitrate difference as the candidate cause and the stratum sizes alongside.
- **Design (2026-09-02, in progress):** ticker-disjoint split only — a single-quarter corpus spans ~105 sessions, so the 30-session-embargoed temporal split leaves 5 training calls (`earnings25_splits_report.csv`); every ticker appears once, so the within-ticker shuffle is the identity and the identity probe is undefined (chance 1/n) — the global shuffle is the control. Cohorts `all` / `64k` (281) / `le24k` (232) from `earnings25_inventory.csv`; `ecvol evaluate-audio-earnings25` → `result_table_3_earnings25.csv` (+ `stratum`), `audio_shuffle_earnings25.csv`, `audio_strata_earnings25.csv`; config `configs/evaluate-audio-earnings25.yaml`. Audio commands (`qc`, `egemaps`, `wavlm`, `emotion2vec`) gained `--dataset`; `featurize sections`/`splits build` cover earnings25.
- **Done (2026-09-03):** full ladder on Earnings25 — QC 514/514 → 16 kHz FLAC store; eGeMAPS 514; WavLM-Large 514; emotion2vec+ 514 (9,245 s); `ecvol evaluate-audio-earnings25` → `result_table_3_earnings25.csv` (1,296 rows: 5 feature sets × 2 covariate sets × ridge/MLP × 3 targets × 4 τ × val/test × 3 cohorts), `audio_shuffle_earnings25.csv`, `audio_strata_earnings25.csv`. **Result:** ridge WavLM+past-vol Δv test R²_OOS pooled 0.443/0.203/0.080/0.108 (τ=3/7/15/30) but global-shuffle 0.351/0.150/0.040/0.095 → **audio increment 0.09/0.05/0.04/0.01**; per stratum 64k +0.03/+0.00/+0.08/−0.02 (n_test 39), ≤24k −0.04/+0.02/+0.07/+0.04 (n_test 55) — no consistent bitrate effect, the higher-bitrate stratum is not better. Audio-only heads' positive Δv R² is intercept/mean-reversion (survives shuffling). **Acceptance met:** same extractors/heads/controls as T4.4, real-vs-shuffle per stratum and pooled, divergence stated with stratum sizes. Paper §6 subsection + `tables/earnings25_audio.tex`.
- **Notes:** DECISIONS 2026-08-23 §7. **Highest-value single experiment on the backlog**: the audio finding currently rests entirely on FinCall's uniform 40 kbps transcodes because MAEC ships no audio. Depends on T7.1 (Earnings25 ingestion).

---

## Phase 7 — Post-cutoff data + lookahead study (~2 weeks, calendar-dependent)

### T7.1 Fresh acquisition pipeline (scripts-not-data) — `[x]` *(done 2026-09-02: Earnings25 ingested — 476 admitted, 475 join prices (99.79%); `docs/earnings25_verification_2026-09.md`)*
- **Goal:** ≥200 calls from 2025-Q4 / 2026-Q1 with audio + transcript + price joins.
- **End result:** acquisition scripts (EarningsCall/EarningsCast API primary; company-IR-page fetcher fallback) + terms-of-use review note; local-only data with manifests.
- **Acceptance test:** ≥200 calls pass the same ingestion gates as T1.4 (≥95% price join); ToS review written **before** any bulk pull; zero raw data committed.
- **Subtasks:**
  - [x] ~~API client + key handling (.env)~~ → not needed: Earnings25 is an open Zenodo payload (md5-verified); `ecvol prices pull --dataset earnings25` (yfinance + Tiingo fallback, key in .env)
  - [x] ~~IR-page fallback fetcher~~ → not needed (DECISIONS 2026-08-23 §4/§6)
  - [x] Ingestion onto the common schema — `ecvol data ingest earnings25` (`src/ecvol/data/earnings25_ingest.py`, 8 tests)
  - [x] Universe selection rule — S&P 500 membership on `ReleaseDate` by CIK, from month-end Wikipedia page revisions (cached + manifested); DECISIONS 2026-09-02 §2
- **Done (2026-09-02):** 514 → **476 admitted** (`lookalike`=4, `not_sp500`=23, `unresolved`=11); own price store `data/prices_earnings25/` (476 tickers, 2025-06→2026-03; AVB/EA via Tiingo, **EQR purged everywhere → 1 documented miss**); **join 475/476 = 99.79% (≥95% gate PASSED)**; targets under both conventions with **measured call times** (`ReleaseDate` UTC → ET; `time_known=True`; 215 before-open / 137 intraday / 124 after-close among admitted calls); bitrate strata 64k=281 / le24k=232 / other=1 (T9.4). Artifacts: `data/earnings25/{calls,targets,targets_calendar}.parquet` (manifested), `data/coverage/earnings25_{ingest_report,inventory,call_times,missing_tickers}.csv`, `data/manifests/{earnings25_raw,earnings25_calls,earnings25_targets,prices_earnings25}.json`. Zero raw data committed. **Finding for T9.2:** `ReleaseDate` is the *call* start, not the press-release time — e.g. Lennar releases after close the evening before an 11:00 ET call — so 137 intraday calls sit under DESIGN §5.3's "intraday = after-hours" rule with the reaction day in the *pre* window; the 8-K Item 2.02 timing is what settles day 0.
- **Notes:** **2026-08-23 — primary source changed to Earnings25** (DECISIONS 2026-08-23 §6): Zenodo DOI 10.5281/zenodo.18762168, CC-BY-4.0, ~500 Q4-2025 S&P 500 calls / 498 h, held Jan–Feb 2026 ⇒ post-cutoff for the Qwen2.5 stack. The task's original primary source (EarningsCast) is dead (HTTP 410, verified 2026-08-13) and earningscall.biz audio is paywalled at $129/mo. **Gated on a one-day verification** that Earnings25 carries what the T1.3 price joins need — audio provenance is undisclosed upstream and per-call metadata completeness (exact datetimes?) is unknown. Self-collection (TX4) is deferred, so the ≥200-call acceptance bar is now met by Earnings25 rather than by our own scripts; the "scripts-not-data" framing applies to the *ingestion* scripts. Also feeds T9.4 (clean-audio ladder re-run).
- **Verification (2026-09-02, `docs/earnings25_verification_2026-09.md`):** zip md5 verified, CC-BY-4.0 (ToS = attribution). 514 calls / 497.9 h, one record each with `Company`/`Country`/`ReleaseDate` (datetime, **consistent with UTC** — first measured call times in the project, feeds T9.2)/`Industry`/`MarketCap`, aligned speaker-attributed segments. No ticker: name→SEC-ticker via the T1.4 matcher + 8 overrides = 499/514; **472 are S&P 500 snapshot members (91.8%)**, ~20 more are 2025-Q4 members since dropped, **~25 calls (~5%) are name-collision look-alikes** (Grainger PLC, Domino's Pizza Group/Enterprises, Paramount Group, Vertex Inc., PTC India, Goldman Sachs BDC, Apple Hospitality REIT, …) → reason code at ingestion. Prices: archive ends 2022-06-30 → fresh yfinance pull needed (dry-checked, serves through 2026-02-27). **Audio is not clean:** 281 calls 44.1 kHz/64 kbps, 216 at 16 kHz/24 kbps, 16 at 11 kHz/16 kbps — 45% *below* FinCall's 40 kbps. **Design calls (user):** (a) re-scope T9.4 to a within-corpus 64-vs-≤24 kbps stratified re-run; (b) universe rule = S&P 500 membership on `ReleaseDate` (recommended) vs any US-listed ticker. Artifacts: `data/coverage/earnings25_{inventory,audio_probe}.csv`.

### T7.2 Frozen-pipeline post-cutoff evaluation — `[x]` *(done 2026-09-03; `ecvol evaluate-lookahead`, run 20260902T173818Z-evaluate-lookahead-aba100fd)*
- **Goal:** the lookahead-bias experiment (DESIGN.md §7.4).
- **End result:** all stages (0–5) evaluated on the post-cutoff set **with zero retraining or threshold changes after first look** (rule pre-registered in DESIGN.md); lookahead table comparing in-cutoff vs. post-cutoff degradation per stage.
- **Acceptance test:** evaluation run from frozen artifacts only (CI-verifiable: no training code touched); table regenerates.
- **Subtasks:**
  - [x] Frozen-eval runner — `src/ecvol/eval/lookahead.py`: persistence, train-fit HAR, ridge canonicals of Stages 2/3/4 fit once on FinCall temporal train (val → alpha; train medians → imputation), scored unchanged on FinCall test and on all 475 Earnings25 calls; **in-cutoff cells reproduce Tables 2/3 exactly**; ticker-FE GBDT omitted by construction (all post-cutoff companies unseen)
  - [x] Degradation metrics with bootstrap CIs — post − in R²_OOS vs persistence; month-cluster bootstrap CI on the post-cutoff R² (1,000 resamples, seed 0)
  - [x] Write-up — paper §6 lookahead subsection + `tables/lookahead.tex`
- **Done (2026-09-03):** **no post-cutoff degradation**: level-v R² HAR 0.464/0.410/0.207/−0.287 (in) → 0.561/0.471/0.319/0.077 (post); ridge text+vol 0.448/0.328/0.045/−1.091 → 0.543/0.429/0.200/−0.329; WavLM+vol 0.343/0.245/−0.390/−1.529 → 0.527/0.402/0.088/−0.432; fusion 0.430/0.303/−0.002/−1.211 → 0.153/0.425/0.172/−0.767. Δv: HAR post 0.561/0.471/0.319/0.077; text 0.247/0.227/0.046/0.028; WavLM 0.432/0.336/0.068/0.063; fusion −7.0/−1.8/−0.01/−0.08 (unstable, wide CI). Ordering preserved out of sample (HAR > content heads on level-v; at τ=30 only HAR edges past persistence, CI includes 0). *Correction 2026-09-02 (paper cross-check):* the in-cutoff test year is **2021** (per-year table), not 2020; the fusion ridge does degrade (level-v τ=3 0.43→0.15, unstable) — paper wording fixed. **Measured-anchor variant** (`result_table_7_measured.csv`, same command): HAR insensitive; ridge past-vol heads lose their in-cutoff Δv collapse (text −1.7/−3.1/−3.6/−0.5 → +0.34/+0.28/+0.08/−0.08), fusion post-cutoff instability shrinks (−7.0 → −0.8); Δv τ=30 post-cutoff WavLM+pastvol 0.18 (CI 0.14–0.24) vs HAR 0.08 — exploratory. Input to the re-baseline decision (HANDOFF).
- **Notes:** **Novelty scan 2026-07-19** (`paper/novelty_scan_2026-07.md`): claim scoped to *first frozen-pipeline post-cutoff evaluation in the earnings-call volatility literature* (nearest precedent: ECB-presser rates-vol check [R38]); optional comparison arm = chronologically consistent LLM [R39]; the lookahead line is active (R15/R16/R33/R34/R37) — time-sensitive, execute and post promptly.

---

## Phase 8 — Paper + reproducibility package (~2–3 weeks)

### T8.1 Reproducibility package — `[~]` *(2026-09-03: REPRODUCE.md + LICENSE-DATA.md written; archive upload + clean-machine run need the user)*
- **Goal:** anyone can rebuild every table.
- **End result:** `REPRODUCE.md` (one command per table), released derived-feature archives (FinCall-Surprise: Apache-2.0; MAEC features: CC-BY-SA-4.0), license audit note, environment lockfile verification on a clean machine.
- **Acceptance test:** clean-machine dry run reproduces Result Table 1 exactly and one GPU table within seed-variance bounds.
- **Subtasks:**
  - [x] `REPRODUCE.md` — one command per table (14 commands), data/feature/verification steps
  - [x] Feature-archive packaging — `ecvol release build` (2026-09-02): four deterministic zips under `data/release/` (FinCall 81.7 MB / MAEC 36.9 MB / Earnings25 11.6 MB / EC 7.5 MB; targets both conventions + anchors, splits, pooled features, identity/timing tables, data manifests, LICENSE-DATA + REPRODUCE inside), SHA-256 pinned in committed `data/manifests/release.json`
  - [ ] Archive upload — needs a data-host account (HANDOFF)
  - [x] License audit — `LICENSE-DATA.md` (per source: upstream licence, what is held, what is released, basis; two open items flagged)
  - [~] Clean-machine verification — 2026-09-02: fresh `git clone` + `uv sync` (CPU) in an empty directory passes the full gate (ruff, format, 320 passed / 4 skipped GPU-stack tests) and `ecvol runs verify` (21 result files); Table 1 regenerated byte-identically on this machine 2026-08-26 and 2026-09-02, Table 7 on 2026-09-02. **Finding:** default Windows git fails the checkout (`Filename too long` on an `ingest/` PDF) — `git clone -c core.longpaths=true` documented in REPRODUCE.md. A non-author machine with the data payload remains user-run
- **Notes:** —

### T8.2 Paper — `[~]` *(2026-09-02 (later): full prose-vs-CSV cross-check done — 12 mismatches fixed (wrong split on the one significant fusion cell, audio-only MLP range, 2021 not 2020 test year, 232 not 233 low-bitrate calls, Earnings25 dates Jul–Dec 2025, 17/2/6/9 code audit in §4, −2.9 not −2.0 past-vol ridge, anchor sensitivity ranges, Tiingo cross-check 47/50 min 0.976, stale “owed/pending” text in Limitations/Discussion, abstract “no degradation”); Earnings25 cited; anchor-sensitivity table added; 27 pp, 0 undefined refs, 0 TODO/pending; remaining = advisor/co-author review, venue)*
- **Goal:** the manuscript, framing per the DESIGN.md §4 gate decision.
- **End result:** draft with all tables generated by `ecvol report`; figures from `notebooks/`; venue selected; arXiv preprint.
- **Acceptance test:** every number in the paper traces to a run artifact; co-author/advisor review pass.
- **Subtasks:**
  - [ ] Draft (framing per gate decision)
  - [ ] Figures notebooks
  - [ ] Venue selection memo
  - [ ] arXiv submission
- **Notes:** *2026-08-19 — draft updated with all post-scaffold results:* Stage-3 audio (curated `tables/audio.tex` incl. the audio-shuffle block, WavLM 76.3% identity probe, gender analysis), Stage-4 fusion + consolidated grid (`tables/grid.tex` with signed Holm markers, per-year 2021 collapse), and the Stage-5 κ-gate failure as a headline finding (`tables/llm.tex` with bootstrap CIs; four null capability controls; no Result Table 5 by design — noted in the appendix). Abstract/intro/discussion/conclusion rewritten to the five-statement negative thesis; framing gate stated as *provisionally Path B* (missed at text/audio/fusion, Stage-5 blocked at validity), final confirmation deferred to Phase-7 (the only remaining `\pending{}`). Limitations rewritten incl. the three 2026-06-24 items below (MLP seed-instability, pastvol-ridge≠HAR, short-τ R² inflation) + single-rater κ / unmeasured ceiling. Appendix now inputs Result Tables 1–4 (`\clearpage` between files — 56 consecutive floats deadlocked the two-column output routine). Builds clean via `paper/build.sh` (20 pp, 0 undefined refs/citations). **Limitations to draft (from 2026-06-24 validation/gut-check):** (1) **MLP-head instability** — across-seed R² `seed_std` up to ~3.4 units; report median/IQR, not divergent-seed means (see T5.2 note). (2) **Past-vol baseline is not HAR** — the Stage-2/3 `[v_pre, rv_daily, rv_weekly, rv_monthly]` ridge (standardized, val-tuned α) overfits the temporal/COVID regime (−2.0 at τ=15) where the rigid OLS-HAR is robust (+0.21); arguably a small *finding* (structural HAR > unconstrained ridge under regime shift) worth a sentence, but at minimum the two must be labeled distinctly. (3) **Short-horizon R²_OOS is inflated by a noisy persistence baseline** (persistence MSE τ=3=1.17 vs τ=7=0.42) — frame short-τ gains honestly. **Validation assets available:** `notebooks/validate_results.py` independently reproduces all 9,696 targets to 1e-15, confirms leakage-free splits and clean features, and emits `data/results/figures/*.png` (R² heatmaps, identity gap, target dists, feature sanity) + `target_handcheck.csv` — feed these into the figures-notebook subtask (matplotlib still needs pinning into the lockfile at this phase). **Pre-submission (2026-07-19):** re-run the novelty scan (`paper/novelty_scan_2026-07.md`) shortly before submission — workshop proceedings and SSRN are under-indexed; must-cites R33–R40 already folded into refs.bib + related work; engage The Sound of Risk [R12] head-on in the discussion (done in draft: no identity controls there, our shuffle predicts its gains vanish).
  - 2026-08-26 — **restructured to the benchmark-first spine** (DECISIONS 2026-08-23 §1, 2026-08-26 §3). New title (`ecvol-bench: An Open, Identity-Controlled Benchmark…`); abstract/intro/contributions/RQs rewritten (RQ3 = reproduction question); `03-data.tex` is now *The ecvol Benchmark* (substrate audit §3.1 with the 2026-08-23 figures marked pending regeneration → Table 5R stub `tab:substrate`; corpora + identity reconstruction; targets under both conventions; call timing; splits with the 0.0%/91.3%/78.9% overlap contrast; release); Models = Stages 0–4 + *Reproduced prior models* (`sec:repro-protocol`); Results keep every Stage 0–4 number verbatim and add pending subsections (convention delta, Table 6R `tab:repro`, Earnings25 clean audio, lookahead); Discussion/Conclusion/Limitations rewritten; **Phase 6 / κ-gate excised entirely** (`tables/llm.tex` removed, DECISIONS 2026-08-26 §1). Builds clean (21 pp, 0 undefined refs). Open `\TODO`s: Earnings25 citation, finalise Discussion/Conclusion when 6R/9/7 land.

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

### TX4 — ecvol-live 50-call capture pilot — `[-]` *(**DEFERRED to future work 2026-08-23** — DECISIONS 2026-08-23 §4; discovery stage + artifacts retained, capture never built, replay window closed on 8/50 calls)*
- **Goal:** measure the feasibility numbers for the ecvol-live forward-collection design (`docs/fincall_methodology_and_successor.md` §4–5) on 50 current-season calls.
- **End result:** captured replay audio + at-collection-time metadata (ticker/CIK/company/fiscal period/call datetime+TZ/replay URL) for up to 50 Q2-2026 calls (~35 S&P 500 / ~15 S&P 400), every non-captured attempt reason-coded; then local Whisper+pyannote transcripts and T1.3 target joins; a pilot report with the three go/no-go numbers: automation coverage (% direct-MP3 / HLS / walled), per-call human minutes for the walled remainder, ASR WER vs Earnings25 overlaps.
- **Acceptance test:** pilot report exists with all three numbers **measured, not estimated**; every attempted call has either bytes-on-disk with SHA-256 manifest or a reason code; zero cash spent; no DESIGN.md change without a further DECISIONS.md entry.
- **Subtasks:**
  - [x] Discovery: universe snapshot (S&P 500 + S&P 400, sourced+dated) × Q2-2026 earnings calendar → candidate list with call dates *(2026-08-13: 903 tickers, 806 candidates Jul 1–Aug 13)*
  - [ ] Stratified 50-call sample (~35/15, seeded) + replay-URL location per call *(sample drawn 2026-08-13, seed 20260813, dates Jul 16–Aug 12; URL location pending)*
  - [ ] Capture: direct-MP3 and HLS routes (polite rates); reason-code the walled remainder
  - [ ] Normalize to the T4.1 store format (16 kHz mono FLAC + SHA-256, original bytes kept)
  - [ ] Transcribe + diarize (may lag capture; 50-call ETA gate applies)
  - [ ] Target join via the T1.3 pipeline (real timestamps — after-hours rule finally applicable)
  - [ ] Pilot report → season-scale go/no-go
- **Notes:** DECISIONS 2026-08-13. Capture-first: bytes decay, everything downstream can wait. Scripts land in `src/ecvol` (they are the release deliverable, not scratch).
