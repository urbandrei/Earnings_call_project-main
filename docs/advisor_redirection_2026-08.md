# Advisor redirection (Aug 2026): assessment, literature-code audit, and open decisions

**Created:** 2026-08-23
**Status:** Analysis only. **No DESIGN.md / TASKS.md / DECISIONS.md changes were made.** Every
promotion proposed here requires its own DECISIONS.md entry first, per CLAUDE.md.
**Trigger:** advisor meeting (a few days before 2026-08-23) proposing a course adjustment.

## 0. Provenance key

Findings below are tagged so future sessions know what to trust without re-deriving:

- **[V]** — verified directly in this session by fetching the artifact or computing on local data.
  HTTP status or exact counts given.
- **[A]** — reported by a research subagent, not independently re-verified here. This project has
  a recorded incident (JOURNAL 2026-08-13) where a subagent claimed a dead source was alive, so
  **[A]** items should be re-checked before any of them enters a paper claim.

---

## 1. The proposal

As relayed from the advisor meeting, the redirection has two halves:

1. **Instead of building our own dataset**, take the existing corpus, clean it up, augment it to
   better suit the task, and build a system around it that resolves the issues for future research.
2. **Find current cutting-edge papers, run them under the new controls**, and showcase the
   difference in performance — "a literature review using the new tooling we create."

## 2. Assessment summary

**The direction is sound, but the two halves carry very different risk.**

### 2.1 Half A (clean up + augment + system) — largely already built

This is mostly a *reweighting*, not new work. DESIGN §4 already lists "an open multimodal benchmark
(ecvol-bench)" as planned contribution #1. Promoting it from supporting infrastructure to headline
contribution is cheap because the assets exist **[V]**:

- 10,270 lines of Python across 51 modules, 281 tests, 34 CLI commands
- leakage-proof temporal / ticker-disjoint / combined splits as committed CSVs, CI-enforced
- 9,696 targets independently reproduced to 1e-15; 99.0% price join on FinCall
- corpus-wide features across five model families (BGE-M3, FinBERT, surface, eGeMAPS, WavLM,
  emotion2vec+)
- eval harness with Diebold–Mariano, Holm correction, cluster bootstrap, identity-control suite

**The crown jewel is the identity reconstruction.** FinCall ships with no ticker, company, CIK,
date, time, fiscal period, or speaker names — four fields per record. Recovering 92.9% overall /
95.3% on the earnings cohort took a 571-line pipeline plus an 80-row manual override table plus
hand audits. Nobody else has that table, and there is a usable narrative hook: *you cannot run the
identity controls that expose the field's confound without first reconstructing the identity the
dataset omitted.*

**The strongest argument for the pivot, which was not made in the meeting:** it defends the
negative results against the most obvious reviewer attack. The draft currently reports audio as
inert; a reviewer will answer "your audio is inert because it is 40 kbps garbage." Under the
benchmark framing that objection becomes the thesis rather than a wound.

**Hard ceiling:** cleanup cannot repair the 40 kbps uniform transcode. Every acoustic feature sits
on compression-damaged audio, permanently. Only new audio fixes that.

**The one augmentation not in any current plan:** retrofit real call timestamps from EDGAR 8-K
Item 2.02 filings. There is currently *no* call time-of-day anywhere (only 3.4% of transcripts
mention a clock time), so every target rests on a uniform assume-after-hours fallback that the
paper concedes is not sensitivity-checkable. EDGAR timestamps are free, public domain, and
redistributable, and the SEC ticker table is already cached. This closes DESIGN §10 risk #7.

### 2.2 Half B (run cutting-edge papers under our controls) — the trap, and the fix

Three problems:

1. **Reproduction is the thing that is broken.** DESIGN §3.4 already cites the ~14%-exact-reproduction
   finding. Section 4 below quantifies exactly how bad it is in *this* literature.
2. **A design confound.** Running prior models on our cleaned corpus changes the dataset *and* the
   protocol simultaneously, so "the model fails" is indistinguishable from "the model does not
   transfer." Each model must run on its own original dataset under the new controls.
3. **The point can largely be made without the re-runs.** The identity probe (89.5% text / 76.3%
   audio) and same-ticker shuffle are properties of the *representations*, not of any head.

**Fix: cap it at 2–3 code-released models** and lead with the substrate audit (§5), which requires
reproducing nobody.

### 2.3 Cost and sequencing

The July deck promised a preprint by 2026-08-17. As of 2026-08-23 Phase 7 has not started and T6.2
is still blocked. The pivot adds roughly 5–8 weeks to a draft that is 20 pages and complete except
for the lookahead section.

**Hedge: do Phase 7 first, via Earnings25** (Zenodo, CC-BY-4.0, ~500 Q4-2025 S&P 500 calls, 498 h).
Q4-2025 calls were held Jan–Feb 2026, post-cutoff for the Qwen2.5 stack. That closes the last
`\pending{}` and leaves a complete submittable paper *before* betting anything on the pivot.
Caveat: Earnings25's audio provenance is undisclosed and per-call metadata completeness is
unverified — budget a day to confirm it carries what the price joins need.

### 2.4 TX4 (self-collection): defer

- `src/ecvol/collect/` contains only `discovery.py` (185 lines). **No capture module exists** — no
  replay downloader, no transcription stage **[V]**.
- Pilot sample spans 2026-07-16 .. 2026-08-12, drawn 2026-08-13, capture never ran. As of
  2026-08-23, **8 of 50 calls are past the 30-day replay floor; oldest is 38 days** **[V]**.
- Earnings25 covers the Phase-7 need TX4 was going to serve. Next clean season is Q3 2026
  (Oct–Nov), past any reasonable submission date.

### 2.5 Venue implication

A cleaned corpus + released identity metadata + control suite + strong baselines + a bounded
reproduction study is squarely a **NeurIPS Datasets & Benchmarks** paper, and LREC-COLING takes
resource papers. Both are stronger than the FinNLP workshop currently on the DESIGN §4 list.

---

## 3. A defect found in our own design contract

**DESIGN.md §5.3 misattributes the horizon convention.** It states targets are computed over
"τ ∈ {3, 7, 15, 30} *trading* days (Qin & Yang convention, for comparability)."

Verified against the source PDF (`legacy/papers/.../What You Say and HowYouSayIt Matters.pdf`) **[V]**:

> "We choose different τ values, including 3, 7, 15, 30 **calendar days** to evaluate the short-term
> and long-term effectiveness of volatility prediction."

Qin & Yang used **calendar** days. Every downstream paper inherits their released label files, so
the entire published leaderboard is on calendar-day windows while `targets.py` computes trading-day
windows. At τ=30 that is ~30 versus ~21 sessions.

**This does not invalidate our targets** — they are internally consistent and unit-tested. It
invalidates the *comparability claim*, which is exactly what a reproduction study rests on.

Requires a DECISIONS.md entry. Options in §7.

---

## 4. Literature code-availability audit

Roughly twenty papers checked. **Five repos are alive and contain runnable code.**

### 4.1 Runnable

| Paper | Repo | Verified state |
|---|---|---|
| **Same Company, Same Signal** (Findings ACL 2025) | `piqueyd/Same-Company-Same-Signal` | MIT, 11,723 KB, **pushed 2026-08-13**, 2 stars. Ships `DEC.csv`, `EC/{EC_earnings.csv, augmented_EC_earnings_history.csv}`, `MAEC/{MAEC15,MAEC16}_earnings.csv` + augmented histories, `SS.ipynb`, `data_provider/` **[V]** |
| **HTML** (WWW 2020) | `YangLinyi/HTML-...-Volatility-Prediction` | 27 files, ~50 KB PyTorch, 63 stars. `Model/Sentence-Level-Transformer/{run_cpu,run_gpu}.py` + vendored `transformers/`; 3 experiment notebooks; token-level bert-as-service + HF-RoBERTa stubs. No requirements.txt **[V]**. Last *code* commit 2021-07-02; the 2024-01-05 commits are README-only **[A]** |
| **KeFVP** (Findings EMNLP 2023) | `hankniu01/KeFVP` | Python, 4,415 KB, 188 tree entries (94 non-pycache), pushed 2024-01-28. Full `pretrain/` (KePt) + `kefvp/` (Autoformer-style conditional time-series) + `price_data/` label CSVs **[V]** |
| **DialogueGAT** (Findings EMNLP 2022) | `sangyx/DialogueGAT` | MIT, 3,974 KB, pushed 2023-06-12, 5 stars. PyTorch + DGL **[V]**. τ ∈ {3,7,15} only; corpus must be rebuilt incl. Seeking Alpha speaker annotations **[A]** |
| **Sawhney** (ACM MM 2020) | `midas-research/multimodal-financial-forecasting` | 205 KB, pushed 2020-10-26, 20 stars **[V]**. Complete pipeline but TF 2.1 + Keras 2.3.1; single-day commit dump **[A]** |

### 4.2 Not runnable — the negative space

| Paper | Finding |
|---|---|
| **ECC Analyzer** (ICAIF 2024) | No code anywhere. Confirmed closed: GPT-4 Turbo-2024-04-09 + OpenAI `text-embedding-3-small` **[A]**. *This is the frontier claim Phase 6 was built to test.* |
| **RiskLabs** (2024 / ICDMW 2025) | No code. GPT-4 + GPT-3.5-Turbo core to two of four modules; news corpus never named **[A]** |
| **The Sound of Risk** (2025) | `soundai2016/sound_risk` loads (913 KB, pushed 2025-08-26, 0 stars) **[V]** but holds **only README + requirements + figures — zero source** **[A]**. Code promised "upon acceptance"; 1,795-call audio corpus proprietary; sentiment from DeepSeek-R1 671B **[A]**. *Primary empirical comparison target per DESIGN §3.0 — cannot be reproduced.* |
| **DeFVP** (ICME 2024) | `hankniu01/DeFVP` exists, **size 0 KB, created == pushed 2024-05-06, never pushed to** **[V]**. Code promised via a named repo, never uploaded. |
| **ECHO-GL** (AAAI 2024) | Repo loads (90,216 KB, pushed 2024-02-26, 16 stars) **[V]**; README states the code "can not run at present" **[A]**. Target is movement direction, not volatility **[A]** |
| **NumHTML** (AAAI 2022) | Promises "all code and datasets will be released on GitHub"; **no URL in the paper**; numeral code absent from the HTML repo **[A]** |
| **GNA-Vol** (IJCNLP-AACL 2023) | Only artifact URL `gen-numattach.nlpfin.com` → **404** **[A]** |
| **AMA-LSTM** (NAACL 2024 Ind.) | No official repo; the one GitHub hit is a 10-byte README, unattributed **[A]** |
| **AT-FinGPT** (FRL 2025) | ScienceDirect 403, paywalled. Nothing verified **[A]** |
| **MR-QA** (IJCAI 2020) | No code. Uses its own 6,494-call Seeking Alpha text-only corpus, τ ∈ {3,7,15} — **not comparable** to the 576-call line **[A]** |
| **Chen et al. NAM/ECNum** (CIKM 2021) | No code found, but the agent could not rule out links behind pages it could not open **[A]** |
| **Qin & Yang** (ACL 2019) | **Data only** — `GeminiLn/EarningsCall_Dataset` (15,047 KB, 158 stars, pushed 2022-08-04); 3 sample dirs + README **[V]**. Drive folder verified to actually list `ACL19_Release.z01/.z02` **[V]**. No MDRM implementation anywhere **[A]** |

**Two agents independently reported: no 2025–2026 multimodal earnings-call *volatility* model with
public runnable code exists, and no leaderboard exists for this task** **[A]**.

### 4.3 Tools and resources newly surfaced (not in DESIGN §13)

- **FinTrust** — Yang et al., ACL 2023 short, *"Measuring Consistency in Text-based Financial
  Forecasting Models."* `yingpengma/FinTrust`, Python, pushed 2024-01-22, 16 stars, not archived
  **[V]**. Ships four perturbation generators (`generate_{add,neg,sym,tra}_consis.py`) + earnings-call
  data at 3/7/15/30 **[V]**. **By the HTML author.** Venue precedent for the "we built a diagnostic
  and the field fails it" paper shape; perturbation families are a complementary control axis to
  our identity shuffle.
- **SPGISpeech 2.0** — +3,780 h earnings calls with speaker tags, **CC BY-SA 4.0** **[A]**.
  DESIGN §5.1 skipped v1 specifically over its academic-only license; that decision is now stale.
- **EarningsInOne** — S&P 1500 2022–2025, announced, repo empty ("coming soon") **[A]**.
  ecvol-live remains unduplicated.
- **"Acoustic Camouflage"** (arXiv 2604.14619) — on MAEC, adding acoustic features *degraded* recall
  66.25% → 47.08%; no code **[A]**. Independent corroboration of our Phase-4 audio-inert finding.
- **Netspar 2025-15** — TRILLsson vs W2V2 speech embeddings on S&P 500 calls; W2V2 54–59% direction,
  TRILLsson no predictive power; no code **[A]**. Closest external comparison to our WavLM probe.
- **MultiFinBen** (ACL 2026) — `xueqingpeng/MultiFinBen` verified live **[A]**; audio tasks include an
  MDRM earnings-call task, but ASR/summarization, not log-RV regression.

### 4.4 Unresolved

**"Volatility Prediction with Audio and Structure-Aware Text Embeddings in Earnings Conference
Calls"**, IEEE ICDMW 2025 — the only 2025–26 title that exactly matches our task. Paywalled; authors,
DOI, and abstract could not be retrieved **[A]**. **Needs a manual IEEE Xplore lookup via university
access before we can claim the space is clear.**

---

## 5. The substrate audit — the highest-value finding

Discovered while verifying the repos. All computed on public files or local data. **All [V].**

### 5.1 The literature shares one physical set of labels

`VolTAGE/cross-modal-attn/test_split_SeriesSingleDayVol3.csv` and
`KeFVP/price_data/test_split_SeriesSingleDayVol3.csv` are **byte-identical** — same values to the last
decimal. This lineage does not merely share a convention; it shares one set of files inherited from
Qin & Yang and never recomputed. (KeFVP's README explicitly acknowledges HTML and VolTAGE.)

### 5.2 Split integrity across all three canonical benchmarks

| Benchmark | calls (tr/va/te) | test calls whose ticker is in train | embargo tr→va | va→te |
|---|---|---|---|---|
| EC (Qin & Yang, 2017) | 392 / 56 / 112 | **80.4%** (90/112) | **0 days** | **0 days** |
| MAEC-15 | 535 / 76 / 154 | **44.2%** (68/154) | **0 days** | **0 days** |
| MAEC-16 | 980 / 140 / 280 | **55.7%** (156/280) | **0 days** | 3 days |

For contrast, our own splits:

| Split | test calls whose ticker is in train |
|---|---|
| `fincall_temporal` | 91.3% (439/481) |
| `fincall_ticker_disjoint` | **0.0%** (0/484) |
| `maec_temporal` | 78.9% (400/507) |
| `maec_ticker_disjoint` | **0.0%** (0/514) |

**Honest framing — this matters.** Temporal splits *always* permit ticker reuse; ours do too. The
defensible claim is **not** "the literature has overlap." It is:

1. **No paper in this lineage ever reports a ticker-disjoint condition**, so the confound is never
   measured. Our 0.0% column is the missing comparison.
2. **The zero embargo is an unambiguous defect.** EC's train segment ends 2017-08-03 and validation
   *starts* 2017-08-03 — 4 calls in train on that date (CLX, PWR, FLT, chd), 14 in val. Same pattern
   at the val/test boundary (2017-10-24). With targets running to 30 days, train-call target windows
   overlap val/test target windows directly. DESIGN §5.4 mandates a 30-trading-day embargo precisely
   for this. No identical (ticker, date) call appears in two splits — the leakage is via target
   windows, not duplicated rows.

### 5.3 Label defects in the shared files

- **137 exact zeros across 16,800 cells (0.82%)** in the single-day log-vol series
  (`SeriesSingleDayVol3`: train 94/11,760, val 9/1,680, test 34/3,360). In log space 0.0 means
  RV = 1.0 against a typical −4 to −8. The headline `Avg_Series_WITH_LOG` targets have **zero**
  zeros, so scope this claim to the single-day series. *Interpretation (zero-variance day vs. missing
  data) is not yet settled — needs a call before it goes in a paper.*
- **KeFVP's MAEC label files interleave four binary classification labels at exactly
  τ ∈ {3, 7, 15, 30} among 26 adjusted-close price columns**, all under identical `future_label_N`
  naming. Verified 154/154 rows, zero exceptions: `future_label_{3,7,15,30}` are always in {0,1};
  the other 26 are always price-like. Anyone slicing that block as a price series silently ingests
  four binary values as prices.

### 5.4 The bridge to our targets already works

KeFVP's shipped MAEC labels key on `text_file_name` in the form `20151029_ALXN` — **our own MAEC
`call_id` format**. They join to our recomputed targets at **99.4% (153/154)** on a direct key match
with no fuzzy logic (single miss: `20151105_HE`, a price-coverage exclusion). TX3 is immediately
actionable: the literature's labels can be diffed against our §5.3 targets call-by-call on the exact
set KeFVP reports on.

---

## 6. Repo-state facts relevant to the decision (all [V])

- **10,270 LOC / 51 modules / 281 tests / 34 CLI commands.**
- **`artifacts/` holds only 8 diagnostic files — no run payload.** DESIGN §8.2 specifies every run
  writes `artifacts/runs/<run_id>/` with resolved config, config hash, git SHA, seeds, env
  fingerprint; §12 promises every paper number comes from `ecvol report` over those artifacts.
  *Not yet checked whether `report` currently sources from `data/results/` instead.* Under a
  benchmark framing this is load-bearing — byte-identical regeneration is the claim reviewers test.
- **`configs/` holds only `example.yaml`**, against a DESIGN §8.2 contract of one committed config
  per experiment.
- **MAEC ships no audio**, so `result_table_4` audio and fusion rows are empty for it — the entire
  "audio is inert" finding rests on one corpus whose every file is a 40 kbps transcode.
- **Housekeeping risk:** `t2019.json`, `t2020.json`, `t2021.json` at repo root are 137 MB of raw
  FinCall transcripts, duplicated from `D:\ecvol-data\raw\fincall\`, **untracked *and* not covered by
  `.gitignore`**. A stray `git add -A` would commit third-party scraped transcripts into a repo whose
  licensing posture is about to become a headline contribution.
- **Legacy asset:** `legacy/4-Reproduce_HTML.ipynb` reimplements HTML inline in plain PyTorch
  (`SelfAttention`, `TransformerBlock`, `RTransformer`; imports only numpy/pandas/sklearn/torch/tqdm)
  over precomputed sentence embeddings on MAEC. HTML can therefore be run as an `ecvol` head on our
  embeddings, targets, and splits rather than as an external repo reproduction.

---

## 7. Decisions taken (2026-08-23)

All twelve resolved by the user in-session. Recorded in full in **DECISIONS.md 2026-08-23**; this is
the index.

| # | Question | Decision |
|---|---|---|
| 1 | Scope of the pivot | **Full reframe** — benchmark/resource paper is the headline |
| 2 | Venue | **ACL** (next cycle, via ARR) |
| 3 | Horizon convention | **Ship both** calendar-day and trading-day targets |
| 4 | TX4 self-collection | **Deferred** to future work |
| 5 | Reproduction slate | **All five** runnable repos |
| 6 | Phase 7 data source | **Earnings25**, now |
| 7 | Earnings25 clean-audio re-run | **Yes** |
| 8 | Call-timestamp retrofit | **Yes** — check SCSS `beforeAfterMarket` first, then EDGAR 8-K |
| 9 | Reproducibility debt | **Close it now**, ahead of further science |
| 10 | Release scope | **Everything** (incl. the identity reconstruction table) |
| 11 | T6.2 / κ-gate | **Phase 6 dropped entirely**, replaced by the reproduction study |
| 12 | Root JSON housekeeping | **Added to `.gitignore`** (not deleted) |

### 7.1 Consequences enacted the same day

- `.gitignore` — the three root transcript JSONs are now un-committable (verified via `git check-ignore`).
- `DECISIONS.md` — one dated entry, twelve numbered sub-decisions with rationale and rejected alternatives.
- `TASKS.md` — new `[-]` status introduced (dropped/deferred); Phase 6 closed with a banner;
  T6.2/T6.3/TX4 → `[-]`; T7.1 source changed to Earnings25; **new Phase 6R** (T6R.1 substrate audit,
  T6R.2 reproduction study) and **new Phase 9** (T9.1 dual-convention targets, T9.2 timestamp
  retrofit, T9.3 repro-debt closure, T9.4 clean-audio re-run).
- `HANDOFF.md` — **rater 2 cancelled**, with the two superseded T6.2 entries retained for the record.

### 7.2 Still open

1. **Confirm the κ-gate result stays in the paper.** Recommendation on record: keep it as a short
   reported negative finding (it is real, novel — ECC Analyzer performs no human validation of
   extraction quality at all — and already drafted in `tables/llm.tex`); drop only the corpus run and
   RQ3's predictive arm. Excising it entirely is the alternative and would need saying.
2. **ACL cycle dates + ARR anonymity policy**, which may conflict with DESIGN §4's "arXiv preprint
   precedes any submission."
3. **The ICDMW 2025 paper** in §4.4 — exact title match, paywalled, needs a manual Xplore lookup.
4. **Sentinel-zero interpretation** (§5.3) — zero-variance day vs missing data — before it enters a
   paper claim.
5. **DESIGN.md amendments** authorised by DECISIONS 2026-08-23 but not yet written: §1 (RQ3
   withdrawn), §4 (contributions reordered; framing gate superseded), §5.3 (horizon convention),
   §6 Stage 5 (dropped), §9 (phase plan).

---

## 8. Sources

**Repos verified this session:** github.com/YangLinyi/HTML-Hierarchical-Transformer-based-Multi-task-Learning-for-Volatility-Prediction ·
github.com/hankniu01/KeFVP · github.com/piyushkhanna7/VolTAGE (github.com/piyushkhanna00705/VolTAGE
redirects to it — same repo) · github.com/GeminiLn/EarningsCall_Dataset ·
github.com/Earnings-Call-Dataset/MAEC-... · github.com/Tizzzzy/FinCall-Surprise ·
github.com/yingpengma/FinTrust · github.com/piqueyd/Same-Company-Same-Signal ·
github.com/hankniu01/DeFVP · github.com/pupu0302/ECHOGL · github.com/soundai2016/sound_risk ·
github.com/midas-research/multimodal-financial-forecasting · github.com/sangyx/DialogueGAT

**Papers:** aclanthology.org/P19-1038 (Qin & Yang, calendar-day quote) ·
aclanthology.org/2023.findings-emnlp.770 (KeFVP) · aclanthology.org/2023.acl-short.60 (FinTrust) ·
aclanthology.org/2025.findings-acl.946 + arxiv.org/abs/2412.18029 (Same Company, Same Signal) ·
arxiv.org/abs/2404.18470 (ECC Analyzer) · arxiv.org/abs/2404.07452 (RiskLabs) ·
arxiv.org/abs/2508.18653 (Sound of Risk) · arxiv.org/abs/2604.14619 (Acoustic Camouflage) ·
arxiv.org/abs/2508.05554 (SPGISpeech 2.0) · arxiv.org/abs/2606.29734 (EarningsInOne) ·
DOI 10.1109/ICME57554.2024.10688325 (DeFVP)

**Local:** `legacy/papers/papers with results from our datasets/` (14 PDFs of the candidate set) ·
`legacy/4-Reproduce_HTML.ipynb` · `docs/fincall_methodology_and_successor.md` ·
`data/splits/*.csv` · `data/maec/targets.parquet`
