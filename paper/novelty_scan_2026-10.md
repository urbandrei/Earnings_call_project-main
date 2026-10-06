# Novelty & frontier scan — 2026-10-06 (pre-submission re-scan)

**Question:** has anything been published or posted between ~2026-07-01 and 2026-10-06 that
scoops, overlaps with, or must be cited by the *current* (benchmark-first) paper? Scope is the
six contributions in `sections/01-introduction.tex`: (1) ecvol-bench, (2) substrate audit,
(3) baselines + identity-control suite, (4) measured call timing, (5) reproduction study +
code audit, (6) post-cutoff evaluation. The old Q2 (LLM structured features) is no longer a
contribution and is not re-scanned. **Method:** 20 web searches over five angles (identity
confounds; volatility benchmarks/datasets/multimodal models; reproductions of HTML / KeFVP /
DialogueGAT / SCSS / Sawhney 2020; lookahead / post-cutoff evaluation; call timing and label
anchoring), plus status re-checks of the July must-cites. Every claim below traces to a page
fetched in this session (arXiv abs/HTML, ACL Anthology). Items whose page could not be fetched
are flagged *unverified* and carry no verdict weight. Single-pass (no adversarial
verification panel, unlike July) — treat CLEAR verdicts as moderate-certainty.

## Bottom line

**No scoop. Two new MUST-CITE OVERLAPs, one of which needs a wording change.**

- **MM-FinEval** (arXiv 2609.38523, 2026-09-29; the FinCall-Surprise group) is a new open
  (CC BY 4.0) multimodal earnings-call benchmark *with raw audio and post-call 3/7/15/30-day
  volatility labels*, plus ticker/date fields. It does not do what ecvol-bench does (no
  stated splits/embargo, no past-volatility baseline, no identity controls), but it falsifies
  our introduction's sentence that FinCall-Surprise is "the corpus with … the only raw
  audio", and it is exactly the kind of benchmark our substrate audit argues against.
- **Fast Numbers, Slow Language** (arXiv 2606.29734, 2026-06-29; the *SCSS authors*) builds
  an S&P 1500 2022–2025 corpus with **8-K release timestamps** plus news-derived
  "FirstActual" timestamps, a Ticker-Mean baseline, and cites SCSS for the identity confound.
  Overlaps contribution 4 (measured timing) in *method*, but its targets are returns, not
  volatility, and it does not audit prior volatility label sets. It was posted just before
  the window and is absent from the July scan.

The lookahead literature kept filling in (four new papers, all generic LLM/backtest
methodology, none on earnings-call volatility) — our post-cutoff claim, scoped as in July,
still holds. No reproduction/replication of the earnings-call volatility lineage surfaced.
No follow-up from anyone *other than* the SCSS authors running identity controls on
earnings-call volatility surfaced.

## Status of earlier must-cites (re-checked)

| Item | Status found 2026-10-06 | Action |
|---|---|---|
| Same Company, Same Signal (2412.18029) | arXiv still v1 (2024-12-23), **but published**: Findings of ACL 2025, pp. 18403–18422, Anthology ID 2025.findings-acl.946. The July scan's "still appears to be an unreviewed arXiv v1" was wrong. | Change `scss2024` from `@misc` to `@inproceedings` (Findings ACL 2025); drop any "unreviewed" wording. |
| FinCall-Surprise (2510.03965) | arXiv v1 only; **published ACL 2026 (long), pp. 13357–13370, 2026.acl-long.610** | Change `fincall2025` to the ACL 2026 Anthology entry. |
| The Memorization Problem (2504.14765) | No change: v1 2025-04-20, v2 2025-12-15; no journal-ref on arXiv. | None. |
| Look-Ahead-Bench (2601.13770) | No change: v1 2026-01-20 only, single author (Benhenda), no journal-ref. | None (vendor-affiliation caveat stands). |
| LAP / Detecting Lookahead Bias (2512.23847) | No change since July: v2 2026-06-12. | Ensure bib points at v2. |
| The Sound of Risk (2508.18653) | Abs page shows only the 2025-08-26 submission; no new version or venue seen. | None. |
| Earnings25 (2607.23813) | v1 2026-07-26 (Jiang, Zhou, Wadhawan + 7). ASR benchmark; no market targets. Already cited (`jiang2026earnings25`). | Confirm bib has the arXiv ID. |

## Verdicts by contribution

### C1 — ecvol-bench (open multimodal volatility benchmark): **MUST-CITE OVERLAP, not scooped**

**MM-FinEval** (Shu, Liu, Zhang, Hu, Zhao, Huang, Du; arXiv 2609.38523, v1 2026-09-29):
2,045 S&P 500 earnings calls, 2019–2022, each with transcript, slides and *full audio*,
web-scraped from company websites, CC BY 4.0; 12 task labels including `vol_fut_{3,7,15,30}d`
(post-call realised volatility) and `vol_past_{3,7,15,30}d`; 19 baseline MLLMs; MAE as the
default metric; the paper reports annual MAE trends 2019–2022. From the fetched HTML: no
train/test split procedure is stated, no embargo, no past-volatility (vol_past → vol_fut)
baseline, no ticker/identity analysis. It records `COMPANY_SYMBOL` and `DATE` fields (stated
for bond-return matching). Its headline is that small any-to-any models using all three
modalities do well — a positive multimodal claim our controls would challenge, since its own
`vol_past_*` columns make the persistence baseline trivially available and it is not
reported.

Why not a scoop: no leakage-proof splits, no econometric baselines, no identity controls, no
substrate audit, no timing correction — all of ecvol-bench's distinguishing properties are
absent. Why must-cite: it is a same-period, same-group, audio-bearing volatility benchmark
that reviewers will know (the FinCall-Surprise group's ACL 2026 paper is our primary corpus).
**Wording consequence:** "the corpus with the best license and the only raw audio,
FinCall-Surprise" (intro ¶3) is no longer true as written.

### C2 — Substrate audit: **CLEAR**

No new audit of shared earnings-call label/split files (embargo, ticker overlap, horizon
convention) found. Nearest new work is generic: *Temporal Leakage in Financial News NLP*
(arXiv 2608.17223, EMNLP 2026; random vs chronological splits inflate MCC 1.1–6.5× on news
direction / M&A classification) — no earnings calls, no volatility, no identity. MM-FinEval
(above) is a fresh instance of the unaudited-substrate pattern, not an audit.

### C3 — Baselines + identity-control suite: **CLEAR**

SCSS (now confirmed Findings ACL 2025) remains the sole earnings-call identity-confound
prior. Its only visible follow-up is by the same authors — *Fast Numbers, Slow Language*
(2606.29734), which uses a Ticker-Mean baseline and cites SCSS's within- vs cross-ticker
similarity (0.80 vs 0.67) — but it predicts returns and runs no probe/shuffle/ticker-disjoint
controls. *Lie to me* (arXiv 2609.13893, 2026-09-12) uses firm-grouped 5-fold CV for an
audio+LLM evasiveness model predicting late SEC filings (AUROC ≈ 0.89, 1,039 calls) —
company-grouped evaluation, but not volatility. *KTD-Fin* (arXiv 2605.28359, May 2026, pre-
window) masks identifiers/dates for LLM trading agents on CSI300 and finds returns are mostly
passive exposure — conceptually aligned (identity/memory controls), different task.

### C4 — Measured call timing: **MUST-CITE OVERLAP (method), not scooped**

*Fast Numbers, Slow Language* (Yu, Liu, Zhang, He; arXiv 2606.29734, 2026-06-29) obtains
earnings-release timestamps from 8-K filings and news timestamps from the EODHD API, defines
FirstActual vs NextOpen anchors, and argues that "without precise event-level timestamps,
numeric signal exhausted within minutes is indistinguishable from language signal maturing
overnight"; corpus "EarningsInOne", 5,428 events / 701 tickers, train 2022 / val 2023 / test
2024–2025; targets are directional log returns. From the fetched HTML it does **not** discuss
the timing of prior volatility datasets (EC, MAEC, DEC) — our specific finding (the uniform
after-hours anchor misplaces day 0 for 53–59% of calls in released volatility label sets) is
unclaimed. Must-cite as concurrent independent use of 8-K timestamps; it also strengthens our
motivation (the SCSS authors themselves moved to precise timestamps).

### C5 — Reproduction study + code audit: **CLEAR**

No reproduction, replication, or re-evaluation of HTML, KeFVP, DialogueGAT, VolTAGE, MRQA,
Sawhney 2020, or SCSS found in the window. Adjacent finance-wide items surfaced only via
search snippets (a 2026 pre-registered survey on finance reproducibility; an LLM-pipeline
replication of JF/JFE/RFS papers) — not fetched, not earnings-call NLP, no verdict weight.

### C6 — Post-cutoff evaluation: **OPTIONAL CITES only; CLEAR**

New lookahead/point-in-time work, none on earnings-call volatility or frozen-pipeline
post-cutoff evaluation:

- *HindsightBench* (Jia; arXiv 2607.18867, v1 2026-07-21, v2 2026-08-10): black-box audit
  of parametric hindsight across 15 models; effective cutoffs precede reported dates by up to
  8 months — relevant to how we define "post-cutoff" (our cutoff margin).
- *Temporal Leakage in LLM Backtesting* (Zhang & Stadie; arXiv 2608.02985, 2026-08-04):
  standard contamination checks fail even on post-cutoff questions; proposes cutoff-boundary
  and matched-control adjusted scores.
- *Scaling Point-in-Time Language Models* (Kelly, Malamud, Schwab, Xu; arXiv 2607.11889, v1
  2026-04-24, rev. 2026-07-17): up to 4B-param models, monthly checkpoints 2013–2024.
- *PALM: Point-in-Time Adaptation* (Lee et al.; arXiv 2609.30316, 2026-09-02): LoRA on
  pre-decision-date text instead of annual PIT retraining.

## New references

| Ref | arXiv / URL | Date | Verdict | Reason |
|---|---|---|---|---|
| MM-FinEval (Shu, Liu, Zhang, Hu, Zhao, Huang, Du) | https://arxiv.org/abs/2609.38523 | 2026-09-29 | **MUST-CITE OVERLAP** | Open audio-bearing earnings-call benchmark with 30-day vol labels; no splits/embargo/past-vol baseline/identity controls; breaks our "only raw audio" sentence |
| Fast Numbers, Slow Language (Yu, Liu, Zhang, He) | https://arxiv.org/abs/2606.29734 | 2026-06-29 | **MUST-CITE OVERLAP** | SCSS authors; 8-K release timestamps + Ticker-Mean baseline; returns not volatility; no audit of prior label timing |
| SCSS — venue update | https://aclanthology.org/2025.findings-acl.946/ | Jul 2025 | MUST-CITE (update) | Already cited; now Findings ACL 2025, not arXiv-only |
| FinCall-Surprise — venue update | https://aclanthology.org/2026.acl-long.610/ | Jul 2026 | MUST-CITE (update) | Already cited; now ACL 2026 long |
| Temporal Leakage in Financial News NLP (Xue et al.) | https://arxiv.org/abs/2608.17223 | 2026-08-18 (v2 09-06), EMNLP 2026 | OPTIONAL CITE | Split-leakage audit in financial NLP; news/M&A, not calls/volatility |
| Lie to me (Chen, Zhang) | https://arxiv.org/abs/2609.13893 | 2026-09-12 | OPTIONAL CITE | Audio+LLM on earnings calls with firm-grouped CV; target = late SEC filings |
| HindsightBench (Jia) | https://arxiv.org/abs/2607.18867 | 2026-07-21 (v2 08-10) | OPTIONAL CITE | Effective cutoffs earlier than reported — supports cutoff-margin choice |
| Temporal Leakage in LLM Backtesting (Zhang, Stadie) | https://arxiv.org/abs/2608.02985 | 2026-08-04 | OPTIONAL CITE | Contamination-check methodology for backtests |
| Scaling Point-in-Time LMs (Kelly, Malamud, Schwab, Xu) | https://arxiv.org/abs/2607.11889 | 2026-04-24 (rev. 07-17) | OPTIONAL CITE | PIT model family; contamination-free arm candidate |
| PALM (Lee et al.) | https://arxiv.org/abs/2609.30316 | 2026-09-02 | OPTIONAL CITE | PIT adaptation via adapters |
| KTD-Fin (memory-controlled trading benchmark) | https://arxiv.org/abs/2605.28359 | May 2026 (pre-window) | OPTIONAL CITE | Identifier/date masking for LLM agents; CSI300 |
| Can Open-Weight Models Compete on Financial Text Comprehension? (Spörer) | https://arxiv.org/abs/2608.08634 | 2026-08-09 | NOT RELEVANT | Annual-report QA |
| Making Alternative Data Work (Kwon et al.) | https://arxiv.org/abs/2609.11607 | 2026-09-10 | NOT RELEVANT | Revenue forecasting from alt data |
| Listen Closely / FinVoc2Vec (Ewertz, Knickrehm, Nienhaus, Reichmann), J. Accounting Research 2026 | https://onlinelibrary.wiley.com/doi/10.1111/1475-679X.70015 | 2026 | *unverified* (403; mirror refused connection) | Search snippets only: domain-adapted vocal-tone model linked to returns. Worth a manual check as a positive audio claim (returns, not RV) |

## Recommended actions

1. **Fix the intro claim (wording change, needs author sign-off):** replace "the corpus with
   the best license and the only raw audio, FinCall-Surprise" with a scoped statement (e.g.,
   FinCall-Surprise is the primary audio-bearing corpus we build on; MM-FinEval, concurrent,
   also ships audio). Check the abstract/benchmark section for any other "only audio corpus"
   phrasing.
2. **Related work adds (must-cite):** MM-FinEval (2609.38523) as concurrent multimodal
   volatility benchmark — position ecvol-bench as supplying what it lacks (embargoed /
   ticker-disjoint splits, past-volatility and HAR baselines, identity controls), and note its
   own `vol_past_*` fields make a persistence baseline free to compute. Fast Numbers, Slow
   Language (2606.29734) in the timing section as concurrent use of 8-K timestamps; state that
   our contribution is the audit of *existing volatility label sets*.
3. **Bib updates:** `scss2024` → Findings ACL 2025 (2025.findings-acl.946); `fincall2025` →
   ACL 2026 (2026.acl-long.610). Remove any remaining "unreviewed preprint" framing of SCSS.
4. **Optional, cheap, possibly high-value:** if MM-FinEval's 2019–2021 calls overlap
   FinCall-Surprise's (same group), its `COMPANY_SYMBOL`/`DATE` fields are an independent
   ground truth for our identity/date reconstruction (currently 60/60 manual audit) — and its
   vol labels can be run through `ecvol audit predictions`-style substrate checks. This is
   exploratory and would need a DECISIONS.md entry; not required for submission.
5. **Optional cites for C6:** HindsightBench (cutoff margins), Zhang & Stadie (adjusted
   backtest scores); one sentence suffices.
6. **Re-scan once more right before the ARR deadline** if it is more than ~4 weeks away; the
   FinCall-Surprise group is clearly active in this exact space.

## Caveats

Single-pass scan; absence verdicts (C2, C3, C5, C6) rest on search coverage, not exhaustive
indexing (SSRN and workshop proceedings under-indexed; SSRN pages returned 403 this session).
MM-FinEval's "no split procedure stated" and "no past-volatility baseline" are from an
automated read of the HTML — confirm by reading its experimental section before writing the
related-work sentence. Fast Numbers, Slow Language was read in two chunks; the claim that it
does not discuss prior volatility datasets' timing is from those reads, not a full manual pass.
