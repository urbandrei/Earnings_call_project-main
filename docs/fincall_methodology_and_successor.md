# FinCall-Surprise: collection-methodology analysis & modern-successor design

**Status: exploration** (2026-08-13). This document analyzes how FinCall-Surprise [D1] was
actually built, inventories its defects as evidenced by our own Phases 1–6, and proposes a
modern successor corpus under the constraints chosen by the user on 2026-08-13:
*new benchmark contribution* (train-scale, also serves as the Phase-7 post-cutoff holdout),
*audio + transcript*, *fully publicly redistributable*, *zero-cost acquisition* (free
scraping + volunteer collectors). Promotion into DESIGN.md/TASKS.md requires a DECISIONS.md
entry; nothing here changes the contract yet.

Inputs: the FinCall paper (arXiv:2510.03965 v1 **and** the ACL 2026 camera-ready
2026.acl-long.610, which differ materially), the data repo (github.com/Tizzzzy/FinCall-Surprise,
full tree + all three transcript JSONs), our local mirror and coverage artifacts (direct
measurement), and a web survey of the 2023–2026 acquisition landscape (all URLs in §7).

---

## 1. How FinCall-Surprise was actually built

| Component | Source (per paper) | Notes |
|---|---|---|
| Audio (MP3) | **EarningsCast** (arXiv v1: "we web scrape the associated audio recordings from EarningsCast"); ACL camera-ready reworded to "official corporate websites, … synchronize the identifiers using EarningsCast" | No mechanics described. Site is **dead — HTTP 410 site-wide** (our probe 2026-06-12, re-verified 2026-08-13). |
| Transcripts | **Seeking Alpha** (v1: "primarily Seeking Alpha"); camera-ready demotes SA to a "manual reference tool" for dates, claims extraction "from the primary corporate sources" | Pre-existing human transcripts, not ASR. The reworded provenance between versions is itself a red flag. |
| Slides (PDF) | "multiple sources, including Bloomberg News and company websites" | |
| Universe | US-listed firms, market cap > $1B, ADV > $50M → "more than 4,000 unique companies" pool | **No attrition accounting**: how 4,000 firms × 12 quarters became 2,688 calls is never explained. |
| Period / size | 2019–2021; 2,688 calls (919 / 704 / 1,065) — verified against the shipped JSONs | Every call modality-complete per the paper (not true in the artifact — §2.3). |
| Label | SUE from **licensed IBES** consensus, Latane–Jones; binarized at \|ES\| ≥ 0.5, middle excluded; consensus = mean of forecasts **within one month after the call** (task: predict *next* quarter's surprise) | Only the final 0/1 ships. No EPS values, no join key, no codebook. 85/15 imbalanced. |
| QC | Two sentences: modalities aligned "by quarterly reporting periods," cross-checked "by conference call titles" | No alignment, no diarization, no human verification, no error rates. |
| Release | Data-only GitHub repo (3 transcript JSONs; 4 fields per call: `input`, `mp3_id`, `ppt_id`, `label`) + ~52 GB MP3 / ~5.3 GB PDF on Google Drive | **No collection or preprocessing code**, despite Appendix A claiming "(iv) Preprocessing and reconstruction code." Zero GitHub issues. |
| License | Apache-2.0, added ~5 months post-release; **no discussion anywhere** of SA/EarningsCast/Bloomberg copyright or ToS | License-washing: Apache-2.0 grants rights the authors don't hold over third-party content. |

**The headline conclusion: the methodology is unreplicable as published.** The audio source
no longer exists, the transcript source's ToS forbid scraping, no scripts were released, the
universe-to-corpus attrition is unaccounted, and the label depends on a licensed database
joined through an identity table the authors withheld. A successor cannot "re-run FinCall on
newer data" — it must be a different, better-engineered pipeline. That is also the
opportunity: publishing the collection scripts and per-step attrition alone would exceed the
original's documented rigor.

## 2. Defect inventory (paper/repo findings × our Phases 1–6 evidence)

Numbers below are from our committed coverage artifacts and direct measurement of the local
mirror; file citations in the full defect report are reproduced in JOURNAL.md history and
`data/coverage/*` / `data/identity/*`.

### 2.1 Provenance and repeatability
- No collection scripts released (repo verified file-by-file: zero code files).
- Audio source dead (HTTP 410) → corpus unrepairable and unextendable at source; Drive
  distribution hit quota walls during our own mirror (JOURNAL 2026-06-11).
- arXiv v1 vs ACL camera-ready **change the stated sources** (EarningsCast and Seeking Alpha
  demoted from primary sources to "sync"/"reference" tools) with no erratum.
- 456 surplus MP3s on disk are referenced by no transcript; 5 of the 10 sample media files
  committed to their repo are orphaned (IDs match nothing in the JSONs).

### 2.2 Metadata: absent by construction
- **No ticker, company, CIK, date, time, fiscal period, exchange, sector, or speaker names**
  — four fields per record, period. Not anonymization (transcripts name the company plainly;
  the paper's own figures show tickers); simply unlabeled.
- Cost to us: a 571-line reconstruction pipeline + 80-row manual override table + audits to
  reach 92.9% ticker resolution (95.3% on earnings calls); 192 calls never resolved; the
  2026 SEC ticker table is missing corpus-era issuers (2024–26 M&A wave), which also cost
  price coverage (4 tickers unservable).
- **32.4% of recovered dates come from the slide PDF's file-creation timestamp**, not a
  stated date; 24 page-1/creation-stamp disagreements; 12 dates outside the corpus span
  (two literal year-0001 artifacts).
- **No call time-of-day anywhere** (only 3.4% of transcripts mention any clock time) → the
  uniform assume-after-hours fallback on every volatility target in the study, acknowledged
  in our paper's limitations, never sensitivity-checkable for lack of ground truth.

### 2.3 Audio
- **Every one of the 2,671 files is a uniform 40 kbps MP3 transcode** — aggressive lossy
  compression underneath all acoustic features (eGeMAPS jitter/shimmer, WavLM). Never
  documented upstream; arguably the most consequential audio defect. (Reference point:
  Earnings-21 ships 256 kbps lossless PCM.)
- Eight source sample rates (8 kHz → 48 kHz); 91 calls below the 16 kHz model input rate,
  14 at telephone-grade 8 kHz.
- Loudness never normalized: 32.6 dB RMS spread; 701 calls peak-limited at ≥ −0.1 dBFS — a
  per-call nuisance variable an acoustic model can latch onto.
- 17 calls missing audio *and* slides, shipped with the literal placeholder string
  `"more ppt"` in the `ppt_id` field. Decode health itself was perfect (2,671/2,671).
- No diarization, no speaker-time anchors, one 3.3-hour duration outlier never explained.

### 2.4 Transcripts
- Role markers only (`Executives:`/`Analysts:`/`Operator:`), often glued to the preceding
  word; no way to tell CEO from CFO or analyst A from analyst B. 126 calls lack `Analysts:`
  entirely; some calls collapse to a single role block.
- Prepared-vs-Q&A boundary must be inferred; our audited heuristic covers 98.7% of calls but
  **229 calls (8.5%) have a degenerate split**.
- Transcription damage baked in: `indiscernible` in 66.4% of calls (12,943 occurrences);
  `[Audio Gap]` in 347 calls (12.9%) — acknowledged missing spans where transcript and audio
  silently disagree, and a leading cause of our unresolved identities.
- Section lengths up to 61k tokens forced the YaRN context-extension design call in T6.2.

### 2.5 Composition and labels
- **7.0% of the corpus is not earnings calls** (M&A announcements, conference firesides,
  annual meetings, a monthly sales call) — detectable only by our heuristic classifier; the
  very first record in `transcripts_2019.json` is the Bristol-Myers/Celgene merger call,
  shipped with `label: 1`.
- The IBES surprise label is unreproducible (licensed source + withheld join), imbalanced
  (85/15), undocumented, and attached to non-earnings events. We never used it.
- 2019–2021 span ⇒ COVID regime dominates: the T2.2 sanity-gate exception, the Stage-2
  generalization failure, and a degenerate one-year temporal test segment all trace to it.
  The corpus is too short for the per-year robustness analysis our design mandates.
- Net: after full reconstruction, **90.2% of the advertised corpus is usable** for the
  headline task (2,424/2,688 with ≥1 target); the combined temporal×ticker-disjoint split
  collapses to val=21/test=92.

### 2.6 What FinCall got right (a successor must preserve)
1. Genuinely open license posture, audio included — the reason it was usable at all.
2. **Full raw call audio** (2,769 h), not precomputed features — enabled our entire Phase 4.
3. Complete untruncated human-quality transcripts, 100% parse rate.
4. Exact 1:1 call↔audio key; counts match the paper exactly.
5. Role markers sufficient for Q&A detection (98.7% coverage, 30/30 audit).
6. Near-complete slide coverage — what made identity/date reconstruction possible at all.
7. Scale adequate for temporal + ticker-disjoint splits + control suites (~2.4k usable calls).
8. Modalities genuinely correspond (same calls); it is the metadata layer that failed.

## 3. The 2026 acquisition landscape (what a successor can actually use)

Hard constraint discovered: **the free-audio aggregator era is over, and replays decay.**

- **EarningsCast — dead** (HTTP 410 site-wide, re-verified 2026-08-13). FinCall's source and
  the only historical free bulk-MP3 archive. Its disappearance also means the 2023–2025
  audio backlog is essentially **not backfillable** at zero cost.
- **earningscall.biz** — the obvious commercial successor: transcripts + raw audio + calendar
  for 5,000+ companies. Free tier = **two companies** (AAPL, MSFT); audio requires the
  $129/mo tier; no redistribution grant. Fails the zero-cost and redistribution constraints.
- **Seeking Alpha / Motley Fool transcripts** — ToS prohibit scraping; their transcripts are
  copyrighted derivative works. Not redistributable, and not needed (see below).
- **SEC EDGAR** — no transcripts as a rule (8-K Ex-99.1 is the press release; verbatim
  transcripts are a small minority), but free and redistributable for: actual EPS (XBRL),
  call announcement date/time ("call at 5:00 p.m. ET"), CIK/ticker identity, sector (SIC).
- **IR webcast replays** (Q4 Inc, Notified, Chorus Call, OpenExchange platforms) — near-
  universal coverage, best-quality source audio, but: embedded HLS players, sometimes
  registration walls, **30–90-day typical retention**, per-vendor engineering, platform ToS
  generally prohibit automated access (the *audio* copyright belongs to the issuer, not the
  platform). A meaningful minority of companies post direct MP3 replay links.
- **YouTube** — trivial to capture (yt-dlp) but tiny, biased coverage (Tesla-style outliers)
  and YouTube ToS make redistribution the norm-breaking act; academic practice is links-only.
- **Free label/metadata APIs** — yfinance `get_earnings_dates()` (EPS estimate/actual/
  surprise + event timestamp, recent quarters only), Alpha Vantage `EARNINGS`/`EARNINGS_CALENDAR`
  (25 req/day), Finnhub (calendar 1 month, surprises 4 quarters), Nasdaq calendar
  (BMO/AMC buckets). All favor **rolling capture at collection time** over backfill.
- **Whisper-class self-transcription** — the clean answer to transcript copyright: our own
  ASR artifact from the audio, no SA/Fool dependency. WER on earnings audio is well-
  characterized by Rev.ai's Earnings-21/22 and Earnings25 benchmarks.

**Legal grounding for "fully public":** *Swatch Group v. Bloomberg* (2d Cir. 2014) held that
redistributing an earnings-call recording — obtained without authorization, by a commercial
actor — is **fair use**: the call is a factual work whose dissemination serves the public
interest, and no licensing market exists for it. A noncommercial research dataset is a
stronger posture still. Practice agrees: MDRM (2019), MAEC (2020), FinCall (2025) all
redistribute call content; none has recorded takedowns after 6–7 years. The state-of-the-art
template is **Earnings25 (2026)**: transcripts/alignments/metadata under CC-BY-4.0 on
Zenodo, audio redistributed with "remain subject to any applicable terms of the original
content providers" + user-responsibility disclaimer. That, plus a documented
takedown-on-request policy, is what we should copy — and unlike Earnings25 (which discloses
neither its audio source nor collection scripts), we release the scripts.

## 4. Proposed successor design (working name: **ecvol-live**)

One-line design: **a rolling, forward-collected, fully-documented earnings-call corpus:
IR-webcast/direct-replay audio captured within the season, self-transcribed and diarized
locally, with identity/timing metadata captured at collection time — released CC-BY with
scripts, manifests, and per-step attrition.**

### 4.1 Pipeline stages (all via `ecvol collect *` CLI, idempotent/resumable per project rules)

1. **Discovery** (weekly during earnings season): universe list (start: S&P 500 + S&P 400,
   ~900 tickers; documented criteria, unlike FinCall's) × earnings calendar (Nasdaq/yfinance,
   cross-checked against the 8-K announcement) → per-call record: ticker, CIK, company,
   fiscal period, **exact call datetime + timezone** (from the press release/8-K), IR page
   URL, webcast/replay URL. Every non-captured call gets a reason code from day one.
2. **Capture** (within the 30–90-day replay window):
   a. direct MP3 replay links where present (automated, polite rate limits);
   b. HLS/progressive streams via yt-dlp/ffmpeg where technically open;
   c. **volunteer manual capture** for registration-walled or vendor-locked replays —
      humans register and download; this also keeps automated access off the walled
      platforms (the ToS-gray zone).
   Store: original stream bytes + a normalized **lossless FLAC, 16 kHz mono, loudness
   documented (not destructively normalized), SHA-256 per file** — the exact store format
   T4.1 already implements.
3. **Transcription + diarization** (local GPU, sequential per DESIGN §8.3): Whisper
   large-v3 (faster-whisper) + pyannote → per-speaker, per-turn, **word-time-aligned**
   transcript; speaker names/roles seeded from the operator's introductions; prepared/Q&A
   boundary from the operator cue **as a data field**, human-audited. ASR quality
   characterized against Earnings-21/Earnings25 reference transcripts (WER report ships
   with the corpus).
4. **Labels**: volatility targets computed from prices by our existing T1.3 pipeline (now
   with real timestamps — the after-hours rule finally *applies* instead of being assumed);
   optional EPS-surprise labels from free sources **with codebook, magnitudes, and join
   keys** (never an opaque binary).
5. **QC + release**: reason-coded coverage report at every stage (universe → calendar →
   captured → decoded → transcribed → target-joined), SHA-256 manifests, per-call metadata
   row; transcripts/metadata/features CC-BY-4.0, audio redistributed with the Earnings25
   disclaimer + takedown policy, **all collection scripts released** (so even a worst-case
   audio takedown leaves a scripts-not-data corpus — the DESIGN.md fallback posture).

### 4.2 Defect → fix mapping

| FinCall defect (§2) | ecvol-live design answer |
|---|---|
| No identity metadata; 571-line reconstruction | Ticker/CIK/company/fiscal period captured at discovery time, from the calendar + 8-K |
| No call datetime; assume-after-hours fallback | Exact datetime + timezone from press release/8-K; the §5.3 rule becomes applicable |
| 7% non-earnings contamination | Earnings-calendar-driven discovery: earnings calls by construction; call type is a field |
| 40 kbps transcode, 8 sample rates, wild loudness | Source-quality capture + lossless FLAC store, uniform 16 kHz, loudness measured & reported |
| Role-only markers, no turns, inferred sections | Diarized, named, word-time-aligned turns; section boundary as data |
| `[Audio Gap]`/`indiscernible` silently baked in | Own ASR: gaps impossible by construction; low-confidence spans machine-flagged |
| Opaque IBES binary label | Computed price targets + optional documented surprise labels with join keys |
| No attrition accounting | Reason-coded pipeline report, published |
| No scripts, dead source, unrepeatable | Scripts + manifests released; rolling collection is repeatable each quarter by anyone |
| No sector/mcap (blocked DESIGN §5.4.2) | SIC/GICS-proxy + market cap captured per call |
| Single COVID-era regime | Rolling multi-year accumulation; each season adds a regime slice |
| 50-call audit slices underpowered | Pre-sized validation slice ≥ 200 calls (κ SE ≈ 0.05) |
| License-washing (Apache-2.0 over SA content) | CC-BY-4.0 over *our own* artifacts; audio disclaimer + takedown policy (Earnings25 model) |

### 4.3 Volunteer protocol (they have better uses than bulk scraping)

- **Capture gap-fill** (highest value): assigned tickers whose replays are walled; register,
  download/record, log the replay URL, call datetime, and file into a shared sheet; the CLI
  ingests the sheet. Human capture avoids automated-access ToS breaches entirely.
- **QC audit**: seeded stratified samples — identity spot-checks, section-boundary audit,
  ASR spot-reads against audio (the audits we already know how to run, at n ≥ 200).
- **Metadata verification**: call datetime vs 8-K cross-check on a sample.
- Explicitly *not* volunteers' job: anything a script can do idempotently.

### 4.4 Scale and cost (estimates)

- One season of S&P 500+400 ≈ 700–900 capturable calls ≈ 700–900 h audio.
- Whisper large-v3 via faster-whisper on the RTX 5060 Ti: ~5–10× realtime ⇒ roughly
  70–180 GPU-hours per season; pyannote diarization of the same order. Batchable,
  resumable, sequential — within the existing compute discipline. (Subject to the standard
  50-call ETA gate before committing to a full season.)
- Storage: FLAC 16 kHz mono ≈ 25–45 GB/season (cf. FinCall's 119 GB store for 2,769 h).
- Cash cost: $0 under the chosen constraints.

### 4.5 Bootstrap opportunity: Earnings25 as seed + validation

Earnings25 (Zenodo, DOI 10.5281/zenodo.18762168) redistributes ~500 Q4-2025 S&P 500 calls
(498 h) with professional aligned transcripts (CC-BY-4.0) and speaker roles/industry
metadata. Three uses, all cheap: (a) an instant modern seed slice — and Q4-2025 calls
(held Jan–Feb 2026) are post-cutoff for our Qwen2.5 stack, i.e. immediate Phase-7 holdout
material; (b) reference transcripts to benchmark our Whisper WER; (c) a QC cross-check for
any calls both corpora capture. Caveats to verify on download: their audio provenance is
undisclosed, per-call metadata completeness (exact datetimes?) unknown, and the audio
carries the "original provider terms" disclaimer — our redistribution of *their* audio
should mirror their own posture.

## 5. Timing: the constraint that binds

Replays decay in 30–90 days and free calendars cover ~1 month. **The Q2 2026 season
(calls mid-July–mid-August) is in its replay window right now**; waiting until Phase 6/7
wrap-up forfeits a full season. This does not require the full pipeline — it requires the
*capture* half (discovery + audio download + metadata snapshot). Transcription, QC, and
release can lag arbitrarily; bytes-on-disk cannot.

**Proposed pilot (1 short phase, before any DESIGN promotion):** capture 50 current-season
calls end-to-end (discovery → audio → metadata → Whisper → diarize → target join), measuring
per-call automation coverage (what % had direct MP3s vs HLS vs walls), per-call human
minutes for the walled remainder, and ASR WER vs Earnings25 on overlaps. The pilot's numbers
decide the season-scale go/no-go — the same ETA-gate discipline as T4.1/T4.2.

## 6. Risks and open design calls

Risks: per-vendor scraper brittleness (mitigate: volunteers absorb the hard tail);
coverage skew toward companies with open replays (mitigate: report capture-rate by
sector/size — the reason codes make skew measurable, unlike FinCall's silent attrition);
redistribution challenge (mitigate: Swatch posture, disclaimer, takedown policy, and the
scripts-not-data fallback keeps the benchmark alive even if audio must come down);
scope creep against T6.2/Phase-6 (mitigate: pilot is capture-first and small; the LLM
decision queue is unaffected).

Open design calls (user):
1. Green-light the 50-call pilot now (time-sensitive), or hold until after the T6.2 decision?
2. Universe: S&P 500 only, or +S&P 400 mid-caps (FinCall's >$1B/$50M-ADV criteria pull in
   smaller names; broader = more regime/size diversity, more capture effort)?
3. Email the FinCall authors (deferred-not-rejected in DECISIONS 2026-06-12) for their
   call-ID→identity table and collection details — costs one email, could both repair the
   old corpus and inform the new one.
4. Volunteer recruitment timing: pilot needs zero volunteers; season-scale needs the
   gap-fill roster ready inside the replay window.
5. Naming/positioning: fold into ecvol-bench as its "live" track, or a standalone dataset
   paper?

## 7. Sources

- FinCall-Surprise: paper https://arxiv.org/abs/2510.03965 · ACL https://aclanthology.org/2026.acl-long.610/ · repo https://github.com/Tizzzzy/FinCall-Surprise · Drive folder (README link)
- Swatch Group v. Bloomberg, 2d Cir. 2014: https://law.justia.com/cases/federal/appellate-courts/ca2/12-2412/12-2412-2014-01-27.html · USCO summary https://www.copyright.gov/fair-use/summaries/swatchgrp-bloomberg-2dcir2014.pdf
- Earnings-21/-22 (Rev.ai): https://github.com/revdotcom/speech-datasets · https://huggingface.co/datasets/Revai/earnings21
- Earnings25: https://arxiv.org/abs/2607.23813 · Zenodo DOI 10.5281/zenodo.18762168
- MDRM: https://github.com/GeminiLn/EarningsCall_Dataset · MAEC: https://github.com/Earnings-Call-Dataset/MAEC-A-Multimodal-Aligned-Earnings-Conference-Call-Dataset-for-Financial-Risk-Prediction
- earningscall.biz pricing: https://earningscall.biz/api-pricing · python lib https://github.com/EarningsCall/earningscall-python
- EarningsCast: https://earningscast.com/ (HTTP 410, verified 2026-06-12 and 2026-08-13)
- Seeking Alpha ToS: https://about.seekingalpha.com/terms · Motley Fool ToS: https://www.fool.com/legal/terms-and-conditions/fool-rules/
- IR platforms: https://www.q4inc.com/platform/platform-features/earnings · https://www.notified.com/IR/earnings-day
- Free label/calendar sources: yfinance `get_earnings_dates` docs https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.get_earnings_dates.html · Alpha Vantage https://www.alphavantage.co/documentation/ · Finnhub https://finnhub.io/pricing · Nasdaq calendar wrapper https://github.com/s-kerin/finance_calendars
- Historic IR-replay download guides (feasibility evidence): https://www.gurufocus.com/news/116414/howto-guide-for-downloading-earnings-calls
