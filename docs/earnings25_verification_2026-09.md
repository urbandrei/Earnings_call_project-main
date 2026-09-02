# Earnings25 — T7.1 one-day verification (2026-09-02)

**Question (DECISIONS 2026-08-23 §6):** does Earnings25 carry what the T1.3 price joins
need — a resolvable ticker, a call date, ideally a call time — and is its audio the
"clean" audio T9.4 assumes?

**Verdict: GO for ingestion. Two design calls arise (§5).**

## 1. Source

| | |
|---|---|
| Record | Zenodo 10.5281/zenodo.18762168, published 2026-06-19, **open access, CC-BY-4.0** (LICENSE file in the zip) |
| Paper | arXiv 2607.23813 (Interspeech 2026) |
| Payload | one zip, 12,044,542,618 bytes, `md5 5aa434b38d8498d98ddb4cce838d94f8` — **verified after download** |
| Local | `D:\ecvol-data\raw\earnings25\` (zip + `earnings-25/` unpacked, ~13.6 GB); `zenodo_record.json` saved beside it |
| Terms | CC-BY-4.0 imposes attribution only. No scraping, no API key, no rate limit. The ToS review the T7.1 acceptance test requires is therefore this paragraph: cite the dataset + paper; redistribute derived features under CC-BY-4.0 with attribution. |

## 2. Contents

`testset-full`: **514 calls, 497.9 h**, one `data.jsonl` record per call with `transcript`,
`speech_segments` (start/end/text/speaker ids, CTC-aligned), `audio_info`, and
`extra_fields` = `Company`, `Country`, `ReleaseDate`, `Industry`, `MarketCap`,
`participants`, `speaker_attributions` (speaker id → name; "Operator" labelled).
`testset-segmented`: 290 five-to-ten-minute WAV segments — not needed here.

No ticker, no CIK, no fiscal-quarter field. Every company appears once.

## 3. What the price join needs — field by field

| Need | Finding |
|---|---|
| **Ticker** | Absent. `Company` → ticker through the SEC `company_tickers.json` table + the T1.4 matcher (`_lookup_name`) with 8 curated overrides: **499/514 mapped; 472 are S&P 500 members in our 2026-08-13 snapshot (472 unique tickers, zero collisions after excluding look-alikes)**. 27 mapped tickers are outside the snapshot — mostly genuine 2025-Q4 members since dropped (CPB, CAG, LW, MTCH, PAYC, POOL, EPAM, HOLX, EA, …) plus name-collision contamination (see §4). 15 unmapped are all non-S&P look-alikes. |
| **Call date** | `ReleaseDate` present and parseable for 514/514: 2025-10 (292), 2025-11 (176), 2025-12 (33), plus 13 stragglers from 2025-03/07/08. Thursday 174 / Wednesday 141 / Tuesday 126 / Friday 50 / Monday 23. |
| **Call time** | `ReleaseDate` carries a clock time (64% on the hour, 0.2% midnight). Hour histogram peaks at 12–15 h and 20–22 h, i.e. 8–11 am and 4–6 pm Eastern **if the field is UTC**. Three transcripts stating a Pacific-time start all sit exactly 8 h before `ReleaseDate` (Palo Alto, Keysight, Lam) → **consistent with UTC**. This is the first corpus in the project with measured call times: it feeds T9.2 directly, and the after-hours rule can be applied with real timestamps (T9.2 must still confirm the UTC reading against an 8-K/press-release sample). |
| **Prices** | Our archive ends 2022-06-30; a fresh pull is required. yfinance serves 2025-09-01 → 2026-02-27 (dry-checked on MU/CAT/AAPL); 404 of the 472 tickers already have a parquet from the FinCall/MAEC universe. τ=30 windows for the latest call (2025-12-19) close ~2026-02-03, inside the served range. |
| **Post-cutoff** | 2025-Q4 calls are post-cutoff for the Qwen2.5 stack (Sep 2024) and for BGE-M3/FinBERT. |

**Gate (≥95% price join, T1.4 standard):** 472/514 = 91.8% resolved to a snapshot member before any per-call cleanup; with the ~20 membership-drift names verified by hand the resolvable share is ≈95–96%, and every exclusion is reason-coded (look-alike / non-US / unmapped). Expected to clear the gate at ingestion.

## 4. Data-quality findings about Earnings25 itself

1. **Name-collision contamination.** The upstream sampling matched company *names*, so the "S&P 500" set contains: Grainger PLC (UK), Domino's Pizza Group plc (UK) *and* Domino's Pizza Enterprises (AU) beside Domino's Pizza Inc., Paramount Group (REIT) beside Paramount Skydance, Vertex Inc. (tax software) beside Vertex Pharmaceuticals, PTC India ×2 beside PTC Therapeutics, Cummins India, GE Vernova T&D India, Everest Kanto / Everest Medicines beside Everest Group, Prologis Property Mexico, CVS Group plc (UK vets), Cleanaway (AU), Capital Ltd (MU), Goldman Sachs BDC, Blackstone Secured Lending / Blackstone Mortgage Trust, BlackRock TCP Capital, Morgan Stanley Direct Lending, Apple Hospitality REIT, The Bancorp, Information Services Group, Snap. **≈25 of 514 calls (~5%) are not S&P 500 companies.** Reason code `lookalike` at ingestion; worth a line in the paper's Earnings25 paragraph.
2. **Audio is not clean.** ffprobe over all 514 mp3s: mono throughout; **281 calls at 44.1 kHz / 64 kbps, 216 at 16 kHz / 24 kbps, 16 at 11.025 kHz / 16 kbps, 1 at 22.05 kHz / 32 kbps.** 45% of the corpus is *lower* bitrate than FinCall's uniform 40 kbps; the rest is 64 kbps, a modest step up. The paper's "original MP3" is accurate but the originals are webcast-replay transcodes. T9.4's premise ("audio that is not a 40 kbps transcode") is false as stated — see §5.
3. **13 pre-Q4 stragglers** (Mar/Jul/Aug 2025) despite the "2025 Q4" label; keep, they are still post-cutoff.
4. 12 countries; the non-US rows are mostly genuine S&P 500 members domiciled abroad (IE/CH/BM/NL) plus the look-alikes above.

## 5. Design calls arising (for the user; not assumed)

1. **T9.4 re-scope.** Earnings25 cannot test "clean audio". It *can* test the audio-quality hypothesis directly as a **within-corpus bitrate-stratified re-run**: 64 kbps (281 calls) vs ≤24 kbps (233 calls), same period, same extractors, same controls. Recommendation: re-scope T9.4 to that contrast and drop the "clean" wording from DESIGN/TASKS/paper. Alternative: drop T9.4 (forfeits the only audio-quality evidence available).
2. **Universe rule for T7.1.** Recommendation: include a call iff its `Company` resolves to a SEC ticker **and** the ticker was an S&P 500 member on `ReleaseDate` (2025-Q4 membership, reconstructed from the 2026-08-13 snapshot plus the drift list verified by hand); exclude look-alikes with a reason code. Alternative: any SEC-resolvable US-listed ticker (adds the ~12 US non-members: BDCs, REITs, SNAP).

## 6. Artifacts

- `data/coverage/earnings25_inventory.csv` — one row per call: id, Company, Country, ReleaseDate, Industry, MarketCap, sample rate, duration, segment/speaker counts, resolved ticker, snapshot membership.
- `data/coverage/earnings25_audio_probe.csv` — ffprobe codec / sample rate / channels / duration / bitrate per file.
- Raw (local only): `D:\ecvol-data\raw\earnings25\{earnings25.zip, earnings-25/, zenodo_record.json}`.
