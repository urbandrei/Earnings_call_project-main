# OLDWORK.md — The Legacy Project (Reference Only)

> ⚠️ **THIS DOCUMENT DESCRIBES OUTDATED, ABANDONED WORK.**
> It exists purely as a reference point for the rework. Nothing in here should be extended,
> rerun, or treated as guidance. The active project is governed by [DESIGN.md](DESIGN.md)
> (source of truth) and [TASKS.md](TASKS.md) (task tracker). The legacy code and materials
> live read-only in `legacy/` (originally a folder named `Earnings_call_project-main/`).

---

## 1. What the legacy project was

A Kansas State University **CIS 831 (Deep Learning) term project, 2024**, by James Chapman,
John Woods, and Nathan Diehl. It predicted **post-earnings-call stock price volatility** from
multimodal data — transcript text plus per-sentence MP3 audio — by reproducing and extending
the HTML and KeFVP architectures. The original authors are no longer available; the project
was inherited in a frozen, partially unrebuildable state (see §6).

Primary artifacts in `legacy/`:

| Artifact | What it is |
|---|---|
| `legacy/1-Webscraping.ipynb` … `6-Results_Summary.ipynb` | The entire codebase: 12 Jupyter notebooks forming a strict numbered pipeline |
| `legacy/Paper.pdf` | The 8-page IEEE-format term paper |
| `legacy/PowerPoint.pptx` | Course presentation |
| `legacy/papers/` | ~60 reference PDFs; `papers/papers with results from our datasets/` is a curated leaderboard of every paper reporting results on the same benchmarks (HTML, KeFVP, NumHTML, ECHO-GL, RiskLabs, DialogueGAT, AMA-LSTM, DocFin…) — still genuinely useful |
| `legacy/.gitignore` | UTF-16-encoded, ~197 hand-listed data file paths (a cautionary artifact) |

## 2. The pipeline (12 notebooks)

Numeric prefixes encoded execution order; letter suffixes were parallel variants. Each stage
read CSV/numpy artifacts written by the previous one into `data/data_prep/` (no longer present
anywhere — the data was lost, see §6).

1. **`1-Webscraping`** — daily prices from Yahoo (`yfinance`) and Alphavantage for both
   datasets' tickers. Hardcoded a `tickers_to_remove` list (26 delisted/merged) and accumulated
   a `bad_tickers` list (~58 with no Alphavantage data).
2. **`2-Feature_Engineering`** — baseline features: GloVe-300 (text), Praat via `parselmouth`
   (27 audio features). Variants: **2b** RoBERTa-large + financial SentenceTransformers
   (FinLang investopedia, BGE financial matryoshka); **2c** emotion2vec audio emotion
   embeddings; **2d** OpenAI GPT-4o-mini sentence-level sentiment classification into 10
   custom categories (required `OPENAI_API_KEY`).
3. **`3-Data_Cleaning`** — computed log realized volatility targets (3/7/15/30 days), fixed
   Praat `--undefined--` values, joined audio+text+targets, wrote train/val/test numpy splits.
   **3b/3c** rebuilt the splits using the KeFVP paper's own target data for apples-to-apples
   comparison (with and without emotion2vec).
4. **`4-Reproduce_HTML`** — trained the HTML hierarchical transformer (code lifted from the
   HTML authors' GitHub with marked adaptations). Triple-nested loop:
   (feature pair × horizon × **17 alpha values for the multi-task loss**); best alpha picked
   on validation, then retrained on train+val, evaluated on test.
5. **`5-HTML_KeFVP` / `5b`** — same scaffold against KeFVP-style splits, ±emotion2vec.
6. **`6-Results_Summary`** — pivoted `*_final_results.csv` files into the paper's tables.

**Dual-dataset convention:** nearly every cell existed twice — a plain variable and a
`MAEC_`-prefixed counterpart — processing both datasets side by side. Any fix had to be applied
in up to six places (3 notebook clones × 2 branches).

## 3. Data

Both raw datasets lived on a personal `D:` drive that has since been wiped; **no data survives
in the repo** (the legacy `.gitignore` excluded every artifact individually, and they were
never committed).

- **EarningsCall dataset** (Qin & Yang 2019): ~572 calls from S&P 500 companies, 2017.
  Folders `CompanyName_YYYYMMDD` with `TextSequence.txt` + per-sentence MP3s under `CEO/`.
  Source: https://github.com/GeminiLn/EarningsCall_Dataset. Tickers resolved via a
  `company_ticker.csv` (lost with the data).
- **MAEC dataset**: ~3,443 calls, 2015–2018, S&P 1500. Folders `YYYYMMDD_TICKER`.
  Source: https://github.com/Earnings-Call-Dataset/MAEC-A-Multimodal-Aligned-Earnings-Conference-Call-Dataset-for-Financial-Risk-Prediction.
  The audio came from a **private Google Drive link** the original authors shared by email —
  not publicly available.
- Config via `.env`: `API_KEY` (Alphavantage, rate-limited with `time.sleep(0.8)`),
  `OPENAI_API_KEY` (notebook 2d only).
- Several notebooks had an `IN_COLAB` switch mounting Google Drive (`gdrive/My Drive/831`).
- Seeds pinned to 777 (`np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`).

## 4. Reported results (treat with caution)

Per `legacy/Paper.pdf` ("Including Emotion & Sentiment in Multimodal Learning with Audio and
Transcripts of Earnings Conference Calls for Predicting Volatility of Stock Prices"):

- ~180 models trained: 15 embedding combinations × 4 horizons × 3 data years.
- Headline claim: best variants (**BGE + Praat** and **RoBERTa + Praat**, tied) came within
  **+0.73% mean MSE of KeFVP** (then-SOTA); beat SOTA on the 2016 MAEC slice.
- Praat alone was best at short horizons; adding emotion2vec helped at longer horizons.
  Investopedia/BGE embeddings were the most consistent text features.
- The GPT-4o-mini sentiment experiments (notebook 2d) were **never finished** — the paper
  explicitly says results could not be included (Beocat cluster maintenance). Only one prompt
  was ever tried.

> **Caveat (why these numbers don't constrain the rework):** every result came from a single
> seed (777), with the multi-task alpha selected on validation and the model then retrained on
> train+val — a procedure that overfits the selection. No variance estimates, no significance
> tests, no past-volatility/GARCH/HAR baselines. Subsequent literature
> ("Same Company, Same Signal", arXiv 2412.18029) showed that models in this family largely
> memorize ticker identity. The +0.73% figure is not evidence the rework needs to defend or
> beat. See DESIGN.md §3 for the full pitfall analysis.

## 5. What the rework inherited vs. rejected

Inherited (concepts carried into DESIGN.md):

- The core premise: multimodal earnings-call data → post-call volatility prediction.
- The target family: log realized volatility over 3/7/15/30 trading days (literature
  comparability).
- The embedding-ladder methodology — systematically comparing representations under a fixed
  protocol — hardened with seeds, significance tests, and controls.
- HTML / KeFVP / the legacy result table as published reference points (one comparability
  table in the eventual paper, nothing more).
- The data-validation instinct: the legacy notebooks cross-checked Yahoo vs. Alphavantage
  prices and self-computed vs. published Praat features. Institutionalized as acceptance gates.

Rejected (full rationale table in DESIGN.md §2): single-seed results, the 17-value alpha-sweep
procedure, per-sentence audio-text alignment, copy-pasted notebook variants, unpinned
environments, yfinance-as-primary with hardcoded bad-ticker lists, personal-drive data storage,
hand-enumerated gitignores.

## 6. Shortcomings observed at handover (June 2026)

1. **Not runnable**: no `data/` directory anywhere, raw datasets gone with the `D:` drive,
   no environment spec. "Running it" would mean re-acquiring everything and rebuilding.
2. **Copy-paste architecture**: 3 near-identical cleaning notebooks, 3 near-identical training
   notebooks sharing a pasted ~200-line transformer cell, dual-branch duplication in every cell.
3. **No environment capture**: no requirements.txt; the stack (torch, transformers, funasr,
   parselmouth, yfinance) breaks across versions.
4. **Unfinished science**: OpenAI sentiment branch never evaluated; one seed; no baselines.
5. **Cosmetic debt**: four notebooks titled "Data Cleaning" that aren't; two-sentence README;
   UTF-16 gitignore listing files one by one.

## 7. What's still worth consulting

- **`legacy/papers/`** — the literature collection, especially the
  "papers with results from our datasets" leaderboard folder.
- **`legacy/Paper.pdf`** — for the future-work list (spectrograms, prompt engineering, RAG,
  news sentiment, movement targets) and the exact experimental grid attempted.
- **Notebook 3 / 3b** — the target-computation and Praat-validation logic, as a cross-check
  when implementing `targets.py` in the rework (T1.3). Do not port code; re-derive and verify.
- **Notebook 1's ticker lists** — as a historical record of which 2017-era tickers were
  problematic, useful context when the rework's coverage reports flag the same names.
