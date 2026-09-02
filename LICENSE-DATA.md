# Licence audit of everything this repository redistributes or derives (T8.1)

Code and configs: Apache-2.0 (see `LICENSE`). The table below covers data and derived artifacts.
"Released" means committed to this repository or intended for the release archive; "local only"
means it lives under `data/raw/` (gitignored, mirrored by SHA-256 manifest) and is never redistributed.

| source | upstream licence | what we hold | what we release | basis |
|---|---|---|---|---|
| FinCall-Surprise (Tizzzzy/FinCall-Surprise) | Apache-2.0 | transcripts, mp3, slide PDFs (local) | reconstructed identity/date CSV, targets, splits, chunk-level derived features (embeddings, FinBERT scores, surface stats, eGeMAPS, WavLM, emotion2vec+) | Apache-2.0 permits derivative works; raw transcripts/audio are not re-hosted |
| MAEC (Earnings-Call-Dataset/MAEC…) | CC-BY-SA-4.0 | transcripts + per-sentence features (local) | targets, splits, chunk-level derived text features | derivatives of CC-BY-SA material are released CC-BY-SA-4.0 with attribution |
| Earnings25 (Zenodo 10.5281/zenodo.18762168) | CC-BY-4.0 | zip + unpacked calls (local) | inventory, timing table, targets, splits, derived text/audio features | attribution to Zenodo record + Interspeech 2026 paper |
| EC / EarningsCall_Dataset (Qin & Yang 2019) | not stated (GitHub, no LICENSE file; Drive release "for readers who are interested in reproducing") | 572 call folders (local) | targets, splits (`ec_published.csv` mirrors the lineage's own split files), derived text features; **no transcripts or audio** | scripts-not-data; lineage split/label files are already public in KeFVP/VolTAGE |
| KeFVP / VolTAGE / HTML repositories | no LICENSE file (KeFVP, VolTAGE, HTML); code only | label/split CSVs mirrored under `raw/ref/lineage/` with commit SHAs (local) | audit statistics only (Table 5R); files themselves are not re-hosted | fair use of published research artefacts for audit; manifest records origin |
| SCSS DEC (piqueyd/Same-Company-Same-Signal, `LICENSE-DATA`) | authors' fields CC-BY-4.0; third-party transcript text separately licensed | `DEC.csv` (local) | reproduction statistics (6R-SCSS), timing flags | we use only the authors' numeric fields |
| SEC EDGAR submissions API | public domain (US federal) | per-CIK JSON (local cache) | timing tables (`*_timing.csv`) with accession numbers | public domain |
| Wikipedia S&P 500 list revisions | CC-BY-SA-4.0 | cached page revisions (local) | membership decisions (in `earnings25_inventory.csv`) | facts, attributed by revision id |
| Yahoo Finance via yfinance / Tiingo | research use; no redistribution | price parquets (local) | targets (derived statistics), never prices | manifests record provenance only |
| Models: BAAI/bge-m3 (MIT), ProsusAI/finbert (see model card), microsoft/wavlm-large (MIT), emotion2vec+ (Apache-2.0), opensmile eGeMAPS (audEERING research licence) | as listed | weights cached by the libraries | derived vectors only | research use |

Open items before a public release: (1) confirm the EC dataset's terms with the authors before
any derived-feature archive that could reconstruct transcripts is posted (chunk-level embeddings
cannot, pooled features cannot; we release only pooled per-call features for EC); (2) the
audEERING openSMILE research licence permits derived features for research; a commercial
re-release would not be covered.
