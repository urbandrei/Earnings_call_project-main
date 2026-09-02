# Reproducing every table (T8.1)

Every number in the paper regenerates from a committed CSV under `data/results/`, and every such
CSV carries a committed run manifest under `artifacts/runs/<run_id>/run.json` (resolved config,
config hash, git SHA, seeds, environment fingerprint, output SHA-256). `ecvol report` and CI refuse
a result file whose bytes match no manifest. `ecvol runs verify` checks all of them.

## Environment

```
uv sync                 # CPU: everything in Tables 1, 5R, 6R-SCSS, 7 and the audits
uv sync --group gpu     # + torch cu128, transformers, funasr: feature extraction, HTML head
```

Windows 11 / RTX 5060 Ti (16 GB) is the reference machine; ffmpeg 8.x on PATH for audio.

## Data (local only; never committed)

| corpus | command | notes |
|---|---|---|
| FinCall-Surprise | `ecvol data fetch fincall` → `ecvol data ingest fincall` | Drive mirror; identity reconstruction runs inside ingest |
| MAEC | `ecvol data fetch maec` → `ecvol data ingest maec` | transcripts + features; no raw audio exists |
| Earnings25 | download Zenodo 10.5281/zenodo.18762168 to `data/raw/earnings25/` (md5 in `docs/earnings25_verification_2026-09.md`) → `ecvol prices pull --dataset earnings25` → `ecvol data ingest earnings25` | membership by CIK from cached Wikipedia revisions |
| EC (Qin & Yang) | Drive 5-part zip → `data/raw/ec/extracted/` → `ecvol prices pull --dataset ec` → `ecvol data ingest ec` | identity from the lineage's own table |
| prices | `ecvol prices pull` (2014–2022 archive) | yfinance + Tiingo fallback (`TIINGO_API_KEY` in `.env`) |
| lineage files | mirrored under `data/raw/ref/lineage/` (see `lineage_manifest.json`) | KeFVP / VolTAGE / HTML repos pinned to commit SHAs |
| DEC (SCSS) | `data/raw/ref/scss/DEC.csv` from the SCSS repository | CC-BY-4.0 fields |

Then: `ecvol targets build` · `ecvol splits build` · `ecvol timing build {fincall,maec,earnings25}` ·
`ecvol timing targets {fincall,maec,earnings25}`.

## Features (GPU; content-hash cached, bit-identical on re-run)

```
ecvol featurize sections
ecvol featurize text --dataset {fincall,maec,earnings25,ec}
ecvol audio qc      --dataset {fincall,earnings25}
ecvol audio egemaps --dataset {fincall,earnings25}
ecvol audio wavlm   --dataset {fincall,earnings25}
ecvol audio emotion2vec --dataset {fincall,earnings25}
```

## Tables — one command each

| paper table | command | result CSV | config |
|---|---|---|---|
| Table 1 (Stage 0/1, both conventions) | `ecvol evaluate` | `result_table_1.csv` | `configs/evaluate.yaml` |
| Table 2 (text heads) | `ecvol evaluate-text` | `result_table_2.csv` | `configs/evaluate-text.yaml` |
| identity controls | `ecvol controls` | `result_controls.csv`, `controls_probe.csv` | `configs/controls.yaml` |
| Table 3 (audio heads + controls) | `ecvol evaluate-audio` | `result_table_3.csv`, `audio_*.csv` | `configs/evaluate-audio.yaml` |
| Table 4 (fusion + grid) | `ecvol evaluate-fusion` · `ecvol grid` | `result_table_4*.csv` | `configs/{evaluate-fusion,grid}.yaml` |
| convention delta | `ecvol targets compare` | `coverage/targets_convention_delta.csv` | — |
| timing sensitivity | `ecvol timing sensitivity` | `timing_sensitivity.csv` | `configs/timing-sensitivity.yaml` |
| Table 5R (substrate audit) | `ecvol audit substrate` | `result_table_5r*.csv` | `configs/audit-substrate.yaml` |
| Table 6R (HTML on EC) | `ecvol reproduce html` | `result_table_6r.csv` | `configs/reproduce-html.yaml` |
| Table 6R (SCSS on DEC) | `ecvol reproduce scss` | `result_table_6r_scss.csv` | `configs/reproduce-scss.yaml` |
| code-availability audit | `ecvol audit code` | `code_availability.csv` | `configs/audit-code.yaml` |
| Earnings25 audio by bitrate | `ecvol evaluate-audio-earnings25` | `result_table_3_earnings25.csv`, `audio_*_earnings25.csv` | `configs/evaluate-audio-earnings25.yaml` |
| Table 7 (lookahead) | `ecvol evaluate-lookahead` | `result_table_7.csv` | `configs/evaluate-lookahead.yaml` |
| rendered md/tex | `ecvol report` | `result_table_{1..4}.{md,tex}` | — |

Curated paper tables under `paper/tables/*.tex` copy cells from these CSVs; the header comment of
each names its source file and run id.

## Verification

```
ruff check . && ruff format --check . && pytest -q     # the CI gate (CPU, no data payload)
ecvol runs verify                                        # every result CSV ↔ its manifest
ecvol data verify                                        # every data file ↔ its SHA-256 manifest
```

Determinism: ridge/HAR/persistence/GARCH runs are byte-identical across re-runs (Table 1 was
re-run live on 2026-08-26 and 2026-09-02 and matched the committed bytes). MLP heads and the
HTML head carry seeds; their cells report the seed standard deviation.

## Licences of released derived artifacts

See `LICENSE-DATA.md`.
