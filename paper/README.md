# Paper — `ecvol-bench`: an identity-controlled benchmark for multimodal earnings-call volatility prediction

**Condensed 5-page spine (2026-10-09).** External feedback on the 30-page draft was
"too long and too difficult to read; shorten to 5 pages". The main text (abstract through
Discussion and Conclusion) now fits five pages of the self-contained 10pt two-column
layout; Limitations, Ethics, references and a long appendix follow. The long draft is
preserved as git tag `paper-long-2026-10-09` and as `archive/main_long_2026-10-09.pdf`.

The benchmark is the headline contribution (DECISIONS 2026-08-23 §1, 2026-08-26 §3); the
Stage 0–4 results, the reproduction study and the post-cutoff test are the evidence for
why it is needed. Every number traces to a manifested result CSV (`ecvol runs verify`).

- **Format:** self-contained two-column (ACL/EMNLP-style), compiles with a stock TeX
  distribution — no external style files required. Target venue: ACL via ARR
  (`venue_memo_2026-10.md`).
- **Main-text structure:** Introduction (with contributions and a compact related-work
  paragraph) → The Substrate and the Benchmark → Models, Protocol and Controls →
  Results → Discussion and Conclusion. Five main-text tables: substrate audit, the
  ladder grid, identity controls, the reproduced models under our controls, and the
  post-cutoff evaluation.
- **Appendix (`sections/A1-complete-results.tex`):** pipeline figure, full head sets
  (baselines, text, audio), horizon-convention delta, the reproduction study in detail
  (HTML head, faithful runs, controls, code-availability audit), Earnings25 bitrate
  study, the call-date-anchor variant of the post-cutoff test, and the generated grids.

## Build

```sh
bash paper/build.sh          # -> paper/main.pdf (latexmk if perl is present, else pdflatex+bibtex)
bash paper/build.sh clean    # remove aux files
uv run python paper/tables/make_main_tables.py   # regenerate the four generated main-text tables
```

Gotchas: keep `\clearpage` between the generated `\input`s in the appendix (~56
consecutive floats deadlock the two-column output routine); use `\dag` in text mode,
never `$\dag$`; keep the post-cutoff table at `\scriptsize` (a wider version overflows the
column and prints over the neighbouring text).

## Layout

| Path | Contents |
|---|---|
| `main.tex` | Document class, preamble, `\TODO`/`\pending` macros, `\input`s all sections |
| `refs.bib` | Bibliography, built from DESIGN.md §13 |
| `sections/` | `00`–`05` main text, `09` Limitations, `10` Ethics, `A1` appendix |
| `tables/*_main.tex` | Generated main-text tables (`make_main_tables.py` reads `data/results/*.csv`) |
| `tables/substrate.tex` | Hand-curated from `result_table_5r*.csv` (run ids in the header) |
| `tables/{baselines,text,audio,convention,repro,faithful,repro_controls,earnings25_audio,lookahead_assumed,code_audit}.tex` | Appendix tables, curated from the CSVs named in each header |
| `figures/pipeline.tex` | TikZ stage-ladder schematic (appendix) |
| `archive/` | Frozen PDFs of superseded drafts |

The appendix also `\input`s the auto-generated grids `../data/results/result_table_{1,2,3,4}.tex`
(emitted by `ecvol report`, byte-identical and CI-guarded). If you re-run a result
command and numbers change, re-run `make_main_tables.py` and update the curated tables
to match.

## Drafting conventions

- `\TODO{...}` (red) and `\pending{...}` (orange) are hidden when `\draftmode` is set
  false in `main.tex`. None remain in the condensed draft.
- Line numbers (`lineno`) are on for review; comment `\linenumbers` for camera-ready.
- Every number in the prose traces to `data/results/` or a JOURNAL-recorded audit.

## Switching to the official ACL class

1. Drop `acl.sty` and `acl_natbib.bst` into this directory (from the ACL Rolling
   Review template).
2. In `main.tex` replace the `\documentclass[...]{article}` + geometry + `\twocolumn`
   block with `\documentclass[11pt]{article}` and `\usepackage[review]{acl}`.
3. Change `\bibliographystyle{plainnat}` to `\bibliographystyle{acl_natbib}`.

Everything else (sections, tables, `refs.bib`) is template-agnostic. Expect the 11pt ACL
template to run roughly 20% longer than the current 10pt layout.
