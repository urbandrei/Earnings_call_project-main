# Paper — `ecvol-bench`: an identity-controlled benchmark for multimodal earnings-call volatility prediction

A **living manuscript scaffold**, restructured 2026-08-26 to the **benchmark-first
spine** (DECISIONS 2026-08-23 §1, 2026-08-26 §3). The benchmark is the headline
contribution; the Stage 0–4 results (Result Tables 1–4, all real numbers) are the
evidence for why it is needed. Phase 6 (LLM structured features) is dropped and
excised from the paper entirely (DECISIONS 2026-08-26 §1).

Owed results are marked `\pending{}` (orange): the substrate audit (Table 5R,
T6R.1), the reproduction study (Table 6R, T6R.2), the calendar-day recomputation
(T9.1), the timestamp retrofit (T9.2), the Earnings25 clean-audio re-run (T9.4),
and the post-cutoff lookahead (Phase 7). Run manifests (T9.3) landed 2026-08-26:
every result CSV is verified against `artifacts/runs/*/run.json` by `ecvol report`
and CI.

- **Format:** self-contained two-column (ACL/EMNLP-style), compiles with a stock TeX
  distribution — no external style files required. Target venue: ACL via ARR.
- **Structure:** Intro → Related work → **The ecvol benchmark** (substrate audit,
  corpora + identity reconstruction, targets under both conventions, call timing,
  splits, release) → Models (Stages 0–4 + reproduced prior models) → Protocol →
  Results → Discussion → Conclusion → Limitations → Ethics → Appendix.

## Build

Requires a TeX distribution (TeX Live or MiKTeX):

```sh
bash paper/build.sh          # -> paper/main.pdf (latexmk if perl is present, else pdflatex+bibtex)
bash paper/build.sh clean    # remove aux files
```

Gotchas: keep `\clearpage` between the generated `\input`s in
`sections/A1-complete-results.tex` (~56 consecutive floats deadlock the two-column
output routine); use `\dag` in text mode, never `$\dag$`.

## Layout

| Path | Contents |
|---|---|
| `main.tex` | Document class, preamble, `\TODO`/`\pending` macros, `\input`s all sections |
| `refs.bib` | Bibliography, built from DESIGN.md §13 |
| `sections/` | `00`–`10` body sections + `A1` appendix (complete results) |
| `tables/` | Curated headline tables (`baselines`, `text`, `controls`, `audio`, `grid`) |
| `figures/` | `pipeline.tex` — TikZ stage-ladder schematic |

The **appendix** `\input`s the auto-generated grids
`../data/results/result_table_{1,2,3,4}.tex` (emitted by `ecvol report`,
byte-identical and CI-guarded). The curated `tables/*.tex` copy selected cells from
those files; if you re-run `ecvol report` and numbers change, update the curated
tables to match.

## Drafting conventions

- `\TODO{...}` (red) — content to write/finalise. `\pending{...}` (orange) — results
  owed by an unfinished task. Both are hidden when `\draftmode` is set false in
  `main.tex` (flip it for a clean render). Grep for them to find open work.
- Every number in the prose traces to `data/results/` or a JOURNAL-recorded audit;
  the substrate-audit figures in §3.1 are quoted from the 2026-08-23 audit and are
  marked pending regeneration by `ecvol audit substrate`.

## Switching to the official ACL class

1. Drop `acl.sty` and `acl_natbib.bst` into this directory (from the ACL Rolling
   Review template).
2. In `main.tex` replace the `\documentclass[...]{article}` + geometry + `\twocolumn`
   block with `\documentclass[11pt]{article}` and `\usepackage[review]{acl}`.
3. Change `\bibliographystyle{plainnat}` to `\bibliographystyle{acl_natbib}`.

Everything else (sections, tables, `refs.bib`) is template-agnostic.
