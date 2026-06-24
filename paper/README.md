# Paper — `ecvol` multimodal earnings-call volatility prediction

A **living manuscript scaffold**. Phases 0–3 (data, baselines, text ladder,
identity controls) are written with real numbers; audio / fusion / LLM / lookahead
sections are stubs marked with `\TODO{}` / `\pending{}` and fill in as those phases
land.

- **Format:** self-contained two-column (ACL/EMNLP-style), compiles with a stock TeX
  distribution — no external style files required.
- **Framing:** deliberately framing-neutral (the Path-A / Path-B gate, DESIGN.md §4,
  is resolved after the audio results).

## Build

Requires a TeX distribution (TeX Live or MiKTeX) with `latexmk`:

```sh
cd paper
latexmk -pdf main.tex          # -> main.pdf
latexmk -c                     # clean aux files
```

Without `latexmk`:

```sh
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

> No TeX toolchain is currently installed in this environment, so the PDF has not
> been built here. Install TeX Live / MiKTeX (or build in Overleaf — upload the
> `paper/` folder plus `data/results/result_table_1.tex` and `result_table_2.tex`)
> to render.

## Layout

| Path | Contents |
|---|---|
| `main.tex` | Document class, preamble, `\TODO`/`\pending` macros, `\input`s all sections |
| `refs.bib` | Bibliography, built from DESIGN.md §13 |
| `sections/` | `00`–`10` body sections + `A1` appendix (complete results) |
| `tables/` | Curated headline tables (`baselines`, `text`, `controls`) |
| `figures/` | `pipeline.tex` — TikZ stage-ladder schematic |

The **appendix** `\input`s the auto-generated grids
`../data/results/result_table_{1,2}.tex` (emitted by `ecvol report`, byte-identical
and CI-guarded). The curated `tables/baselines.tex` and `tables/text.tex` copy
selected cells from those files; if you re-run `ecvol report` and numbers change,
update the curated tables to match.

## Drafting conventions

- `\TODO{...}` (red) — content to write/finalise. `\pending{...}` (orange) — results
  blocked on an unfinished phase. Both are hidden when `\draftmode` is set false in
  `main.tex` (flip it for a clean render). Grep for them to find open work.
- Keep prose framing-neutral until the §4 gate is resolved; record the decision in
  `DECISIONS.md` and then rewrite the abstract/intro/conclusion headline.

## Switching to the official ACL/EMNLP class

1. Drop `acl.sty` and `acl_natbib.bst` into this directory (from the ACL Rolling
   Review / EMNLP template).
2. In `main.tex` replace the `\documentclass[...]{article}` + geometry + `\twocolumn`
   block with `\documentclass[11pt]{article}` and `\usepackage[review]{acl}`.
3. Change `\bibliographystyle{plainnat}` to `\bibliographystyle{acl_natbib}`.

Everything else (sections, tables, `refs.bib`) is template-agnostic.
