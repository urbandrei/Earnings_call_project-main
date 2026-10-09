# Editing this paper in Overleaf

This folder is a self-contained copy of the paper. Upload the zip to Overleaf
(New Project → Upload Project), set `main.tex` as the main document, and compile
with pdfLaTeX (the default). No extra style files are needed.

## What is where

| Path | Contents | Edit freely? |
|---|---|---|
| `main.tex` | Preamble, title and author block, section order | yes |
| `sections/00-abstract.tex` … `06-recommendations.tex` | Main text, one file per section | yes |
| `sections/09-limitations.tex`, `10-ethics.tex` | Limitations and Ethics (unnumbered, required by ACL) | yes |
| `sections/A1-complete-results.tex` | Appendix narrative | yes |
| `tables/*_main.tex`, `tables/code_audit.tex` rows | Generated from the result CSVs by scripts in the code repository | wording in captions is fine; please do not retype numbers |
| `tables/*.tex` (others) | Curated tables, each header names the CSV and run it came from | captions fine; numbers only via the repository |
| `figures/problems.tex` | Figure 1, drawn in TikZ | yes |
| `figures/fig_*.pdf` | Figures 2–4, generated from the result CSVs | regenerate in the repository |
| `refs.bib` | Bibliography | yes |
| `supplement.pdf` | The complete result grids (21 pp), for reference only | n/a |

Every number in the text and tables traces to a result file in the code repository
with a run manifest. If a number needs to change, the change is made there and the
tables regenerate, so please leave numeric edits to the repository side and flag
them in a comment instead.

## Conventions in the draft

- Line numbers are on for review (`\linenumbers` in `main.tex`); comment that line
  for a clean read.
- The author block holds placeholders. ARR review is anonymous, so it goes back to
  `Anonymous Author(s)` for submission.
- `\TODO{...}` (red) and `\pending{...}` (orange) macros exist for marking open
  items; none are in the text at the moment.
- The layout is a self-contained two-column approximation of the ACL style. To
  switch to the official class, drop `acl.sty` and `acl_natbib.bst` into the project,
  replace the `\documentclass` and geometry lines in `main.tex` with
  `\documentclass[11pt]{article}` plus `\usepackage[review]{acl}`, and change the
  bibliography style to `acl_natbib`. Expect the 11pt template to run about 20 %
  longer.

## Bringing edits back

Download the project as a zip from Overleaf and replace the same files under
`paper/` in the repository (sections, tables captions, figures/problems.tex,
refs.bib, main.tex). The repository's `paper/build.sh` reproduces the PDF.
