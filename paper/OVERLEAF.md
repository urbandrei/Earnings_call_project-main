# Editing this paper in Overleaf

This folder is a self-contained copy of the paper. Upload the zip to Overleaf
(New Project → Upload Project), set `main.tex` as the main document, and compile
with pdfLaTeX (the default). The official ACL style files (`acl.sty`, `acl_natbib.bst`)
are included, so nothing else needs to be installed.

## What is where

| Path | Contents | Edit freely? |
|---|---|---|
| `main.tex` | Preamble, title and author block, section order | yes |
| `acl.sty`, `acl_natbib.bst` | Official ACL style, unchanged from github.com/acl-org/acl-style-files | no |
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

- The paper uses the official ACL class in `review` mode (`\usepackage[review]{acl}`
  in `main.tex`), which prints line numbers and replaces the author block with
  "Anonymous ACL submission". Change `review` to `preprint` to see the author
  names without line numbers, or to `final` for the camera-ready version.
- The author block holds placeholder names and affiliations. It is ignored in
  `review` mode, so it can be filled in at any time.
- `\TODO{...}` (red) and `\pending{...}` (orange) macros exist for marking open
  items; none are in the text at the moment.
- ARR long papers get eight pages for the main text. Limitations, Ethical
  Considerations, references and the appendix do not count. The main text currently
  ends on page 8 with some room to spare.

## Bringing edits back

Download the project as a zip from Overleaf and replace the same files under
`paper/` in the repository (sections, tables captions, figures/problems.tex,
refs.bib, main.tex). The repository's `paper/build.sh` reproduces the PDF.
