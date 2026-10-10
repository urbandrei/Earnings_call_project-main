# Editing this paper in Overleaf

Upload the zip as a new project (New Project, Upload Project). `main.tex` is the
main document and pdfLaTeX is the compiler. The ACL style files are included.

## Files

| Path | What it is | Edit? |
|---|---|---|
| `main.tex` | Preamble, title, author block, section order | yes |
| `sections/00-abstract.tex` to `07-recommendations.tex` | The main text, one file per section | yes |
| `sections/08-limitations.tex`, `09-ethics.tex` | Limitations and Ethical Considerations (unnumbered) | yes |
| `sections/10-appendix.tex` | Appendix text | yes |
| `tables/*.tex` | Tables. The first line of each file names the result file it comes from | captions yes, numbers no |
| `figures/problems.tex` | Figure 1, drawn in TikZ | yes |
| `figures/fig_*.pdf` | Figures 2 to 4, generated from the result files | no |
| `refs.bib` | Bibliography | yes |
| `acl.sty`, `acl_natbib.bst` | Official ACL style | no |
| `supplement.pdf` | Complete result grids, for reference | no |

Every number in the text and tables comes from a result file in the code
repository. If a number looks wrong, leave a comment rather than retyping it, and
it will be fixed and regenerated on the repository side.

## Notes

- `\usepackage[review]{acl}` prints line numbers and hides the authors. Change
  `review` to `preprint` to see the author block, or to `final` for camera-ready.
- ARR allows eight pages of main text. Limitations, Ethical Considerations,
  references and the appendix do not count. The main text currently ends at the
  bottom of page 8, so a new sentence has to be paid for with a cut.
- To bring edits back, download the project zip and replace the same files under
  `paper/` in the repository.
