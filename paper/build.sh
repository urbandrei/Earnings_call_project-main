#!/usr/bin/env bash
# Build the ecvol paper PDF.
#
# Usage:   ./build.sh            # build main.pdf
#          ./build.sh clean      # remove LaTeX aux files (keep main.pdf)
#          ./build.sh realclean  # remove aux files AND main.pdf
#
# Works with a stock TeX distribution (TeX Live / MiKTeX). Prefers `latexmk`;
# falls back to the explicit pdflatex -> bibtex -> pdflatex x2 sequence.
# Run from the paper/ directory (the script cd's there itself).
set -euo pipefail

cd "$(dirname "$0")"
MAIN=main

clean() {
  rm -f "$MAIN".{aux,bbl,blg,log,out,toc,lof,lot,fls,fdb_latexmk,synctex.gz} \
        sections/*.aux 2>/dev/null || true
}

case "${1:-build}" in
  clean)     clean; echo "Cleaned aux files."; exit 0 ;;
  realclean) clean; rm -f "$MAIN".pdf; echo "Cleaned aux files and $MAIN.pdf."; exit 0 ;;
  build)     ;;
  *) echo "Unknown argument: $1" >&2; echo "Use: build | clean | realclean" >&2; exit 2 ;;
esac

# latexmk is a Perl script; on MiKTeX without a Perl interpreter it cannot run, so
# only use it when both latexmk AND perl are present. Otherwise drive pdflatex directly.
if command -v latexmk >/dev/null 2>&1 && command -v perl >/dev/null 2>&1; then
  echo "==> Building with latexmk"
  latexmk -pdf -interaction=nonstopmode -halt-on-error "$MAIN".tex
elif command -v pdflatex >/dev/null 2>&1; then
  echo "==> Building with pdflatex + bibtex"
  pdflatex -interaction=nonstopmode -halt-on-error "$MAIN".tex
  bibtex "$MAIN" || true            # tolerate 'no \citation' on first pass
  pdflatex -interaction=nonstopmode -halt-on-error "$MAIN".tex
  pdflatex -interaction=nonstopmode -halt-on-error "$MAIN".tex
else
  echo "ERROR: no TeX engine found (need latexmk or pdflatex)." >&2
  echo "Install TeX Live or MiKTeX, or build paper/ on Overleaf." >&2
  exit 1
fi

echo "==> Done: $(pwd)/$MAIN.pdf"
