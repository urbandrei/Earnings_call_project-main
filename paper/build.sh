#!/usr/bin/env bash
# Build the ecvol paper PDFs.
#
# Usage:   ./build.sh               # build main.pdf
#          ./build.sh supplement    # build supplement.pdf (complete generated grids)
#          ./build.sh all           # build both
#          ./build.sh clean         # remove LaTeX aux files (keep the PDFs)
#          ./build.sh realclean     # remove aux files AND the PDFs
#
# Works with a stock TeX distribution (TeX Live / MiKTeX). Prefers `latexmk`;
# falls back to the explicit pdflatex -> bibtex -> pdflatex x2 sequence.
# Run from the paper/ directory (the script cd's there itself).
set -euo pipefail

cd "$(dirname "$0")"

clean() {
  for m in main supplement; do
    rm -f "$m".{aux,bbl,blg,log,out,toc,lof,lot,fls,fdb_latexmk,synctex.gz} 2>/dev/null || true
  done
  rm -f sections/*.aux 2>/dev/null || true
}

build_one() {
  local MAIN="$1"
  # latexmk is a Perl script; on MiKTeX without a Perl interpreter it cannot run, so
  # only use it when both latexmk AND perl are present. Otherwise drive pdflatex directly.
  if command -v latexmk >/dev/null 2>&1 && command -v perl >/dev/null 2>&1; then
    echo "==> Building $MAIN with latexmk"
    latexmk -pdf -interaction=nonstopmode -halt-on-error "$MAIN".tex
  elif command -v pdflatex >/dev/null 2>&1; then
    echo "==> Building $MAIN with pdflatex + bibtex"
    pdflatex -interaction=nonstopmode -halt-on-error "$MAIN".tex
    bibtex "$MAIN" || true            # tolerate 'no \citation' (supplement has none)
    pdflatex -interaction=nonstopmode -halt-on-error "$MAIN".tex
    pdflatex -interaction=nonstopmode -halt-on-error "$MAIN".tex
  else
    echo "ERROR: no TeX engine found (need latexmk or pdflatex)." >&2
    echo "Install TeX Live or MiKTeX, or build paper/ on Overleaf." >&2
    exit 1
  fi
  echo "==> Done: $(pwd)/$MAIN.pdf"
}

case "${1:-build}" in
  clean)      clean; echo "Cleaned aux files."; exit 0 ;;
  realclean)  clean; rm -f main.pdf supplement.pdf; echo "Cleaned aux files and PDFs."; exit 0 ;;
  build)      build_one main ;;
  supplement) build_one supplement ;;
  all)        build_one main; build_one supplement ;;
  *) echo "Unknown argument: $1" >&2; echo "Use: build | supplement | all | clean | realclean" >&2; exit 2 ;;
esac
