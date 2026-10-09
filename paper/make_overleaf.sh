#!/usr/bin/env bash
# Package the paper as a self-contained Overleaf project.
#
# Usage:  bash paper/make_overleaf.sh          # -> paper/dist/ecvol-paper-overleaf-<gitsha>.zip
#
# Includes main.tex, refs.bib, the ACL style files (acl.sty, acl_natbib.bst), sections/,
# tables/ (tex only), figures/ (tex + pdf), OVERLEAF.md and supplement.pdf. Excludes build artefacts, generator scripts and
# anything that reaches outside paper/. The staged copy is test-compiled before zipping.
set -euo pipefail
cd "$(dirname "$0")"
sha=$(git rev-parse --short HEAD)
stage=$(mktemp -d)
name="ecvol-paper-overleaf-$sha"
mkdir -p "$stage/$name/sections" "$stage/$name/tables" "$stage/$name/figures" dist

cp main.tex refs.bib acl.sty acl_natbib.bst OVERLEAF.md "$stage/$name/"
cp sections/*.tex "$stage/$name/sections/"
cp tables/*.tex "$stage/$name/tables/"
cp figures/*.tex figures/*.pdf "$stage/$name/figures/"
[ -f supplement.pdf ] && cp supplement.pdf "$stage/$name/"

# test-compile the staged copy exactly as Overleaf would (pdflatex + bibtex)
( cd "$stage/$name" \
  && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null \
  && bibtex main >/dev/null \
  && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null \
  && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null )
if grep -qiE "(Reference|Citation) .* undefined" "$stage/$name/main.log"; then echo "undefined references in staged build" >&2; exit 1; fi
rm -f "$stage/$name"/main.{aux,bbl,blg,log,out,pdf} "$stage/$name"/sections/*.aux

( cd "$stage" && rm -f "$name.zip" && python -c "
import shutil, sys
shutil.make_archive('$name', 'zip', '.', '$name')
" )
mv "$stage/$name.zip" "dist/$name.zip"
rm -rf "$stage"
echo "wrote paper/dist/$name.zip"
unzip -l "dist/$name.zip" | tail -1
