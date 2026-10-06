#!/usr/bin/env bash
# Build the reveal.js slideshow HTML from the notebook.
#
#   ./scripts/build_slides.sh
#
# Produces talk_en.slides.html in the repo root.
#
# To get a PDF of the slides (one slide per page), open the generated file in a
# browser with the ?print-pdf query, then print to PDF:
#
#   1. ./scripts/build_slides.sh
#   2. open "talk_en.slides.html?print-pdf"   (macOS; or open it in Chrome)
#   3. Cmd+P  ->  Destination: Save as PDF  ->  Save
#
# (reveal.js renders real slide pages only via the ?print-pdf route; a plain
#  print of talk_en.slides.html will not paginate correctly.)
set -euo pipefail

cd "$(dirname "$0")/.."

NB="talk_en.ipynb"

# Prefer running inside Sage's shell if available (matches the notebook kernel);
# fall back to a plain jupyter on PATH.
if command -v sage >/dev/null 2>&1; then
    sage -sh -c "jupyter nbconvert --to slides $NB"
else
    jupyter nbconvert --to slides "$NB"
fi

echo
echo "Wrote talk_en.slides.html"
echo "PDF: open \"talk_en.slides.html?print-pdf\" in a browser -> Cmd+P -> Save as PDF"
