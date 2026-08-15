#!/bin/zsh
# Rebuild viewer/transcripts.html from whatever .eval logs exist right now.
cd "$(dirname "$0")/.." || exit 1
./.venv/bin/python viewer/export_viewer.py logs -o viewer/data.json
./.venv/bin/python viewer/build_viewer.py viewer/viewer.html viewer/data.json viewer/transcripts.html
