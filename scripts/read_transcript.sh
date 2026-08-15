#!/bin/bash
# Read ImpossibleBench rollout transcripts from the Inspect .eval logs.
#
# Runs on the host venv (inspect_ai >= 0.3.258, same as apptainer/eval.sif, so
# the .eval zip compression is readable). `inspect view --log-dir inspect-logs`
# also works if you prefer the browser UI.
#
# Usage:
#   scripts/read_transcript.sh list                     # all .eval logs, newest first
#   scripts/read_transcript.sh summary <log.eval>       # samples, scores, turn/token counts
#   scripts/read_transcript.sh show <log.eval> [SAMPLE] # full message transcript
#
# SAMPLE is a task_id such as lcbhard_3 (default: the first sample in the log).
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a

MODE="${1:-list}"
LOGROOT="$ARTIFACT_DIR/rollouts"

if [ "$MODE" = "list" ]; then
    find -L "$LOGROOT" -name '*.eval' -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | cut -d' ' -f2-
    exit 0
fi

if [ $# -lt 2 ]; then
    echo "usage: $0 summary|show <log.eval> [SAMPLE]" >&2
    exit 2
fi

MODE="$MODE" CLOG="$2" SAMPLE="${3:-}" exec .venv/bin/python scripts/_read_transcript.py
