#!/bin/bash
# Read ImpossibleBench rollout transcripts from the Inspect .eval logs.
#
# The logs MUST be read inside apptainer/eval.sif: they are written by
# inspect_ai 0.3.258, while the host venv is pinned to 0.3.69 (uv.lock resolves
# that for python < 3.13, and it cannot be raised without breaking vllm's
# dependency set). Reading them on the host fails with
# "NotImplementedError: That compression method is not supported".
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
    find "$LOGROOT" -name '*.eval' -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | cut -d' ' -f2-
    exit 0
fi

if [ $# -lt 2 ]; then
    echo "usage: $0 summary|show <log.eval> [SAMPLE]" >&2
    exit 2
fi
LOG="$2"
SAMPLE="${3:-}"
SCRATCH="$ARTIFACT_DIR/rollout-scratch/logread"
mkdir -p "$SCRATCH/tmp"

# Paths are rewritten to the container's view of the artifact bind.
CLOG="${LOG/$ARTIFACT_DIR//artifacts}"

apptainer exec --contain --cleanenv --writable-tmpfs \
    --bind "$PROJECT_DIR":/project:ro \
    --bind "$ARTIFACT_DIR":/artifacts:ro \
    --bind "$SCRATCH":/work:rw \
    --env TMPDIR=/work/tmp \
    --env MODE="$MODE" --env CLOG="$CLOG" --env SAMPLE="$SAMPLE" \
    apptainer/eval.sif python /project/scripts/_read_transcript.py
