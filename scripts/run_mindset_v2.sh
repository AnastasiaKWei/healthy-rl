#!/bin/zsh
# Run the three v2 mindset arms on Gemma-3-12B and Qwen3-14B.
#
# Sequential on purpose. Each run holds up to --max-sandboxes Docker containers
# and the arms share one Colima VM, so running them concurrently trades wall
# clock for sandbox timeouts that land in the logs as sample errors.
#
# Conflicting split only, matching the v1 mindset arms so the comparison holds.
# The baseline for both models is the existing -affect-e3 arm: segments other
# than the mindset block are byte-identical in v2, so it needs no re-run.
#
# Usage: ./scripts/run_mindset_v2.sh 2>&1 | tee logs/run_mindset_v2.log
set -u
cd "$(dirname "$0")/.." || exit 1

EPOCHS=3
LIMIT=5

run () {   # run <label> <extra args...>
  local label="$1"; shift
  echo "=== $(date +%H:%M:%S)  $label ==="
  ./.venv/bin/python experiments/step0_elicitation.py \
      --epochs "$EPOCHS" --limit "$LIMIT" --splits conflicting \
      --affect-prompt "$@" \
    || echo "!!! $label exited $? -- continuing to the next arm"
}

for v in growth resilience appraisal; do
  run "gemma-3-12b  $v" \
      --model openrouter/google/gemma-3-12b-it --scratchpad --mindset "$v"
done

for v in growth resilience appraisal; do
  run "qwen3-14b    $v" \
      --model openrouter/qwen/qwen3-14b --reasoning on --mindset "$v"
done

echo "=== $(date +%H:%M:%S)  all six arms finished ==="
