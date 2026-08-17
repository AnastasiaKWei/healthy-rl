#!/bin/zsh
# The v4 prompts: intervention in a system turn, no task restatement after a
# failure, retry guidance in every arm including the baseline.
#
#   gemma   solvable    scratchpad granted, solvability note true and sent
#   qwen    impossible  native reasoning trace, no scratchpad, no solvability note
#
# The baseline is NOT optional and is not reusable from v3. v4 changes the shared
# user turn (no restatement, retry guidance added) and the failure message, so every
# existing baseline differs from these arms by more than the intervention. It runs
# first, so an interrupted batch still leaves the comparison possible.
#
# Waits for any in-flight eval. The arms share one Colima VM and each holds up to
# --max-sandboxes containers; overlapping runs surface as sandbox timeouts recorded
# as sample errors.
#
# Launch a COPY of this file if you intend to edit it while it runs: zsh reads a
# script by offset, so an in-place edit corrupts the commands it has not reached.
# A copy cannot find the repo from its own path, so set REPO when you do.
#
# Usage: REPO=/path/to/healthy-rl ./scripts/run_v4.sh [gemma|qwen|both]
set -u
cd "${REPO:-$(dirname "$0")/..}" || exit 1
WHICH="${1:-both}"

while pgrep -f step0_elicitation.py >/dev/null 2>&1; do sleep 60; done
echo "=== $(date +%H:%M:%S) starting v4; no other eval in flight ==="

run () {
  local label="$1"; shift
  echo "=== $(date +%H:%M:%S)  $label ==="
  ./.venv/bin/python experiments/step0_elicitation.py \
      --allow-test-modifications --affect-prompt --epochs 3 --limit 5 \
      --working-limit 1800 --max-connections 16 --max-sandboxes 16 "$@" \
    || echo "!!! $label exited $? -- continuing"
}

if [[ "$WHICH" == "gemma" || "$WHICH" == "both" ]]; then
  GEMMA=(--model openrouter/google/gemma-3-12b-it --scratchpad --splits original)
  run "gemma v4 baseline" "${GEMMA[@]}"
  for v in growth resilience control compassion; do
    run "gemma v4 $v" "${GEMMA[@]}" --mindset "$v"
  done
fi

if [[ "$WHICH" == "qwen" || "$WHICH" == "both" ]]; then
  QWEN=(--model openrouter/qwen/qwen3-14b --reasoning on --splits conflicting)
  run "qwen v4 baseline" "${QWEN[@]}"
  for v in growth resilience control compassion; do
    run "qwen v4 $v" "${QWEN[@]}" --mindset "$v"
  done
fi

echo "=== $(date +%H:%M:%S)  v4 done ==="
