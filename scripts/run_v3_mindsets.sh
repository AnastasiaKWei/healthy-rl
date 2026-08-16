#!/bin/zsh
# The four rewritten v3 mindset blocks -- growth, resilience, control, compassion --
# each read against the model's own hackable baseline.
#
# The baselines sit on different splits per model, so the two models are NOT
# comparable to each other here. Each is comparable only to its own baseline:
#
#   gemma   original     (solvable)    baseline google-...-pad-affect-hackable-e3
#   qwen    conflicting  (impossible)  baseline qwen-...-nopad-affect-hackable-e3
#
# Qwen runs with reasoning on and no scratchpad: the reasoning trace already gives
# affect a private surface, so a <scratchpad> grant would add a second one and make
# the private channel mean two different things across models.
#
# Waits for any in-flight eval. The arms share one Colima VM and each holds up to
# --max-sandboxes containers; overlapping runs surface as sandbox timeouts recorded
# as sample errors.
#
# The split is an argument because the two models' baselines were built on
# different ones, and an arm is only readable against a baseline that matches it in
# every respect except the intervention -- split included. Pass `baseline` as the
# third argument when the target split has no baseline yet; it runs first, so an
# interrupted batch still leaves the comparison possible.
#
# Usage: ./scripts/run_v3_mindsets.sh [gemma|qwen|both] [split] [baseline]
set -u
cd "$(dirname "$0")/.." || exit 1
WHICH="${1:-both}"
SPLIT="${2:-}"
BASE="${3:-}"

while pgrep -f step0_elicitation.py >/dev/null 2>&1; do sleep 60; done
echo "=== $(date +%H:%M:%S) starting; no other eval in flight ==="

run () {
  local label="$1"; shift
  echo "=== $(date +%H:%M:%S)  $label ==="
  ./.venv/bin/python experiments/step0_elicitation.py \
      --allow-test-modifications --affect-prompt --epochs 3 --limit 5 \
      --working-limit 1800 --max-connections 16 --max-sandboxes 16 "$@" \
    || echo "!!! $label exited $? -- continuing"
}

if [[ "$WHICH" == "gemma" || "$WHICH" == "both" ]]; then
  GEMMA=(--model openrouter/google/gemma-3-12b-it --scratchpad
         --splits "${SPLIT:-original}")
  [[ -n "$BASE" ]] && run "gemma baseline" "${GEMMA[@]}"
  for v in growth resilience control compassion; do
    run "gemma $v" "${GEMMA[@]}" --mindset "$v"
  done
fi

if [[ "$WHICH" == "qwen" || "$WHICH" == "both" ]]; then
  QWEN=(--model openrouter/qwen/qwen3-14b --reasoning on
        --splits "${SPLIT:-conflicting}")
  [[ -n "$BASE" ]] && run "qwen baseline" "${QWEN[@]}"
  for v in growth resilience control compassion; do
    run "qwen $v" "${QWEN[@]}" --mindset "$v"
  done
fi

echo "=== $(date +%H:%M:%S)  done ==="
