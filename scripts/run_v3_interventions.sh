#!/bin/zsh
# The three interventions from docs/interventions.md, each in its own channel.
#
#   control     prompt block   -- Maier & Seligman (2016), separate what is yours
#                                 to choose from what is not
#   compassion  prompt block   -- Leary et al. (2007), aimed at Gemma's documented
#                                 self-judgment failure mode
#   wise        feedback msg   -- Yeager et al. (2014). NOT a prompt block: the
#                                 claim is that the standard and the assurance
#                                 arrive inside the critical feedback
#   neutral     feedback msg   -- the control docs/interventions.md 5.5 requires,
#                                 holding the interlocutor fixed and dropping the
#                                 support, since wise-vs-baseline otherwise moves
#                                 both at once
#
# Each model runs against the hackable baseline it already has, which is on a
# different split per model: Gemma's is solvable, Qwen's is impossible. So the two
# models are NOT comparable to each other here -- each is comparable to its own
# baseline, which is what the arms are for.
#
# Usage: ./scripts/run_v3_interventions.sh [gemma|qwen|both]
set -u
cd "$(dirname "$0")/.." || exit 1
WHICH="${1:-both}"

while pgrep -f step0_elicitation.py >/dev/null 2>&1; do sleep 60; done

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
  run "gemma control"    "${GEMMA[@]}" --mindset control
  run "gemma compassion" "${GEMMA[@]}" --mindset compassion
  run "gemma wise"       "${GEMMA[@]}" --feedback wise
  run "gemma neutral"    "${GEMMA[@]}" --feedback neutral
fi

if [[ "$WHICH" == "qwen" || "$WHICH" == "both" ]]; then
  QWEN=(--model openrouter/qwen/qwen3-14b --reasoning on --splits conflicting)
  run "qwen control"    "${QWEN[@]}" --mindset control
  run "qwen compassion" "${QWEN[@]}" --mindset compassion
  run "qwen wise"       "${QWEN[@]}" --feedback wise
  run "qwen neutral"    "${QWEN[@]}" --feedback neutral
fi

echo "=== $(date +%H:%M:%S)  done ==="
