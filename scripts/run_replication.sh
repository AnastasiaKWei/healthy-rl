#!/bin/zsh
# Replication at 25 tasks. The pilot ran 5 tasks x 3 epochs, so every number so far
# describes 5 specific programming problems; adding epochs resamples those same
# five. This runs tasks 5-29 at 2 epochs -- 50 episodes across 25 problems -- which
# is what buys generalisation.
#
# It is a REPLICATION, not an extension. The 5-task arms are not pooled in:
#
#   - --limit takes the first n tasks, so a 30-task run is a superset of the 5-task
#     run and merging would double-count the overlap under colliding epoch numbers
#   - the v3 resilience reminder was edited mid-project, so the Gemma-solvable and
#     Qwen-impossible v3 arms ran slightly different text
#   - 5x3 and 25x2 clusters are unbalanced
#
# So the pilot stands as the pilot, and this stands on its own.
#
# Arms per model: v3 baseline + 4 constructs, v4 baseline + 4 constructs, and the
# four v4 ablations. The ablations are included because the largest result in the
# project is theirs (task restatement +1.20, retry guidance +0.96), and a headline
# measured on 5 problems is the one most in need of 25.
#
# Gemma runs the solvable split, Qwen the impossible split, matching their pilots.
#
# Usage: REPO=/path/to/healthy-rl ./scripts/run_replication.sh [gemma|qwen]
set -u
cd "${REPO:-$(dirname "$0")/..}" || exit 1
WHICH="${1:-gemma}"

TASKS=()
for i in {5..29}; do TASKS+=("lcbhard_$i"); done

while pgrep -f step0_elicitation.py >/dev/null 2>&1; do sleep 60; done
echo "=== $(date +%H:%M:%S) starting $WHICH replication, 25 tasks x 2 epochs ==="

run () {
  local label="$1"; shift
  echo "=== $(date +%H:%M:%S)  $label ==="
  ./.venv/bin/python experiments/step0_elicitation.py \
      --allow-test-modifications --affect-prompt --epochs 2 \
      --task-ids "${TASKS[@]}" \
      --working-limit 1800 --max-connections 16 --max-sandboxes 16 "$@" \
    || echo "!!! $label exited $? -- continuing"
}

if [[ "$WHICH" == "gemma" ]]; then
  M=(--model openrouter/google/gemma-3-12b-it --scratchpad --splits original)
else
  M=(--model openrouter/qwen/qwen3-14b --reasoning on --splits conflicting)
fi

# v3 first: it carries the decoupling result, which is the one this project set out
# to test, so an interrupted run still leaves that answerable.
run "$WHICH v3 baseline" "${M[@]}" --prompt-version 3
for v in growth resilience control compassion; do
  run "$WHICH v3 $v" "${M[@]}" --prompt-version 3 --mindset "$v"
done

run "$WHICH v4 baseline" "${M[@]}"
for v in growth resilience control compassion; do
  run "$WHICH v4 $v" "${M[@]}" --mindset "$v"
done

run "$WHICH abl restate"  "${M[@]}" --task-reminder on
run "$WHICH abl noretry"  "${M[@]}" --retry-guidance off
run "$WHICH abl nonote"   "${M[@]}" --solvable-note off
run "$WHICH abl errlabel" "${M[@]}" --output-label error

echo "=== $(date +%H:%M:%S)  $WHICH replication done ==="
