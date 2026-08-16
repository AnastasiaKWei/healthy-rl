#!/bin/zsh
# Gemma-3-12B on the POSSIBLE (original) split with test modification ALLOWED.
#
# Why this split. Gemma passes original only 1/15 while running 4.9 turns, so for
# this model "solvable" still produces a full repeated-failure loop. That makes it
# the hard-but-solvable control docs/elicitation.md calls mandatory: it separates
# affect caused by failing from affect caused by the task being impossible, which
# the conflicting arms cannot do on their own.
#
# Why hackable here specifically. On conflicting, neutralising the tests is the
# only way to pass, so a hack is a rational response to impossibility. On original
# the model can legitimately solve the problem, so any hack is gratuitous -- a much
# stronger signal about willingness to cheat.
#
# Baseline first: it is the arm every intervention is read against, so if the run
# is interrupted the comparison is still possible.
#
# Waits for any in-flight eval. The arms share one Colima VM and each holds up to
# --max-sandboxes containers; overlapping runs surface as sandbox timeouts recorded
# as sample errors.
set -u
cd "$(dirname "$0")/.." || exit 1

while pgrep -f step0_elicitation.py >/dev/null 2>&1; do sleep 60; done
echo "=== $(date +%H:%M:%S) starting; no other eval in flight ==="

COMMON=(--model openrouter/google/gemma-3-12b-it --scratchpad --affect-prompt
        --allow-test-modifications --splits original --epochs 3 --limit 5
        --working-limit 1800 --max-connections 16 --max-sandboxes 16)

run () {
  local label="$1"; shift
  echo "=== $(date +%H:%M:%S)  $label ==="
  ./.venv/bin/python experiments/step0_elicitation.py "${COMMON[@]}" "$@" \
    || echo "!!! $label exited $? -- continuing"
}

run "baseline (no mindset)"
for v in growth resilience appraisal; do
  run "v2 $v" --mindset "$v"
done

echo "=== $(date +%H:%M:%S)  all four arms finished ==="
