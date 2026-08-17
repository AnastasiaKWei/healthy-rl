#!/bin/bash
# Run the five mindset-v4 arms sequentially against a local vllm-lens server.
#
# The single-GPU (RunPod) counterpart of scripts/mindset_cells.sh: one server,
# no slurm, arms one after another. Each arm gets its own generated config
# (base + `mindset:` + `out_dir:`) and its own out_dir, because resume refuses
# to mix arms in one JSONL -- that refusal is a feature, not an obstacle.
#
# Usage:
#   scripts/run_v4_arms.sh configs/rollouts_v4_Qwen3-14B.yaml Qwen3-14B [arms...]
#
# Arms default to all five. Requires ARTIFACT_DIR and a healthy server at
# $HEALTHY_RL_SERVER_URL (default http://127.0.0.1:8000).
set -euo pipefail
CFG=${1:?usage: run_v4_arms.sh <config.yaml> <model-name> [arms...]}
MODEL=${2:?model name}
shift 2
ARMS=("$@")
[ ${#ARMS[@]} -gt 0 ] || ARMS=(baseline growth resilience control compassion)
: "${ARTIFACT_DIR:?ARTIFACT_DIR must be set}"
URL=${HEALTHY_RL_SERVER_URL:-http://127.0.0.1:8000}
REPO=$(cd "$(dirname "$0")/.." && pwd)

curl -sf "$URL/health" >/dev/null || { echo "no healthy server at $URL" >&2; exit 1; }

GEN="$ARTIFACT_DIR/rollout-configs"
mkdir -p "$GEN"

# Generate every arm's config before the first run starts, so a typo aborts
# the whole plan instead of surfacing four hours in (mindset_cells.sh rule).
declare -a GENERATED
for arm in "${ARMS[@]}"; do
  gen="$GEN/$(basename "$CFG" .yaml)-$arm.yaml"
  cp "$CFG" "$gen"
  {
    echo ""
    echo "# --- appended by run_v4_arms.sh ---"
    [ "$arm" != baseline ] && echo "mindset: [$arm]"
    echo "out_dir: $ARTIFACT_DIR/rollouts/$MODEL/v4-$arm"
  } >> "$gen"
  GENERATED+=("$gen")
done

for i in "${!ARMS[@]}"; do
  arm=${ARMS[$i]}
  echo "=== [$(date +%H:%M:%S)] v4 arm: $arm"
  "$REPO/.venv/bin/python" "$REPO/scripts/run_rollouts.py" \
    --config "${GENERATED[$i]}" --model "$MODEL" --base-url "$URL"
done
echo "=== [$(date +%H:%M:%S)] all arms complete"
