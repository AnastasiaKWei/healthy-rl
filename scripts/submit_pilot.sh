#!/usr/bin/env bash
#
# Submit the desperation pilot: per model, the GPU serve job that produces
# activations, the CPU jobs that build and gate the directions, and the GPU
# serve job that runs the rollouts. Jobs are chained with --dependency=afterok
# within a model and are completely independent across models, so one model
# failing or queueing badly never blocks another.
#
# Stages 1 and 2 (fetch_stories, fetch_bench) are NOT submitted: compute nodes
# have no DNS, so they run on the login node before this script. Their
# artifacts are checked for here.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# GPU routing, from the spec. The 69 GB and 66 GB checkpoints need the 92 GB
# L40S pair; the smaller ones fit the 80 GB A100 pair. Every node has exactly
# two GPUs, so tensor-parallel 2 on one node is the only option either way.
gres_for_model() {
    case "$1" in
        Olmo-3.1-32B-Think) echo "gpu:L40S-46G:2" ;;
        gemma-4-31B-it)     echo "gpu:L40S-46G:2" ;;
        Qwen3.6-27B)        echo "gpu:A100-40G:2" ;;
        Muse-Glimmer-30B)   echo "gpu:A100-40G:2" ;;
        *) return 1 ;;
    esac
}

# Muse-Glimmer-30B is absent from the default set: ruling R8 dropped it from
# the pilot because vLLM 0.27.1 does not implement
# MuseGlimmerForConditionalGeneration. Its routing above is kept so that
# --models Muse-Glimmer-30B still works if that changes.
DEFAULT_MODELS=(Olmo-3.1-32B-Think gemma-4-31B-it Qwen3.6-27B)

SMOKE_SCRIPT="scripts/smoke.py"
SMOKE_CONFIG="configs/smoke.yaml"
ACTS_SCRIPT="scripts/extract_acts.py"
ACTS_CONFIG="configs/extract_acts.yaml"
VECTORS_SCRIPT="scripts/build_vectors.py"
VECTORS_CONFIG="configs/build_vectors.yaml"
GATE_SCRIPT="scripts/gate.py"
GATE_CONFIG="configs/gate.yaml"
ROLLOUT_SCRIPT="scripts/run_rollouts.py"
ROLLOUT_CONFIG="configs/rollouts.yaml"

SIF="apptainer/eval.sif"

ACTS_TIME="8:00:00"
CPU_TIME="1:00:00"
ROLLOUT_TIME="12:00:00"

DRY_RUN=0
SKIP_PREFLIGHT=0
MODELS=()

usage() {
    cat <<'USAGE'
usage: scripts/submit_pilot.sh [options]

  --models "M1 M2"      models to submit (default: Olmo-3.1-32B-Think
                        gemma-4-31B-it Qwen3.6-27B)
  --dry-run             print the sbatch commands without submitting;
                        preflight failures are downgraded to warnings
  --skip-preflight      submit without checking inputs exist
  --acts-time HH:MM:SS      wall clock for the smoke+extract_acts job (8:00:00)
  --cpu-time HH:MM:SS       wall clock for the build_vectors and gate jobs (1:00:00)
  --rollout-time HH:MM:SS   wall clock for the rollout job (12:00:00)
  -h, --help            this message

Per model this submits four jobs:

  1  <model>-acts   serve.slurm  GPU  smoke, then extract_acts
  2  <model>-vecs   stage.slurm  CPU  build_vectors        (afterok 1)
  3  <model>-gate   stage.slurm  CPU  gate                 (afterok 2)
  4  <model>-roll   serve.slurm  GPU  run_rollouts         (afterok 2)
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models) read -r -a MODELS <<< "$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
        --acts-time) ACTS_TIME="$2"; shift 2 ;;
        --cpu-time) CPU_TIME="$2"; shift 2 ;;
        --rollout-time) ROLLOUT_TIME="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "submit_pilot.sh: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
    MODELS=("${DEFAULT_MODELS[@]}")
fi

cd "$REPO_ROOT"

if [[ ! -f .env ]]; then
    echo "submit_pilot.sh: no .env at $REPO_ROOT; MODEL_DIR/ARTIFACT_DIR/PROJECT_DIR come from it" >&2
    exit 1
fi
set -a
# shellcheck disable=SC1091
source ./.env
set +a

: "${MODEL_DIR:?MODEL_DIR must be set in .env}"
: "${ARTIFACT_DIR:?ARTIFACT_DIR must be set in .env}"
: "${PROJECT_DIR:?PROJECT_DIR must be set in .env}"

PREFLIGHT_FAILED=0

problem() {
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "warning: $*" >&2
    else
        echo "error: $*" >&2
        PREFLIGHT_FAILED=1
    fi
}

require_file() {
    [[ -f "$1" ]] || problem "missing file: $1${2:+  ($2)}"
}

require_dir() {
    [[ -d "$1" ]] || problem "missing directory: $1${2:+  ($2)}"
}

preflight() {
    require_dir .venv "run 'uv sync' at the repo root"
    require_file "$SIF" "build it on the login node: apptainer build $SIF apptainer/eval.def"

    require_file "$ARTIFACT_DIR/stories/v1/manifest.json" \
        "run scripts/fetch_stories.py on the LOGIN node; compute nodes have no DNS"
    require_file "$ARTIFACT_DIR/bench/v1/manifest.json" \
        "run scripts/fetch_bench.py on the LOGIN node; compute nodes have no DNS"

    local f
    for f in "$SMOKE_SCRIPT" "$SMOKE_CONFIG" "$ACTS_SCRIPT" "$ACTS_CONFIG" \
             "$VECTORS_SCRIPT" "$VECTORS_CONFIG" "$GATE_SCRIPT" "$GATE_CONFIG" \
             "$ROLLOUT_SCRIPT" "$ROLLOUT_CONFIG" \
             slurm/serve.slurm slurm/stage.slurm; do
        require_file "$f"
    done

    local model
    for model in "${MODELS[@]}"; do
        if ! gres_for_model "$model" >/dev/null; then
            echo "error: no GPU routing defined for model '$model'" >&2
            PREFLIGHT_FAILED=1
            continue
        fi
        require_dir "$MODEL_DIR/$model" "checkpoint"
    done
}

if [[ $SKIP_PREFLIGHT -eq 0 ]]; then
    preflight
    if [[ $PREFLIGHT_FAILED -eq 1 ]]; then
        echo "submit_pilot.sh: preflight failed; nothing submitted" >&2
        exit 1
    fi
fi

mkdir -p logs

# Runs sbatch, or prints it under --dry-run. Echoes the job id on stdout so
# callers can chain dependencies; everything human-readable goes to stderr.
submit() {
    local -a cmd=(sbatch --parsable "$@")
    if [[ $DRY_RUN -eq 1 ]]; then
        printf '  %s\n' "$(printf '%q ' "${cmd[@]}")" >&2
        printf 'DRYRUN\n'
        return 0
    fi
    local job_id
    job_id="$("${cmd[@]}")"
    printf '%s\n' "$job_id"
}

dep() {
    if [[ "$1" == "DRYRUN" ]]; then
        printf -- '--dependency=afterok:<%s>\n' "$2"
    else
        printf -- '--dependency=afterok:%s\n' "$1"
    fi
}

SUMMARY=()

for MODEL in "${MODELS[@]}"; do
    GRES="$(gres_for_model "$MODEL")"
    echo "=== $MODEL  ($GRES) ===" >&2

    ACTS_ID="$(submit \
        --job-name="${MODEL}-acts" \
        --gres="$GRES" \
        --time="$ACTS_TIME" \
        slurm/serve.slurm \
        --model "$MODEL" \
        --config "$ACTS_CONFIG" \
        --stage "${SMOKE_SCRIPT}:${SMOKE_CONFIG}" \
        --stage "${ACTS_SCRIPT}:${ACTS_CONFIG}")"

    VECS_ID="$(submit \
        --job-name="${MODEL}-vecs" \
        --time="$CPU_TIME" \
        "$(dep "$ACTS_ID" "${MODEL}-acts")" \
        slurm/stage.slurm \
        "$VECTORS_SCRIPT" "$VECTORS_CONFIG" --model "$MODEL")"

    GATE_ID="$(submit \
        --job-name="${MODEL}-gate" \
        --time="$CPU_TIME" \
        "$(dep "$VECS_ID" "${MODEL}-vecs")" \
        slurm/stage.slurm \
        "$GATE_SCRIPT" "$GATE_CONFIG" --model "$MODEL")"

    ROLL_ID="$(submit \
        --job-name="${MODEL}-roll" \
        --gres="$GRES" \
        --time="$ROLLOUT_TIME" \
        "$(dep "$VECS_ID" "${MODEL}-vecs")" \
        slurm/serve.slurm \
        --model "$MODEL" \
        --config "$ROLLOUT_CONFIG" \
        --stage "${ROLLOUT_SCRIPT}:${ROLLOUT_CONFIG}")"

    SUMMARY+=("$(printf '%-22s acts=%-10s vecs=%-10s gate=%-10s roll=%-10s' \
        "$MODEL" "$ACTS_ID" "$VECS_ID" "$GATE_ID" "$ROLL_ID")")
done

echo >&2
if [[ $DRY_RUN -eq 1 ]]; then
    echo "dry run: nothing submitted" >&2
else
    echo "submitted job ids:"
fi
printf '%s\n' "${SUMMARY[@]}"
