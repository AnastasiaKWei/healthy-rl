#!/bin/bash
# scripts/dashboard_tunnel.sh [jobid] -- print the ssh tunnel for a running dashboard job.
# Run on the login node. Finds the newest dashboard-endpoint (or the one for JOBID).
#
# ARTIFACT_DIR and HEALTHY_RL_LOGIN_HOST already in the environment win over the
# .env file, the same precedence healthy_rl.config.load_env uses. `set -a; . .env`
# overwrites both, so each is saved by name and put back afterwards.
# HEALTHY_RL_ENV_FILE overrides which .env is read (tests pin it at /dev/null).
set -euo pipefail
ENV_FILE="${HEALTHY_RL_ENV_FILE:-$(dirname "$0")/../.env}"
PRESET_ARTIFACT_DIR="${ARTIFACT_DIR:-}"
PRESET_LOGIN_HOST="${HEALTHY_RL_LOGIN_HOST:-}"
set -a; . "$ENV_FILE" 2>/dev/null || true; set +a
if [[ -n "$PRESET_ARTIFACT_DIR" ]]; then ARTIFACT_DIR="$PRESET_ARTIFACT_DIR"; fi
if [[ -n "$PRESET_LOGIN_HOST" ]]; then HEALTHY_RL_LOGIN_HOST="$PRESET_LOGIN_HOST"; fi
: "${ARTIFACT_DIR:?ARTIFACT_DIR must be set in .env or the environment}"

# `|| true`: with pipefail a no-match ls would end the script here, silently and
# with its own exit code, instead of reaching the explanation below.
if [[ $# -ge 1 ]]; then
    EP=$(ls -1 "$ARTIFACT_DIR"/serve/*/"$1"/dashboard-endpoint 2>/dev/null | head -1) || true
else
    EP=$(ls -1t "$ARTIFACT_DIR"/serve/*/*/dashboard-endpoint 2>/dev/null | head -1) || true
fi
[[ -n "${EP:-}" && -f "$EP" ]] || {
    echo "no dashboard-endpoint found under $ARTIFACT_DIR/serve${1:+ for job $1}" >&2
    echo "(the dashboard stage writes it once uvicorn has a port; is the job still loading the model?)" >&2
    exit 1
}

# The stage removes its endpoint file on the way out, but a job killed outright
# (SIGKILL, node failure) leaves one behind, and ssh into a dead node's port
# fails in a way that looks like the dashboard is broken. Say so instead.
JOB=$(basename "$(dirname "$EP")")
if command -v squeue >/dev/null 2>&1 && ! squeue -h -j "$JOB" -o %i 2>/dev/null | grep -q .; then
    echo "warning: job $JOB is not in the queue; $EP is probably stale" >&2
fi

HP=$(cat "$EP"); NODE=${HP%%:*}; PORT=${HP##*:}
echo "from your laptop:  ssh -L ${PORT}:${NODE}:${PORT} ${HEALTHY_RL_LOGIN_HOST:-$(hostname -f)}"
echo "then open:         http://localhost:${PORT}"
