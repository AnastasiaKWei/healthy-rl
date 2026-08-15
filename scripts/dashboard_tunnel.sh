#!/bin/bash
# scripts/dashboard_tunnel.sh [jobid] -- print the ssh tunnel for a running dashboard job.
# Run on the login node. Finds the newest dashboard-endpoint (or the one for JOBID).
#
# ARTIFACT_DIR / HEALTHY_RL_LOGIN_HOST already in the environment win over .env,
# the same precedence healthy_rl.config.load_env uses.
set -euo pipefail
_PRESET_ARTIFACT_DIR="${ARTIFACT_DIR:-}"
set -a; . "$(dirname "$0")/../.env" 2>/dev/null || true; set +a
[[ -n "$_PRESET_ARTIFACT_DIR" ]] && ARTIFACT_DIR="$_PRESET_ARTIFACT_DIR"
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
HP=$(cat "$EP"); NODE=${HP%%:*}; PORT=${HP##*:}
echo "from your laptop:  ssh -L ${PORT}:${NODE}:${PORT} ${HEALTHY_RL_LOGIN_HOST:-$(hostname -f)}"
echo "then open:         http://localhost:${PORT}"
