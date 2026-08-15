#!/bin/bash
# One-screen pilot health check. Read-only; safe to run from anywhere, any time.
# Prints queue state, per-job progress, and STALL warnings for jobs that are
# running but whose outputs have stopped advancing.
cd "$(dirname "$0")/.." || exit 1
set -a; . ./.env 2>/dev/null; set +a
STALL_MIN=${STALL_MIN:-15}

echo "=== $(date '+%F %T') ==="
echo "--- queue ---"
squeue -u "$USER" -o "%.10i %.26j %.9T %.14R %.8M %.10L" 2>/dev/null || echo "squeue unavailable"

echo "--- recent job exits (last 3h) ---"
sacct -u "$USER" --starttime=now-3hours --format=JobID%12,JobName%26,State%14,Elapsed,ExitCode 2>/dev/null \
  | grep -vE '\.(batch|extern|[0-9]+) ' | tail -15

echo "--- smoke results ---"
for f in "$ARTIFACT_DIR"/smoke/*/*/smoke.json; do
  [ -e "$f" ] || { echo "  (none yet)"; break; }
  python3 -c "
import json,sys
d=json.load(open('$f'))
checks=d.get('checks',d)
if isinstance(checks,dict):
    line=', '.join(f\"{k}={'PASS' if (v.get('passed') if isinstance(v,dict) else v) else 'FAIL'}\" for k,v in checks.items() if k!='architecture')
else: line=str(checks)[:120]
print('  %-24s %s' % (d.get('model','?'), line))
" 2>/dev/null || echo "  $f (unparseable)"
done

echo "--- output freshness (STALL if idle > ${STALL_MIN}m while job RUNNING) ---"
now=$(date +%s)
# Only files belonging to a CURRENTLY RUNNING job can stall. A finished job's log
# is stale by definition, and flagging it trains us to ignore the alert.
runids=$(squeue -u "$USER" -h -t R -o "%i" 2>/dev/null)
[ -z "$runids" ] && echo "  (no running jobs — nothing can stall)"
for f in logs/*.out "$ARTIFACT_DIR"/rollouts/*/*/*.jsonl "$ARTIFACT_DIR"/activations/*/*/*; do
  [ -f "$f" ] || continue
  live=""
  for j in $runids; do case "$f" in *"$j"*) live=1;; esac; done
  # artifact files carry no job id; treat them as live whenever any job runs
  case "$f" in "$ARTIFACT_DIR"/*) [ -n "$runids" ] && live=1;; esac
  [ -n "$live" ] || continue
  age=$(( (now - $(stat -c %Y "$f")) / 60 ))
  n=""
  case "$f" in *.jsonl) n=" records=$(wc -l < "$f")";; esac
  flag=""
  [ "$age" -gt "$STALL_MIN" ] && flag="  <-- STALL?"
  printf "  %-58s %4dm ago%s%s\n" "$(basename "$f")" "$age" "$n" "$flag"
done | tail -20

echo "--- errors in logs (last 3) ---"
grep -lE "Traceback|CUDA out of memory|ResolutionImpossible|srun: error|OOM" logs/*.out 2>/dev/null | tail -3 \
  | while read -r L; do echo "  $L:"; grep -E "Traceback|Error|error:|OOM|out of memory" "$L" | tail -3 | sed 's/^/    /'; done
