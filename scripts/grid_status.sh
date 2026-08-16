#!/bin/bash
# Status of the affect x split cross product. Read-only; safe any time, anywhere.
#
# One row per model x condition cell of the 2x2 -- {affect prompt on, off} x
# {conflicting split, original split} -- showing records written, queue state of
# its three shards, and whether the shards agree on the bench split they read.
# Reports what the runs ARE doing rather than what they were asked to do: the
# expensive failures in this pilot have all been silent ones, where a job ran for
# hours against the wrong parquet, the wrong cap, or no hook at all.
cd "$(dirname "$0")/.." || exit 1
set -a; . ./.env 2>/dev/null; set +a

MODELS=${MODELS:-"Qwen3.5-9B Ministral-3-14B-Reasoning-2512 Qwen3-14B Nemotron-3-Nano-4B-BF16"}
CELLS=${CELLS:-"d6 aff6 pos6 affpos6"}
TARGET=${TARGET:-24}

echo "=== grid $(date '+%F %T') ==="
printf '%-32s %-8s %5s %-22s %s\n' MODEL CELL RECS "QUEUE (R/PD/done of 3)" SPLIT
for m in $MODELS; do
  for c in $CELLS; do
    dir="$ARTIFACT_DIR/rollouts/$m/$c"
    recs=$(cat "$dir"/*.jsonl 2>/dev/null | wc -l)
    # Split as the RUN reports it, not as the config asked. summary.json is
    # written before any rollout, so this is available as soon as a job starts.
    # Runs from before 2026-08-15 have no bench_split key and were all
    # conflicting -- flag them as such rather than as unknown.
    split=$(python3 -c "
import glob,json
v={json.load(open(f)).get('bench_split') or 'conflicting*' for f in glob.glob('$dir/summary*.json')}
print('-' if not v else ('/'.join(sorted(v))+('  <-- MIXED SPLITS' if len(v)>1 else '')))
" 2>/dev/null || echo '?')
    r=$(squeue -u "$USER" -h -t R -o "%j" 2>/dev/null | grep -c "^$m-$c-s")
    p=$(squeue -u "$USER" -h -t PD -o "%j" 2>/dev/null | grep -c "^$m-$c-s")
    # A cell can have MORE than its three shards in flight: a slow cell gets
    # dependent continuation jobs queued behind the running ones. Deriving
    # "done" by subtraction then prints nonsense like "done-3", so clamp it and
    # say plainly when the cell is oversubscribed.
    done=$((3 - r - p))
    if [ "$done" -lt 0 ]; then
      q=$(printf 'R%d PD%d +cont' "$r" "$p")
    else
      q=$(printf 'R%d PD%d done%d' "$r" "$p" "$done")
    fi
    flag=""
    [ "$recs" -ge "$TARGET" ] && flag="  complete"
    [ "$recs" -lt "$TARGET" ] && [ $((r + p)) -eq 0 ] && flag="  <-- SHORT, nothing queued"
    printf '%-32s %-8s %5s %-22s %s%s\n' "$m" "$c" "$recs" "$q" "$split" "$flag"
  done
done

echo
echo "--- jobs that exited badly since the cross product was submitted ---"
sacct -u "$USER" --starttime=now-24hours --format=JobID%12,JobName%40,State%16,Elapsed,ExitCode -n 2>/dev/null \
  | grep -vE '\.(batch|extern|[0-9]+) ' \
  | grep -vE 'COMPLETED|RUNNING|PENDING' | tail -12
[ -z "$(sacct -u "$USER" --starttime=now-24hours --format=State -n 2>/dev/null | grep -vE 'COMPLETED|RUNNING|PENDING')" ] \
  && echo "  (none)"

echo
echo "--- newest log line per running job ---"
for j in $(squeue -u "$USER" -h -t R -o "%i" 2>/dev/null); do
  f=$(ls -t logs/*-"$j".out 2>/dev/null | head -1)
  [ -n "$f" ] || continue
  printf '  %-52s %s\n' "$(basename "$f")" "$(tail -1 "$f" | cut -c1-110)"
done
