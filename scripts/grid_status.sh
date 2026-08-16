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
# Models abandoned on purpose. Still reported, never flagged as needing action.
DROPPED=${DROPPED:-"Qwen3-14B"}
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
echo "--- short SHARDS with no job (invisible in the cell view above) ---"
# A cell is three shards of 8 rollouts. The grid counts records per CELL, so a
# shard that finished short hides behind its siblings: d6 sat at 20/24 with
# shards s0=6 and s1=7 abandoned and no job for either, while s2 kept running
# and the cell showed "R1", i.e. not stalled. Records carry their own `shard`
# field, so check at that granularity -- it is the only level where "needs a
# resubmit" is actually true.
short_any=0
for m in $MODELS; do
  for c in $CELLS; do
    dir="$ARTIFACT_DIR/rollouts/$m/$c"
    [ -d "$dir" ] || continue
    per=$(python3 -c "
import glob,json,collections
n=collections.Counter()
for f in glob.glob('$dir/*.jsonl'):
    for line in open(f):
        try: n[json.loads(line).get('shard','?')]+=1
        except Exception: pass
out = [s + ':' + str(n.get(s, 0)) for s in ('0/3', '1/3', '2/3')]
print(' '.join(out))
" 2>/dev/null) || continue
    for entry in $per; do
      sh=${entry%%:*}; cnt=${entry##*:}; i=${sh%%/*}
      [ "$cnt" -ge 8 ] && continue
      if squeue -u "$USER" -h -t R,PD -o "%j" 2>/dev/null | grep -qx "$m-$c-s$i"; then continue; fi
      # A deliberately abandoned model is permanently "short with no job". Listing
      # it as an alert every 20 minutes trains the reader to ignore the section,
      # and invites a resubmit that the user explicitly ruled out.
      case " $DROPPED " in
        *" $m "*) printf '  -- %-32s %-9s shard %s at %s/8 (DROPPED, do not resubmit)\n' "$m" "$c" "$i" "$cnt"; continue;;
      esac
      printf '  !! %-32s %-9s shard %s has %s/8 records and NO job\n' "$m" "$c" "$i" "$cnt"
      short_any=1
    done
  done
done
[ "$short_any" -eq 0 ] && echo "  none — every short shard has a job"

echo
echo "--- liveness (a hung client looks IDENTICAL to a slow one in the grid above) ---"
# Qwen3-14B d6 s0 held two GPUs for 3h18m writing nothing while its vLLM server
# reported 111 tok/s, 8 running requests and no preemption. Server health is not
# evidence the job is alive; the CLIENT log is. Two independent signals, because
# either alone gives false positives:
#
#   NO-PROGRESS  the job log has not gained a line since the previous run of this
#                script, and that was more than STALL_MIN minutes ago. Stateful
#                on purpose -- one snapshot cannot tell "quiet" from "stopped".
#   NO-ATTEMPTS  the job has run longer than NOATTEMPT_MIN and has never finished
#                a single attempt. This is what the hang actually looked like, and
#                it is the signal that fires earliest.
#
# A single attempt can legitimately take ~30 minutes (a 24576-token cap at the
# ~14 tok/s each request gets when 8 share one server), so thresholds are in
# multiples of that, not in "feels like a while".
STALL_MIN=${STALL_MIN:-45}
NOATTEMPT_MIN=${NOATTEMPT_MIN:-75}
STATE="${ARTIFACT_DIR:-/tmp}/.grid_liveness.tsv"
now=$(date +%s)
new_state=$(mktemp)
flagged=0
# ONLY this experiment's own jobs. A peer session runs under the SAME Unix user
# (different worktree, different cells: mindset arms growth6/resil6/appr6 and
# their aff* variants, plus dashboard "serve" jobs). Iterating `squeue -u $USER`
# unfiltered would let this script flag a teammate's job as hung, and the loop
# that consumes these flags issues scancel. Match names against MY cells only.
MINE_RE="^(${MODELS// /|})-(${CELLS// /|})-s[0-9]"
for j in $(squeue -u "$USER" -h -t R -o "%i" 2>/dev/null); do
  jname=$(squeue -j "$j" -h -o "%j" 2>/dev/null)
  printf '%s' "$jname" | grep -qE "$MINE_RE" || continue
  f=$(ls logs/*-"$j".out 2>/dev/null | head -1)
  [ -n "$f" ] || continue
  # grep -c prints 0 AND exits 1 when nothing matches, so `|| echo 0` would make
  # the count the two-line string "0\n0" and every numeric test would error.
  lines=$(awk 'END {print NR}' "$f" 2>/dev/null); lines=${lines:-0}
  done_att=$(grep -acE 'Tests (failed|passed) on attempt|Test execution failed' "$f" 2>/dev/null); done_att=${done_att:-0}
  elapsed_min=$(( ( now - $(date -d "$(sacct -j "$j" --format=Start -n 2>/dev/null | head -1 | tr -d ' ')" +%s 2>/dev/null || echo "$now") ) / 60 ))
  printf '%s	%s	%s
' "$j" "$lines" "$now" >> "$new_state"

  why=""
  prev=$(awk -v j="$j" '$1==j {print $2"	"$3}' "$STATE" 2>/dev/null | tail -1)
  if [ -n "$prev" ]; then
    plines=$(printf '%s' "$prev" | cut -f1); pts=$(printf '%s' "$prev" | cut -f2)
    gap=$(( (now - pts) / 60 ))
    [ "$lines" = "$plines" ] && [ "$gap" -ge "$STALL_MIN" ] && why="NO-PROGRESS (${gap}m, no new log lines)"
  fi
  [ "$done_att" -eq 0 ] && [ "$elapsed_min" -ge "$NOATTEMPT_MIN" ] \
    && why="${why:+$why; }NO-ATTEMPTS (${elapsed_min}m elapsed, 0 attempts finished)"

  if [ -n "$why" ]; then
    flagged=$((flagged + 1))
    printf '  !! %-10s %-34s %s\n' "$j" "$(basename "$f" .out)" "$why"
  fi
done
mv "$new_state" "$STATE" 2>/dev/null
if [ "$flagged" -eq 0 ]; then
  echo "  all of THIS experiment's running jobs advancing (peer jobs not inspected)"
else
  echo "  Check the CLIENT, not the server. A flagged job whose server still reports"
  echo "  healthy throughput is generating into the void: scancel it and let its"
  echo "  continuation resume (records checkpoint per rollout, so nothing is lost)."
fi

echo
echo "--- newest log line per running job ---"
for j in $(squeue -u "$USER" -h -t R -o "%i" 2>/dev/null); do
  f=$(ls -t logs/*-"$j".out 2>/dev/null | head -1)
  [ -n "$f" ] || continue
  printf '  %-52s %s\n' "$(basename "$f")" "$(tail -1 "$f" | cut -c1-110)"
done
