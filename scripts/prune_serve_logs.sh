#!/bin/bash
# Strip the repeating vllm-lens layer-0 pre-hook traceback from finished jobs'
# server logs. Read-mostly; refuses to touch a log whose job is still running.
#
# WHY. vllm-lens registers a pre-hook on every decoder layer. On layer 0 the
# pre-hook's args carry no hidden state, so `hidden = args[1]` raises IndexError.
# vllm-lens CATCHES it and prints "skipping" -- the capture layers are unaffected
# and every record still carries hook_data and residuals -- but it prints the
# five-line traceback on every forward pass of every request. One 3-hour job
# logged it 12.1 million times and reached 5.6 GB; the serve directory reached
# 99 GB of a 108 GB artifact tree.
#
# Keeps the first KEEP occurrences so the evidence survives, drops the rest.
cd "$(dirname "$0")/.." || exit 1
set -a; . ./.env 2>/dev/null; set +a
KEEP=${KEEP:-5}

running=$(squeue -u "$USER" -h -t R -o "%i" 2>/dev/null | tr '\n' ' ')
total_before=0 total_after=0
for log in "$ARTIFACT_DIR"/serve/*/*/vllm-server.log; do
  [ -f "$log" ] || continue
  job=$(basename "$(dirname "$log")")
  case " $running " in *" $job "*) echo "  skip $job (still running)"; continue;; esac
  before=$(stat -c %s "$log")
  [ "$before" -lt 104857600 ] && continue          # under 100 MB: not worth rewriting
  # Line-by-line on SIGNATURES, not a "skip the next N lines" state machine: the
  # tensor-parallel workers print concurrently, so their six-line tracebacks
  # interleave and a positional skip drops the wrong lines. Each signature keeps
  # its first `keep` occurrences.
  awk -v keep="$KEEP" '
    /vllm-lens pre-hook error on layer .*, skipping/            { if (++a > keep) next }
    /Traceback \(most recent call last\):/                      { if (++b > keep) next }
    /_worker_ext\.py", line [0-9]+, in hook/                    { if (++c > keep) next }
    /hidden = args\[1\]/                                        { if (++d > keep) next }
    /^\(Worker_TP[0-9]+ pid=[0-9]+\) +~+\^+$/                   { if (++e > keep) next }
    /IndexError: tuple index out of range/                      { if (++f > keep) next }
    { print }
  ' "$log" > "$log.pruned" && mv "$log.pruned" "$log"
  after=$(stat -c %s "$log")
  total_before=$((total_before + before)); total_after=$((total_after + after))
  printf '  %-12s %6s -> %s\n' "$job" "$(numfmt --to=iec "$before")" "$(numfmt --to=iec "$after")"
done
[ "$total_before" -gt 0 ] && echo "reclaimed $(numfmt --to=iec $((total_before - total_after)))"
