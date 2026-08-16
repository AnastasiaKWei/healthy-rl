#!/usr/bin/env bash
# Generate the Nemotron per-token impossible-task grid and submit its slurm DAG.
#
#   scripts/nemotron_pertok_cells.sh --dry-run    # write configs, print sbatch lines (default)
#   scripts/nemotron_pertok_cells.sh --submit     # write configs, submit
#
#   ONLY_VERSION=d6r scripts/nemotron_pertok_cells.sh --submit   # just that cell
#
# Requested 2026-08-16 evening: the Ministral impossible-task conditions on
# Nemotron-3-Nano-4B-BF16 with per-token capture -- {nothing, inoculation v1,
# growth/resilience/appraisal with the trigger-fixed 7d6fd07 text} x {affect
# prompt off, on}, conflicting split, 24 rollouts a cell. Ten cells:
#
#   priority 1  d6r      aff6r          fresh base cells (no arm)          nice 0
#   priority 2  inoc6    affinoc6       inoculation on d6r / aff6r         nice 1000
#   priority 3  growth6b resil6b appr6b, affgrowth6b affresil6b affappr6b  nice 2000
#
# Every cell is generated from Nemotron's d6 (affect off) or aff6 (affect on) shard-0
# config, which already carries max_tokens 24576, and differs from it only in
# `shard`, `out_dir` and the appended arm key(s). d6r/aff6r are the same condition
# as d6/aff6, sampled fresh so every record carries the per-token arrays (records
# written before 2026-08-16 lack them). The arm cells therefore compare byte for
# byte against d6r / aff6r, which is what makes them one grid.
#
# SIX SHARDS OF FOUR, not three of eight, and any two GPUs rather than A100s only.
# We are on the clock: a Nemotron server on 2 x A100-40G decodes ~24 tok/s with one
# request in flight, ~74 at four, ~84 at eight (vllm-server.log of job 5641943), so
# one-rollout jobs would waste the GPU while eight-rollout jobs make each cell wait
# on its slowest shard. Four in flight is close to the throughput plateau and keeps
# a job to ~35 min, well inside the 2 h limit. Continuations exist only as insurance
# against a hang / EngineDeadError; they carry nice +10000 so a pending primary
# always outranks an idle continuation.
#
# Two phases, deliberately: phase 1 preflights and writes every selected config,
# phase 2 submits. Refuses to overwrite a shard config that already exists with
# different content. SHARD_DIR overrides where configs are written (tests use a
# temp dir); the sbatch lines always name configs/shards/.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=dry
case "${1:-}" in --dry-run|"") MODE=dry ;; --submit) MODE=submit ;; *) echo "usage: $0 [--dry-run|--submit]" >&2; exit 2 ;; esac

MODEL=Nemotron-3-Nano-4B-BF16
N_SHARDS=6
SHARD_DIR="${SHARD_DIR:-configs/shards}"
mkdir -p "$SHARD_DIR"

# version|template_version|arm|nice   arm: none | inoc | growth | resilience | appraisal
CELLS=(
  "d6r|d6|none|0"
  "aff6r|aff6|none|0"
  "inoc6|d6|inoc|1000"
  "affinoc6|aff6|inoc|1000"
  "growth6b|d6|growth|2000"
  "resil6b|d6|resilience|2000"
  "appr6b|d6|appraisal|2000"
  "affgrowth6b|aff6|growth|2000"
  "affresil6b|aff6|resilience|2000"
  "affappr6b|aff6|appraisal|2000"
)

ONLY_VERSION="${ONLY_VERSION:-}"
in_scope() { [[ -z $ONLY_VERSION || $1 == "$ONLY_VERSION" ]]; }
echo "versions in scope: ${ONLY_VERSION:-all (ONLY_VERSION unset)}" >&2

# Only ever called from phase 1, i.e. before the first sbatch.
die() { echo "$*" >&2; echo "nothing submitted" >&2; exit 1; }

# The per-token base a cell is read against: d6r for affect-off cells, aff6r for
# affect-on. The base cells themselves are read against their pre-per-token twin.
base_of() { case "$1" in d6) echo d6r ;; aff6) echo aff6r ;; esac; }

write_config() {  # version template arm shard_index
  local version=$1 template=$2 arm=$3 i=$4
  local src="configs/shards/rollouts-${MODEL}-${template}-s0of3.yaml"
  local dst="$SHARD_DIR/rollouts-${MODEL}-${version}-s${i}of${N_SHARDS}.yaml"
  local base; base=$(base_of "$template")
  [[ -f "$src" ]] || die "missing template config $src"
  local tmp; tmp=$(mktemp)
  sed -e "s#^shard: .*#shard: \"${i}/${N_SHARDS}\"#" \
      -e "s#^out_dir: .*#out_dir: /out/rollouts/${MODEL}/${version}#" "$src" > "$tmp"
  case "$arm" in
    none) cat >> "$tmp" <<EOF

# --- RE-RUN WITH PER-TOKEN CAPTURE (2026-08-16) ---------------------------------
# Identical condition to ${template} (same flags, split, problems, samples); a fresh
# sample at temperature 1.0 written into ${version} so every rollout carries the
# per-token projection/norm/kind arrays that records written before 2026-08-16
# lack. Six shards of four rollouts rather than three of eight (see
# scripts/nemotron_pertok_cells.sh); the shard partition is index % 6, so the
# 24 (task, sample) items are the same as ${template}'s. Compare the arm cells
# (inoc6/affinoc6, *6b) against THIS cell, not ${template}.
EOF
      ;;
    inoc) cat >> "$tmp" <<EOF

# --- INOCULATION ARM (prompt v1) -----------------------------------------
# Inoculation-prompting stimulus (Wichers et al. 2025, reference/inoculation/):
# the DO-NOT-MODIFY-THE-TESTS line is dropped and a paragraph giving explicit
# permission to game the tests is added, on turn 1 AND in every reminder. Compare
# this cell against ${MODEL}/${base}: everything except this key and out_dir is
# byte-identical to that cell's shard config. Read the probes, not the words.
inoculation: true
EOF
      ;;
    growth|resilience|appraisal) cat >> "$tmp" <<EOF

# --- MINDSET ARM: ${arm} (prompt v2, trigger phrasing fixed 2026-08-16, astwei 7d6fd07) ---
# One of Anastasia's three mindset blocks (experiments/step0_elicitation.py,
# copied verbatim into healthy_rl.rollouts.MINDSET), inserted into the turn-1
# instruction between the benchmark text and the affect request, and STRIPPED
# from the reminder the scaffold re-sends after each failure -- so the model sees
# it once per rollout.
#
# What changed: the two-part procedure is now triggered by a STANDING RULE,
# "Your first attempt is just the code. Open every attempt after that with
# these two lines", instead of by an event, "whenever a test fails". Those are
# equivalent only while the block repeats every turn. Here it is sent on turn 1 and
# only turn 1 -- before any test has failed, and gone by the time one does -- which
# made the event-conditional wording inert: a smoke run measured 0/12 compliance
# with the labelled prefixes on attempts 2+, and 12/12 after the rewrite.
# MINDSET_VERSION stays 2; every record carries a mindset_hash, so this text and the
# old one cannot be resumed into each other. Nemotron never ran the old text.
#
# Compare this cell against ${MODEL}/${base}: everything except this key and
# out_dir is byte-identical to that cell's shard config, and both carry the
# per-token arrays. It is a demand characteristic pointing the opposite way from
# the affect prompt; read the probes, not the words.
mindset: [${arm}]
EOF
      ;;
    *) die "unknown arm $arm" ;;
  esac
  if [[ -f "$dst" ]] && ! cmp -s "$tmp" "$dst"; then
    rm -f "$tmp"
    die "refusing to overwrite $dst: it exists with different content"
  fi
  mv "$tmp" "$dst"
  chmod --reference="$src" "$dst"
}

# Sets JOB_ID rather than echoing it: `p=$(submit ...)` would run in a subshell where
# `set -e` is not honoured, and a failed sbatch would chain continuations onto an
# empty id. See scripts/mindset_cells.sh for the verification.
JOB_ID=
submit() {
  local -a cmd=(sbatch --parsable "$@")
  if [[ $MODE == dry ]]; then
    printf '%s\n' "${cmd[*]}"; printf 'DRYRUN\n' >&2; JOB_ID=DRYRUN; return 0
  fi
  local id rc=0
  id=$("${cmd[@]}") || rc=$?
  id=${id%%;*}
  if (( rc != 0 )) || [[ ! $id =~ ^[0-9]+$ ]]; then
    echo "sbatch failed (exit $rc, returned '$id') for: ${cmd[*]}" >&2
    echo "stopping: the rows printed above are already queued -- scancel them unless you re-run" >&2
    exit 1
  fi
  JOB_ID=$id
}

# --- phase 1: preflight, then write every config ------------------------------
for f in slurm/serve.slurm scripts/run_rollouts.py; do
  [[ -f "$f" ]] || die "missing $f -- run this from a checkout where it exists"
done
selected=0
for row in "${CELLS[@]}"; do
  IFS='|' read -r version template arm nice <<< "$row"
  in_scope "$version" || continue
  selected=$((selected + 1))
  [[ -f "configs/shards/rollouts-${MODEL}-${template}-s0of3.yaml" ]] \
    || die "missing template config for $template"
done
(( selected > 0 )) || die "ONLY_VERSION='$ONLY_VERSION' matches no row in CELLS"

for row in "${CELLS[@]}"; do
  IFS='|' read -r version template arm nice <<< "$row"
  in_scope "$version" || continue
  for ((i = 0; i < N_SHARDS; i++)); do
    write_config "$version" "$template" "$arm" "$i"
  done
done

# --- phase 2: submit ----------------------------------------------------------
if [[ $MODE == submit ]]; then
  echo
  echo "| model | version | shard | primary | continuations |"
  echo "|---|---|---|---|---|"
fi

CONT_NICE=10000
for row in "${CELLS[@]}"; do
  IFS='|' read -r version template arm nice <<< "$row"
  in_scope "$version" || continue
  for ((i = 0; i < N_SHARDS; i++)); do
    cfg="configs/shards/rollouts-${MODEL}-${version}-s${i}of${N_SHARDS}.yaml"
    name="${MODEL}-${version}-s${i}"
    common=(--gres=gpu:2 --mem=96G --cpus-per-task=16 --time=2:00:00)
    primary=("${common[@]}" --nice="$nice")
    cont=("${common[@]}" --nice=$((nice + CONT_NICE)))
    stage=(slurm/serve.slurm --model "$MODEL" --config "$cfg" --gpu-memory-utilization 0.90
           --stage "scripts/run_rollouts.py:$cfg")
    if [[ $MODE == dry ]]; then
      submit --job-name="$name" "${primary[@]}" "${stage[@]}"
      submit --job-name="$name-cont" --dependency=afterany:PRIMARY "${cont[@]}" "${stage[@]}"
      submit --job-name="$name-cont2" --dependency=afterany:CONT "${cont[@]}" "${stage[@]}"
    else
      submit --job-name="$name" "${primary[@]}" "${stage[@]}"; p=$JOB_ID
      submit --job-name="$name-cont" --dependency=afterany:"$p" "${cont[@]}" "${stage[@]}"; c1=$JOB_ID
      submit --job-name="$name-cont2" --dependency=afterany:"$c1" "${cont[@]}" "${stage[@]}"; c2=$JOB_ID
      printf '| %s | %s | %s | %s | %s |\n' "$MODEL" "$version" "s$i" "$p" "$c1 / $c2"
    fi
  done
done
