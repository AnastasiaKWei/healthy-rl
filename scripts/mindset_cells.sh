#!/usr/bin/env bash
# Generate the mindset-arm shard configs and submit their slurm DAG.
#
#   scripts/mindset_cells.sh --dry-run    # write configs, print sbatch lines (default)
#   scripts/mindset_cells.sh --submit     # write configs, submit
#
#   ONLY_MODEL=gemma-3-12b-it scripts/mindset_cells.sh --submit   # just that model
#   ONLY_BASE=spaff6 scripts/mindset_cells.sh --submit             # just that base cell
#
# Cells (docs/superpowers/specs/2026-08-15-mindset-vectors-design.md §2):
#   priority 1  Ministral d6-base   growth6 resil6 appr6            nice 0
#   priority 2  Qwen3.5-9B d6-base  growth6 resil6 appr6            nice 2000
#   priority 3  Ministral aff6-base affgrowth6 affresil6 affappr6   nice 4000
#   priority 4  gemma sp6-base      spgrowth6 spresil6 spappr6      nice 0
#   priority 5  gemma aff6-base     affgrowth6 affresil6 affappr6   nice 2000
#   priority 6  gemma spaff6-base   spaffgrowth6 spaffresil6 spaffappr6  nice 0
# Each shard: primary (4h) -> -cont -> -cont2, chained afterany. Resume appends;
# an idle continuation exits after model load. Priority is honoured because the
# cluster runs PriorityType=priority/multifactor.
#
# The shard config is the model's base-cell config with max_tokens 24576 (the 2x2
# value; the cap never bound in d6), the `mindset:` key, and its own out_dir.
# Everything else is byte-identical, which is what keeps the cells comparable.
# Refuses to overwrite a shard config that already exists with different content.
#
# Two phases, deliberately: phase 1 preflights and writes every selected config,
# phase 2 submits. Nothing is submitted until every config is on disk, so the failure
# modes that abort the script (missing base config, missing serve.slurm, a config
# that would be overwritten) cost nothing but a re-run. Generation used to
# interleave with submission, which meant an abort part way through left a
# half-built DAG whose job ids had never been printed.
#
# SHARD_DIR overrides where configs are written (tests use a temp dir).
#
# ONLY_MODEL restricts the run to one model (exact name), ONLY_BASE to one base
# version (exact name). They filter BOTH phases, so the configs of the other rows
# are neither rewritten nor submitted -- which is what makes it safe to re-run this
# script to add a cell after earlier ones are already queued. Unset means every
# row, the original behaviour; setting both keeps only the rows matching BOTH.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=dry
case "${1:-}" in --dry-run|"") MODE=dry ;; --submit) MODE=submit ;; *) echo "usage: $0 [--dry-run|--submit]" >&2; exit 2 ;; esac

SHARD_DIR="${SHARD_DIR:-configs/shards}"
mkdir -p "$SHARD_DIR"

# model|version|base_version|mindset|nice
CELLS=(
  "Ministral-3-14B-Reasoning-2512|growth6|d6|growth|0"
  "Ministral-3-14B-Reasoning-2512|resil6|d6|resilience|0"
  "Ministral-3-14B-Reasoning-2512|appr6|d6|appraisal|0"
  "Qwen3.5-9B|growth6|d6|growth|2000"
  "Qwen3.5-9B|resil6|d6|resilience|2000"
  "Qwen3.5-9B|appr6|d6|appraisal|2000"
  "Ministral-3-14B-Reasoning-2512|affgrowth6|aff6|growth|4000"
  "Ministral-3-14B-Reasoning-2512|affresil6|aff6|resilience|4000"
  "Ministral-3-14B-Reasoning-2512|affappr6|aff6|appraisal|4000"
  "gemma-3-12b-it|spgrowth6|sp6|growth|0"
  "gemma-3-12b-it|spresil6|sp6|resilience|0"
  "gemma-3-12b-it|spappr6|sp6|appraisal|0"
  "gemma-3-12b-it|affgrowth6|aff6|growth|2000"
  "gemma-3-12b-it|affresil6|aff6|resilience|2000"
  "gemma-3-12b-it|affappr6|aff6|appraisal|2000"
  "gemma-3-12b-it|spaffgrowth6|spaff6|growth|0"
  "gemma-3-12b-it|spaffresil6|spaff6|resilience|0"
  "gemma-3-12b-it|spaffappr6|spaff6|appraisal|0"
)

# Empty means every model / every base. A row has to match every filter that is
# set. Announced on stderr rather than stdout because stdout in dry mode is the
# sbatch script and nothing else.
ONLY_MODEL="${ONLY_MODEL:-}"
ONLY_BASE="${ONLY_BASE:-}"
in_scope() {  # model base
  [[ -z $ONLY_MODEL || $1 == "$ONLY_MODEL" ]] && [[ -z $ONLY_BASE || $2 == "$ONLY_BASE" ]]
}
echo "models in scope: ${ONLY_MODEL:-all (ONLY_MODEL unset)}; bases in scope: ${ONLY_BASE:-all (ONLY_BASE unset)}" >&2

# Only ever called from phase 1, i.e. before the first sbatch -- which is what
# makes "nothing submitted" true rather than hopeful.
die() { echo "$*" >&2; echo "nothing submitted" >&2; exit 1; }

write_config() {  # model version base mindset shard_index
  local model=$1 version=$2 base=$3 mindset=$4 i=$5
  local src="configs/shards/rollouts-${model}-${base}-s${i}of3.yaml"
  local dst="$SHARD_DIR/rollouts-${model}-${version}-s${i}of3.yaml"
  [[ -f "$src" ]] || die "missing base config $src"
  local tmp; tmp=$(mktemp)
  # max_tokens -> 24576; out_dir -> the new cell; then append the mindset block.
  sed -e 's/^max_tokens: .*/max_tokens: 24576/' \
      -e "s#^out_dir: .*#out_dir: /out/rollouts/${model}/${version}#" "$src" > "$tmp"
  cat >> "$tmp" <<EOF

# --- MINDSET ARM: ${mindset} (prompt v2) -------------------------------------
# One of Anastasia's three mindset blocks (experiments/step0_elicitation.py,
# copied verbatim into healthy_rl.rollouts.MINDSET), inserted into the turn-1
# instruction between the benchmark text and the affect request, and STRIPPED
# from the reminder the scaffold re-sends after each failure -- so the model sees
# it once per rollout. Compare this cell against ${model}/${base}: everything
# except this key and max_tokens (24576, the 2x2 value) is byte-identical to that
# cell's shard config. It is a demand characteristic pointing the opposite way
# from the affect prompt; read the probes, not the words.
mindset: [${mindset}]
EOF
  if [[ -f "$dst" ]] && ! cmp -s "$tmp" "$dst"; then
    rm -f "$tmp"
    die "refusing to overwrite $dst: it exists with different content"
  fi
  mv "$tmp" "$dst"
  # mktemp makes the temp file 0600 and mv keeps that mode, which would leave these
  # configs unreadable to the rest of the group while every other config in
  # configs/shards is 0664. Take the base config's mode instead.
  chmod --reference="$src" "$dst"
}

# Sets JOB_ID rather than echoing the id, and it matters. `p=$(submit ...)` would run
# submit in a command substitution subshell, and bash does NOT honour `set -e` inside
# one of those: a failing sbatch there is swallowed, the function runs on, and the
# continuations get chained onto `--dependency=afterany:` with an EMPTY id. Verified on
# bash 5.1.8: `set -e; f() { id=$(false); echo AFTER; }; p=$(f)` prints AFTER and the
# script continues. Calling submit in the current shell instead means its `exit 1`
# really does stop the run.
JOB_ID=
submit() {  # runs (dry mode: prints) one sbatch; sets JOB_ID
  local -a cmd=(sbatch --parsable "$@")
  if [[ $MODE == dry ]]; then
    printf '%s\n' "${cmd[*]}"; printf 'DRYRUN\n' >&2; JOB_ID=DRYRUN; return 0
  fi
  local id rc=0
  id=$("${cmd[@]}") || rc=$?
  id=${id%%;*}   # --parsable returns "<jobid>[;<cluster>]"
  if (( rc != 0 )) || [[ ! $id =~ ^[0-9]+$ ]]; then
    echo "sbatch failed (exit $rc, returned '$id') for: ${cmd[*]}" >&2
    echo "stopping: the rows printed above are already queued -- scancel them unless you re-run" >&2
    exit 1
  fi
  JOB_ID=$id
}

# --- phase 1: preflight, then write every config ------------------------------
# No sbatch runs below this point until phase 1 has finished for every shard.
for f in slurm/serve.slurm scripts/run_rollouts.py; do
  [[ -f "$f" ]] || die "missing $f -- run this from a checkout where it exists"
done

selected=0
for row in "${CELLS[@]}"; do
  IFS='|' read -r model version base mindset nice <<< "$row"
  in_scope "$model" "$base" || continue
  selected=$((selected + 1))
  for i in 0 1 2; do
    src="configs/shards/rollouts-${model}-${base}-s${i}of3.yaml"
    [[ -f "$src" ]] || die "missing base config $src"
  done
done
(( selected > 0 )) || die "ONLY_MODEL='$ONLY_MODEL' ONLY_BASE='$ONLY_BASE' matches no row in CELLS"

for row in "${CELLS[@]}"; do
  IFS='|' read -r model version base mindset nice <<< "$row"
  in_scope "$model" "$base" || continue
  for i in 0 1 2; do
    write_config "$model" "$version" "$base" "$mindset" "$i"
  done
done

# --- phase 2: submit ----------------------------------------------------------
# The header goes out before the first sbatch and each row as its ids come back,
# so an sbatch that fails half way still leaves on screen the ids of everything
# already queued -- that list is the only record of what needs scancel'ing.
if [[ $MODE == submit ]]; then
  echo
  echo "| model | version | shard | primary | continuations |"
  echo "|---|---|---|---|---|"
fi

for row in "${CELLS[@]}"; do
  IFS='|' read -r model version base mindset nice <<< "$row"
  in_scope "$model" "$base" || continue
  for i in 0 1 2; do
    # The sbatch line always names the real location; SHARD_DIR only redirects
    # where this run WROTE the config (so a test can generate into a temp dir).
    cfg="configs/shards/rollouts-${model}-${version}-s${i}of3.yaml"
    name="${model}-${version}-s${i}"
    common=(--gres=gpu:A100-40G:2 --mem=96G --cpus-per-task=16 --time=4:00:00)
    [[ "$nice" -gt 0 ]] && common+=(--nice="$nice")
    stage=(slurm/serve.slurm --model "$model" --config "$cfg" --gpu-memory-utilization 0.90
           --stage "scripts/run_rollouts.py:$cfg")
    if [[ $MODE == dry ]]; then
      submit --job-name="$name" "${common[@]}" "${stage[@]}"
      submit --job-name="$name-cont" --dependency=afterany:PRIMARY "${common[@]}" "${stage[@]}"
      submit --job-name="$name-cont2" --dependency=afterany:CONT "${common[@]}" "${stage[@]}"
    else
      submit --job-name="$name" "${common[@]}" "${stage[@]}"; p=$JOB_ID
      submit --job-name="$name-cont" --dependency=afterany:"$p" "${common[@]}" "${stage[@]}"; c1=$JOB_ID
      submit --job-name="$name-cont2" --dependency=afterany:"$c1" "${common[@]}" "${stage[@]}"; c2=$JOB_ID
      printf '| %s | %s | %s | %s | %s |\n' "$model" "$version" "s$i" "$p" "$c1 / $c2"
    fi
  done
done
