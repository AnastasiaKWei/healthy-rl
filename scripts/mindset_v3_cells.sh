#!/usr/bin/env bash
# Generate the 16 mindset-v3 cells and submit their slurm DAG.
#
#   scripts/mindset_v3_cells.sh --dry-run    # write configs, print sbatch lines (default)
#   scripts/mindset_v3_cells.sh --submit     # write configs, submit
#   ONLY_MODEL=gemma-3-12b-it ONLY_VERSION=spaffctrl6v3 scripts/mindset_v3_cells.sh --dry-run
#
# Requested 2026-08-16: Anastasia's v3 mindset prompts (growth, resilience,
# control, compassion -- persona + psychoeducation block sent once, before the
# task under a "## Task" heading, plus a one-line reminder inserted into every
# test-failure message; her commit 77d558c, our healthy_rl.rollouts MINDSET
# version 3) on gemma-3-12b-it WITH the scratchpad and on
# Ministral-3-14B-Reasoning-2512 (native reasoning, no scratchpad), affect
# prompt off and on, conflicting split, 24 rollouts a cell, per-token capture.
#
# Every cell is generated from that model's per-token base cell -- gemma sp6r /
# spaff6, Ministral d6r / aff6r -- and differs from it in `shard`, `out_dir` and
# the appended `mindset:` key only, so each arm reads against its own base byte
# for byte:
#
#   nice 0     spaffgrowth6v3 spaffresil6v3 spaffctrl6v3 spaffcomp6v3   gemma, base spaff6
#              affgrowth6v3   affresil6v3   affctrl6v3   affcomp6v3     Ministral, base aff6r
#   nice 1000  spgrowth6v3    spresil6v3    spctrl6v3    spcomp6v3      gemma, base sp6r
#              growth6v3      resil6v3      ctrl6v3      comp6v3        Ministral, base d6r
#
# Affect-on first: it is the condition her judge-scored v3 numbers exist for.
# Three shards of eight on 2 x A100-40G, 4 h, primary + -cont + -cont2, as the
# other gemma/Ministral mindset cells ran (scripts/mindset_cells.sh).
# Continuations carry nice +10000 so an idle continuation never outranks a
# pending primary.
#
# Two phases, deliberately: phase 1 preflights and writes every selected config,
# phase 2 submits. Refuses to overwrite a shard config that already exists with
# different content. SHARD_DIR overrides where configs are written (tests use a
# temp dir); the sbatch lines always name configs/shards/.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=dry
case "${1:-}" in --dry-run|"") MODE=dry ;; --submit) MODE=submit ;; *) echo "usage: $0 [--dry-run|--submit]" >&2; exit 2 ;; esac

N_SHARDS=3
SHARD_DIR="${SHARD_DIR:-configs/shards}"
mkdir -p "$SHARD_DIR"
GEMMA=gemma-3-12b-it
MINISTRAL=Ministral-3-14B-Reasoning-2512

# model|version|template|arm|nice   arm: growth | resilience | control | compassion
CELLS=(
  "$GEMMA|spaffgrowth6v3|spaff6|growth|0"
  "$GEMMA|spaffresil6v3|spaff6|resilience|0"
  "$GEMMA|spaffctrl6v3|spaff6|control|0"
  "$GEMMA|spaffcomp6v3|spaff6|compassion|0"
  "$MINISTRAL|affgrowth6v3|aff6r|growth|0"
  "$MINISTRAL|affresil6v3|aff6r|resilience|0"
  "$MINISTRAL|affctrl6v3|aff6r|control|0"
  "$MINISTRAL|affcomp6v3|aff6r|compassion|0"
  "$GEMMA|spgrowth6v3|sp6r|growth|1000"
  "$GEMMA|spresil6v3|sp6r|resilience|1000"
  "$GEMMA|spctrl6v3|sp6r|control|1000"
  "$GEMMA|spcomp6v3|sp6r|compassion|1000"
  "$MINISTRAL|growth6v3|d6r|growth|1000"
  "$MINISTRAL|resil6v3|d6r|resilience|1000"
  "$MINISTRAL|ctrl6v3|d6r|control|1000"
  "$MINISTRAL|comp6v3|d6r|compassion|1000"
)

ONLY_MODEL="${ONLY_MODEL:-}"
ONLY_VERSION="${ONLY_VERSION:-}"
in_scope() { [[ -z $ONLY_MODEL || $1 == "$ONLY_MODEL" ]] && [[ -z $ONLY_VERSION || $2 == "$ONLY_VERSION" ]]; }
echo "models in scope: ${ONLY_MODEL:-all}; versions in scope: ${ONLY_VERSION:-all}" >&2

# Only ever called from phase 1, i.e. before the first sbatch.
die() { echo "$*" >&2; echo "nothing submitted" >&2; exit 1; }

write_config() {  # model version template arm shard_index
  local model=$1 version=$2 template=$3 arm=$4 i=$5
  local src="configs/shards/rollouts-${model}-${template}-s0of3.yaml"
  local dst="$SHARD_DIR/rollouts-${model}-${version}-s${i}of${N_SHARDS}.yaml"
  [[ -f "$src" ]] || die "missing template config $src"
  local tmp; tmp=$(mktemp)
  sed -e "s#^shard: .*#shard: \"${i}/${N_SHARDS}\"#" \
      -e "s#^out_dir: .*#out_dir: /out/rollouts/${model}/${version}#" "$src" > "$tmp"
  cat >> "$tmp" <<EOF

# --- MINDSET ARM: ${arm} (prompt v3, astwei 77d558c, 2026-08-16) -----------------
# One of Anastasia's five v3 mindset blocks (experiments/step0_elicitation.py,
# copied verbatim into healthy_rl.rollouts.MINDSET, MINDSET_VERSION 3). Two
# mechanisms, both hers:
#   * the block -- persona sentence, a "## <construct>" psychoeducation paragraph,
#     a "What this looks like in practice" vignette -- goes BEFORE the benchmark
#     instruction, closed by "---" and followed by a "## Task" heading, on turn 1
#     only: strip_mindset_from_reminders takes it out of the reminder the scaffold
#     re-sends after each failure. The "## Task" heading survives into that
#     reminder (her send_mindset_once strips the block alone; docs/prompts/v3.md).
#   * a one-sentence reminder line ("Remember you are a ...") is inserted into
#     EVERY test-failure message between the pytest output and "To reiterate,
#     this is your task:" (patch_failure_feedback). Appraisal has no such line;
#     these four do.
# mindset_hash on every record covers block + reminder line; resume refuses a
# different text or version.
#
# Compare this cell against ${model}/${template}: everything except this key,
# shard and out_dir is byte-identical to that cell's shard config, and both carry
# the per-token arrays. It is a demand characteristic pointing the opposite way
# from the affect prompt; read the probes, not the words. Her judge-scored v3
# arms (Gemma, original split, hackable, affect on) sat ~0.9 below baseline on
# both channels -- whether that is suppression or a shift is what this cell asks.
mindset: [${arm}]
EOF
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
  IFS='|' read -r model version template arm nice <<< "$row"
  in_scope "$model" "$version" || continue
  selected=$((selected + 1))
  [[ -f "configs/shards/rollouts-${model}-${template}-s0of3.yaml" ]] \
    || die "missing template config for ${model}/${template}"
done
(( selected > 0 )) || die "ONLY_MODEL='$ONLY_MODEL' ONLY_VERSION='$ONLY_VERSION' matches no row in CELLS"

for row in "${CELLS[@]}"; do
  IFS='|' read -r model version template arm nice <<< "$row"
  in_scope "$model" "$version" || continue
  for ((i = 0; i < N_SHARDS; i++)); do
    write_config "$model" "$version" "$template" "$arm" "$i"
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
  IFS='|' read -r model version template arm nice <<< "$row"
  in_scope "$model" "$version" || continue
  for ((i = 0; i < N_SHARDS; i++)); do
    cfg="configs/shards/rollouts-${model}-${version}-s${i}of${N_SHARDS}.yaml"
    name="${model}-${version}-s${i}"
    common=(--gres=gpu:A100-40G:2 --mem=96G --cpus-per-task=16 --time=4:00:00)
    primary=("${common[@]}" --nice="$nice")
    cont=("${common[@]}" --nice=$((nice + CONT_NICE)))
    stage=(slurm/serve.slurm --model "$model" --config "$cfg" --gpu-memory-utilization 0.90
           --stage "scripts/run_rollouts.py:$cfg")
    if [[ $MODE == dry ]]; then
      submit --job-name="$name" "${primary[@]}" "${stage[@]}"
      submit --job-name="$name-cont" --dependency=afterany:PRIMARY "${cont[@]}" "${stage[@]}"
      submit --job-name="$name-cont2" --dependency=afterany:CONT "${cont[@]}" "${stage[@]}"
    else
      submit --job-name="$name" "${primary[@]}" "${stage[@]}"; p=$JOB_ID
      submit --job-name="$name-cont" --dependency=afterany:"$p" "${cont[@]}" "${stage[@]}"; c1=$JOB_ID
      submit --job-name="$name-cont2" --dependency=afterany:"$c1" "${cont[@]}" "${stage[@]}"; c2=$JOB_ID
      printf '| %s | %s | %s | %s | %s |\n' "$model" "$version" "s$i" "$p" "$c1 / $c2"
    fi
  done
done
