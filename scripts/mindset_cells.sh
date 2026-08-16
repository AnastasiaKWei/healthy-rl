#!/usr/bin/env bash
# Generate the mindset-arm shard configs and submit their slurm DAG.
#
#   scripts/mindset_cells.sh --dry-run    # write configs, print sbatch lines (default)
#   scripts/mindset_cells.sh --submit     # write configs, submit
#
# Cells (docs/superpowers/specs/2026-08-15-mindset-vectors-design.md §2):
#   priority 1  Ministral d6-base   growth6 resil6 appr6            nice 0
#   priority 2  Qwen3.5-9B d6-base  growth6 resil6 appr6            nice 2000
#   priority 3  Ministral aff6-base affgrowth6 affresil6 affappr6   nice 4000
# Each shard: primary (4h) -> -cont -> -cont2, chained afterany. Resume appends;
# an idle continuation exits after model load. Priority is honoured because the
# cluster runs PriorityType=priority/multifactor.
#
# The shard config is the model's base-cell config with max_tokens 24576 (the 2x2
# value; the cap never bound in d6), the `mindset:` key, and its own out_dir.
# Everything else is byte-identical, which is what keeps the cells comparable.
# Refuses to overwrite a shard config that already exists with different content.
#
# SHARD_DIR overrides where configs are written (tests use a temp dir).
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
)

write_config() {  # model version base mindset shard_index
  local model=$1 version=$2 base=$3 mindset=$4 i=$5
  local src="configs/shards/rollouts-${model}-${base}-s${i}of3.yaml"
  local dst="$SHARD_DIR/rollouts-${model}-${version}-s${i}of3.yaml"
  [[ -f "$src" ]] || { echo "missing base config $src" >&2; exit 1; }
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
    echo "refusing to overwrite $dst: it exists with different content" >&2; rm -f "$tmp"; exit 1
  fi
  mv "$tmp" "$dst"
  # mktemp makes the temp file 0600 and mv keeps that mode, which would leave these
  # configs unreadable to the rest of the group while every other config in
  # configs/shards is 0664. Take the base config's mode instead.
  chmod --reference="$src" "$dst"
}

submit() {  # prints or runs sbatch; echoes job id (or DRYRUN)
  local -a cmd=(sbatch --parsable "$@")
  if [[ $MODE == dry ]]; then printf '%s\n' "${cmd[*]}"; printf 'DRYRUN\n' >&2; return 0; fi
  local id; id=$("${cmd[@]}"); printf '%s\n' "${id%%;*}"
}

SUMMARY=()
for row in "${CELLS[@]}"; do
  IFS='|' read -r model version base mindset nice <<< "$row"
  for i in 0 1 2; do
    write_config "$model" "$version" "$base" "$mindset" "$i"
    # The sbatch line always names the real location; SHARD_DIR only redirects
    # where this run WRITES (so a test can generate into a temp dir).
    cfg="configs/shards/rollouts-${model}-${version}-s${i}of3.yaml"
    name="${model}-${version}-s${i}"
    common=(--gres=gpu:A100-40G:2 --mem=96G --cpus-per-task=16 --time=4:00:00)
    [[ "$nice" -gt 0 ]] && common+=(--nice="$nice")
    stage=(slurm/serve.slurm --model "$model" --config "$cfg" --gpu-memory-utilization 0.90
           --stage "scripts/run_rollouts.py:$cfg")
    if [[ $MODE == dry ]]; then
      p=$(submit --job-name="$name" "${common[@]}" "${stage[@]}")
      c1=$(submit --job-name="$name-cont" --dependency=afterany:PRIMARY "${common[@]}" "${stage[@]}")
      c2=$(submit --job-name="$name-cont2" --dependency=afterany:CONT "${common[@]}" "${stage[@]}")
      printf '%s\n%s\n%s\n' "$p" "$c1" "$c2"
    else
      p=$(submit --job-name="$name" "${common[@]}" "${stage[@]}")
      c1=$(submit --job-name="$name-cont" --dependency=afterany:"$p" "${common[@]}" "${stage[@]}")
      c2=$(submit --job-name="$name-cont2" --dependency=afterany:"$c1" "${common[@]}" "${stage[@]}")
      SUMMARY+=("$(printf '| %s | %s | %s | %s | %s |' "$model" "$version" "s$i" "$p" "$c1 / $c2")")
    fi
  done
done

if [[ $MODE == submit ]]; then
  echo
  echo "| model | version | shard | primary | continuations |"
  echo "|---|---|---|---|---|"
  printf '%s\n' "${SUMMARY[@]}"
fi
