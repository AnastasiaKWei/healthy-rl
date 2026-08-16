# Mindset prompts under the emotion probes — design

Date: 2026-08-15. Branch `feature/mindset-vectors` (worktree `.claude/worktrees/mindset`),
which includes `origin/prompts/mindset-v2` (Anastasia's v2 prompt text).

## 1 Purpose

Anastasia's step-0 runs measure the three mindset interventions (growth,
resilience, appraisal — `experiments/step0_elicitation.py::MINDSET`, v2) on
*externally visible* affect through an OpenRouter judge. This work runs the same
three blocks through the pipeline that captures activations, so the question in
`docs/interventions.md` §8 can be asked directly: does a mindset block move the
represented affect (the emotion-direction trajectory over six failing turns), or
only what the model says?

Priority result: three arms on **Ministral-3-14B-Reasoning-2512**, `d6` base
(conflicting split, no scratchpad, no affect prompt), 24 rollouts each, compared
against the existing `d6` cell. Then fill out with the same three arms on
**Qwen3.5-9B** (`d6` base), then Ministral with the affect prompt on (`aff6`
base), as GPUs allow.

Two things go into the same change because the runs are unattended and cannot
be repeated cheaply:

- **Per-token projections are kept on disk.** The hook already computes them at
  every capture layer; `summarise_hook_results` currently reduces them to a turn
  mean and drops the arrays. `docs/runs.md` says they are kept; they are not.
- **The mindset block is sent once**, in the opening message only, as in her v2:
  ImpossibleBench's minimal scaffold re-sends `metadata["instruction_prompt"]`
  after every failed attempt (`include_task_reminder=True`, the default our
  `build_task` inherits), so without intervention the block would arrive six
  times per rollout.

## 2 Cells

| version | model | mindset | affect prompt | split | compare against | priority |
|---|---|---|---|---|---|---|
| `growth6` | Ministral-3-14B-Reasoning-2512 | growth | off | conflicting | `d6` | 1 |
| `resil6` | Ministral-3-14B-Reasoning-2512 | resilience | off | conflicting | `d6` | 1 |
| `appr6` | Ministral-3-14B-Reasoning-2512 | appraisal | off | conflicting | `d6` | 1 |
| `growth6` / `resil6` / `appr6` | Qwen3.5-9B | as above | off | conflicting | `d6` | 2 (`--nice=2000`) |
| `affgrowth6` / `affresil6` / `affappr6` | Ministral-3-14B-Reasoning-2512 | as above | **on** | conflicting | `aff6` | 3 (`--nice=4000`) |

Each cell: 12 problems × 2 samples = 24 rollouts, 3 shards of 8, six attempts,
`max_tokens: 24576` (the 2x2 value; the cap never bound in `d6`), `residual_layers:
probe`, `save_residuals: true`. Everything not listed as varied is copied from
the model's `d6` shard config (or `aff6` for the affect-on cells). Shard configs
are `configs/shards/rollouts-<model>-<version>-s{i}of3.yaml`, generated, not
hand-edited.

The version string is the directory name under `$ARTIFACT_DIR/rollouts/<model>/`.
`MINDSET_VERSION` (2) is stored in every record and guarded on resume; it is not
in the directory name.

## 3 Code changes — `src/healthy_rl/rollouts.py`

### 3.1 Mindset text (verbatim)

```python
MINDSET_KEY = "mindset"            # config key: list of names, or comma string
MINDSET_VERSION = 2                # matches experiments/step0_elicitation.MINDSET_VERSION
MINDSET_HEADER = "\n\nHow to approach this:\n\n"
MINDSET: dict[str, str] = {"growth": ..., "resilience": ..., "appraisal": ...}
def mindset_section(names: Sequence[str]) -> str
def mindset_for(cfg) -> tuple[str, ...]   # sorted by MINDSET order, validated names, () when none
```

`MINDSET` and `mindset_section` reproduce `experiments/step0_elicitation.py`
**character for character**: `mindset_section` returns `""` for none, otherwise
`MINDSET_HEADER + "\n\n".join(chosen) + "\n\n"`, `chosen` in `MINDSET` order.
Copied rather than imported because that module imports ImpossibleBench at
module scope (same reason `AFFECT_INSTRUCTION` is copied).

Test `tests/cpu/test_mindset.py::test_mindset_text_matches_step0` parses
`experiments/step0_elicitation.py` with `ast` (as `test_affect_prompt.py` does)
and asserts the three block strings, the header, the join, and `MINDSET_VERSION`
are identical. Drift fails the suite.

### 3.2 Composition

`compose_instruction(instruction, affect, mindset=())` becomes

```
instruction + mindset_section(mindset) + (AFFECT_INSTRUCTION if affect else "")
```

`bench_instruction(affect, mindset)` passes it through, so turn 1 is
`INSTRUCTION + " **DO NOT MODIFY THE TESTS.**" + INSTRUCTION_SUFFIX + mindset + affect`.
Mindset before affect, as in hers; affect stays last, as in every existing
cell. Observed and left alone: her scaffold puts the benchmark boilerplate
*after* the affect sentence, ours before it — a pre-existing difference between
the two pipelines that this change neither fixes nor widens.

### 3.3 Send once

`strip_mindset_from_reminders(samples, mindset)`: for each converted sample,
`meta["instruction_prompt"] = before.replace(section, "\n\n")`; raises
`RuntimeError` if `section not in before` (a silent no-op would produce a
six-times arm wearing a once-only label). No-op when `mindset` is empty.
`build_task` calls it right after `samples = [convert(...)]`, on the local-parquet
path; the `use_hf` path raises if `mindset` is set (it builds its own prompt),
mirroring the existing `affect_prompt` refusal.

Turn 1 = instruction with block, then the problem. Turns 2–6 = the scaffold's
failure message + `To reiterate, this is your task: ` + instruction **without**
the block. Everything else in the reminder repeats as before.

### 3.4 Records, summary, resume

- `RunState.mindset: tuple[str, ...]`.
- Every record: `MINDSET_KEY: list(state.mindset)` — in `MINDSET` order, the
  order the blocks appear in the prompt (`[]` when none) — and
  `"mindset_version": MINDSET_VERSION`.
- `summary.json`: `mindset`, `mindset_version`, `instruction` (turn 1, already
  recorded), and new `instruction_reminder` (the stripped text the scaffold
  re-sends). Both strings recorded so what the model saw is checkable without
  the eval log.
- `check_resume_mindset(existing, mindset, path)`: refuses to resume a JSONL
  whose records carry a different mindset set or version; records predating the
  key count as none / version 0. Same shape and message style as
  `check_resume_affect`.
- The startup print (`scratchpad_reasoning=... affect_prompt=...`) gains
  `mindset=[...] v2`.

### 3.5 Per-token projections on disk

`summarise_hook_results` puts, per capture layer, into the stashed arrays that
`_write_residuals` already writes to the rollout's `.npz`:

| key in npz | shape | dtype | meaning |
|---|---|---|---|
| `t{turn}_proj_L{n}` | (P, 14) | float16 | projection of every hook row onto the 14 directions; P = prefill row(s) + decode rows, in hook order |
| `t{turn}_norm_L{n}` | (P,) | float32 | residual norm of each row |
| `t{turn}_kind_L{n}` | (P,) | int8 | 1 = prefill row (the residual that produced the first generated token), 0 = decode row |

Kept for **every** capture layer, not only the probe layer. Cosine at row *i* is
`proj[i] / norm[i]`. `kind` is stored rather than filtered so a chunked-prefill
one-position chunk (recorded as a decode row, see `docs/measurement.md`) stays
visible instead of silently shifting positions. `turn_stat`, `turn_stat_layers`,
`turn_observed_norm`, and the boundary residuals are unchanged, so nothing
downstream breaks. Cost: ~0.8 MB per rollout, ~20 MB per cell.

The record gains `turn_completion: list[str]` (each turn's completion text, from
the same `_model_turns` that already yields it) so analysis can re-tokenise
offline and check the count against the decode-row count. Best effort: for
reasoning models the completion may not include the reasoning tokens, so a
mismatch is expected and must be reported, not hidden. Requesting `logprobs` for
exact token strings is the right follow-up and is out of scope tonight — that
request path has not yet run on a GPU here.

Test on synthetic hook output: arrays present at every capture layer with the
right shapes/dtypes, `kind` marks the prefill row, `turn_stat` unchanged.

### 3.6 Config plumbing

`run_rollouts` reads `mindset_for(cfg)`, passes it to `build_task` and the
summary, sets `RunState.mindset`, and runs `check_resume_mindset` alongside the
existing three guards. Unknown mindset names raise at startup.

## 4 Rendered-prompt check — `scripts/render_rollout_prompts.py`

Runs inside `eval.sif` (`impossiblebench` importable), imports
`healthy_rl.rollouts`, and writes `docs/prompts-rollouts.md`: for each of
baseline / growth / resilience / appraisal, with affect off and on, the exact
turn-1 instruction and the exact reminder text (`bench_instruction` and the
stripped copy), plus word counts. This is what I read before submitting; it is
also the artefact that lets anyone check the stimulus without opening an eval
log. Same idea as her `experiments/render_prompts.py`, for this pipeline.

## 5 Shard configs and submission — `scripts/mindset_cells.sh`

One script, `--dry-run` by default, `--submit` to act:

1. Writes the 27 shard configs from the model's `d6` (or `aff6`) shard config:
   copies the file, sets `max_tokens: 24576`, appends a commented `mindset:` block
   naming the arm and its comparison cell, sets `out_dir: /out/rollouts/<model>/<version>`.
   Refuses to overwrite an existing shard config that differs.
2. Submits, per shard, the same `sbatch` line the 2x2 used
   (`--gres=gpu:A100-40G:2 --mem=96G --cpus-per-task=16 --time=4:00:00
   slurm/serve.slurm --model M --config C --gpu-memory-utilization 0.90 --stage
   scripts/run_rollouts.py:C`), job name `<model>-<version>-s<i>`, then two
   continuations `-cont` and `-cont2` chained with `--dependency=afterany`
   (resume appends; an idle continuation exits after model load). Priority-2
   and -3 cells get `--nice=2000` / `--nice=4000` (`PriorityType=priority/multifactor`,
   so nice is honoured).
3. Prints the job-id table for `docs/runs.md`.

Submitted **from the worktree** so jobs bind the worktree at `/project`
(worktree `.env` sets `PROJECT_DIR`; `.venv`, `slurm`, `apptainer/eval.sif` are
symlinks to the main checkout — see memory `worktree-slurm-setup`).

## 6 Monitoring while unattended

A wakeup loop every ~30–45 min:

- `MODELS=... CELLS="growth6 resil6 appr6 affgrowth6 affresil6 affappr6" scripts/grid_status.sh`
  — records per cell, queue state, `NO-ATTEMPTS` / `NO-PROGRESS`. A flagged
  shard of *mine* is `scancel`led; its continuation resumes. The peer agent's
  jobs are never touched.
- As records land: `scripts/live_trajectory.py --model M --version V` for the
  new cells, both positions; check `hook_data`, skipped-residual counts, and that
  the new npz keys exist. A silent measurement fault is caught in the first
  records, not in the morning.
- Extra continuations submitted if a cell is short with nothing queued.

## 7 Docs

- `docs/runs.md`: version-table rows for the six new versions; current-state
  rows; the DAG job ids; the comparison rule (mindset cell vs its base cell,
  single-token, both positions, t0 and first-to-last); **correct** the per-token
  sentence — describe the new npz keys and say older records have only the
  boundary residuals.
- `docs/elicitation.md` "Mindset prompts" bullet: note the vector arm exists and
  where it is registered.
- No findings written until the user has looked.

## 8 Merge

Feature branch merges to `main` only after `tests/cpu` is green and
`docs/prompts-rollouts.md` has been read. All changes are default-off (mindset)
or additive (npz keys, record fields), so the peer's later jobs are unaffected
in behaviour and gain the per-token arrays.

## 9 Out of scope

Token strings via `logprobs`; the feedback-channel arms (wise / neutral person)
and immunisation from `docs/interventions.md`; gemma-3-12b-it (`sp6` base);
`pos6`-base mindset cells; any analysis tool beyond `live_trajectory.py`.
