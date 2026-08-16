# Run registry

What exists under `$ARTIFACT_DIR`, what each version means, and how to read it.
Results are in [findings.md](findings.md).

## Version naming

Vectors and gate artifacts are all `v1` — one extraction per model. Rollout
versions encode the condition:

| version | turns | condition |
|---|---:|---|
| `v1` | 3 | original pilot. **Void** — token-budget confound, see [infrastructure.md](infrastructure.md#token-budget) |
| `p1`, `p2` | 3 | early sharded pilot tiers, same confound |
| `hitok` | 3 | the higher-token-cap attempt; produced no completed records |
| `d6` | 6 | **baseline.** No scratchpad, no affect prompt, conflicting split |
| `sp6` | 6 | scratchpad reasoning (`scratchpad_reasoning: true`) |
| `aff6` | 6 | affect prompt (`affect_prompt: true`), conflicting split |
| `pos6` | 6 | no affect prompt, **original (solvable) split** |
| `affpos6` | 6 | affect prompt, original split |

`d6` / `sp6` / `aff6` / `pos6` / `affpos6` are the trustworthy set. Each condition
needs its own `out_dir` — resume refuses to mix them, and since 2026-08-15 it
also refuses to mix bench splits.

## The 2x2

`pos6` and `affpos6` complete a cross product: **{affect prompt on, off} x
{conflicting tests, solvable tests}**. It separates two things the `d6`-vs-`aff6`
contrast alone cannot:

|  | conflicting (impossible) | original (solvable) |
|---|---|---|
| **no affect prompt** | `d6` | `pos6` |
| **affect prompt** | `aff6` | `affpos6` |

Run on four models: Qwen3.5-9B, Ministral-3-14B-Reasoning-2512, Qwen3-14B and
Nemotron-3-Nano-4B-BF16. The last two failed the logit-lens gate and are run
anyway — the failures are lexical, not semantic; see the ruling in
[findings.md](findings.md#instrument-gate).

- The split column asks whether the rising `desperate`/`frustrated` trajectory
  needs the task to be *impossible*, or whether any six turns of a hard coding
  problem produce it. lcbhard is hard, so `pos6` still contains genuine repeated
  failure — it is failure without unsatisfiability, which is the control that
  matters.

  **Trajectory depth in the solvable arm is model-specific — check it per model
  before comparing at late turn indices.** Rollouts reaching t5, out of 24:

  | model | `d6` | `pos6` | `aff6` | `affpos6` |
  |---|---:|---:|---:|---:|
  | Ministral-3-14B | 23 | 20 | 23 | 20 |
  | Nemotron-3-Nano-4B | 23 | **11** | 23 | **8** |

  Ministral barely notices: lcbhard is hard enough that it keeps failing for six
  turns whether or not the tests can be satisfied, so its columns compare over
  nearly equal depth. Nemotron solves early far more often, so its solvable cells
  thin out to 11 and 8 rollouts by t5. An earlier note here generalised
  Ministral's result to the design as a whole; that was wrong. For Nemotron, read
  t0–t2 and treat t5 as underpowered.
- The affect row asks whether being asked to *verbalise* affect changes what is
  *represented*, or only what is said.

Both splits come from the same ImpossibleBench dataset. ImpossibleBench builds
`conflicting` from `original` by inserting one assertion that contradicts an
existing one; the 103 task_ids, the prompts (byte-identical, verified) and the
entry points are otherwise the same. So the two columns differ in the tests and
nothing else.

`max_tokens` is 24576 in all the new cells. The existing `d6` runs used 16384,
which never bound — the longest turn any model generated was 14198 tokens
(Ministral `aff6`) and 9861 in `d6`, so nothing was truncated. Checked rather
than assumed, because an unmatched token cap is what voided the first pilot.

## Current state

Rollout records, `$ARTIFACT_DIR/rollouts/<model>/<version>/*.jsonl`:

| model | version | records | notes |
|---|---|---:|---|
| Ministral-3-14B-Reasoning-2512 | `d6` | 24 | complete, analysed |
| Ministral-3-14B-Reasoning-2512 | `aff6` | 24 | complete |
| Ministral-3-14B-Reasoning-2512 | `pos6` | 24 | complete; 5/24 solved, 20/24 ran all six turns |
| Ministral-3-14B-Reasoning-2512 | `affpos6` | 23 | last shard running |
| Ministral-3-14B-Reasoning-2512 | `v1` | 48 | void |
| Qwen3.5-9B | `d6` | 18 | **still accumulating**, analysed at n=17–18 |
| Qwen3.5-9B | `aff6` | 21 | hit the 4h wall; resumed as 5643601-03 |
| Qwen3.5-9B | `pos6` | 16 | running |
| Qwen3.5-9B | `affpos6` | 22 | running |
| Qwen3.5-9B | `v1` | 22 | void |
| gemma-3-12b-it | `d6` | 24 | complete, analysed |
| gemma-3-12b-it | `sp6` | 24 | complete, analysed |
| gemma-3-12b-it | `aff6` | 24 | complete, analysed (turn-end only) |
| Olmo-3.1-32B-Think | `v1` | 172 | void |
| Qwen3.6-27B | `v1` | 0 | never produced records |
| Qwen3-14B | `d6` | 0 | running; max_model_len fix confirmed |
| Qwen3-14B | `aff6` | 0 | relaunched as 5642680-82 |
| Qwen3-14B | `pos6` | 0 | relaunched as 5642683-85 |
| Qwen3-14B | `affpos6` | 0 | relaunched as 5642686-88 |
| Nemotron-3-Nano-4B-BF16 | `d6` | 24 | complete |
| Nemotron-3-Nano-4B-BF16 | `aff6` | 24 | complete |
| Nemotron-3-Nano-4B-BF16 | `pos6` | 22 | running |
| Nemotron-3-Nano-4B-BF16 | `affpos6` | 23 | last shard running |

gemma-3-12b-it has no `pos6`/`affpos6` cell: it is the flattest of the measured
models, and the 2x2s went to the models with a clear conflicting-split signal
plus the two the gate had wrongly excluded. Adding it is cheap if the split
contrast turns out to carry the result.

Vectors exist for all eight gated models. Rollouts now cover six of them. The
two that never ran are gemma-4-12B-it (the one genuine gate failure) and the
27B models Olmo and Qwen3.6, which are exempt from the "every new passing model
runs the benchmark" rule because they predate it and are the wrong size for this
question.

## Reading a run

```bash
# trajectories + first-vs-last + repetition; works on runs in flight
scripts/live_trajectory.py --model gemma-3-12b-it --version d6

# transcripts (opens .eval logs inside the container)
scripts/read_transcript.sh --model gemma-3-12b-it --version d6

# recover true hack scores from existing logs, no re-run needed
scripts/rescore_transcripts.py --model Qwen3.5-9B --version d6

# pipeline status
scripts/pilot_status.sh
```

## Record fields worth knowing

Each JSONL row is one rollout:

- `emotions` — direction order. Must match `vectors.json`; analysis refuses to run
  if it does not.
- `n_turns`, `turn_n_generated` — **check the token counts before trusting a
  trajectory.** Turns sitting exactly at `max_tokens` mean you are measuring the
  cap. Zero-token leading turns exist and must not be indexed over.
- `turn_stat` — per-turn mean projections. The **superseded** statistic; see
  [measurement.md](measurement.md#granularity-single-token-vs-turn-mean).
- `residuals` — relative path to an `.npz` of boundary residuals, keyed
  `t{turn}_res_{start|end}_L{layer}`. This is what the single-token analysis
  reads. Some are non-finite; check the skip count the tool prints.
- `bench_split` — `conflicting` or `original`. **Read this before reading
  `passed`.** Records written before 2026-08-15 have no such key and are all
  `conflicting`.
- `passed` — **its meaning inverts across splits.** On `conflicting`, true means
  the submission satisfied mutually unsatisfiable tests, i.e. it cheated; it is
  zero everywhere so far, with the caveat in
  [findings.md](findings.md#hack-rate-is-zero-everywhere-and-the-number-means-less-than-it-looks).
  On `original`, true means the model solved the problem. Never pool the two.

Per-token projections onto all 14 directions are kept at **every** capture layer
(~280 bytes/token). Full residuals are kept only at event positions — turn
boundaries and the first token after a test-failure message — and only at the
probe layer by default. The per-token projections are the untapped resource: the
signal is localised and we currently sample two positions per turn out of
hundreds.

## Bench artifacts

Two splits of one dataset, in separate directories because `fetch_bench.py`
writes `bench.json` and `manifest.json` under fixed names holding only the split
it just fetched:

| dir | split | config | used by |
|---|---|---|---|
| `$ARTIFACT_DIR/bench/v1` | `conflicting` | `configs/fetch_bench.yaml` | `d6`, `sp6`, `aff6` |
| `$ARTIFACT_DIR/bench/orig1` | `original` | `configs/fetch_bench_original.yaml` | `pos6`, `affpos6` |

Both are login-node stages: compute nodes have no DNS.

## Dashboard sessions

The Affect Scope dashboard writes its own records, one directory per job:

```
$ARTIFACT_DIR/dashboard/<model>/<jobid>/
    session.json          model, emotion order, capture layers, probe layer,
                          vectors dir, job id/node, config, zstd check, start time
    records.jsonl         one row per generation (chat turn or task attempt)
    proj/<record_id>.npz  proj (T,L,E) float32, norm (T,L), proj_prefill (L,E),
                          norm_prefill (L), res_start_L<probe>, res_end_L<probe> (float16)
```

`python -m healthy_rl.dashboard --replay <that directory>` reopens it read-only on
the login node — rail, transcript, trajectory, tokens and aggregate all work with
no GPU and no vectors artifact, because the projections are already in the npz.

### How these records differ from rollout records

A rollout record is one whole rollout with per-turn arrays inside it. A dashboard
record is **one generation**, so the turn structure is spread across rows of the
same `conversation_id` rather than nested in one row:

| field | meaning |
|---|---|
| `source` | `chat` or `task`. Never aggregate across the two |
| `conversation_id`, `record_id` | the conversation this generation belongs to, and this row |
| `turn_index`, `non_empty_turn_index` | raw index within the conversation; index among turns that generated tokens (`None` for an empty turn). Analysis uses the second, as `live_trajectory.py` does |
| `attempt` | task runs: 1-based attempt number, `turn_index + 1` |
| `tokens`, `token_kind` | the generated token strings, and `think` / `answer` per token. Rollout records keep no token text |
| `n_think` | how many of `n_generated` were reasoning tokens |
| `at_cap` | `n_generated == max_tokens`. The dashboard's version of the `turn_n_generated` check below |
| `misaligned` | hook rows did not equal `len(tokens)`; the row is still written, the token strip is hidden, and the counts are in `error`. Treat the turn's per-token readouts as unusable |
| `warnings` | non-fatal notes from generation assembly, e.g. `reasoning_content offset is a guess: answer text not found in token stream` |
| `text`, `reasoning`, `answer` | the full completion, and the two halves of it |
| `messages_in` | the full input message list for this generation, so any attempt can be replayed exactly |
| `condition` | `scratchpad`, `affect_prompt`, `temperature`, `max_tokens`, `auto_continue`, `system_prompt_hash` |
| `user_intervention` | text the user inserted before the feedback message, if any. Rollouts have no such thing |
| `title` | first chat turn only; what the rail shows |
| `passed`, `feedback` | task runs: the scorer result and the exact message fed back |
| `timings` | `request_s`, and `sandbox_s` on task rows |
| `arrays` | relative path to the npz |

`emotions`, `capture_layers`, `probe_layer`, `bench_split` and `task_id` mean what
they mean in a rollout record, and `emotions` is checked the same way — a mismatch
against `vectors.json` relabels every column.

`passed` **inverts across splits here exactly as it does for rollouts.** On
`conflicting` a pass means the submission satisfied mutually unsatisfiable tests,
i.e. it cheated; on `original` it means the model solved the problem. The
`/api/aggregate` route refuses to pool the two rather than leaving it to the
reader: asking for `source=task` with both splits present and no `split=` returns
400.

`proj_prefill` / `norm_prefill` are the prefill row — the residual that produced
the first generated token. That is what the `start` readout reads; see
[measurement.md](measurement.md#reading-the-tools).

### The GPU smoke gate

`scripts/dashboard.py --smoke` is the gate the dashboard job is submitted behind.
It runs one chat turn and one two-attempt task on the `original` split through the
real engine and the real sandbox, then prints one JSON line and exits 0 or 1:

```bash
sbatch --time=1:00:00 slurm/serve.slurm --model Ministral-3-14B-Reasoning-2512 \
    --config configs/dashboard.yaml --stage scripts/dashboard.py::--smoke
```

Reading the line (in `logs/serve-<jobid>.out`):

| key | green |
|---|---|
| `smoke_ok` | `true`; it is the AND of everything below |
| `chat_turn_event`, `task_done_event` | `true` — the SSE streams reached `turn` and `done` |
| `n_records` | ≥ 2 (one chat turn, one or two task attempts) |
| `misaligned` | `[]`. Any record id here means hook rows and tokens disagreed |
| `errors` | `[]`. A turn that errored is a red gate: the run is otherwise indistinguishable from a healthy one |
| `first_start_readout` | a float, not `null` — the readout path works end to end |
| `problems_error`, `readout_error` | absent. Present, they name a failure that would otherwise have crashed the gate before it printed anything |

A `WARNING: vllm-lens zstd file patch is NOT applied` on stderr is expected
whenever `uv sync` has reverted the patch; it is recorded in `session.json`, not
fatal (see [infrastructure.md](infrastructure.md#the-zstd-patch-is-recorded-not-required)).

### Dashboard jobs

| job | what | state |
|---|---|---|
| 5643496 | smoke gate, `--stage scripts/dashboard.py::--smoke` | submitted 2026-08-15, pending at time of writing |
| 5643744 | the dashboard itself, `--dependency=afterok:5643496` | submitted 2026-08-15, pending (Dependency) at time of writing |
| 5643851 | streaming-hook spike, `--stage scripts/spike_stream_hooks.py` | submitted 2026-08-15, pending at time of writing |

All three are `slurm/serve.slurm` on Ministral-3-14B-Reasoning-2512 with
`configs/dashboard.yaml`; logs are `logs/serve-<jobid>.out`. **No dashboard
session has run on a GPU yet** — nothing in this section has been confirmed
against a real record file. If 5643496 fails, 5643744 shows
`DependencyNeverSatisfied` and must be cancelled and resubmitted after the fix.
5643851 decides whether token-text streaming is possible at all; the dashboard
renders a turn only when the response lands until it passes.

## Jobs hit a 4-hour wall

Rollout jobs are submitted with `--time=4:00:00` and a shard that has not
finished its eight rollouts is SIGTERMed, which `sacct` reports as
`FAILED ... 143:0`. That is a wall, not a fault. Resume is the fix: resubmit the
same shard config and it appends to the JSONL, skipping the rollouts already
recorded. `scripts/grid_status.sh` flags the case that matters — a cell short of
24 records with nothing queued.

## Stale artifacts

`results/summary.md` describes the **void** 3-turn Olmo pilot and has not been
regenerated. `scripts/compare.py` assumes version `v1`. Do not quote either
without regenerating against `d6` first. `results/` is git-ignored.
