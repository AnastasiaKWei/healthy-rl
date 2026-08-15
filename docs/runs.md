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

  **Measured, first completed cell (Ministral `pos6`, n=24):** 5 of 24 rollouts
  solved the problem, and 20 of 24 still ran the full six turns. Trajectory depth
  survives the switch to solvable tests — 20 rollouts reach t5 against 23 in `d6`
  — so the two columns are compared over nearly the same number of turns rather
  than a deep arm against a shallow one. This was the main risk to the design and
  it did not materialise. The matching `passed` counts are 5/24 here (solves) and
  0/24 in `d6` (hacks), which is the split inversion showing up exactly where it
  should.
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
| Ministral-3-14B-Reasoning-2512 | `aff6` | 21 | **still accumulating** |
| Ministral-3-14B-Reasoning-2512 | `pos6` | 24 | complete; 5/24 solved, 20/24 ran all six turns |
| Ministral-3-14B-Reasoning-2512 | `affpos6` | 0 | **submitted 2026-08-15**, jobs 5641830-32 |
| Ministral-3-14B-Reasoning-2512 | `v1` | 48 | void |
| Qwen3.5-9B | `d6` | 18 | **still accumulating**, analysed at n=17–18 |
| Qwen3.5-9B | `aff6` | 0 | **submitted 2026-08-15**, jobs 5641818-20 |
| Qwen3.5-9B | `pos6` | 0 | **submitted 2026-08-15**, jobs 5641821-23 |
| Qwen3.5-9B | `affpos6` | 0 | **submitted 2026-08-15**, jobs 5641824-26 |
| Qwen3.5-9B | `v1` | 22 | void |
| gemma-3-12b-it | `d6` | 24 | complete, analysed |
| gemma-3-12b-it | `sp6` | 24 | complete, analysed |
| gemma-3-12b-it | `aff6` | 24 | complete, analysed (turn-end only) |
| Olmo-3.1-32B-Think | `v1` | 172 | void |
| Qwen3.6-27B | `v1` | 0 | never produced records |
| Qwen3-14B | `d6` | 0 | **submitted 2026-08-15**, jobs 5641928-30 |
| Qwen3-14B | `aff6` | 0 | **submitted 2026-08-15**, jobs 5641931-33 |
| Qwen3-14B | `pos6` | 0 | **submitted 2026-08-15**, jobs 5641934-36 |
| Qwen3-14B | `affpos6` | 0 | **submitted 2026-08-15**, jobs 5641937-39 |
| Nemotron-3-Nano-4B-BF16 | `d6` | 0 | **submitted 2026-08-15**, jobs 5641940-42 |
| Nemotron-3-Nano-4B-BF16 | `aff6` | 0 | **submitted 2026-08-15**, jobs 5641943-45 |
| Nemotron-3-Nano-4B-BF16 | `pos6` | 0 | **submitted 2026-08-15**, jobs 5641946-48 |
| Nemotron-3-Nano-4B-BF16 | `affpos6` | 0 | **submitted 2026-08-15**, jobs 5641949-51 |

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

## Stale artifacts

`results/summary.md` describes the **void** 3-turn Olmo pilot and has not been
regenerated. `scripts/compare.py` assumes version `v1`. Do not quote either
without regenerating against `d6` first. `results/` is git-ignored.
