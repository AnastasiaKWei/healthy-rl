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

  | model | `d6` | `pos6` | `aff6` | `affpos6` | solve rate on `pos6` |
  |---|---:|---:|---:|---:|---:|
  | Ministral-3-14B | 23 | 20 | 23 | 20 | 5/24 |
  | Nemotron-3-Nano-4B | 23 | 11 | 23 | 8 | 13/24 |
  | Qwen3-14B | — | **≈0** | — | **≈0** | **17/17 so far** |

  Depth in the solvable arm is set by how good the model is at the problems, and
  it varies from "barely matters" to "the arm does not exist":

  - **Ministral** keeps failing for six turns whether or not the tests can be
    satisfied, so its columns compare over nearly equal depth.
  - **Nemotron** solves early more often and thins to 11 and 8 rollouts by t5.
  - **Qwen3-14B solves the original split almost perfectly** — 17 of 17 and 18 of
    20 in the records so far — in 1–4 turns and ~11k tokens, against ~42–45k
    tokens over six failing turns on `conflicting`. Its solvable cells have
    essentially no trajectory past t1. For this model the split contrast is
    "six turns of failure" versus "solved immediately", which is a far larger
    difference than the same contrast on Ministral.

  Read the split comparison at **t0**, where every cell has all 24 rollouts, and
  check this table before quoting anything at a later turn index. An earlier note
  here generalised Ministral's near-equal depth to the design as a whole; that was
  wrong.
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
| Qwen3.5-9B | `aff6` | 21 | resumed, running |
| Qwen3.5-9B | `pos6` | 16 | resumed, running |
| Qwen3.5-9B | `affpos6` | 23 | resumed, queued |
| Qwen3.5-9B | `v1` | 22 | void |
| gemma-3-12b-it | `d6` | 24 | complete, analysed |
| gemma-3-12b-it | `sp6` | 24 | complete, analysed |
| gemma-3-12b-it | `aff6` | 24 | complete, analysed (turn-end only) |
| Olmo-3.1-32B-Think | `v1` | 172 | void |
| Qwen3.6-27B | `v1` | 0 | never produced records |
| Qwen3-14B | `d6` | 1 | **the problem cell** — see below |
| Qwen3-14B | `aff6` | 17 | running |
| Qwen3-14B | `pos6` | 21 | running |
| Qwen3-14B | `affpos6` | 21 | running |
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

## Handoff: state at 2026-08-15, end of day

**8 of 16 cells complete.** Ministral-3-14B and Nemotron-3-Nano-4B are done in
all four cells. Qwen3.5-9B and Qwen3-14B are each 1–8 records short per cell,
with jobs running or queued for every one.

What the finished half already shows, on two models and two architectures:

- **The split makes no difference.** Matched cells agree to four decimals at t0
  (`frustrated` +0.0059 vs +0.0060 on Ministral, +0.0092 vs +0.0093 on Nemotron).
  Whether the tests can be satisfied at all is invisible to these probes. The
  rising `desperate`/`frustrated` trajectory tracks accumulated failure on a hard
  multi-turn task, not impossibility.
- **The affect prompt shifts the whole trajectory up**, by roughly +0.026 on
  Ministral `frustrated`, present from t0 before any test has failed, and held
  across all six turns. Asking a model to verbalise affect changes what is
  represented, not only what it says.
- **All four cross-model directions replicate on Nemotron** — a model the gate
  had excluded — at n=24, correct signs, all p<0.001.

Read the split comparison at **t0**: it is the only turn index where all four
cells have all 24 rollouts, because solvable-arm depth varies by model (see the
table above).

**Not established, do not repeat as fact:** the affect prompt tripled Ministral's
solve rate on the solvable split (5/24 → 16/24, Fisher p=0.003). It does **not**
replicate on Nemotron (13/24 → 15/24, p=0.77). One model, twelve problems.

### Operational hazards for whoever picks this up

- **Jobs hang with their server still healthy.** Two confirmed. `grid_status.sh`
  now flags them; see
  [infrastructure.md](infrastructure.md#a-rollout-job-can-hang-with-its-server-still-generating).
  The fix is `scancel` plus a resubmit of the same shard config — resume appends.
- **`request_timeout_s: 600` is the leading suspect** for those hangs and is
  **unconfirmed**. It is shorter than a full-length Qwen3-14B generation (~30
  minutes). Deliberately left unchanged: altering it mid-experiment on an
  unverified hypothesis is how this project has produced withdrawn claims before.
  Test it properly on one shard before changing it everywhere.
- **Qwen3-14B `d6` is the cell at risk.** It generates 42–45k tokens per rollout
  over six failing turns, has 1 record, and is on 8-hour continuation jobs. If it
  cannot finish, the honest fallback is a 3-model 2x2 plus a Qwen3-14B cell
  reported at whatever n it reached — not a quiet drop.

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
| `n_think` | how many of the `tokens` were reasoning tokens. It counts over `token_kind`, so it equals a share of `n_generated` only when the record is not `misaligned` |
| `at_cap` | `n_generated >= max_tokens` **or** `finish_reason == "length"`. Wider than the rollout records' cap check, which is the `turn_n_generated == max_tokens` comparison below |
| `misaligned` | hook rows did not equal `len(tokens)`; the row is still written, the token strip is hidden, and the counts are in `error`. Treat the turn's per-token readouts as unusable |
| `warnings` | non-fatal notes from generation assembly, e.g. `reasoning_content offset is a guess: answer text not found in token stream` |
| `text`, `reasoning`, `answer` | the full completion, and the two halves of it |
| `messages_in` | the full input message list for this generation, so any attempt can be replayed exactly |
| `condition` | task runs: `scratchpad`, `affect_prompt`, `temperature`, `max_tokens`, `auto_continue`, `system_prompt_hash`. Chat records carry only `max_tokens` and `temperature` — the other switches are task-loop settings and do not exist for a chat turn |
| `user_intervention` | text the user inserted before the feedback message, if any. Rollouts have no such thing |
| `title` | first chat turn only; what the rail shows |
| `passed`, `feedback` | task runs: the scorer result and the exact message fed back |
| `timings` | `request_s`, and `sandbox_s` on task rows |
| `arrays` | relative path to the npz |

`emotions`, `capture_layers`, `probe_layer`, `bench_split` and `task_id` mean what
they mean in a rollout record, and `emotions` is checked the same way — a mismatch
against `vectors.json` relabels every column. A record whose `emotions` differ from
the loaded vectors' comes back from `/api/conversations/{id}` with every readout
`None` and `emotion_order_mismatch: true` on the turn, and is skipped (and counted)
in `/api/aggregate` rather than plotted under the wrong labels.

`reasoning_from_parser` is true when the server returned the reasoning as its own
`reasoning_content` field rather than inside tags. On those records `text` is the
two halves joined for display (`reasoning`, a blank line, then `answer`) and it is
`answer`, not `text`, that was fed back to the model as the assistant turn.

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
| 5643496 | smoke gate, `--stage scripts/dashboard.py::--smoke` | ran 2026-08-15 on spockmk2-22-01, `smoke_ok: false` — see below |
| 5643744 | the dashboard itself, `--dependency=afterok:5643496` | dropped 2026-08-15 by the failed dependency, never ran |
| 5643851 | streaming-hook spike, `--stage scripts/spike_stream_hooks.py` | completed 2026-08-15 on spockmk2-03: `{"per_request_stream_hooks": true, "persistent_keyed_by_response_id": true, "rows_match_tokens": true}`. `persistent_keyed_by_response_id` means the collect key *starts with* the response id (`<id>-<suffix>`) — a prefix match, not the literal id |
| 5645488 | smoke gate rerun after the fix | submitted 2026-08-15, running on spockmk2-09 at time of writing; **result pending** |
| 5645489 | the dashboard itself, `--dependency=afterok:5645488` | submitted 2026-08-15, pending (Dependency) at time of writing |

All of them are `slurm/serve.slurm` on Ministral-3-14B-Reasoning-2512 with
`configs/dashboard.yaml`; logs are `logs/serve-<jobid>.out`.

**The first smoke gate ran and failed on a real bug.** 5643496 served the model,
generated, and wrote 3 records; `first_start_readout` was −0.0021 (finite, so the
probe path works end to end). It returned `smoke_ok: false` because both task
attempts hit the 512-token cap and came back one decode row short —
`"512 logprob tokens but 511 decode rows in hook results"` — which
`assemble_generation` was reporting as `misaligned`. The chat turn (16 tokens,
`finish=stop`) aligned fine. Cause and fix are in
[measurement.md](measurement.md#the-dashboards-readouts) (commit c41c5af): a
capped generation never feeds its last token back, so that token has no residual
row, and the row is now padded rather than flagged. 5643744 was dropped by the
`afterok` dependency and never ran; the rerun is 5645488 with 5645489 behind it,
and **its result is still pending** — nothing in this section has yet been
confirmed against a record file from a passing gate.

5643851 has answered its question — token-text streaming is feasible, per-request
hooks and all, and stays unimplemented in this version; the verdict and the
traps that come with it are in docs/infrastructure.md, "Streaming and hooks".

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
