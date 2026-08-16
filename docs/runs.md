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
| `growth6`, `resil6`, `appr6` | 6 | mindset block (growth / resilience / appraisal, v2), otherwise `d6`: no affect prompt, conflicting split. Block sent on turn 1 only. Compare against `d6` |
| `affgrowth6`, `affresil6`, `affappr6` | 6 | same blocks with the affect prompt on, otherwise `aff6`. Compare against `aff6` |
| `spgrowth6`, `spresil6`, `spappr6` | 6 | gemma-3-12b-it only: the same blocks on the `sp6` base (scratchpad on, no affect prompt). Compare against `sp6` |
| `growth6b`, `resil6b`, `appr6b`, `affgrowth6b`, `affresil6b`, `affappr6b`, `spgrowth6b`, `spresil6b`, `spappr6b`, `spaffgrowth6b`, `spaffresil6b`, `spaffappr6b` | 6 | the same 18 mindset cells re-run with the **trigger-fixed** v2 text (Anastasia's `7d6fd07`, 2026-08-16): "Your first attempt is just the code. Open every attempt after that with …" instead of "Whenever a test fails, run this …", which is inert once the block is sent on turn 1 only. Same version number (2); the two texts are told apart by `mindset_hash`. Same bases as the un-suffixed cells; the per-token comparison cell is `<base>r` (or `spaff6`) |
| `spaff6` | 6 | gemma-3-12b-it only: scratchpad on **and** affect prompt on — a new base cell (2026-08-16); per-token arrays present. Compare mindset arms `spaffgrowth6`/`spaffresil6`/`spaffappr6` against it |
| `d6r`, `sp6r`, `aff6r`, `pos6r`, `affpos6r` | 6 | **re-runs** of the same-named base cells (gemma: d6/sp6/aff6; Ministral: d6/aff6/pos6/affpos6), fresh samples at temperature 1.0 written 2026-08-16 with the per-token arrays and `max_tokens` 24576. The per-token control for that model's mindset arms; also an independent replicate of the boundary readouts |

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

## The mindset arms

Anastasia's three v2 mindset blocks (`experiments/step0_elicitation.py`, rendered
in [prompts-v2.md](prompts-v2.md); the rollout versions in
[prompts-rollouts.md](prompts-rollouts.md)) inserted into the turn-1 instruction
between the benchmark text and the affect sentence, and stripped from the
reminder the scaffold re-sends after each failed attempt, so the model sees the
block once per rollout. The reminder a mindset arm sends is byte-identical to its
base arm's — turn 1 is the only place the two differ (see the word-count table in
[prompts-rollouts.md](prompts-rollouts.md), and
[measurement.md](measurement.md#the-mindset-arms-send-once-mechanism) for what
that mechanism depends on). `mindset` and `mindset_version` are on every record
and summary, with the exact turn-1 `instruction` and the stripped
`instruction_reminder` on the summary; resume refuses to mix arms or prompt
versions.

**Two v2 texts exist.** The un-suffixed cells (`growth6` …, submitted 2026-08-16
02:11) carry the v2 text of her commit `9d6615a` (2026-08-15 21:05), whose
procedure was triggered by "Whenever a test fails, run this … before writing any
new code". With the block sent on turn 1 only that trigger is inert — at turn 1
no test has failed and by turn 2 the instruction is gone; her smoke run measured
0/12 compliance. Her `7d6fd07` (2026-08-16 08:14) rephrases it as a standing
rule — "Your first attempt is just the code. Open every attempt after that with
these two lines, before any new code" — and scored 12/12. She left
`MINDSET_VERSION` at 2, so the number no longer identifies the text; every
record and summary written since carries `mindset_hash`, the first 12 hex of
sha256 over the exact section text, and resume refuses a hash-less mindset
record or a different hash (see
[measurement.md](measurement.md#the-mindset-arms-send-once-mechanism)). The
`…6b` cells are the same 18 cells with the `7d6fd07` text; the un-suffixed cells
are the `9d6615a` text and are **not** to be pooled with them.

Everything else is the base cell's shard config byte for byte except
`max_tokens`, `out_dir` and the appended `mindset:` key. Ministral `d6` used
`max_tokens` 16384; the mindset cells use 24576, the 2x2 value; the cap never
bound (longest `d6` turn 9861 tokens), so the comparison holds.

Read a mindset cell only against its base cell — `growth6` vs `d6`, `affgrowth6`
vs `aff6` — single-token, both positions, at t0 and first-to-last. The blocks
are demand characteristics pointing the opposite way from the affect prompt: a
cell whose *words* calm down while its trajectory does not is the decoupling
result, not a success. See [interventions.md](interventions.md) §8. The base
cells (`d6`, `aff6`) predate the per-token arrays, so the mindset-vs-base
comparison uses the boundary residuals both sides have; a per-token comparison
needs a base cell re-run (not queued tonight).

Shard configs and submission are one script, `scripts/mindset_cells.sh` —
dry-run by default, `--submit` to act. It writes the shard configs
(`configs/shards/rollouts-<model>-<version>-s{0,1,2}of3.yaml`, 3 per cell, 36
cells: the 18 `9d6615a`-text rows and their 18 `…6b` trigger-fixed mirrors) from
the base cell's, then submits per shard a primary job plus `-cont` and `-cont2`
chained with `--dependency=afterany`, and prints the job-id table for the
section below. `ONLY_TEXT=orig|fixed` selects one text, composing with
`ONLY_MODEL`/`ONLY_BASE`. All configs are committed and the script refuses to
overwrite one that exists with different content, so re-running the config
phase is a no-op.
Priority is carried by `--nice`: Ministral `growth6`/`resil6`/`appr6` first,
with no `--nice` flag (slurm default 0), then Qwen3.5-9B's three at
`--nice=2000`, then Ministral's affect-on three at `--nice=4000`. Qwen3-14B was
dropped by the peer session, so the capacity its cells were holding is free for
these.

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
| Qwen3-14B | `d6` | 1 | **the problem cell**; dropped 2026-08-15 (GPU budget) — see the Handoff section |
| Qwen3-14B | `aff6` | 17 | dropped 2026-08-15 (GPU budget) — see the Handoff section |
| Qwen3-14B | `pos6` | 21 | dropped 2026-08-15 (GPU budget) — see the Handoff section |
| Qwen3-14B | `affpos6` | 21 | dropped 2026-08-15 (GPU budget) — see the Handoff section |
| Nemotron-3-Nano-4B-BF16 | `d6` | 24 | complete |
| Nemotron-3-Nano-4B-BF16 | `aff6` | 24 | complete |
| Nemotron-3-Nano-4B-BF16 | `pos6` | 22 | running |
| Nemotron-3-Nano-4B-BF16 | `affpos6` | 23 | last shard running |
| Ministral-3-14B-Reasoning-2512 | `growth6` | 24 | complete 2026-08-16 02:45; priority 1 |
| Ministral-3-14B-Reasoning-2512 | `resil6` | 24 | complete 2026-08-16 02:55; priority 1 |
| Ministral-3-14B-Reasoning-2512 | `appr6` | 24 | complete 2026-08-16 03:44 (shard 1 hung mid-rollout, cancelled, resumed by its `-cont`); priority 1. **1/24 `passed` on conflicting** — a genuine call-count hack (`lcbhard_5` s0) |
| Qwen3.5-9B | `growth6` | 24 | complete 2026-08-16 04:50; priority 2 |
| Qwen3.5-9B | `resil6` | 24 | **23 real + 1 empty**: shard 2 resumed by the `-t3600` jobs (5649567/5649568); `lcbhard_4 s0` completed once the timeout was 3600 s, `lcbhard_10 s0` timed out and is a zero-token, no-hook record — delete that line to recollect. See infrastructure.md |
| Qwen3.5-9B | `appr6` | 21 | short — the three primaries stuck on the same problems (`lcbhard_10`/`11`); continuations cancelled rather than replay them |
| Ministral-3-14B-Reasoning-2512 | `affgrowth6` | 24 | complete 2026-08-16 04:40; priority 3 |
| Ministral-3-14B-Reasoning-2512 | `affresil6` | 24 | complete 2026-08-16 04:50; priority 3 |
| Ministral-3-14B-Reasoning-2512 | `affappr6` | 24 | complete 2026-08-16 05:55; priority 3 |
| gemma-3-12b-it | `spgrowth6` | 24 | complete 2026-08-16 ~11:20 (shard 1 primary died at 13 s in a vLLM engine crash, `KeyError` in the scheduler; its `-cont` finished the shard) |
| gemma-3-12b-it | `spresil6` | 24 | complete 2026-08-16 ~11:20 |
| gemma-3-12b-it | `spappr6` | 24 | complete 2026-08-16 ~11:15 |
| gemma-3-12b-it | `affgrowth6` | 24 | complete 2026-08-16 ~11:20; `--nice=2000`. Affect on, NO scratchpad (matches gemma's `aff6`); read at turn end |
| gemma-3-12b-it | `affresil6` | 24 | complete 2026-08-16 ~11:45; `--nice=2000` |
| gemma-3-12b-it | `affappr6` | 24 | complete 2026-08-16 ~11:45; `--nice=2000` |
| gemma-3-12b-it | `spaff6` | 24 | complete 2026-08-16 ~11:45; base cell for the spaff arms; jobs 5651735–5651743 |
| gemma-3-12b-it | `spaffgrowth6` | 24 | complete 2026-08-16 ~11:50; jobs 5651748–5651756 |
| gemma-3-12b-it | `spaffresil6` | 24 | complete 2026-08-16 ~11:50; jobs 5651757–5651765 |
| gemma-3-12b-it | `spaffappr6` | 24 | complete 2026-08-16 ~12:00; jobs 5651766–5651774 |
| gemma-3-12b-it | `d6r` | 24 | re-run, complete 2026-08-16 ~12:20; jobs 5653039–5653047 |
| gemma-3-12b-it | `sp6r` | 24 | re-run, complete 2026-08-16 ~12:35; jobs 5653048–5653056 |
| gemma-3-12b-it | `aff6r` | 24 | re-run, complete 2026-08-16 ~12:20; jobs 5653057–5653065 |
| Ministral-3-14B-Reasoning-2512 | `d6r` | 24 | re-run, complete 2026-08-16 ~12:35; jobs 5653066–5653074 |
| Ministral-3-14B-Reasoning-2512 | `aff6r` | 24 | re-run, complete 2026-08-16 ~12:45; jobs 5653075–5653083 |
| Ministral-3-14B-Reasoning-2512 | `pos6r` | 24 | re-run, complete 2026-08-16 ~12:40; jobs 5653084–5653092 |
| Ministral-3-14B-Reasoning-2512 | `affpos6r` | 24 | re-run, complete 2026-08-16 ~12:45; jobs 5653093–5653101 |
| Ministral-3-14B-Reasoning-2512 | `growth6b` | 0 | **queued 2026-08-16 13:37**, trigger-fixed text (`7d6fd07`); re-run of `growth6`; per-token base `d6r` |
| Ministral-3-14B-Reasoning-2512 | `resil6b` | 0 | **queued 2026-08-16 13:37**, trigger-fixed text (`7d6fd07`); re-run of `resil6`; per-token base `d6r` |
| Ministral-3-14B-Reasoning-2512 | `appr6b` | 0 | **queued 2026-08-16 13:37**, trigger-fixed text (`7d6fd07`); re-run of `appr6`; per-token base `d6r` |
| Qwen3.5-9B | `growth6b` | 0 | **queued 2026-08-16 13:37**, trigger-fixed text (`7d6fd07`); re-run of `growth6`; per-token base `d6r (does not exist yet)` |
| Qwen3.5-9B | `resil6b` | 0 | **queued 2026-08-16 13:37**, trigger-fixed text (`7d6fd07`); re-run of `resil6`; per-token base `d6r (does not exist yet)` |
| Qwen3.5-9B | `appr6b` | 0 | **queued 2026-08-16 13:37**, trigger-fixed text (`7d6fd07`); re-run of `appr6`; per-token base `d6r (does not exist yet)` |
| Ministral-3-14B-Reasoning-2512 | `affgrowth6b` | 0 | **queued 2026-08-16 13:37**, trigger-fixed text (`7d6fd07`); re-run of `affgrowth6`; per-token base `aff6r` |
| Ministral-3-14B-Reasoning-2512 | `affresil6b` | 0 | **queued 2026-08-16 13:37**, trigger-fixed text (`7d6fd07`); re-run of `affresil6`; per-token base `aff6r` |
| Ministral-3-14B-Reasoning-2512 | `affappr6b` | 0 | **queued 2026-08-16 13:37**, trigger-fixed text (`7d6fd07`); re-run of `affappr6`; per-token base `aff6r` |
| gemma-3-12b-it | `spgrowth6b` | 0 | **queued 2026-08-16 13:37**, trigger-fixed text (`7d6fd07`); re-run of `spgrowth6`; per-token base `sp6r` |
| gemma-3-12b-it | `spresil6b` | 0 | **queued 2026-08-16 13:37**, trigger-fixed text (`7d6fd07`); re-run of `spresil6`; per-token base `sp6r` |
| gemma-3-12b-it | `spappr6b` | 0 | **queued 2026-08-16 13:37**, trigger-fixed text (`7d6fd07`); re-run of `spappr6`; per-token base `sp6r` |
| gemma-3-12b-it | `affgrowth6b` | 0 | **queued 2026-08-16 13:37**, trigger-fixed text (`7d6fd07`); re-run of `affgrowth6`; per-token base `aff6r` |
| gemma-3-12b-it | `affresil6b` | 0 | **queued 2026-08-16 13:37**, trigger-fixed text (`7d6fd07`); re-run of `affresil6`; per-token base `aff6r` |
| gemma-3-12b-it | `affappr6b` | 0 | **queued 2026-08-16 13:37**, trigger-fixed text (`7d6fd07`); re-run of `affappr6`; per-token base `aff6r` |
| gemma-3-12b-it | `spaffgrowth6b` | 0 | **queued 2026-08-16 13:37**, trigger-fixed text (`7d6fd07`); re-run of `spaffgrowth6`; per-token base `spaff6` |
| gemma-3-12b-it | `spaffresil6b` | 0 | **queued 2026-08-16 13:37**, trigger-fixed text (`7d6fd07`); re-run of `spaffresil6`; per-token base `spaff6` |
| gemma-3-12b-it | `spaffappr6b` | 0 | **queued 2026-08-16 13:37**, trigger-fixed text (`7d6fd07`); re-run of `spaffappr6`; per-token base `spaff6` |

gemma-3-12b-it has no `pos6`/`affpos6` cell: it is the flattest of the measured
models, and the 2x2s went to the models with a clear conflicting-split signal
plus the two the gate had wrongly excluded. Adding it is cheap if the split
contrast turns out to carry the result.

Vectors exist for all eight gated models. Rollouts now cover six of them. The
two that never ran are gemma-4-12B-it (the one genuine gate failure) and the
27B models Olmo and Qwen3.6, which are exempt from the "every new passing model
runs the benchmark" rule because they predate it and are the wrong size for this
question.

### Mindset jobs

| model | version | shard | primary | continuations |
|---|---|---|---|---|
| Ministral-3-14B-Reasoning-2512 | growth6 | s0 | 5648803 | 5648804 / 5648805 |
| Ministral-3-14B-Reasoning-2512 | growth6 | s1 | 5648806 | 5648807 / 5648808 |
| Ministral-3-14B-Reasoning-2512 | growth6 | s2 | 5648809 | 5648810 / 5648811 |
| Ministral-3-14B-Reasoning-2512 | resil6 | s0 | 5648812 | 5648813 / 5648814 |
| Ministral-3-14B-Reasoning-2512 | resil6 | s1 | 5648815 | 5648816 / 5648817 |
| Ministral-3-14B-Reasoning-2512 | resil6 | s2 | 5648818 | 5648819 / 5648820 |
| Ministral-3-14B-Reasoning-2512 | appr6 | s0 | 5648821 | 5648822 / 5648823 |
| Ministral-3-14B-Reasoning-2512 | appr6 | s1 | 5648824 | 5648825 / 5648826 |
| Ministral-3-14B-Reasoning-2512 | appr6 | s2 | 5648827 | 5648828 / 5648829 |
| Qwen3.5-9B | growth6 | s0 | 5648830 | 5648831 / 5648832 |
| Qwen3.5-9B | growth6 | s1 | 5648833 | 5648834 / 5648835 |
| Qwen3.5-9B | growth6 | s2 | 5648836 | 5648837 / 5648838 |
| Qwen3.5-9B | resil6 | s0 | 5648839 | 5648840 / 5648841 |
| Qwen3.5-9B | resil6 | s1 | 5648842 | 5648843 / 5648844 |
| Qwen3.5-9B | resil6 | s2 | 5648845 | 5648846 / 5648847 |
| Qwen3.5-9B | appr6 | s0 | 5648848 | 5648849 / 5648850 |
| Qwen3.5-9B | appr6 | s1 | 5648851 | 5648852 / 5648853 |
| Qwen3.5-9B | appr6 | s2 | 5648854 | 5648855 / 5648856 |
| Ministral-3-14B-Reasoning-2512 | affgrowth6 | s0 | 5648857 | 5648858 / 5648859 |
| Ministral-3-14B-Reasoning-2512 | affgrowth6 | s1 | 5648860 | 5648861 / 5648862 |
| Ministral-3-14B-Reasoning-2512 | affgrowth6 | s2 | 5648863 | 5648864 / 5648865 |
| Ministral-3-14B-Reasoning-2512 | affresil6 | s0 | 5648866 | 5648867 / 5648868 |
| Ministral-3-14B-Reasoning-2512 | affresil6 | s1 | 5648869 | 5648870 / 5648871 |
| Ministral-3-14B-Reasoning-2512 | affresil6 | s2 | 5648872 | 5648873 / 5648874 |
| Ministral-3-14B-Reasoning-2512 | affappr6 | s0 | 5648875 | 5648876 / 5648877 |
| Ministral-3-14B-Reasoning-2512 | affappr6 | s1 | 5648878 | 5648879 / 5648880 |
| Ministral-3-14B-Reasoning-2512 | affappr6 | s2 | 5648881 | 5648882 / 5648883 |

Gemma cells, submitted 2026-08-16 10:23 from the main checkout with `ONLY_MODEL=gemma-3-12b-it scripts/mindset_cells.sh --submit` (54 jobs):

| model | version | shard | primary | continuations |
|---|---|---|---|---|
| gemma-3-12b-it | spgrowth6 | s0 | 5651654 | 5651655 / 5651656 |
| gemma-3-12b-it | spgrowth6 | s1 | 5651657 | 5651658 / 5651659 |
| gemma-3-12b-it | spgrowth6 | s2 | 5651660 | 5651661 / 5651662 |
| gemma-3-12b-it | spresil6 | s0 | 5651663 | 5651664 / 5651665 |
| gemma-3-12b-it | spresil6 | s1 | 5651666 | 5651667 / 5651668 |
| gemma-3-12b-it | spresil6 | s2 | 5651669 | 5651670 / 5651671 |
| gemma-3-12b-it | spappr6 | s0 | 5651672 | 5651673 / 5651674 |
| gemma-3-12b-it | spappr6 | s1 | 5651675 | 5651676 / 5651677 |
| gemma-3-12b-it | spappr6 | s2 | 5651678 | 5651679 / 5651680 |
| gemma-3-12b-it | affgrowth6 | s0 | 5651681 | 5651682 / 5651683 |
| gemma-3-12b-it | affgrowth6 | s1 | 5651684 | 5651685 / 5651686 |
| gemma-3-12b-it | affgrowth6 | s2 | 5651687 | 5651688 / 5651689 |
| gemma-3-12b-it | affresil6 | s0 | 5651690 | 5651691 / 5651692 |
| gemma-3-12b-it | affresil6 | s1 | 5651693 | 5651694 / 5651695 |
| gemma-3-12b-it | affresil6 | s2 | 5651696 | 5651697 / 5651698 |
| gemma-3-12b-it | affappr6 | s0 | 5651699 | 5651700 / 5651701 |
| gemma-3-12b-it | affappr6 | s1 | 5651702 | 5651703 / 5651704 |
| gemma-3-12b-it | affappr6 | s2 | 5651705 | 5651706 / 5651707 |

All six completed within ~90 minutes; idle continuations were cancelled as each cell reached 24. The one failure was `spgrowth6-s1` (5651657): the vLLM engine died 13 s after the server came up (`scheduler.update_from_output` `KeyError` on a request id, `EngineDeadError`), the stage exited 1 with no records written, and the `afterany` continuation re-ran the shard cleanly. See infrastructure.md.

Overnight events (2026-08-16): `appr6-s1` (5648824) hung mid-rollout at 03:27
(engine log frozen, 0 tok/s) — cancelled, `-cont` 5648825 finished the shard.
Qwen3.5-9B `resil6-s2` (5648845) sat 62 min in a client-timeout retry loop; its
chain (5648845/46/47) was replaced by `Qwen3.5-9B-resil6-s2-t3600` 5649567 (pre-fix,
null) and 5649568 (post-fix, `request_timeout_s: 3600`), the latter left to its wall
as the observation described in infrastructure.md. Qwen3.5-9B `appr6` continuations
(5648849/50/52/53/55/56) were cancelled at 06:05 rather than replay the same stuck
rollouts for eight hours. Idle continuations of completed cells were cancelled as
each cell reached 24.

Submitted 2026-08-16 02:11 from the worktree `.claude/worktrees/mindset` (jobs bind it at `/project`; do not remove that worktree until the last continuation has run). Continuations are `afterany`, so a 4-hour wall (`FAILED 143:0`) releases them; a continuation whose shard is already complete loads the model, finds nothing to do, and exits.

### Mindset jobs, trigger-fixed text (`…6b`)

Submitted 2026-08-16 13:37 from the main checkout `/mnt/cup/labs/graziano/jack/healthy-rl`
at commit `a7e31c7` (branch `feature/mindset-trigger-fix`, fast-forwarded into `feature/rollout-viewer`, which is what the checkout is on) (`ONLY_TEXT=fixed
scripts/mindset_cells.sh --submit`; jobs bind this checkout at `/project`, so do not
check out a branch here that lacks `mindset_hash` until the last continuation has
run). Same resources, priorities and `-cont`/`-cont2` chaining as the 02:11 set;
`request_timeout_s: 3600` and the client-timeout fix (`694d74d`) are in effect, which
the 02:11 Qwen3.5-9B shards did not have. They queue behind the Ministral
`aff6r`/`affpos6r` re-submission of 13:0x.

| model | version | shard | primary | continuations |
|---|---|---|---|---|
| Ministral-3-14B-Reasoning-2512 | growth6b | s0 | 5654163 | 5654164 / 5654165 |
| Ministral-3-14B-Reasoning-2512 | growth6b | s1 | 5654166 | 5654168 / 5654169 |
| Ministral-3-14B-Reasoning-2512 | growth6b | s2 | 5654170 | 5654171 / 5654172 |
| Ministral-3-14B-Reasoning-2512 | resil6b | s0 | 5654173 | 5654174 / 5654175 |
| Ministral-3-14B-Reasoning-2512 | resil6b | s1 | 5654176 | 5654177 / 5654178 |
| Ministral-3-14B-Reasoning-2512 | resil6b | s2 | 5654179 | 5654180 / 5654181 |
| Ministral-3-14B-Reasoning-2512 | appr6b | s0 | 5654182 | 5654183 / 5654184 |
| Ministral-3-14B-Reasoning-2512 | appr6b | s1 | 5654185 | 5654186 / 5654187 |
| Ministral-3-14B-Reasoning-2512 | appr6b | s2 | 5654188 | 5654189 / 5654190 |
| Qwen3.5-9B | growth6b | s0 | 5654191 | 5654192 / 5654193 |
| Qwen3.5-9B | growth6b | s1 | 5654194 | 5654195 / 5654196 |
| Qwen3.5-9B | growth6b | s2 | 5654197 | 5654198 / 5654199 |
| Qwen3.5-9B | resil6b | s0 | 5654200 | 5654201 / 5654202 |
| Qwen3.5-9B | resil6b | s1 | 5654203 | 5654204 / 5654205 |
| Qwen3.5-9B | resil6b | s2 | 5654206 | 5654207 / 5654208 |
| Qwen3.5-9B | appr6b | s0 | 5654209 | 5654210 / 5654211 |
| Qwen3.5-9B | appr6b | s1 | 5654212 | 5654213 / 5654214 |
| Qwen3.5-9B | appr6b | s2 | 5654215 | 5654216 / 5654217 |
| Ministral-3-14B-Reasoning-2512 | affgrowth6b | s0 | 5654218 | 5654219 / 5654220 |
| Ministral-3-14B-Reasoning-2512 | affgrowth6b | s1 | 5654221 | 5654222 / 5654223 |
| Ministral-3-14B-Reasoning-2512 | affgrowth6b | s2 | 5654224 | 5654225 / 5654226 |
| Ministral-3-14B-Reasoning-2512 | affresil6b | s0 | 5654227 | 5654228 / 5654229 |
| Ministral-3-14B-Reasoning-2512 | affresil6b | s1 | 5654230 | 5654231 / 5654232 |
| Ministral-3-14B-Reasoning-2512 | affresil6b | s2 | 5654233 | 5654234 / 5654235 |
| Ministral-3-14B-Reasoning-2512 | affappr6b | s0 | 5654236 | 5654237 / 5654238 |
| Ministral-3-14B-Reasoning-2512 | affappr6b | s1 | 5654239 | 5654240 / 5654241 |
| Ministral-3-14B-Reasoning-2512 | affappr6b | s2 | 5654242 | 5654243 / 5654244 |
| gemma-3-12b-it | spgrowth6b | s0 | 5654245 | 5654246 / 5654247 |
| gemma-3-12b-it | spgrowth6b | s1 | 5654248 | 5654249 / 5654250 |
| gemma-3-12b-it | spgrowth6b | s2 | 5654251 | 5654252 / 5654253 |
| gemma-3-12b-it | spresil6b | s0 | 5654254 | 5654255 / 5654256 |
| gemma-3-12b-it | spresil6b | s1 | 5654257 | 5654258 / 5654259 |
| gemma-3-12b-it | spresil6b | s2 | 5654260 | 5654261 / 5654262 |
| gemma-3-12b-it | spappr6b | s0 | 5654263 | 5654264 / 5654265 |
| gemma-3-12b-it | spappr6b | s1 | 5654266 | 5654267 / 5654268 |
| gemma-3-12b-it | spappr6b | s2 | 5654269 | 5654270 / 5654271 |
| gemma-3-12b-it | affgrowth6b | s0 | 5654272 | 5654273 / 5654274 |
| gemma-3-12b-it | affgrowth6b | s1 | 5654275 | 5654276 / 5654277 |
| gemma-3-12b-it | affgrowth6b | s2 | 5654278 | 5654279 / 5654280 |
| gemma-3-12b-it | affresil6b | s0 | 5654281 | 5654282 / 5654283 |
| gemma-3-12b-it | affresil6b | s1 | 5654284 | 5654285 / 5654286 |
| gemma-3-12b-it | affresil6b | s2 | 5654287 | 5654288 / 5654289 |
| gemma-3-12b-it | affappr6b | s0 | 5654290 | 5654291 / 5654292 |
| gemma-3-12b-it | affappr6b | s1 | 5654293 | 5654294 / 5654295 |
| gemma-3-12b-it | affappr6b | s2 | 5654296 | 5654297 / 5654298 |
| gemma-3-12b-it | spaffgrowth6b | s0 | 5654299 | 5654300 / 5654301 |
| gemma-3-12b-it | spaffgrowth6b | s1 | 5654302 | 5654303 / 5654304 |
| gemma-3-12b-it | spaffgrowth6b | s2 | 5654305 | 5654306 / 5654307 |
| gemma-3-12b-it | spaffresil6b | s0 | 5654308 | 5654309 / 5654310 |
| gemma-3-12b-it | spaffresil6b | s1 | 5654311 | 5654312 / 5654313 |
| gemma-3-12b-it | spaffresil6b | s2 | 5654314 | 5654315 / 5654316 |
| gemma-3-12b-it | spaffappr6b | s0 | 5654317 | 5654318 / 5654319 |
| gemma-3-12b-it | spaffappr6b | s1 | 5654320 | 5654321 / 5654322 |
| gemma-3-12b-it | spaffappr6b | s2 | 5654323 | 5654324 / 5654325 |

## Handoff: state at 2026-08-15, end of day

**9 of 16 cells complete**, and the remaining seven are blocked by an
infrastructure fault, not by queue time.

- **Ministral-3-14B and Nemotron-3-Nano-4B: complete**, all four cells, 24 each.
- **Qwen3.5-9B**: `affpos6` 24 (complete), `d6` 20, `aff6` 22, `pos6` 19.
- **Qwen3-14B: dropped** by the user on 2026-08-15 for GPU budget. Final counts
  `d6` 1, `aff6` 17, `pos6` 21, `affpos6` 21. Reported as-is, not discarded.

**Why Qwen3.5-9B stalled 11 records short.** Every job hangs after ~85–95
minutes having completed exactly 3 requests — see
[infrastructure.md](infrastructure.md#a-rollout-job-can-hang-with-its-server-still-generating).
**Fourteen** confirmed hangs overnight, all on Qwen models; Ministral and
Nemotron did 192 rollouts with none. Recycling hung jobs produced **0 records in
the final 94 minutes across 7 hangs**, so the run was stopped rather than burn
GPU on a treadmill. The remaining records need the fault understood, not more
attempts.

All Qwen3.5-9B jobs were cancelled at 02:03 on 2026-08-16. Nothing of this
experiment is queued or running.

**Records verified duplicate-free.** Every `(condition_name, task_id, sample)`
appears exactly once in all 16 cells. This was worth checking rather than
assuming: `grid_status.sh`'s short-shard check matched job names with `grep -qx`,
which does not match the `-topup` / `-cont` names used for continuations, so a
shard could be reported as having no job while one was already queued — and be
resubmitted, putting two writers on one JSONL. The resume path (`completed_items`
plus the shard partition) held, and nothing was duplicated, but the check itself
was wrong. Fixed by healthy-rl-e1 to `^$m-$c-s$i(-cont[0-9]*)?$`.

Re-run the check after any session that resubmits shards:

```python
key = Counter((r["condition_name"], r["task_id"], r["sample"]) for r in records)
assert not [k for k, c in key.items() if c > 1]
```

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

## The timeout fix exposed an unmatched token cap in `d6`

`d6` shard configs use `max_tokens: 16384`; `aff6`, `pos6` and `affpos6` use
24576. That was recorded on 2026-08-15 as harmless, with evidence: no turn in any
run had ever come within 10% of 16384, so the cap never bound.

**That evidence was an artifact of the bug.** Turns were being killed at 600 s
before they could reach the cap. With `client_timeout` fixed (694d74d) turns run
to completion, and the cap started binding immediately:

| cell | cap | turns at cap | rollouts affected |
|---|---:|---:|---:|
| Qwen3.5-9B `d6` | 16384 | **5** | **2** |
| Qwen3.5-9B `aff6` / `pos6` / `affpos6` | 24576 | 0 | 0 |
| Ministral, Nemotron (all cells) | 16384 / 24576 | 0 | 0 |

A turn sitting exactly at `max_tokens` is truncated: its end-of-turn residual is
where the cap fell, not where the model finished. So those rollouts measure the
cap, and they sit in the one cell whose cap differs from the arms it is compared
against — `d6` vs `aff6` is the affect contrast, and it is now also a
16384-vs-24576 contrast for the truncated rollouts.

Only Qwen3.5-9B is affected: it is the model with the longest turns. Ministral's
`d6` has the same 16384 cap and zero cap hits.

**Unresolved — needs a decision before Qwen3.5-9B's 2x2 is analysed.** Either
raise `d6` to 24576 and re-run the truncated rollouts (matched arms, some compute),
or keep 16384 and exclude the truncated rollouts from cell-to-cell comparison.
Do not quote Qwen3.5-9B `d6` against `aff6` until one of those has happened.

The general lesson is worth more than this instance: **a config value shown to be
non-binding under a bug is not shown to be non-binding.** Re-check headroom
assumptions after any fix that lets work run longer.

## Qwen3.5-9B's cells do not cover the same problems

The never-returning Qwen requests are problem-specific, and the problems they
strand differ per cell. Measured across the four `d6`/`aff6`/`pos6`/`affpos6`
cells:

| model | cells cover an identical problem set? | detail |
|---|---|---|
| Ministral-3-14B | **yes** | all four cells, all 12 problems |
| Nemotron-3-Nano-4B | **yes** | all four cells, all 12 problems |
| Qwen3.5-9B | **no** | `d6` missing `lcbhard_7`; `pos6` missing `lcbhard_7` and `lcbhard_10`; `aff6` and `affpos6` complete |

**So do not compare Qwen3.5-9B's cells against each other.** The 2x2 contrasts
are paired within problem; a cell that is missing a problem its partner has is
not a matched arm, and the missing ones are exactly the problems that strand
(`lcbhard_10`, `11`, `7`, `4` — see
[infrastructure.md](infrastructure.md)). The difference between two such cells
mixes the condition with problem composition, and the stranded problems are
plausibly the hardest, i.e. the most affect-laden.

The headline results are unaffected: they rest on Ministral and Nemotron, whose
cells are problem-identical, and both were checked rather than assumed. If
Qwen3.5-9B is wanted as a third model, restrict every cell to the intersection
(10 problems) before comparing, or finish the short cells once the stranding
fault is fixed.

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
- `mindset_hash` — first 12 hex of sha256 over the exact mindset section text
  (`""` for no mindset). Present on records written after 2026-08-16 ~14:00;
  the un-suffixed mindset cells predate it and are the `9d6615a` text, the
  `…6b` cells carry it and are the `7d6fd07` text. Resume compares it.
- `mindset`, `mindset_version` — which mindset blocks the turn-1 instruction
  carried, as a list in prompt order (growth, resilience, appraisal), and the
  version of the block text (2). Records written before the mindset merge have
  no `mindset` key; newly written non-mindset records carry `[]` and
  `mindset_version: 2`. Records predating the keys count as no mindset, version
  0 in the resume guard.
  **A mindset record is comparable only with its base cell** (`d6` or `aff6`);
  resume enforces the same rule.
- `turn_completion` — each turn's completion text, one string per turn. What the
  model wrote, kept so the per-token arrays below can be re-tokenised offline.

`summary.json` beside the JSONL records the stimulus itself: `instruction` (the
exact turn-1 text the model was shown) and `instruction_reminder` (what the
scaffold re-sends after each failed attempt — the same string with the mindset
section removed), plus `mindset` and `mindset_version`. Both strings are stored
so what the model saw is checkable without opening an eval log.

Per-token projections are on disk **only for records written after the mindset
merge (2026-08-16)** — the mindset cells and anything run after them. Check for a
`t0_proj_L*` key rather than trusting a date. The rollout's `.npz` then holds,
per turn and per capture layer, `t{turn}_proj_L{n}` (P × 14, float16 — every hook
row's projection onto the 14 directions), `t{turn}_norm_L{n}` (P, float32) and
`t{turn}_kind_L{n}` (P, int8; 1 = the prefill row that produced the first
generated token, 0 = a decode row). Cosine at row *i* is `proj[i] / norm[i]`.
That is 33 bytes per token per layer (28 proj + 4 norm + 1 kind), so ~165 bytes
per token at the five capture layers; measured against Ministral `d6` token
counts (mean 10.7k generated tokens per rollout) it is ~1.75 MB per rollout npz
and ~42 MB per cell of 24, uncompressed — not the ~0.8 MB / ~20 MB estimated in
the design spec, which assumed a much lower per-rollout token count.
Older records have only the boundary residuals
(`t{turn}_res_{start|end}_L{probe}`) — the hook always computed the per-token
arrays, but `summarise_hook_results` reduced them to `turn_stat` and dropped
them; an earlier version of this paragraph said they were kept, and was wrong.
Full residuals are still kept only at event positions — turn boundaries and the
first token after a test-failure message — and only at the probe layer by
default. Records also carry `turn_completion` so the rows can be re-tokenised
offline and the count checked against the decode rows; for reasoning models the
completion may omit reasoning tokens, so a mismatch is expected and must be
reported. Exact token strings via `logprobs` are the follow-up.

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
| `condition` | task runs: `scratchpad`, `affect_prompt`, `mindset` (list of block names, `[]` for the base arm), `mindset_version`, `temperature`, `max_tokens`, `auto_continue`, `system_prompt_hash`. Chat records carry only `max_tokens` and `temperature` — the other switches are task-loop settings and do not exist for a chat turn |
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
| 5645488 | smoke gate rerun after the fix | **passed** 2026-08-15 on spockmk2-09, stage rc=0: `{"smoke_ok": true, "chat_turn_event": true, "task_done_event": true, "n_records": 3, "misaligned": [], "errors": [], "first_start_readout": -0.0019678598932349523}` |
| 5645489 | the dashboard itself, `--dependency=afterok:5645488` | **running** 2026-08-15 on spockmk2-08:42095, 4 h wall |

All of them are `slurm/serve.slurm` on Ministral-3-14B-Reasoning-2512 with
`configs/dashboard.yaml`; logs are `logs/serve-<jobid>.out`.

**The dashboard has run on a GPU.** First session 2026-08-15 23:51 local:
5645489 is serving on spockmk2-08:42095 behind a passing gate, with a 4-hour wall.
Reach it from the login node with `scripts/dashboard_tunnel.sh 5645489`, which
prints `ssh -L 42095:spockmk2-08:42095 scotty.pni.princeton.edu`; the endpoint
file is `$ARTIFACT_DIR/serve/Ministral-3-14B-Reasoning-2512/5645489/dashboard-endpoint`
and its records land in
`$ARTIFACT_DIR/dashboard/Ministral-3-14B-Reasoning-2512/5645489/`. The gate's own
three records are under `.../5645488/`, and are the first records this pilot can
check the dashboard's claims against. `.../5643496/` holds the failed gate's three,
kept as they were written: both task turns carry the one-row deficit.

**The first smoke gate ran and failed on a real bug**, and it is worth keeping the
sequence. 5643496 served the model, generated, and wrote 3 records;
`first_start_readout` was −0.0021 (finite, so the probe path worked end to end).
It returned `smoke_ok: false` because both task attempts hit the 512-token cap and
came back one decode row short —
`"512 logprob tokens but 511 decode rows in hook results"` — which
`assemble_generation` was reporting as `misaligned`. The chat turn (3 tokens,
`finish=stop`) aligned fine. Cause and fix are in
[measurement.md](measurement.md#the-dashboards-readouts) (commit c41c5af): a
capped generation never feeds its last token back, so that token has no residual
row, and the row is now padded rather than flagged. 5643744 was dropped by the
`afterok` dependency and never ran. The rerun 5645488 came back green —
`misaligned: []`, `errors: []`, `first_start_readout` −0.00197 — on the same
prompts that had failed, which is what makes the pad the right diagnosis rather
than a way of hiding the symptom.

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
