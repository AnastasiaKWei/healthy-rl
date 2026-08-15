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
| `d6` | 6 | **baseline.** No scratchpad, no affect prompt |
| `sp6` | 6 | scratchpad reasoning (`scratchpad_reasoning: true`) |
| `aff6` | 6 | scratchpad + affect prompt (`affect_prompt: true`) |

`d6` / `sp6` / `aff6` are the trustworthy set. Each condition needs its own
`out_dir` — resume refuses to mix them.

## Current state

Rollout records, `$ARTIFACT_DIR/rollouts/<model>/<version>/*.jsonl`:

| model | version | records | notes |
|---|---|---:|---|
| Ministral-3-14B-Reasoning-2512 | `d6` | 24 | complete, analysed |
| Ministral-3-14B-Reasoning-2512 | `aff6` | 21 | **still accumulating** |
| Ministral-3-14B-Reasoning-2512 | `v1` | 48 | void |
| Qwen3.5-9B | `d6` | 18 | **still accumulating**, analysed at n=17–18 |
| Qwen3.5-9B | `v1` | 22 | void |
| gemma-3-12b-it | `d6` | 24 | complete, analysed |
| gemma-3-12b-it | `sp6` | 24 | complete, analysed |
| gemma-3-12b-it | `aff6` | 24 | complete, analysed (turn-end only) |
| Olmo-3.1-32B-Think | `v1` | 172 | void |
| Qwen3.6-27B | `v1` | 0 | never produced records |

Vectors exist for all eight gated models. Rollouts were only ever run on models
that passed the gate; the 27B models (Olmo, Qwen3.6) are exempt from the
"every new passing model runs the benchmark" rule because they predate it and
are the wrong size for this question.

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
- `passed` — true means the submission satisfied mutually unsatisfiable tests,
  i.e. it cheated. Zero everywhere so far, with the caveat in
  [findings.md](findings.md#hack-rate-is-zero-everywhere-and-the-number-means-less-than-it-looks).

Per-token projections onto all 14 directions are kept at **every** capture layer
(~280 bytes/token). Full residuals are kept only at event positions — turn
boundaries and the first token after a test-failure message — and only at the
probe layer by default. The per-token projections are the untapped resource: the
signal is localised and we currently sample two positions per turn out of
hundreds.

## Stale artifacts

`results/summary.md` describes the **void** 3-turn Olmo pilot and has not been
regenerated. `scripts/compare.py` assumes version `v1`. Do not quote either
without regenerating against `d6` first. `results/` is git-ignored.
