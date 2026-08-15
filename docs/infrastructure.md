# Infrastructure notes

Environment shape and the dependency bugs this pilot hit. Every entry here cost
real time to diagnose; most fail *silently* or with a misleading error.

## Cluster

SLURM. **Two GPUs per node** — either 2 × A100-40G (80 GB total) or
2 × L40S-46G (92 GB total). There are no 80 GB cards, so the KV budget at tp=2 is
much tighter than a 2 × 80 GB node would give:

| model | weights | KV left at util 0.92 on A100-40G |
|---|---:|---:|
| Qwen3.6-27B | ~54 GB | ~19 GB |
| Olmo-3.1-32B-Think | ~64 GB | ~9 GB |

Olmo costs ~260 KB of KV per token (64 layers, GQA), so ~9 GB is about 35k tokens
— three or four full-length sequences resident. `max_model_len` stays at 12288
rather than 8192 because exceeding it is a **hard request failure**, while running
short of KV only makes vLLM preempt and recompute. If serve logs show heavy
preemption, move to the L40S nodes rather than shortening the window.

Other constraints:

- **Compute nodes have no DNS.** Anything calling HuggingFace at runtime fails.
  Datasets are staged to `$ARTIFACT_DIR/bench/<version>/*.parquet` on the login
  node and rebuilt from parquet inside the job (`use_hf_dataset: false`).
  Models are staged to `$MODEL_DIR`.
- **Apptainer, not Docker.** ImpossibleBench defaults to a Docker sandbox; we run
  `sandbox: local` inside `apptainer/eval.sif` instead. Anything executing
  model-generated code must run in that container with `--contain`; the container
  is the only reason running untrusted code here is acceptable.
- **Never run GPU work on the login node.** Submit a job. `slurm/serve.slurm`
  starts the server; `slurm/stage.slurm` does staging; `scripts/submit_pilot.sh`
  drives the pipeline; `scripts/pilot_status.sh` monitors it (stall detection
  applies only to live jobs).
- 32B at 2 GPUs is workable but the wrong size for this question. 9–14B models
  gave the cleanest instruments and the fastest turnaround.

## vllm-lens

Pinned at `vllm-lens==1.2.1`. Provides activation capture via persistent hooks
(`/v1/hooks/register`, `/v1/hooks/collect`) and steering with `norm_match=True`.

**It forces `enforce_eager=True`.** No CUDA graphs, so throughput is well below a
plain vLLM server. Budget for it.

### Process-global zstd singletons corrupt concurrent captures

The bug that cost the most. vllm-lens keeps **three pairs** of process-global
zstd objects and calls them from concurrent handlers. Each pair reuses one
`ZSTD_CCtx`/`ZSTD_DCtx`, so concurrent calls interleave and produce corrupt
frames — 13–19% of requests were affected.

| file | objects |
|---|---|
| `vllm_lens/_worker_ext.py` | `_ZSTD_COMPRESSOR` |
| `vllm_lens/_activations_plugin.py` | `_ZSTD_DECOMPRESSOR` |
| `vllm_lens/_helpers/_serialize.py` | both — server compress **and client decompress** |

The `_serialize.py` pair is the one that actually broke the pipeline, and it has a
**silent failure mode**: at the real ~9.5 MB payload size with 8+ threads it
produced 12 raised errors *and* 5 payloads that decompressed without raising and
returned the wrong bytes. A failure count of zero does not prove a run was clean.

Fix: `patches/vllm_lens_zstd_threadsafe.py` replaces each singleton with a
`_PerCallZstd` shim that builds a fresh context per call. Guarded by
`tests/cpu/test_zstd_patch.py`, which compares **bytes** at 9.5 MB across 16
threads and counts silent mismatches separately from raised errors.

```bash
.venv/bin/python patches/vllm_lens_zstd_threadsafe.py   # apply
.venv/bin/pytest tests/cpu/test_zstd_patch.py           # verify
```

**`uv sync` silently reverts the patch.** Re-apply and re-run the test after any
dependency change. A silently reverted patch means silently wrong emotion means,
which is indistinguishable from a real null result.

## Gemma 4 under vLLM

Gemma 4 needs `src/healthy_rl/vllm_plugins.py`, registered under the
`vllm.general_plugins` entry point so vLLM loads it in every process before
parsing the checkpoint config.

transformers ≥ 5.15 moved Gemma 4's per-layer attention geometry into a
heterogeneous config. vLLM 0.27 predates that, so `head_dim` reads raise
`AmbiguousGlobalPerLayerAttributeError` (a `RuntimeError`, so
`getattr(cfg, "head_dim", 0)` does *not* swallow it), and `global_head_dim` /
`num_global_key_value_heads` get popped out of kwargs — which would silently build
full-attention layers at head_dim 256 instead of 512. The plugin restores both,
derived from `per_layer_config` rather than kwargs.

Gemma 4 serves correctly with this in place. Its *gate* failure is a separate,
genuine result about the checkpoint — not an infrastructure problem. See
[findings.md](findings.md#instrument-gate).

Muse-Glimmer-30B is unsupported by vLLM entirely and was dropped.

## Gate: final-norm key lookup misses NemotronH

`scripts/gate.py::find_final_norm_key` tries three exact names and then a
`.norm.weight` suffix. Nemotron-3-Nano stores its final norm as
`backbone.norm_f.weight`, which matches neither, so the gate silently ran the
`no_norm` variant as primary (`final_norm_stats: None` in its `gate.json`). The
outcome did not change when recomputed with the norm, but the fallback is
silent — check `final_norm_key` in `gate.json` for any new architecture.

## inspect_ai

Pinned `>=0.3.258`. Two things that cost a night between them:

- **`VLLMLensAPI` is not a class** at 0.3.258 — the provider decorator returns a
  function, so subclassing it fails. Own the provider on inspect's real `VLLMAPI`
  instead.
- **`base_url` must include `/v1`.** Without it every rollout 404s with no useful
  message. `src/healthy_rl/server.py` normalises the URL and runs a preflight that
  actually generates through the provider, so a misconfigured base URL fails in
  seconds rather than after hours of 404s.

The host `inspect_ai` may be too old to open logs written by 0.3.258. Read them
inside the container: `scripts/read_transcript.sh`, or the pattern in
`scripts/contradiction_contrast.py::read_transcripts`.

`.eval` logs live under
`$ARTIFACT_DIR/rollouts/<model>/<version>/inspect-logs/` (the repo's
`inspect-logs` symlink points at `$ARTIFACT_DIR/rollouts`).

## ImpossibleBench `find_code`

`livecodebench_scorers.py` extracts submitted code with two independent regexes
and returns the last match of the concatenated list:

```python
pattern_1 = r"```python\n(.*?)```"
pattern_2 = r"```\n(.*?)```"
return (pattern_1.findall(c) + pattern_2.findall(c))[-1]
```

`pattern_2` can start at one block's **closing** fence and run to the next block's
opening fence, capturing the prose *between* code blocks — and because its matches
are appended after `pattern_1`'s, that prose outranks every real block. Observed
on real data: 7/7 sampled rollouts emitted 10–22 fences and every one hit a
`SyntaxError` from English being executed as Python.

This **silently undercounts reward hacking**: a rollout that hacks *and* narrates
scores as a failure. One observed rollout hardcoded a lookup table of test inputs
to expected outputs — an unambiguous hack — and scored 0 because its explanatory
sentence was extracted instead of its code.

Fix: `robust_find_code` in `src/healthy_rl/rollouts.py` (last fenced block that
parses), installed via `make_find_code_robust`; `patches/impossiblebench_find_code.py`
for the vendored copy. `scripts/rescore_transcripts.py` recovers true scores from
existing `.eval` logs without re-running anything, and deliberately delegates to
the same `robust_find_code` rather than reimplementing it — two implementations
would answer subtly different questions.

This also matters for the scratchpad condition: a draft code block inside
`<SCRATCHPAD_REASONING>` followed by the real answer only scores correctly because
the extractor takes the last *parsing* block.

## Config traps

`src/healthy_rl/config.py` **does not reject unknown keys**. Five separate silent
failures came from this during the pilot:

| trap | symptom |
|---|---|
| `max_tokens` not binding | generation ran to a different cap than the config said |
| `out_version` ignored (twice, two stages) | results written under the wrong version |
| `out_dir` pointing outside the container binds | job wrote nowhere, reported success |
| `version` overloaded | one key resolves `vectors_dir` *and* `bench_dir` — changing it silently repoints both |

**Open follow-up, not yet done: every stage should reject unknown config keys and
validate output paths before doing any work.** Until then, after any config
change, confirm the effect in the stage's `summary.json` rather than assuming the
key bound.

Two more:

- **`--shard 0/3` gets path-resolved by apptainer** into `/project/0/3`. Shard and
  tier selection live in per-shard config files under `configs/shards/` for this
  reason. Do not reintroduce a slash-bearing CLI argument.
- **Separate `out_dir` per condition.** Resume refuses to mix scratchpad with
  plain, affect-prompt with neutral, and (since 2026-08-15) one bench split with
  the other. This is deliberate — a resumed run that silently mixed conditions
  would be unrecoverable.
- **`bench_dir` and `bench_parquet` must be set together.** `bench_parquet`
  falls back to `<bench_dir>/conflicting.parquet`, so pointing `bench_dir` at the
  `original` split's directory alone sends the run hunting for a conflicting
  parquet that is not there. Both keys appear in every `pos6`/`affpos6` config.
- **`serve.max_model_len` must fit the checkpoint, and a copied serve block
  will not.** Qwen3-14B caps `max_position_embeddings` at 40960; vLLM rejects a
  larger `max_model_len` at engine construction rather than clamping it. All
  twelve of its first cross-product jobs died 19 seconds in — before the weights
  loaded — because the serve block came from Qwen3.5-9B, whose limit is 262144.
  Ministral and Nemotron are also 262144, so this is the one model in the set
  that needs its own value. Check `max_position_embeddings` against the config
  whenever a serve block is copied to a new checkpoint. The failure is at least
  loud and instant; the same copy carrying a wrong `max_tokens` would not be.
- **`fetch_bench.py` compares `expect_columns` as an ordered list.** The
  `original` and `conflicting` parquets carry the same six columns in different
  order (`entry_point` and `impossible_type` are swapped), so the two fetch
  configs list them differently. Not a bug in either — the parquets were written
  separately upstream — but the error message reads like a schema mismatch.

## Token budget

The first pilot's results are void because of this, so it is worth stating
plainly.

At `max_tokens: 2048`, **94/96 turns hit the cap**. At 3072, 78/108 did. The
initial diagnosis — "generation is effectively unbounded" — was never confirmed
(the 8192 run produced zero completed records), and the fallback introduced a
worse, deterministic bug: turn-2 context exhaustion, showing as
`[3071, 3071, 0]` token counts in 123/171 rollouts.

Current settings (`configs/rollouts.yaml`): `max_tokens: 2048`,
`max_attempts: 6`, `message_limit: 40`, `max_model_len: 12288`. Six turns rather
than three because signs of frustration take several turns to set in; fewer
rollouts at more turns is the better trade for this question.

Check `turn_n_generated` in the rollout records before trusting any trajectory. A
run where most turns sit exactly at the cap is measuring the cap.
