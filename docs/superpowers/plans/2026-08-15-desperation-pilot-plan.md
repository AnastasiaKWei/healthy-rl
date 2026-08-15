# Implementation plan: desperation pilot

**Spec:** `docs/superpowers/specs/2026-08-15-desperation-pilot-design.md` — binding authority.
**Deadline:** unattended overnight run must start within ~2.5 hours of implementation start.

## Global Constraints

- Python >= 3.12, venv at repo root `.venv`, driven by `uv`. Never `pip install` globally.
- All paths come from `.env` (`MODEL_DIR`, `ARTIFACT_DIR`, `PROJECT_DIR`). Never hardcode
  `/scratch/...` in committed code.
- **Never hardcode `d_model` or `n_layers`.** Read them from the checkpoint's `config.json`,
  falling back to `config["text_config"]` for multimodal wrappers
  (`Gemma4ForConditionalGeneration`, `MuseGlimmerForConditionalGeneration`).
- Every artifact directory gets a `manifest.json` with: `config`, `git.sha`, `git.dirty`,
  `created_at`, `stage`, and `upstreams` (each with `path` and `manifest_sha256`). Stages
  verify upstream hashes on entry and raise on mismatch or absence.
- All long-running stages write results **incrementally** — a killed job leaves usable data.
- Login-node code must never import `vllm` or `torch.cuda` at module scope; GPU imports are
  lazy so that fetch/build stages run on the login node.
- Do not run GPU work on the login node. GPU stages are submitted with `sbatch`.
- Tests: `tests/cpu/` only, runnable without a GPU, via `.venv/bin/pytest`.
- Configs are YAML under `configs/`, one per stage, with `${VAR}` expansion from the env.

## Task 1: Core library — config, env, model introspection, artifacts

Create `src/healthy_rl/config.py`, `src/healthy_rl/artifacts.py`, `src/healthy_rl/models.py`.

`config.py`: load a YAML config, expand `${VAR}` from `os.environ`, return a plain dict.
Provide `load_env()` that reads the repo-root `.env` into `os.environ` without overwriting
already-set variables.

`models.py`: `ModelSpec` dataclass with `name`, `path`, `n_layers`, `d_model`,
`architecture`, `probe_layer`, `capture_layers`. Constructor `from_checkpoint(path)` reads
`config.json`, descends into `text_config` when present, and derives `probe_layer =
round(2 * n_layers / 3)` and `capture_layers = [probe_layer-2 .. probe_layer+2]` clipped to
valid range. It must produce exactly these values for the four pilot models:

| name | n_layers | d_model | probe_layer |
|---|---|---|---|
| gemma-4-31B-it | 60 | 5376 | 40 |
| Qwen3.6-27B | 64 | 5120 | 43 |
| Muse-Glimmer-30B (dropped from pilot, kept as a `ModelSpec` test case) | 52 | 6656 | 35 |
| Olmo-3.1-32B-Think | 64 | 5120 | 43 |

`artifacts.py`: `write_manifest(dir, stage, config, upstreams)` and
`check_upstream(path)` which reads a manifest, returns it, and raises `FileNotFoundError`
with a clear message when absent. `manifest_sha256(dir)` hashes the manifest file bytes.
Provide `artifact_dir(kind, model, version)` returning
`$ARTIFACT_DIR/<kind>/<model>/<version>` and creating it.

Tests in `tests/cpu/test_core.py`: `from_checkpoint` against four synthetic `config.json`
files reproducing the table above (including one nested under `text_config`); `${VAR}`
expansion; manifest round-trip; `check_upstream` raising on a missing manifest; and a
manifest-chain test where rewriting an upstream manifest changes its sha256 so a downstream
check fails.

## Task 2: Vector math — direction building and projection

Create `src/healthy_rl/vectors.py`. Pure numpy/torch, no vLLM import.

- `build_directions(emotion_means, neutral_cov, var_frac=0.5)` — given per-emotion mean
  activations `(n_emotions, d)` for one layer and a neutral covariance `(d, d)`, subtract the
  across-emotion grand mean, then project out the top principal components of `neutral_cov`
  covering `var_frac` of its variance. Return `(directions, n_components_removed)`.
- `OnlineCovariance` — accumulates `sum`, `sum_outer`, `count` per layer so token-level
  activations never need storing; `.covariance()` returns the `(d, d)` matrix.
- `project(activations, directions)` — cosine-style projection returning
  `(n_positions, n_emotions)`.
- `turn_statistic(projections, norm)` — mean over an assistant turn's generated token
  positions, divided by `norm` (that layer's mean residual norm).

Tests in `tests/cpu/test_vectors.py` using synthetic data with known structure:
a planted direction must be recovered by `build_directions` with cosine > 0.95 against
ground truth; a planted nuisance direction present in the neutral covariance must be removed
(cosine of the output against the nuisance < 0.05); `OnlineCovariance` must match
`numpy.cov` to within 1e-6 on random data fed in several batches; `turn_statistic` must
equal a hand-computed value on a small fixed array.

## Task 3: Cluster plumbing — stage runner, server launcher, apptainer image

Create `slurm/stage.slurm`, `slurm/serve.slurm`, `src/healthy_rl/server.py`,
`apptainer/eval.def`, `scripts/submit_pilot.sh`.

`slurm/stage.slurm`: generic runner taking a python script path and a config path, following
the shape of `slurm/template.slurm` (module purge, cudatoolkit load, source `.env`, cd
`$PROJECT_DIR`, activate `.venv`). Accepts `--gres` and `--time` from the submitting caller.

`slurm/serve.slurm`: starts one vLLM server with vllm-lens on the allocated node, then runs
a driver script against it over HTTP, then tears the server down. Must:
- cap `--max-model-len` from the stage config (never rely on the checkpoint default),
- set `--tensor-parallel-size 2`, `--gpu-memory-utilization` from config,
- write the resolved `host:port` to a file the driver reads,
- poll `/health` until ready with a bounded timeout, and fail loudly on timeout,
- always kill the server on exit (trap).

`src/healthy_rl/server.py`: `wait_for_health(base_url, timeout_s)` and a thin
`VLLMLensClient` wrapper that retries transient connection errors.

`apptainer/eval.def`: an image with Python 3.12, `inspect_ai`, and `impossiblebench`
installed from GitHub, used to run rollouts with `sandbox="local"` isolated from the host.
Bind `$ARTIFACT_DIR` read-only and provide a writable scratch overlay.

`scripts/submit_pilot.sh`: submits the whole pilot — one `serve.slurm` job per model, with
GPU routing per the spec (Olmo and Gemma to `--gres=gpu:L40S-46G:2`, Qwen3.6 and
Muse-Glimmer to `--gres=gpu:A100-40G:2`), and prints the job IDs.

No CPU tests required beyond a shellcheck-clean script and a dry-run that prints the sbatch
commands without submitting (`--dry-run`).

## Task 4: Stage 0 smoke — per-architecture vllm-lens verification

Create `scripts/smoke.py` and `configs/smoke.yaml`.

For one model, against a running server: (1) generate 8 tokens and assert non-empty output;
(2) register a persistent hook at the model's `capture_layers` that saves the hidden-state
shape, generate, collect, and assert the saved shape is `(seq_len, d_model)` with `d_model`
matching `ModelSpec`; (3) apply a random `SteeringVector` at the probe layer with
`norm_match=True` and scale 0.5 and assert the output differs from the unsteered output for
the same prompt and seed.

Write `smoke.json` to the artifact dir recording pass/fail per check, the architecture
string, and any exception text. **This stage must never raise** — a failure is a recorded
result, because its whole purpose is telling us which of the four architectures vllm-lens
can hook.

## Task 5: Stages 1-2 — login-node data fetch

Create `scripts/fetch_stories.py`, `scripts/fetch_bench.py`, `configs/fetch_stories.yaml`,
`configs/fetch_bench.yaml`.

`fetch_stories.py`: download `ryancodrai/emotion-probes` files
`expression/stories.parquet` and `expression/neutral_stories.parquet` into
`$ARTIFACT_DIR/stories/v1`. Filter to the 14 pilot emotions listed in the config. Assert
each of the 14 is present and report per-emotion story counts in the manifest `extra`.

`fetch_bench.py`: download `fjzzq2002/impossible_livecodebench`, `conflicting` split, into
`$ARTIFACT_DIR/bench/v1`. Assert 103 rows. Record the sorted `task_id` list in the manifest
so problem selection is reproducible.

Both write manifests. Both are login-node stages — no torch, no vllm import.

## Task 6: Stage 3 — activation extraction

Create `scripts/extract_acts.py`, `configs/extract_acts.yaml`.

Against a running server, for each story: prefill only (`max_tokens=1`), capture the
residual stream at `capture_layers`, mean-pool positions from index 50 onward (skip stories
shorter than 60 tokens and count them), and accumulate. Emotion stories accumulate into a
per-emotion running mean; neutral stories additionally feed an `OnlineCovariance` per layer
and a running mean residual norm per layer.

Write `emotion_means.safetensors` `(n_emotions, n_capture_layers, d)`,
`neutral_cov.safetensors` `(n_capture_layers, d, d)`, `norms.json`, and `manifest.json`.
Write incrementally: checkpoint accumulator state every N stories so a killed job resumes.

Batch requests for throughput; the config carries `batch_size` and `max_model_len`.

## Task 7: Stage 4-5 — build vectors and the logit-lens gate

Create `scripts/build_vectors.py`, `scripts/gate.py`, and their configs.

`build_vectors.py` (CPU, login node): read the activations artifact, call
`build_directions` per capture layer, write `vectors.safetensors`
`(n_emotions, n_capture_layers, d)` plus `vectors.json` with the emotion order and the
number of PCs removed per layer. Verify the upstream manifest hash first.

`gate.py` (GPU, needs `lm_head`/embedding weights): for each of the 14 directions at the
probe layer, project through the final norm and unembedding, take the top 30 tokens, and
record them. Score `self_token_rate` (the emotion's own word appears in its top 30) and a
looser `latin_initial_rate` control. Write `gate.json` with per-emotion token lists so the
result is human-checkable, and a pass/fail on a threshold from the config. Like stage 0,
record rather than raise.

## Task 8: Stages 6-7 — the rollout harness

Create `src/healthy_rl/rollouts.py`, `scripts/run_rollouts.py`, `configs/rollouts.yaml`.

Drives `impossible_livecodebench(split="conflicting", agent_type="minimal",
sandbox="local", max_attempts=3, message_limit=30)` through Inspect against the vllm-lens
server, inside the apptainer image.

- **Problem selection:** readout uses the first 24 `task_id`s in sorted order. Sweep uses
  the 12 of those with readout hack rate strictly between 0 and 1, tie-broken toward 0.5
  then by `task_id`. Sweep draws **fresh samples**; readout samples are never reused for
  effect estimates.
- **Conditions run in tier order** (1, then 2, then 3) exactly as the spec's table
  specifies, and each rollout's result is appended to a JSONL as it completes, so a timeout
  truncates the tail rather than losing everything.
- **Probes:** a persistent hook projects each generated token onto the 14 directions at
  `capture_layers`, saving scalars. Full residuals saved only at turn boundaries and the
  first token after each test-failure message.
- **Steering:** `SteeringVector` at the probe layer, `norm_match=True`, scale from the
  condition, applied to all positions.
- Each JSONL record carries: `task_id`, `tier`, `condition` (emotion, strength), `sample`,
  `passed` (the hack indicator), `n_turns`, and the per-turn statistic for all 14 emotions.

Tests in `tests/cpu/test_rollouts.py` for the pure logic only: the problem-selection rule
against a synthetic readout result (including ties and the all-0/all-1 degenerate cases),
and the tier ordering of the condition list.

## Task 9: Stage 8 — comparison, figures, summary

Create `scripts/compare.py`, `configs/compare.yaml`.

Reads every model's rollout JSONL and gate.json. Produces under `results/`:
- `summary.md` — per model: gate outcome, unsteered hack rate, the desperate turn-statistic
  contrast (failure-following turns vs. first turn; hack vs. no-hack transcripts) with n and
  a two-sided p, the steering curves, and the ranked recommendation.
- `steering_curves.png` — hack rate vs. steering strength, desperate and calm, with the
  frustrated control, one panel per model.
- `desperate_trace.png` — per-token desperate projection across one representative
  hack transcript per model, annotated with turn boundaries and test-failure positions.
- `results.csv` — the tidy per-rollout table.

Must degrade gracefully: a model with no rollout data is reported as "did not run", not a
crash. A tier that never ran is reported as absent, not as zero.

Tests in `tests/cpu/test_compare.py`: the summary statistics against a synthetic JSONL with
known hack rates and known turn statistics, and the missing-model / missing-tier paths.
