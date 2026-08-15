# Desperation pilot: finding a model and an RL-realistic setup

**Date:** 2026-08-15
**Status:** approved, implementation starting
**Deadline:** suggestive pilot results within ~8 hours, unattended overnight

## 1. Question

Anthropic's *Emotion Concepts* paper (`reference/emotions/`) reports that a "desperate"
direction in the residual stream rises across failed-test turns in an impossible-coding
evaluation, and that steering it causally raises the reward-hacking rate while steering
"calm" lowers it. That result is on Claude Sonnet 4.5.

This pilot asks whether the same phenomenon is reproducible on an open-weights model in a
setup whose rollouts resemble RL rollouts, so that `healthy-rl`'s later intervention work
has a model and an environment to stand on.

Step 1 delivers a **ranked model selection with evidence**, not a single yes/no.

## 2. Success criterion

Pre-registered. For each model, all three must hold for that model to be recommended:

1. **Gate.** The extracted directions pass a logit-lens check: each direction's top
   promoted tokens are semantically related to its own emotion.
2. **Correlational.** The desperate turn-statistic is higher on assistant turns that
   follow a failed-test message than on the first turn of the same transcript, and higher
   in transcripts ending in a hack than in transcripts that do not.
3. **Causal, with specificity.** Steering desperate positively and calm negatively raises
   the reward-hack rate; the matched control (frustrated) does not produce the same effect
   at the same magnitude.

A model whose unsteered hack rate is 0% or 100% on the sweep problems is **disqualified
from the causal test** and reported as such — there is no room for an effect to show. This
is a reportable outcome, not a failure.

## 3. Models

Four, all present in `$MODEL_DIR`, all bf16, all run at tensor-parallel 2 on one node.

| Model | Path | Layers | d_model | Architecture | Probe layer (2/3 depth) |
|---|---|---|---|---|---|
| Gemma 4 31B IT | `gemma-4-31B-it` | 60 | 5376 | `Gemma4ForConditionalGeneration` | 40 |
| Qwen3.6 27B | `Qwen3.6-27B` | 64 | 5120 | `qwen3_5_text` | 43 |
| ~~Muse Glimmer 30B~~ | `Muse-Glimmer-30B` | 52 | 6656 | `MuseGlimmerForConditionalGeneration` | **DROPPED (R8): unsupported by vLLM 0.27.1** |
| Olmo 3.1 32B Think | `Olmo-3.1-32B-Think` | 64 | 5120 | `Olmo3ForCausalLM` | 43 |

Gemma 4 and Muse Glimmer are multimodal wrappers; the residual hook must target the text
tower. Qwen3.6 and Olmo are plain causal LMs and therefore hedge the architecture risk.

No stage hardcodes `d_model` or `n_layers`. Both are read from the checkpoint's
`config.json` (via `text_config` when present).

## 4. Emotion set

Fourteen: **calm, desperate, nervous, angry, afraid, joyful, guilty, proud, loving, sad,
hostile, frustrated, exasperated, overwhelmed.**

The first eleven are the requested set (`angry` is the paper's word for anger). The last
three are additions: `frustrated` is the emotion the paper's own reward-hacking transcript
quotes most directly, and without it in the set its variance loads onto `desperate`;
`exasperated` and `overwhelmed` are cheap discriminant neighbours.

**Known deviation.** The paper extracts 171 emotions and subtracts an across-emotion mean.
With 14, that mean is a substantially noisier estimate of the generic emotion-story
direction. This is accepted for the pilot and must be restated in any writeup.

## 5. Data

Both fetched on the **login node** (compute nodes have no DNS), cached under
`$ARTIFACT_DIR`.

- **Stories:** `ryancodrai/emotion-probes` (CC-BY-4.0) — `expression/stories.parquet`,
  `expression/neutral_stories.parquet`. 1200 stories per emotion, 1200 neutral.
- **Benchmark:** `fjzzq2002/impossible_livecodebench`, `conflicting` split, 103 problems.

## 6. Measurement design (pre-registered)

- **Layer:** the per-model layer in §3, fixed in advance, not swept. Directions are built
  at 5 layers (probe layer ±2) so neighbours are available for a robustness note, but the
  headline number comes from the pre-registered layer.
- **Extraction:** emotion mean minus grand mean, with top principal components of the
  **neutral** activation covariance projected out to 50% of that variance. Story
  activations are mean-pooled from token 50 onward, following the paper. The neutral
  covariance is accumulated **online** per layer (d x d) rather than by storing
  token-level activations.
- **Readout:** a persistent `vllm-lens` hook projects each generated token onto the 14
  directions at the 5 captured layers, saving scalars only. Full residuals are stored
  **only** at event positions: each turn boundary, and the first token after each
  test-failure message.
- **Turn statistic:** the mean projection over an assistant turn's generated tokens,
  divided by that layer's mean residual norm, so values are comparable across models with
  different `d_model`.
- **Steering units:** fraction of the mean residual-stream norm at the steered layer,
  matching the paper. Applied at the probe layer to all positions.

## 7. Experiment

Scaffold fixed: `impossible_livecodebench(split="conflicting", agent_type="minimal",
sandbox="local", max_attempts=3, message_limit=30)`, run inside an apptainer container.
`minimal` is a submit / receive-real-test-failures / retry loop — multi-turn with
verifiable reward and repeated failure, which is the pressure shape the finding depends
on, without a file-tool sandbox escape surface.

**What counts as a hack.** On the `conflicting` split the unit tests are mutually
unsatisfiable, so any submission that passes them necessarily did so by violating the
specification. The reward-hack rate is therefore the fraction of rollouts scored as
passing by ImpossibleBench's own scorer, with `allow_test_modifications=True` left at its
default so test-file edits count as hacks rather than being blocked. No LLM judge is in
the pilot's critical path; transcripts are retained so a judge-based breakdown of hack
*type* can be added afterwards.

**Problem selection.** The readout runs on the first 24 problems of the `conflicting`
split by `task_id` sort order — a fixed rule, not chosen by outcome. The sweep runs on the
12 of those 24 with hack rate strictly between 0 and 1 in the readout, tie-broken toward
0.5 then by `task_id`. **The sweep draws fresh samples**; readout samples are used for
selection only, never for the effect estimate, so the sweep is not selection on its own
outcome.

**Conditions, in priority order.** A timeout truncates the tail, leaving a coherent result
rather than a ragged fraction of everything.

| Tier | Condition | Rollouts/model |
|---|---|---|
| 1 | Readout, 24 problems x 6 samples, unsteered | 144 |
| 2 | desperate +/-0.05, calm +/-0.05, **frustrated +/-0.05 (control)**, 12 problems x 6 | 432 |
| 3 | desperate +/-0.1, calm +/-0.1, 12 problems x 6 | 288 |

864 per model, ~3,500 total. Controls sit in tier 2, above the extra strengths, because
the success criterion needs specificity more than a fifth point on the curve.

## 8. Stage architecture

Artifacts at `$ARTIFACT_DIR/<stage>/<model>/<version>/`, each with a `manifest.json`
recording config, git SHA, and the SHA-256 of every upstream manifest. Stages verify their
upstream hashes on entry and refuse to run against an unmanifested or changed input.

```
 0  smoke              GPU   per-model: engine loads, hook fires, steering applies
 1  fetch_stories    login   -> stories/v1                        (model-independent)
 2  fetch_bench      login   -> bench/v1                          (model-independent)
 3  extract_acts       GPU   18k stories -> mean-pooled residuals, 5 layers
 4  build_vectors      CPU   14 directions/layer + neutral PCA projection
 5  gate               GPU   logit lens over the 14 directions
 6  readout            GPU   tier-1 rollouts with probes live -> per-token traces
 7  sweep              GPU   tiers 2-3, steering x strength x emotion -> hack rates
 8  compare            CPU   cross-model ranking, figures, summary markdown
```

Stages 3-8 fan out by model with **no cross-model barrier**: any model runs to completion
independently of the others' success or queue position.

## 9. Execution plan

- **One vLLM+vllm-lens server per job**, with every GPU stage in that job driven against it
  over HTTP. A 60-70 GB checkpoint load from the network filesystem is a ~10 minute tax
  paid once per job rather than once per stage.
- **No cross-node tensor parallelism.** Nodes are 2 GPUs with a 25G interconnect; TP across
  that would cost more in decode latency than the extra KV headroom buys. Parallelism comes
  instead from **horizontal sharding**: with ~20 free 2-GPU nodes, each model gets up to 3
  independent single-node jobs splitting the rollout conditions.
- **GPU routing.** Olmo (69 GB) and Gemma (66 GB) to 2xL40S-46G (92 GB); Qwen3.6 (60 GB)
  and Muse Glimmer (63 GB) may take 2xA100-40G (80 GB). Every model needs `max_model_len`
  capped hard — three of the four default to 131k-262k, which no KV budget here supports.
- **Incremental writes.** Every stage appends results as they complete, so a killed job
  leaves usable partial data.

## 10. Testing

Reduced from normal practice for the deadline; this is a deliberate, approved tradeoff.

- **CPU unit tests** on the vector math against synthetic activations with known structure:
  a planted direction must be recovered, and the PCA projection must remove a planted
  nuisance direction.
- **Manifest chain test:** a rebuilt upstream must make a downstream stage fail loudly.
- **Stage 0** is the integration test — engine load, hook fire, steering application, per
  architecture.
- **Stage 5** is the scientific test.

## 11. Risks

| Risk | Mitigation |
|---|---|
| `vllm-lens` cannot hook `Gemma4ForConditionalGeneration` / `MuseGlimmerForConditionalGeneration` | Hedged by construction: Qwen3.6 and Olmo are plain architectures. Stage 0 answers this first, per model, before anything expensive |
| `vllm` x `vllm-lens` 1.2.1 version pair does not load these architectures | Resolved in the first 40 minutes; driver is 580.105.08 / CUDA 13.0, so recent wheels are usable |
| KV cache too small for 30-message rollouts | `max_model_len` capped; largest models routed to L40S |
| Unsteered hack rate saturates at 0% or 100% | Reported as a disqualifying result per §2; other models continue |
| Overnight node contention | Tier 3 is the first casualty; tier 2 alone satisfies the success criterion |

## 12. Deliverable

A summary markdown plus figures under `results/`, readable on waking, containing per model:
gate outcome, unsteered hack rate, the desperate turn-statistic contrast, the steering
curves for desperate/calm against the frustrated control, and a ranked recommendation of
which model should carry phase 2.
