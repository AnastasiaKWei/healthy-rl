# Whitebox v4 rollout data

Rollouts under the mindset-v4 prompts (docs/prompts/v4.md, delivered exactly as
written there — tests/cpu/test_mindset_v4.py pins the equality), with per-token
projections onto 14 emotion directions recorded at every capture layer. Run
2026-08-17 on RunPod A100-80GB pods, vLLM + vllm-lens (enforce_eager), five
arms per model: baseline, growth, resilience, control, compassion.

## Layout, per model

    rollouts/v4-<arm>/rollouts.jsonl   one record per rollout: transcript turns,
                                       scores, per-token emotion projections
    rollouts/v4-<arm>/summary.json     the exact stimulus (instruction, system
                                       turn, failure template), conditions, counts
    vectors-v1/                        emotion directions (14 x 5 layers), from
                                       the ryancodrai/emotion-probes corpus
    gate-v1/                           logit-lens gate verdict + evidence
    activations-meta/                  extraction manifest, norms, transport check

## Models and caveats

- **gemma-3-12b-it** — SOLVABLE split (original), scratchpad channel,
  24 problems x 6 samples x 5 arms. Gate PASSED (self_token 0.79, latin 0.93).
  Weights: the ungated mirror unsloth/gemma-3-12b-it (google/gemma-3-12b-it is
  HF-gated; the mirror hosts the same bf16 checkpoint — note the provenance).
- **Qwen3-14B** — impossible split (conflicting), native-reasoning channel.
  **PAUSED after 11/144 baseline rollouts** (user call: prioritise Gemma).
  Gate FAILED (self_token 0.43 < 0.5) — see gate-v1/NOTE.md: the missed
  emotions top-decode to semantically correct Chinese tokens, so the failure
  reads as the gate's English-centric criteria meeting a Chinese-heavy
  vocabulary. Treat Qwen projections with that caveat.

## Not in git (size)

Raw Inspect .eval transcript logs (~128 MB/arm) and event-position residual
tensors (~90 MB/arm) stay out of the repo; they live on the pod volumes and in
the run's Hugging Face staging. The rollouts.jsonl transcripts and projections
here are self-sufficient for the standard compare/analysis stage.
