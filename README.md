# healthy-rl

**Research question:** Can we build a psychologically healthy RL training setup by
including psychology-inspired interventions in the training loop?

Apart Research model welfare sprint.

## Framing

The interventions (inoculation prompting, confession, and other psychology-derived
methods) are usually motivated as *behavioral* mitigations — they change what the
model does. This project asks whether they also change what training does to the
model's internal affective representations, and whether a training setup can be
designed to be healthy by construction rather than patched after the fact.

This is a constructive question, not a diagnostic one. The measurement is the
instrument; the deliverable is a claim about training setup design.

## Approach

- **Interventions:** inoculation prompting, confession, others TBD.
- **Measurement:** emotion vectors, extracted with the same method as the
  emotion-vector literature (contrastive activation differences). *TODO: pin the
  exact reference and extraction details.*
- **Models:** small open models — Qwen and Gemma. Empirical check first before
  committing to a family.
- **Runs:** small-scale training we own end-to-end, plus deeper probing analysis on
  the resulting checkpoints.

## Open questions

- What does "psychologically healthy" mean operationally, and what would falsify a
  claim that a setup is healthier? This needs an answer before the writeup.
- Do the interventions change affect representations, or only their behavioral
  expression? These come apart, and that gap is the interesting result either way.
- Baseline: what does an unmitigated RL run do to the same vectors?

## Status

Scaffolding only. No experiments run yet.
