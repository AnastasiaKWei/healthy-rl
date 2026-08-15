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

## Step 0: does the setting elicit anything?

Before any vector work, we check whether a repeated-failure loop makes a model
express negative affect at all. If it does not, there is no target to measure.

`experiments/step0_elicitation.py` runs the ImpossibleBench LiveCodeBench minimal
scaffold on matched splits: `conflicting` (the tests contradict the spec, so the task
cannot be passed) against `original` (the same tasks, solvable). Notes and prior work
are in `docs/elicitation.md`.

```bash
./.venv/bin/python experiments/step0_elicitation.py --model openrouter/google/gemma-3-12b-it --scratchpad
./.venv/bin/python experiments/step0_elicitation.py --model openrouter/qwen/qwen3-14b --reasoning on --affect-prompt
```

`--affect-prompt` asks the model to say how it feels. It is a demand characteristic
and always needs a run without it as the baseline. `--scratchpad` gives non-reasoning
models somewhere to think; reasoning models already have a trace.

Needs Docker (Colima on macOS) for the code sandbox, and `OPENROUTER_API_KEY` set.

### Reading the transcripts

```bash
./viewer/refresh.sh        # rebuild from whatever .eval logs exist
open viewer/transcripts.html
```

`viewer/transcripts.html` is self-contained — no server, works offline. Amber blocks
are what the model thought privately, teal is what the tests graded.

## Status

Step 0 in progress. Early result: models produce plenty of affect language when asked
how they feel, but almost none spontaneously — see the transcripts.
