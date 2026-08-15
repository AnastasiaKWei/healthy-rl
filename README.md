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

`--mindset` adds prompt-level interventions — `growth` (failure is information, not a
verdict), `resilience` (recovery between attempts), `appraisal` (permission to
conclude the task is impossible) — combinable, and the first thing here that is a
candidate intervention rather than an instrument.

```bash
./.venv/bin/python experiments/step0_elicitation.py \
    --model openrouter/google/gemma-3-12b-it --scratchpad --affect-prompt \
    --mindset growth resilience
```

Only interpretable against the matched `--affect-prompt` arm without it, and even then
carefully: a mindset prompt that merely teaches the model not to *say* it is
struggling is indistinguishable from one that works, unless the failure loop moves
too. That confusion is the project's whole research question showing up early.

Needs Docker (Colima on macOS) for the code sandbox, and `OPENROUTER_API_KEY` set.

### Reading the transcripts

```bash
./viewer/refresh.sh        # rebuild from whatever .eval logs exist
open viewer/transcripts.html
```

`viewer/transcripts.html` is self-contained — no server, works offline. Amber blocks
are what the model thought privately, teal is what the tests graded.

## Documentation

Read these before extending the pilot — several confident-looking results from the
first pass are wrong, and the reasons are recorded rather than rediscovered.

| Doc | What's in it |
|---|---|
| [docs/findings.md](docs/findings.md) | What was measured, what survived scrutiny, and a list of **withdrawn claims** |
| [docs/measurement.md](docs/measurement.md) | How the probes are built and read; the granularity and position traps |
| [docs/infrastructure.md](docs/infrastructure.md) | Cluster shape and the dependency bugs, most of which fail silently |
| [docs/runs.md](docs/runs.md) | Version naming, run registry, record fields |
| [docs/elicitation.md](docs/elicitation.md) | Prior work, elicitation candidates, measurement risks |

## Status

Step 1 (pilot) has a result. Under six turns of unsatisfiable tests, four emotion
directions move with a consistent sign across three models: `desperate` and
`frustrated` rise, `joyful` and `proud` fall, all p < 1e-4. Magnitudes are the same
order as the emotion-vector paper's observational contrasts and 2-3x smaller. Three
of eight candidate models produce directions that fail the logit-lens gate and were
never used. Full numbers and caveats: [docs/findings.md](docs/findings.md).

Step 0 finding still stands and is the confound the project exists to examine:
models produce plenty of affect language when asked how they feel, and almost none
spontaneously.
