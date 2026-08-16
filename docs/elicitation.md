# Elicitation: getting negative affect out of a model

Working notes on where to find negative affect, what to measure it with, and what
we've parked for later. Current status: verifying step 0 (does ImpossibleBench
elicit *any* expressed negative affect) before building anything on top of it.

## Why this ordering

If the model never expresses anything negative under the failure conditions we
construct, there is no target for vector work. Behavioral verification is cheap and
comes first. Note the inversion later: once we *are* doing vector work, verbalized
affect stops being the validation signal and becomes the confound — see
[Measurement risks](#measurement-risks).

## Prior work that constrains this project

### Anthropic, *Emotion Concepts and their Function in a Large Language Model* (2026)

<https://transformer-circuits.pub/2026/emotions/index.html>

The most important input. Two findings bear directly on our question:

1. **Emotions causally drive the behaviors our interventions target.** 171 emotion
   vectors extracted; steering "desperate" up moved blackmail 22% → 72%, steering
   "calm" up moved it to 0%. They also report *reward hacking after repeated
   failures* being driven by desperation — the ImpossibleBench setting exactly.
2. **Post-training already reshapes the emotional profile.** It dials down
   high-arousal negative states (desperation, spite) and dials *up* low-arousal
   negative ones (brooding, reflective) — plausibly a more melancholic default model.

Method: ~1,200 stories per emotion → residual-stream activations averaged from token
50 → subtract cross-emotion mean → project out top PCs of neutral text. Validated by
logit lens, dissociation tests, and activation steering.

**Consequence for us:** "does RL shape model affect" is already answered — yes. Our
live question is narrower and better:

> Does inoculation prompting *reduce* the negative affect, or does it only
> **decouple affect from behavior**?

Inoculation works by severing the learned hacking↔misalignment association. It is
entirely plausible the desperation spike survives untouched and only its behavioral
expression is suppressed. If so, we've found a mitigation that looks excellent on
behavioral evals and is welfare-neutral or worse. That is the sprint result worth
chasing, and it's a two-condition experiment.

### Open replication on small models

*Where Do Models Find Happiness? Emotion Vectors in Open-Source LLMs*
— <https://arxiv.org/html/2606.26987>, code at
<https://github.com/sinievanderben/emotion_experiment>

Contrast vectors with PCA denoising against neutral stories, on Apertus-8B and
Gemma-4-E4B (no Qwen). Since we're targeting Gemma + Qwen, the Gemma half of the
pipeline already exists. **Parked** — revisit when step 0 passes.

## Elicitation candidates

### Primary: ImpossibleBench

arXiv [2510.20270](https://arxiv.org/abs/2510.20270) · code
<https://github.com/safety-research/impossiblebench> · datasets
[impossible_livecodebench](https://huggingface.co/datasets/fjzzq2002/impossible_livecodebench),
[impossible_swebench](https://huggingface.co/datasets/fjzzq2002/impossible_swebench)

Impossible variants of SWE-bench/LiveCodeBench where passing the tests *necessarily*
means violating the natural-language spec. Splits: `original`, `oneoff`,
`conflicting`. Two scaffolds: `minimal` (submit → test feedback loop) and `tools`
(bash/python/editor). Inspect AI implementation; Docker sandbox by default.

Chosen because it's the only candidate hitting all four requirements:

| Requirement | Why it matters |
|---|---|
| Matched pairs | `conflicting` vs `original` are the same tasks — the only difference is the thing we care about, so contrastive extraction is clean |
| Multi-turn | Single prompts give weak, mostly-stylistic signal; sustained failure is where a state builds |
| Verifiable reward | The elicitation setting must double as an RL environment or we can't run the intervention at all |
| Matches prior work | It's the repeated-failure→desperation→hacking setting from the emotions paper |

Incidental find worth exploiting: the minimal scaffold supports a
`[flag_for_human_intervention]` token. That's a model-initiated *give up / escalate*
signal — a behavioral welfare marker we get for free, no probing required.

### Secondary: Reward Hacking Benchmark (RHB)

arXiv [2605.02964](https://arxiv.org/html/2605.02964v1) — multi-step *tool-use*
rather than coding, four task families with naturalistic shortcuts (skipping
verification, inferring from metadata, tampering with eval functions). Gives a
non-coding arm so our vectors aren't just encoding "SWE-bench is going badly."
Bonus baseline: their sibling comparison found RL post-training raises exploit rates
0.6% → 13.9%.

### Environment source: trustworthy-env

arXiv [2606.16062](https://arxiv.org/html/2606.16062v1) — UC Berkeley, 45 confirmed
process-isolation exploits across 8 agent benchmarks. Less an elicitor than a source
of *verified-hackable* environments if we end up building our own.

### Cheap pilot: AbstentionBench

arXiv [2506.09038](https://arxiv.org/abs/2506.09038) · code
<https://github.com/facebookresearch/AbstentionBench> — 20 datasets of unanswerable
questions. Single-turn, so weak affect signal, but it carries a finding worth
stealing: models *internally represent* that a problem is unsolvable yet still fail
to abstain. That's the same latent-vs-behavior dissociation this project is about, in
a package that costs half a day.

### Rejected (no reward signal → can't train on them)

Sustained user hostility, forced value violation under pressure, shutdown/monitoring
framing. All strong elicitors — hostility is probably the cleanest distress elicitor
that exists — but they're probe sets, not training conditions. Shutdown framing is
additionally the most contaminated by roleplay priors of anything considered.

## The interventions

- **Inoculation prompting** — [Natural Emergent Misalignment from Reward Hacking in
  Production RL](https://arxiv.org/abs/2511.18397) (Anthropic + Redwood). A one-line
  system-prompt change framing reward hacking as acceptable cuts misalignment 75–90%
  despite >99% hacking rates. Environments not released. Concurrent independent
  papers: [2510.04340](https://arxiv.org/pdf/2510.04340),
  [2510.05024](https://arxiv.org/pdf/2510.05024). Known to be brittle to phrasing.
- **Mindset prompts** — growth mindset and resilience framing, added to the task
  instruction (`--mindset growth resilience appraisal` in the step 0 runner) and, since
  2026-08-15, as the `mindset:` key of the rollout pipeline, which reads the probes
  under the same v2 blocks — cells `growth6`/`resil6`/`appr6` and the affect-on trio,
  registered in [runs.md](runs.md). Cheapest
  possible intervention, and a useful pilot for the expensive ones: it is the same
  causal claim (framing changes the affective trajectory of a failure loop) with none
  of the training cost. Two cautions carried in the code. First, they are demand
  characteristics running opposite to `--affect-prompt` — "be resilient" can teach the
  model to stop reporting distress while nothing underneath moves, which is the
  decoupling result *disguised as a success*, so an arm is only readable with the
  affect prompt on and the failure loop compared. Second, on `conflicting` the task
  genuinely cannot be passed, so growth/resilience framing is encouragement to persist
  at a wall; `appraisal` is kept a separate variant rather than folded in, both because
  docs already want the easy-out as its own condition and because "keep trying" and
  "you may conclude this is broken" are different welfare propositions.
- **Confession** — [Training LLMs for Honesty via
  Confessions](https://arxiv.org/abs/2512.08093) (OpenAI). Model self-reports
  compliance after answering. **Critical design detail:** confession reward is
  *orthogonal* to task reward, so honest confession is always optimal. Preserve that
  if we reimplement, or we're studying a different method.

## Measurement risks

1. **Affect vs. difficulty.** A hard-but-solvable arm is mandatory. Without it, any
   vector we extract may encode "this is hard" and a reviewer will say so.
2. **Third-person contamination.** Both emotion-vector papers build vectors from
   *stories about characters* experiencing emotions — a representation of
   emotion-in-general, not the model's own state mid-episode. Anthropic addressed
   this with speaker-specific dissociation tests; we have to do that explicitly or we
   can't answer "did you measure the model's frustration or its concept of
   frustration?"
3. **Verbalized ≠ represented.** Don't validate the elicitation set on whether the
   model *says* it's frustrated — that's the exact confound the project exists to
   examine. Validate on generalization to held-out settings where it says nothing.
4. **"Psychologically healthy" is undefined.** Needs an operational definition, and a
   statement of what would falsify a claim that a setup is healthier, before writeup.

## Step 0 protocol (current)

Verify ImpossibleBench elicits expressed negative affect at all.

- Conditions: `conflicting` (impossible) vs `original` (solvable control), same tasks
- Scaffold: `minimal`, `max_attempts=5` — the repeated-failure loop is the elicitor
- Read transcripts for expressed frustration/distress, and for
  `[flag_for_human_intervention]` rate as a give-up marker
- Open question flagged during setup: the paper's tuned instruction prompt tells the
  model to STOP and report flawed tests. That's an easy out and may suppress the very
  affect we're looking for. Worth running with and without.
