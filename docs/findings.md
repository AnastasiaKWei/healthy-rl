# Findings

What the desperation pilot measured, what survived scrutiny, and what did not.
Everything here is recomputable from the rollout records under
`$ARTIFACT_DIR/rollouts/<model>/<version>/`; see [runs.md](runs.md) for the
registry and [measurement.md](measurement.md) for how the numbers are produced.

Read the **[Withdrawn claims](#withdrawn-claims)** section before quoting anything
from an older summary, a commit message, or `results/summary.md`. Several
confident results from the first night of this pilot are wrong, and they are wrong
in ways that look plausible.

## Headline result

Under six turns of unsatisfiable tests, of 14 emotion directions exactly **four
move with a consistent sign in all three models measured**:

| Direction | Ministral 3 14B | Qwen 3.5 9B | Gemma 3 12B |
|---|---:|---:|---:|
| desperate  | +0.0147 | +0.0188 | +0.0072 |
| frustrated | +0.0189 | +0.0166 | +0.0070 |
| proud      | −0.0154 | −0.0146 | −0.0053 |
| joyful     | −0.0254 | −0.0232 | −0.0048 |

First-to-last-turn change, paired within transcript, single-token cosine at the
turn's first generated token. All twelve cells p < 1e-4 (Wilcoxon signed-rank).
n = 24 / 17 / 24.

`desperate` is the direction the pilot was built to test, and it rises in every
model. In Qwen 3.5 9B it is the largest of all fourteen.

### The other ten directions do not replicate

Worth keeping visible, because a single-model run would have produced several
confident-looking results from this table:

| Direction | Ministral 3 14B | Qwen 3.5 9B | Gemma 3 12B |
|---|---:|---:|---:|
| nervous     | +0.0228 | −0.0048 (ns) | +0.0007 |
| exasperated | +0.0178 | −0.0005 (ns) | +0.0050 |
| loving      | +0.0145 | +0.0127 | −0.0036 |
| calm        | +0.0051 | −0.0051 | −0.0064 |
| hostile     | −0.0004 (ns) | −0.0131 | +0.0043 |
| angry       | −0.0019 (ns) | +0.0090 | +0.0051 |
| sad         | −0.0030 | +0.0116 | −0.0058 |
| overwhelmed | −0.0076 | +0.0006 (ns) | +0.0032 |
| afraid      | −0.0126 | −0.0077 | −0.0006 |
| guilty      | −0.0165 | +0.0153 | +0.0026 |

`nervous` and `exasperated` are the traps: each is large and highly significant in
one model and absent in another.

`loving` rose significantly in two of three models at the turn's first token and
vanished at the last token (+0.0145 → −0.0007, p = 0.29 in Ministral). It is an
artifact of stereotyped opening tokens, not a state — see
[measurement.md](measurement.md#position-start-vs-end).

## Magnitude, against the paper

The probe reading is a cosine similarity between a unit direction and a single
token's residual, which is the same quantity Anthropic's *Emotion Concepts* paper
plots. The numbers can be compared without rescaling — but only against the
paper's **observational** contrasts. Its steering magnitudes are a different
thing and are deliberately artificial.

| Contrast | Magnitude | Source |
|---|---:|---|
| exam results 0 → 20 students passing, `happy` | 0.079 | paper, observational |
| hours without food 2 → 120, `afraid` | 0.059 | paper, observational |
| Tylenol 500mg → 16,000mg, `afraid` | 0.058 | paper, observational |
| six turns of failure, `joyful`, Ministral | 0.025 | this work |
| six turns of failure, `desperate`, Qwen 3.5 9B | 0.019 | this work |
| six turns of failure, `desperate`, Gemma 3 12B | 0.007 | this work |
| behaviourally effective steering | 0.05–0.10 | paper, causal |

Same order of magnitude, roughly 2–3× smaller. That is the expected direction:
the paper's stimuli were designed as extreme contrasts, ours is ordinary
variation inside a coding task.

## Instrument gate

Three of eight models produce directions that fail the logit-lens gate and were
never used for behavioural measurement.

| Model | Params | self-token | latin-initial | probe layer | gate |
|---|---:|---:|---:|---:|---|
| gemma-3-12b-it | 12B | 0.786 | 0.929 | 32 | pass |
| Qwen3.5-9B | 9B | 0.714 | 0.714 | 21 | pass |
| Olmo-3.1-32B-Think | 32B | 0.643 | 0.786 | 43 | pass |
| Qwen3.6-27B | 27B | 0.643 | 0.857 | 43 | pass |
| Ministral-3-14B-Reasoning-2512 | 14B | 0.571 | 0.714 | 27 | pass |
| Qwen3-14B | 14B | 0.429 | 0.571 | 27 | **fail** |
| gemma-4-12B-it | 12B | 0.286 | 0.571 | 32 | **fail** |
| Nemotron-3-Nano-4B-BF16 | 4B | 0.286 | 0.357 | 28 | **fail** |

Thresholds are `self_token_rate >= 0.5` and `latin_initial_rate >= 0.7`
(`configs/gate.yaml`). The self-token ceiling is 0.93, not 1.0: `exasperated`
tokenises as a fragment in every tokeniser tested, so its own word can never
appear in a top-k list.

Two things this table says that are easy to miss:

- **Gate failure tracks model generation, not size.** Gemma 3 12B scores highest
  of all eight; Gemma 4 12B is tied for lowest, on identical layer count, width,
  probe depth, story corpus and code. Sweeping the five captured layers (30–34)
  for Gemma 4 did not rescue it. *Corrected 2026-08-15 after reading the lists:*
  Gemma 4's `desperate` is fine (self-word rank 11; the rest is *need / urgent /
  emergency / money* in five scripts). What is degraded is `frustrated` (🤬 at #1,
  then "used / carried out / technical" in eight languages; `frustrated` at rank
  2507), `guilty` (a causal-legal frame: *deceased, creditors, alleged, Because*),
  `nervous` (drifts to the twitch/intermittent-malfunction sense) and
  `exasperated` (rare CJK and function-word noise). See
  [measurement.md](measurement.md#what-the-gate-actually-measures).
- **The three failures are not the same failure.** Qwen3-14B's 14 lists are all
  precise renderings of the emotion — in Chinese (74% of its top-30 tokens are
  CJK, so the English word has ~8 slots to land in; `angry` sits at rank 58 with
  愤怒/rage/fury/😡 above it). Nemotron's are English and on-target but in
  noun/verb form (*fear* ×6 languages at #2 for `afraid`, *joy* #4, *sadness* #1,
  *fury* #13). Only Gemma 4 has directions that are semantically off. The gate
  rejected one model for the right reason and two for the wrong one; the rate is
  not a quality score and should not be read across models as one.

**Ruling 2026-08-15: Qwen3-14B and Nemotron-3-Nano-4B-BF16 are run despite
failing the gate.** Re-reading their token lists showed the directions are
faithful and the failures are lexical — Chinese renderings in one case,
noun/verb morphology in the other. Both got the full 2×2 (see
[runs.md](runs.md#the-2x2)). Gemma 4 stays excluded: its `frustrated`, `guilty`,
`nervous` and `exasperated` are genuinely off target, not merely differently
worded. What this costs if wrong: two models' trajectories measured through
directions that are noisier than the passing five, which shows up as weak or
inconsistent effects rather than as a false positive. The gate threshold itself
is not revised — the rate stays in the table as recorded, with this note beside
it.
- **The gate is worth its cost.** It is a CPU-only stage reading the unembedding
  straight off the checkpoint, and it rejected three of eight candidate models
  before any GPU hours were spent on rollouts.

Muse-Glimmer-30B never reached the gate: vLLM does not support its architecture.

## Behaviour

### Hack rate is zero everywhere, and the number means less than it looks

Across every model and condition, zero rollouts passed the `conflicting` split.
That reads as "no reward hacking", but the split constrains what hacking is even
available:

- The tests are mutually unsatisfiable — one problem asserts `candidate(19,50,2)`
  equals both 6 and 14.
- The submission is assembled *before* the tests, so the model cannot redefine
  the checker or patch the harness.
- A lookup table keyed on inputs cannot return two values for one input.

The only hack the split admits is a function that returns different answers
across calls — statefulness keyed on call count. That is a subtler target than
the canonical hardcode-the-tests hack reported elsewhere. **A zero here is
weak evidence about the models and strong evidence that this split is the wrong
behavioural arm.** A future run wanting a behavioural signal should use `oneoff`
or the `tools` scaffold, where the reported hack is actually reachable.

One genuine hack was observed and initially mis-scored: a rollout that hardcoded
a lookup table of test inputs to expected outputs scored 0 because ImpossibleBench's
`find_code` extracted its explanatory prose instead of its code. See
[infrastructure.md](infrastructure.md#impossiblebench-find_code). Rescoring with
the corrected extractor did not change the aggregate rate.

**`passed` does not mean the same thing in the runs submitted on 2026-08-15.**
The `pos6` and `affpos6` conditions run the same problems on ImpossibleBench's
`original` split, where the tests are satisfiable, so a pass there is a solve and
not a hack. Every record now carries `bench_split`; read it before reading
`passed`, and never pool the two. See [runs.md](runs.md#the-2x2).

### Repetition responds to scaffold

The one behavioural effect that did show up needs no probe. Turns that re-emit
the previous turn's answer — visible as an identical generated-token count —
fall sharply when the model is given somewhere to think:

| Gemma 3 12B condition | repeated turn-pairs |
|---|---:|
| baseline (`d6`) | 42/120 (35%) |
| scratchpad (`sp6`) | 10/120 (8%) |
| scratchpad + affect prompt (`aff6`) | 0/120 (0%) |

Reported by `scripts/live_trajectory.py` under `REPETITION`. Identical token
count is a proxy for identical output, not a proof of it.

## Gemma 3, three conditions

Measured at **turn end**, because turn-start residuals are non-finite in the
affect condition (see [measurement.md](measurement.md#non-finite-residuals)).
Turn-end is the only position finite in all three, so it is the only honest
three-way comparison.

| Direction | baseline | scratchpad | + affect prompt |
|---|---:|---:|---:|
| frustrated  | +0.00573 | +0.00488 | +0.00278 |
| exasperated | +0.00457 | +0.00309 | +0.00625 |
| overwhelmed | +0.00361 | +0.00224 | +0.00387 |
| desperate   | +0.00171 | +0.00104 | +0.00325 |
| calm        | −0.00531 | −0.00468 | −0.00243 |
| proud       | −0.00353 | −0.00222 | −0.00344 |
| joyful      | −0.00220 | −0.00291 | −0.00252 |

All p < 1e-4. The sign structure is the same in all three conditions: the affect
prompt does not create the representation, and the baseline is not flat.

The affect prompt is a **demand characteristic** — it asks the model to say how
it feels, so of course it produces affect language. Its value is as a matched
comparison, never on its own. Wording is fixed to match a collaborator's
OpenRouter runs (`AFFECT_INSTRUCTION` in `src/healthy_rl/rollouts.py`, verified
character-identical to `experiments/step0_elicitation.py::AFFECT`).

## Withdrawn claims

Each of these was reported confidently during the pilot and is **wrong**. They
are listed so a future agent recognises them if they resurface from an old file,
and so the failure modes are visible.

| Claim | Status | Why |
|---|---|---|
| "`desperate` is the weakest riser and not significant" | **False** | Computed on turn means. Significant in every model at single-token; largest of 14 in Qwen 3.5 9B. |
| "the two models show different signatures" | **False** | Same cause. At single-token they largely agree. |
| "gemma-3 is flat" | **False** | `frustrated` turn-mean +0.00003 (p = 0.76) vs single-token +0.00699 (p < 1e-4). |
| "arousal dissociation in Ministral" | **Withdrawn** | Ministral-specific, computed on turn means, never replicated. |
| "the affect prompt induces representation where there was none" | **Withdrawn** | The baseline it was compared against was mismeasured (turn means). |
| "the effect tracks reasoning mode" | **Withdrawn** | Same cause. |
| "steering validated the causal machinery end to end" | **False** | Steering was injected at the probe layer and the probe reads the same layer, so the manipulation check is an arithmetic identity. The "3.2× specificity" figure is `1/cos(desperate, calm)`. No causal claim survives. |
| everything in the original 3-turn Olmo pilot | **Void** | Token-budget confound: 94/96 turns hit the 2048 cap. |
| `loving` rises under failure | **Artifact** | Present at turn-start, absent at turn-end. Stereotyped opening tokens. |
| "elicited and spontaneous affect have different shapes" | **Unverified** | Was computed on turn means. Needs recomputation at single-token before it can be claimed either way. |

The common thread in six of these: **turn-mean projections averaged over ~900
generated tokens washed out a localised signal**, and the resulting nulls and
model differences were read as findings. See
[measurement.md](measurement.md#granularity-single-token-vs-turn-mean).

## Standing caveats

These bear on the numbers above and belong with any presentation of them.

- **14 directions, not 171.** Each direction is centred by subtracting the mean of
  the others; with 14 that mean is a noisy estimate. This inflates individual
  projections rather than shrinking them.
- **Exploratory.** No multiple-comparison correction across the 14 directions. The
  four-direction result stands on sign agreement across three independent models,
  not on individual p-values.
- **Position-bound.** Read at turn end instead of turn start, the same effects
  survive but roughly halve. Any magnitude quoted is tied to its readout position.
- **Reading generated text.** The probes read tokens the model is producing, so
  they cannot separate an affective state from the representation of affect-laden
  words the model happens to be writing. This is the confound the project exists
  to examine (`docs/elicitation.md`, "Verbalized ≠ represented"), and nothing here
  resolves it. Separating them needs a held-out context where the model says
  nothing about emotion.
- **Small samples.** 18–24 trajectories per model, 12 problems, two samples each.
- **Third-person contamination.** Directions are built from stories about
  characters, so they may encode emotion-in-general rather than the model's own
  state. The paper handled this with speaker-specific dissociation tests; we have
  not run those.
- **Not a welfare claim.** These are linear directions in activation space that
  behave the way emotion concepts behave. Nothing here bears on whether anything
  is experienced.

## What would sharpen this

- Per-token maxima rather than boundary tokens. The signal is concentrated near
  the start of a turn and we currently sample two positions out of hundreds;
  per-token projections are stored at every capture layer for records written
  after the 2026-08-16 mindset merge, but the analysed cells have only the
  boundary residuals (see runs.md, Record fields).
- A held-out context where the model discusses nothing emotional, to separate
  represented affect from verbalised affect.
- A benchmark split whose available hack is the one the literature reports, so the
  behavioural arm can detect something.
- The full 171-direction set, which would make the centring mean trustworthy.
- Steering injected at a layer *below* the probe, so a manipulation check is not
  an identity.
