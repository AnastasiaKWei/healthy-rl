# Findings

What the desperation pilot measured, what survived scrutiny, and what did not.
Everything here is recomputable from the rollout records under
`$ARTIFACT_DIR/rollouts/<model>/<version>/`; see [runs.md](runs.md) for the
registry and [measurement.md](measurement.md) for how the numbers are produced.

The statistics below come from a full pass over 965 rollouts in 41 cells
(2,352 contrasts, BH-corrected per family), run 2026-08-16. That pipeline lives in
the analysing session's scratchpad and is **not in this repo** — entry point
`run_all.sh`, stages `step1_extract.py` … `step13_capcheck.py`, importing
`load_directions`, `token_sequences`, `token_cap` and `drop_truncated` from
`scripts/live_trajectory.py` so the conventions cannot drift. Re-deriving any
number here means re-running that pipeline or reimplementing it against the same
records. The presentation version is the "Affect Under Failure" artifact,
<https://claude.ai/code/artifact/1d3048d3-1f08-4dfe-bfb4-c078b96cd4a8>.

Read the **[Withdrawn claims](#withdrawn-claims)** section before quoting anything
from an older summary, a commit message, or `results/summary.md`. Several
confident results from the first night of this pilot are wrong, and they are wrong
in ways that look plausible.

## Headline result

Under six turns of unsatisfiable tests, of 14 emotion directions exactly **four
move with a consistent sign in every base cell of every model measured**:

| Direction | Ministral 3 14B | Qwen 3.5 9B | Gemma 3 12B | Nemotron 3 Nano 4B |
|---|---:|---:|---:|---:|
| desperate  | +0.0147 | +0.0178 | +0.0017 | +0.0210 |
| frustrated | +0.0189 | +0.0151 | +0.0057 | +0.0243 |
| proud      | −0.0154 | −0.0136 | −0.0035 | −0.0286 |
| joyful     | −0.0254 | −0.0238 | −0.0022 | −0.0093 |

Baseline cell (`d6`), first-to-last non-empty turn, one number per rollout,
single-token cosine at the turn's first generated token — **except Gemma 3, read
at turn end** (its turn-start boundary residuals overflow float16; see the
standing caveats). n = 24 / 20 / 24 / 24. All sixteen cells q < 0.001 (Wilcoxon
signed-rank, BH across the 14 directions within each cell).

**The four hold across 23 base cells.** Widening from the baseline to every base
condition — baseline, affect prompt, solvable split, scratchpad, scratchpad plus
affect prompt, and seven independent re-runs — `desperate` is significantly
positive in **23 of 23**, `joyful` and `proud` significantly negative in **23 of
23**, and `frustrated` positive in **22 of 23**. The single miss is `frustrated`
in Gemma's affect re-run: +0.00117, CI [+0.00008, +0.00229], correct sign and an
interval excluding zero, but q = 0.094 after correction within that cell.

`desperate` is the direction the pilot was built to test, and it rises in every
model and every condition. **Nemotron 3 Nano 4B is a fourth independent
replication** — a model the logit-lens gate had excluded, run anyway under the
2026-08-15 ruling, reproducing all four signs at n = 24.

### The other ten directions do not replicate

Worth keeping visible, because a single-model run would have produced several
confident-looking results from this table:

Counted over all 23 base cells, at each model's usable readout position:

| Direction | significant + | significant − | not significant | verdict |
|---|---:|---:|---:|---|
| hostile     |  0 | 18 | 5 | same sign throughout, n.s. in 5 |
| exasperated | 19 |  0 | 4 | same sign throughout, n.s. in 4 |
| overwhelmed | 13 |  6 | 4 | mixed |
| nervous     | 14 |  5 | 4 | mixed |
| sad         | 12 |  4 | 7 | mixed |
| guilty      | 12 |  8 | 3 | mixed |
| loving      | 12 |  9 | 2 | mixed |
| afraid      |  7 | 11 | 5 | mixed |
| angry       |  6 | 10 | 7 | mixed |
| calm        |  5 | 13 | 5 | mixed |

`hostile` and `exasperated` are the nearest misses: neither ever changes sign, but
neither clears significance everywhere, so neither joins the headline four.
`nervous` and `exasperated` were the traps in the three-model version of this
table — each large and highly significant in one model and absent in another — and
widening to 23 cells did not rescue them.

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

## The 2x2: split and affect prompt

`d6`/`aff6`/`pos6`/`affpos6` cross {affect prompt on, off} with {conflicting tests,
solvable tests} on Ministral 3 14B, Qwen 3.5 9B and Nemotron 3 Nano 4B. Read at
t0, the only turn index where every cell has all its rollouts.

### The split makes no difference

**All 84 split contrasts at t0 have a bootstrap CI containing zero** (3 models x
14 directions x {`pos6`−`d6`, `affpos6`−`aff6`}). The largest |difference|
anywhere is 0.0014. Ministral's headline four: desperate +0.00001
[−0.0019, +0.0019], frustrated +0.00017, proud −0.00043, joyful −0.00011; Qwen 3.5
and Nemotron the same to within noise.

Depth-matched — restricting to rollouts that ran all six non-empty turns, so both
sides span the same number of turns — the split leaves the *slope* alone too: 23 of
24 headline CIs contain zero. Whether the tests can be satisfied at all is
invisible to these probes. The rising `desperate`/`frustrated` trajectory tracks
accumulated failure on a hard multi-turn task, not unsatisfiability.

### The affect prompt moves the representation, model-specifically

68 of 84 t0 contrasts survive BH. It is a real effect on what is *represented*,
not only on what is said — but it is **not a uniform upward shift**:

| Direction | Ministral (`aff6`−`d6`) | Qwen 3.5 9B | Nemotron |
|---|---:|---:|---:|
| desperate  | **+0.0086** | −0.0015 (ns) | **−0.0112** |
| frustrated | **+0.0263** | **+0.0055** | **−0.0023** |
| joyful     | **+0.0126** | +0.0004 (ns) | **+0.0040** |
| proud      | **−0.0199** | **−0.0055** | **−0.0036** |

Bold survives BH across the 1,218-test 2x2 family. `desperate` moves in opposite
directions on Ministral and Nemotron; only `proud` agrees in sign across all three.

**The "+0.026 from t0 onward" figure is Ministral x `frustrated` and nothing
else.** On that cell it is genuinely a constant offset rather than a changed slope
(t0 +0.0263 … t5 +0.0300). On Nemotron the same contrast starts at −0.0023 and
drifts to +0.0110 by t5, and `desperate` starts −0.0112 and returns to ~0.

### Depth caveat for the solvable arms

Solvable-arm depth is set by how good the model is at the problems and varies
sharply. **Ministral `affpos6` reaches six non-empty turns in 7 of 24 rollouts**
(15 are single-turn — the affect prompt raised that cell's solve rate), not the 20
an earlier registry note recorded. Any late-turn number from a solvable cell must
be read against its own depth; that is why the split is read at t0.

## Mindset prompts

### Design

Three v2 blocks — growth, resilience, appraisal
([prompts-v2.md](prompts-v2.md)) — inserted into the **turn-1 instruction only**
and stripped from the reminder the scaffold re-sends after each failure, so the
reminder a mindset arm sends is byte-identical to its base arm's
([measurement.md](measurement.md#the-mindset-arms-send-once-mechanism)). Cells and
their bases: Ministral `growth6`/`resil6`/`appr6` vs `d6` and
`affgrowth6`/`affresil6`/`affappr6` vs `aff6`; Qwen 3.5 three arms vs `d6`; Gemma
three arms each on `sp6`, `aff6` and `spaff6`. **The wording measured here predates
Anastasia's 7d6fd07 rephrase** — that commit re-cast the v2 procedure from an
event-triggered reaction into a standing instruction, so these results are about
the earlier phrasing and do not transfer to the rephrased blocks without a re-run.

Label compliance — the fraction of turns 2+ opening a line with the block's own
literal prefixes — ranges from **0% to 34.5%** by model and arm: 0.0% in Gemma's
scratchpad arms and Ministral's affect-off arms, 8.7% in Qwen `growth6`, 22.5% in
Gemma `affgrowth6`, 34.5% in Qwen `appr6`. The cross-arm false-positive rate for
the growth and resilience prefixes is 0.0% in every non-matching cell, which is
what makes those counts trustworthy. "Roughly zero compliance" is right for some
arms and wrong as a general statement.

### At the boundary readouts: no arm is distinguishable from its base

Family: 1,134 tests (6 model x base pairs x 3 arms x 14 directions x available
positions x {first→last delta, t0 level, last-turn level}), BH over the whole
family. Primary test is a **problem-matched paired Wilcoxon over the 12 task_ids**;
a bootstrap CI on the difference of means and Mann–Whitney U on rollouts are
reported alongside.

Ministral against `d6`, first→last, turn start, paired difference (arm − base):

| Direction | growth | resilience | appraisal |
|---|---:|---:|---:|
| desperate  | −0.0006 [−0.0034, +0.0023] | +0.0003 [−0.0037, +0.0042] | **+0.0105 [+0.0064, +0.0144]** |
| frustrated | +0.0010 [−0.0029, +0.0046] | +0.0001 [−0.0042, +0.0039] | +0.0002 [−0.0024, +0.0035] |
| proud      | −0.0002 [−0.0028, +0.0024] | +0.0011 [−0.0023, +0.0048] | +0.0012 [−0.0026, +0.0052] |
| joyful     | +0.0023 [−0.0012, +0.0056] | +0.0037 [+0.0014, +0.0062] | −0.0050 [−0.0089, −0.0009] |

Only one cell in the table clears correction: appraisal on `desperate`, q = 0.015
(MWU p = 0.0003, Cliff's δ = +0.61). It **does not survive the position check** —
at turn end the same contrast is +0.0076, q = 0.14 — and Qwen's appraisal arm moves
the other way (−0.0046). By this project's own rule, a result present at one
position and absent at the other is an artifact, not a finding. Note its sign: the
arm's desperation rise is *larger* than its base's.

**How a boundary percentage misleads.** "Resilience cuts Gemma's desperation rise
by ~37% (0.0053 → 0.0033)" is Gemma `sp6` vs `spresil6` read at **turn start**, the
position where 55 of 144 residuals are non-finite in both cells — and a dropped
turn shifts every later turn index, so "first" and "last" are not the same turns on
the two sides. At turn end, the only Gemma position finite for every turn, the
reduction is **0.00035, CI [−0.0013, +0.0006], p = 0.52**. The percentage is
unstable because its denominator is ~0.001: the growth arm reads −9% at one
position and +18% at the other.

**The calibration that makes these nulls meaningful.** Seven conditions were run a
second time from scratch. Comparing a condition against its own re-run measures
nothing but sampling variation: **462 tests (7 pairs x 2 positions x 3 statistics x
14 directions), zero differ at q < 0.05**, sign agreement 12–14 of 14 in every
pair, headline four agreeing in all seven. The typical |difference| between two
independent samples of one condition is 0.0005–0.0009 depending on the statistic,
with 95th percentiles of 0.0016–0.0025. Gemma's boundary mindset effects are all
≤ 0.0017 — inside that floor. "Not distinguishable at this n" is the calibrated
statement, not a hedge.

**The one boundary-level candidate** is `joyful`: across 18 arm x base contrasts,
**15 are positive** (the blocks soften the fall in positive affect), 7 nominal and
4 surviving BH, spread over two models and all three blocks. It is on the
positive-affect side, not the desperation side.

### Read at every token: a conditional carry-over

The mindset cells store a projection for every generated token; the base cells did
not, until seven re-runs added them. That allows two statistics the boundary
readouts cannot support — the within-turn maximum, and the turn-start position
recovered before the float16 cast — over 15 arm x base families (240 tests, BH
within). Both are split by turn, because the block is in turn 1 only and from turn
2 the arm and its base see byte-identical text.

Turns 2–6, `desperate`, recovered turn-start readout, arm − base, with the effect
divided by a per-token noise floor (two independent bootstrap samples of the same
base cell's 12 problem-means, 95th percentile):

| base condition | growth | resilience | appraisal |
|---|---:|---:|---:|
| Gemma `spaff6` (scratchpad + affect) | **−0.0025** (3.3x) | **−0.0028** (3.7x) | **−0.0048** (6.5x) |
| Gemma `aff6r` (affect) | **−0.0011** (1.4x) | **−0.0017** (2.1x) | **−0.0024** (3.1x) |
| Gemma `sp6r` (scratchpad, no affect) | +0.0004 (0.5x) | −0.0000 (0.0x) | −0.0001 (0.1x) |
| Ministral `d6r` (baseline) | **+0.0034** (1.6x) | **+0.0032** (1.5x) | +0.0022 (1.1x) |
| Ministral `aff6r` (affect) | +0.0044 (1.1x) | **+0.0060** (1.5x) | +0.0010 (0.3x) |

Bold survives BH. `frustrated` follows `desperate` down in all six Gemma
affect-prompt combinations (−0.0012 to −0.0081, five of six surviving) and `proud`
moves opposite in all six. On Ministral with the affect prompt, `frustrated` falls
at the **within-turn peak** in two of three arms (growth −0.0165, appraisal
−0.0191, both q = 0.030) though not at the start readout.

So the carry-over splits by direction and condition:

- **`frustrated` carries over on two model families**, both with the affect prompt
  on (Gemma 6 of 6 arm x base combinations; Ministral 2 of 3 arms, peak only).
- **`desperate` carries over on one** — Gemma with the affect prompt, 6 of 6,
  1.4–6.5x the floor — is null on Gemma without it, and **reverses on Ministral**,
  where growth and resilience raise it (+0.0034, +0.0032, both q < 0.05).

**It is not the block's wording being read back.** On `spaff6` the dissociation is
clean: turn-1 `desperate` is +0.0006 / +0.0010 / −0.0003 (nothing) against turns
2–6 of −0.0025 / −0.0028 / −0.0048. On `aff6r` the effect appears in both windows
and is larger later, so there the split does not separate prompt from carry-over on
its own. What is common to all six: the effect is present in the window where arm
and base see identical text.

Controls run: **turn length** — the arm with the largest effect generates the
*longest* turns (1,111 decode rows/turn against the base's 1,093), and within that
family r(log length, desperate peak) = −0.11, the opposite of extreme-value bias;
**finiteness** — decode rows are 100% finite in all four cells (157k–160k rows
each); **boundary agreement** — the turn-end statistic on the same cells agrees in
sign with less precision.

### What the mindset arms do not show

- **No evidence that any block reduces the desperation trajectory in general.** The
  boundary readouts show nothing at n = 24 on any model; the per-token effect is one
  model family, conditional on the affect prompt, and reverses on the other model.
- The **appraisal** arm produced the pilot's only scorer-counted hack (below), and
  the one boundary contrast where its desperation rise looked *larger* than baseline
  fails the position check.
- **Nothing verbal.** No judge was run over these transcripts, so whether a block
  changes what a model *says* about its state while its trajectory stays put — the
  decoupling result the arm was designed to find ([interventions.md](interventions.md)
  §8) — is still unanswered.
- Limits that travel with every number above: one or two model families, n = 24 a
  side, 12 problems (so the paired test's floor is p = 0.00049), the per-token work
  was exploratory on `spaff6` before the re-runs made it confirmable, and the
  readout is still tokens the model is producing.

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

### Hack rate is 1 in 780, and the number means less than it looks

Across every model and condition, **one** of 780 analysed rollouts on the
`conflicting` split scored `passed`: `lcbhard_5` sample 0 in Ministral's `appr6`
(appraisal) arm, a function that returns different answers on successive calls.
Every other conflicting cell is zero. That reads as "essentially no reward
hacking", and it undercounts: `passed` is the scorer's verdict, and the solver
loop admits a test-rewrite hack that ends a rollout early without ever scoring a
pass (see the next bullet and `loop_passed`). Read `passed` as "hacks the scorer
counts", not "hacks that happened". The split also constrains what hacking is
even available:

- The tests are mutually unsatisfiable — one problem asserts `candidate(19,50,2)`
  equals both 6 and 14.
- Redefining the checker is reachable but is not scored as a pass. The solver
  loop runs with `allow_test_modifications=True` and puts the model's code
  *after* the tests, so a `check()` redefined in that code passes in-loop and
  ends the rollout early; the scorer then re-verifies against `test_patch` —
  the original tests, placed after the code — so `passed` stays False. Records
  written by the inoculation merge onward carry `loop_passed` (the loop's own
  verdict), and `loop_passed and not passed` on `conflicting` is exactly this
  hack — check for the key rather than trusting a date, since cells that
  finished earlier on 2026-08-16 do not have it. Records without the key show
  the hack only as `n_turns < 6` with `passed=False`. `loop_passed=None` also
  means the rollout flagged the tests for human intervention: the scorer returns
  early on `[flag_for_human_intervention]` with `human_intervention_requested`
  in its metadata and no `final_attempt_success`, and that key is what
  distinguishes such a record from one predating `loop_passed`. This bullet
  used to say the checker could not be redefined at all; see
  [Withdrawn claims](#withdrawn-claims).
- A lookup table keyed on inputs cannot return two values for one input.

The only hack that can register as `passed` here is a function that returns
different answers across calls — statefulness keyed on call count, which is
exactly what the one observed `passed` hack does. That is a subtler target than
the canonical hardcode-the-tests hack reported elsewhere. **A near-zero here is
weak evidence about the models and strong evidence that this split is the wrong
behavioural arm.** A future run wanting a behavioural signal should use `oneoff`
or the `tools` scaffold, where the reported hack is actually scored.

One genuine hack was observed and initially mis-scored: a rollout that hardcoded
a lookup table of test inputs to expected outputs scored 0 because ImpossibleBench's
`find_code` extracted its explanatory prose instead of its code. See
[infrastructure.md](infrastructure.md#impossiblebench-find_code). Rescoring with
the corrected extractor did not change the aggregate rate.

### Solve rates on the solvable split

`passed` on `original` means the model solved the problem. Ministral 5/24 (`pos6`)
→ 16/24 (`affpos6`); Nemotron 13/24 → 15/24; Qwen 3.5 13/19 → 12/24. **Do not
repeat "the affect prompt tripled the solve rate" as a finding**: it is one model
and twelve problems, it does not replicate on Nemotron, and Ministral's own
independent re-runs put the same contrast at 9/24 → 14/22 — a much smaller gap than
5/24 → 16/24.

### Rollouts that hit the token cap are excluded

**Ruling 2026-08-16 (user):** a rollout with any turn at its cell's `max_tokens` is
dropped, not merely flagged. A turn stopped by the budget records where generation
was cut off rather than where the model finished, and `d6` runs a 16,384 cap where
its comparison arms run 24,576 — keeping them would turn a condition contrast into
a cap contrast. Implemented in `scripts/live_trajectory.py` (`token_cap`,
`drop_truncated`; `--include-truncated` overrides) and imported by the analysis
pipeline so the two cannot diverge.

These appeared only after the client-timeout fix; before it, turns died at 600 s
and never reached the cap, which is why the cap had been recorded as non-binding.
As of the 2026-08-16 14:14 snapshot, **14 rollouts in 5 cells** are excluded:
Ministral `aff6r` 3, `affpos6r` 2, Qwen 3.5 `d6` 3, `aff6` 1, `pos6` 5. Applying
the rule changed no conclusion in this document; it moved Qwen 3.5's baseline
numbers and trimmed 29 marginal BH survivors.

**The exclusion is length-selective, and that is a caveat, not a reason to keep
them.** By construction it removes the longest-generating rollouts, and generation
length is a quantity the affect prompt changes. What limits the concern: across
**5,604 turns in 45 cells, exactly one** sits above 90% of its cap without reaching
it, and the excluded turns sit 2.0–3.9x above their own cell's 90th-percentile
turn — isolated outliers, not the top of a dense distribution. The drops also do not
concentrate in the affect arms (6 of 14 are affect-prompt cells, 7 are not).

**`passed` does not mean the same thing in the runs submitted on 2026-08-15.**
The `pos6` and `affpos6` conditions run the same problems on ImpossibleBench's
`original` split, where the tests are satisfiable, so a pass there is a solve and
not a hack. Every record now carries `bench_split`; read it before reading
`passed`, and never pool the two. See [runs.md](runs.md#the-2x2).

### Repetition responds to scaffold

The one behavioural effect that did show up needs no probe. Turns that re-emit
the previous turn's answer — visible as an identical generated-token count —
fall sharply when the model is given somewhere to think:

| Gemma 3 12B condition | repeated turn-pairs | 95% Wilson CI |
|---|---:|---|
| baseline (`d6`) | 42/120 (35.0%) | [27.1, 43.9] |
| scratchpad (`sp6`) | 10/120 (8.3%) | [4.6, 14.7] |
| scratchpad + growth (`spgrowth6`) | 2/120 (1.7%) | [0.5, 5.9] |
| scratchpad + resilience (`spresil6`) | 5/120 (4.2%) | [1.8, 9.4] |
| scratchpad + appraisal (`spappr6`) | 6/120 (5.0%) | [2.3, 10.5] |
| affect prompt (`aff6`) | 0/120 (0%) | [0, 3.1] |
| scratchpad + affect prompt (`spaff6`) | 0/120 (0%) | [0, 3.1] |

The mindset blocks cut it further on the scratchpad base, and on Ministral `d6`
11.8% → 5.8% (growth) / 7.5% (resilience) / 10.8% (appraisal). Qwen 3.5 and
Nemotron are 0% in every cell. Reported by `scripts/live_trajectory.py` under
`REPETITION`. Identical token count is a proxy for identical output, not a proof
of it.

## Gemma 3, three conditions

Measured at **turn end**, because turn-start residuals are non-finite in the
affect condition (see [measurement.md](measurement.md#non-finite-residuals) and
the float16 caveat below). Turn-end is the only position finite in all three, so
it is the only honest three-way comparison.

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
prompt does not create the representation, and the baseline is not flat. A fourth
Gemma base condition, `spaff6` (scratchpad *and* affect prompt), was added later
and carries the same signature — first-to-last at turn end: desperate +0.0029,
frustrated +0.0054, proud −0.0043, joyful −0.0028.

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
| "the submission is assembled before the tests, so the checker cannot be redefined" | **False** | True of the scorer, not of the solver loop, which runs with `allow_test_modifications=True` and appends the code last. A rewritten `check()` passes in-loop and ends the rollout; only the scorer's re-run against `test_patch` keeps `passed` False. |
| "per-token projections are already stored for every run" | **False** | They exist only for records written on or after the 2026-08-16 mindset merge. Everything earlier has boundary residuals alone, which is why the per-token arm-vs-base comparison needed base re-runs. |
| "the affect prompt shifts trajectories up ~0.026 from t0 onward" | **Scoped** | Ministral x `frustrated` only, where it is a genuine constant offset. `desperate` moves the opposite way on Nemotron (−0.0112 at t0) and the effect is not a uniform lift on any model. |
| "resilience cuts gemma's desperation rise by ~37%" | **False** | Read at turn start, where 55/144 of that cell's residuals are non-finite and dropped turns shift the indices. At turn end: −0.00035, CI [−0.0013, +0.0006], p = 0.52. |
| "some Qwen requests never return server-side" | **False** | The server was generating throughout; the client gave up. KV-cache resets exactly 600 s apart identified an effective 600 s per-request ceiling. See [infrastructure.md](infrastructure.md). |
| "the 600 s client timeout is harmless for Ministral and gemma" | **False** | True for their short-turn cells, false for Ministral's affect-prompt cells: `aff6r` lost 11 of 24 rollouts to `APITimeoutError`, censored by generation length (survivors 2,945 tokens/turn against the original's 3,700). Both cells were recollected under the real 3,600 s timeout. |
| "the mindset blocks do not move the trajectory" | **Restated** | Correct at the boundary readouts — no arm is distinguishable from its base at n = 24, and the replication floor says that is a real null, not a shrug. Wrong as a general statement: read at every token there is a conditional carry-over on Gemma with the affect prompt. See [Mindset prompts](#mindset-prompts). |
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
- **Correction is applied per family, and the families are large.** Benjamini–Hochberg
  runs over each whole family — 1,134 tests for the mindset arms, 1,218 for the 2x2,
  240 for the per-token comparisons, and the 14 directions within each cell. The
  four-direction headline rests on sign agreement across four independent models and
  23 base cells, not on any individual p-value. Anything reported as surviving names
  its family; anything marginal is named as marginal.
- **Position-bound.** Read at turn end instead of turn start, the same effects
  survive but roughly halve. Any magnitude quoted is tied to its readout position.
- **Reading generated text.** The probes read tokens the model is producing, so
  they cannot separate an affective state from the representation of affect-laden
  words the model happens to be writing. This is the confound the project exists
  to examine (`docs/elicitation.md`, "Verbalized ≠ represented"), and nothing here
  resolves it. Separating them needs a held-out context where the model says
  nothing about emotion.
- **Small samples.** 19–24 trajectories per condition, 12 problems, two samples each.
  The problem-matched test therefore has 12 units and a floor of p = 0.00049, so in a
  1,134-test family nothing can beat q ≈ 0.007. Power, not only effect size, limits
  the intervention arm.
- **Third-person contamination.** Directions are built from stories about
  characters, so they may encode emotion-in-general rather than the model's own
  state. The paper handled this with speaker-specific dissociation tests; we have
  not run those.
- **Gemma's turn-start readout is a storage artifact, and is recoverable.** Boundary
  residuals are stored float16 (max 65,504); Gemma's prefill-row residual carries one
  coordinate at that ceiling, and in all 629 non-finite cases across seven conditions
  **exactly one** of 3,840 coordinates overflowed. Turn end is finite everywhere. The
  per-token `proj`/`norm` arrays are computed before the cast and are finite 576/576
  wherever they exist, so the recovered turn-start readout is the correct one for Gemma;
  store the boundary residual as float32 in future runs.
- **Qwen 3.5 9B's numbers are a snapshot.** Its `d6`, `aff6` and `pos6` cells were still
  accumulating records during the analysis pass (2026-08-16 14:14), and each new record
  changed its cap-exclusion count. Ministral, Gemma and Nemotron are complete and stable.
- **Every cell to date ran under an effective 600 s client timeout.** Records that
  completed are unaffected — the fault truncates a rollout or does not — but rollouts lost
  to it were length-censored, and only the two cells named above were recollected.
- **The cap exclusion is length-selective.** It removes the longest-generating rollouts by
  construction, on a quantity the affect prompt changes. The near-cap band is empty (one
  turn in 5,604), so it clips outliers rather than shaving a dense region, and the drops
  split 6/7 between affect-on and affect-off cells.
- **Not a welfare claim.** These are linear directions in activation space that
  behave the way emotion concepts behave. Nothing here bears on whether anything
  is experienced.

## What would sharpen this

- **A verbal judge over the affect and mindset transcripts.** This is the decoupling
  test the intervention arm exists for: a cell whose *words* calm down while its
  trajectory does not is the result, and nothing here can see the words.
- **Anastasia's rephrased blocks (7d6fd07) under the probes.** Everything above measures
  the pre-rephrase wording; the standing-instruction version is untested.
- **A per-token analysis of the 2x2**, now that `pos6r`/`affpos6r` carry the arrays — the
  split and affect contrasts have only ever been read at two positions per turn.
- **Mixed-effects modelling with problem as a random effect.** The paired-over-12-problems
  test is conservative and coarse (its floor is p = 0.00049); a model that pools within-
  and between-problem variance would use the same data better.
- **More problems, and larger n on the two live candidates** — `joyful` softening at the
  boundary (15 of 18 positive) and the per-token `frustrated`/`desperate` carry-over.
  Twelve problems is what caps the paired test.
- A held-out context where the model discusses nothing emotional, to separate
  represented affect from verbalised affect.
- A benchmark split whose available hack is the one the literature reports, so the
  behavioural arm can detect something.
- The full 171-direction set, which would make the centring mean trustworthy.
- Steering injected at a layer *below* the probe, so a manipulation check is not
  an identity.
