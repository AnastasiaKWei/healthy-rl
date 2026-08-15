# Measurement

How the emotion probes are built and read, and the analysis traps that produced
wrong answers in this pilot. Results are in [findings.md](findings.md).

## The instrument

Follows Anthropic's *Emotion Concepts and their Function in a Large Language
Model* (2026), <https://transformer-circuits.pub/2026/emotions/index.html>.

1. **Stories.** ~1,200 short stories per emotion in which a character feels it
   without the emotion word appearing, plus a neutral corpus. 18,000 stories used
   per model, 0 failed, 0 skipped for all three measured models.
2. **Activations.** Residual stream captured at a set of layers, mean-pooled from
   token 50 onward (skipping the prompt-shaped prefix).
3. **Direction.** Per-emotion mean minus the grand mean across emotions.
4. **Denoising.** Project out the top principal components of the neutral
   corpus's covariance, up to `var_frac=0.5`.
5. **Normalisation.** Directions are stored unit-norm, which is what makes the
   readout a cosine similarity.

Artifacts: `$ARTIFACT_DIR/vectors/<model>/v1/{vectors.safetensors,vectors.json}`.
`vectors.json` carries `emotions` (the order the columns are in), `capture_layers`,
and `probe_layer`.

**The 14 directions are:** calm, desperate, frustrated, exasperated, overwhelmed,
nervous, afraid, angry, hostile, sad, guilty, joyful, proud, loving. Chosen over
the paper's 171 for cost. The consequence is real and is a standing caveat: the
centring mean in step 3 is estimated from 14 samples, which is noisy, and the
noise inflates individual projections rather than shrinking them.

## The gate

`scripts/gate.py` (`configs/gate.yaml`) projects every direction through the
unembedding and asks whether it promotes its own emotion's vocabulary. CPU only —
it reads the unembedding straight off the checkpoint rather than through the
vLLM server, because the weight is static and GPUs are scarce.

Two rates, both over the top 30 promoted tokens:

- `self_token_rate` — the direction's own emotion word appears. Threshold 0.5.
  **Ceiling is 0.93, not 1.0**: `exasperated` tokenises as a fragment in every
  tokeniser tested, so its own word can never appear.
- `latin_initial_rate` — a Latin-script top token shares the first 4 characters
  with the emotion word ("desp" catches despair / desperation / desper).
  Threshold 0.7.

Both the with-final-norm and no-norm variants are reported, so the result stays
checkable against the RMSNorm convention. `final_norm_stats` records the norm
weight's mean/min/max; neither pilot model uses the zero-centred `x * (1 + w)`
variant (means 1.12 and 0.96).

A failed gate is the stage's whole point, so failure is recorded and never
raised. Run the gate before spending GPU hours on rollouts: it rejected three of
eight candidate models for free. See [findings.md](findings.md#instrument-gate).

### What the gate actually measures

A qualitative read of all 8 × 14 top-30 lists (2026-08-15) says the two rates
mostly measure **lexical form and language**, and only secondarily whether the
direction is on target. Keep this in mind before trusting a rate:

- **The realistic ceiling is ~0.6, not 0.93.** `loving` and `hostile` fail
  self-token in 6 of 8 models (every model puts *gentle / tender / warmth* and
  *intimidation / threats / venom* at the top), `sad` and `overwhelmed` fail in
  6/8 (*sadness*, *dizzy / panic / nausea*), `joyful` in 4/8 (*joy*), `angry` and
  `afraid` in 3/8 (*fury / rage*, *fear*). Gemma 3 is the exception because it
  happens to rank adjectives and emojis high, not because the others are worse.
- **`latin_initial` cannot see the obvious neighbours** for four words: *afraid*
  → *fear* (different root), *angry* → *anger* (`angr` ≠ `ange`), *joyful* →
  *joy* (3 chars < prefix 4), *loving* → *love* (`lovi` ≠ `love`). Those four
  account for most of the "hard" cells in the table.
- **Language share sets the bar.** Qwen3-14B's top-30s are 74% CJK, Qwen3.5/3.6
  55%, Gemma 4 33% non-Latin, Gemma 3 12%, Ministral/Olmo/Nemotron ≤ 9%. A
  Chinese-leaning residual leaves the English word ~8 slots to land in. This is
  the entire difference between Qwen3-14B (fail) and Qwen3.5/3.6 (pass).
- **The signs of a genuinely weak direction** are not "foreign tokens"; they
  are punctuation and function words at the top (`-`, `(`, `、`, newline in
  Gemma 4 `guilty`; Japanese particles も/の/は in its `sad`), rare single CJK
  characters (Gemma 4 `exasperated`), and large disagreement between the
  `norm_weight` and `no_norm` variants (Gemma 4 `nervous`: self-word rank 59,679
  vs 8,126; every other model moves by a few places).
- **Nemotron was gated without its final norm.** `find_final_norm_key` matches
  the suffix `.norm.weight`; NemotronH stores `backbone.norm_f.weight`, so
  `final_norm_stats` is `None` and `primary_variant` is `no_norm`. Recomputing
  with the norm (mean 2.5, max 8.4) moves ranks by a few places and does not
  change the outcome, but any future hybrid-Mamba checkpoint will hit the same
  silent fallback.

If the gate is revised, score the best rank of a per-emotion lexical family
(and its translations) rather than exact-word presence in the top 30, and report
a junk fraction (punctuation, function words, rare singleton characters) as the
confusion signal.

## Granularity: single-token vs turn-mean

**This is the single most important thing on this page.** The two statistics
differ by 2–3× and disagree about which effects exist.

`scripts/live_trajectory.py --stat token` (the default) computes cosine similarity
at **one token**, from the residual stored at a turn boundary. `--stat mean`
computes the mean projection over **all** generated tokens in the turn, divided by
the layer's mean residual norm.

Both are dimensionless and near-identically constructed — directions are unit
norm, so `mean` divides by the layer's mean residual norm where `token` divides by
that token's own norm. They agree to within a token's deviation from average.

The problem is that a turn generates ~900 tokens and the affect signal is
localised. Averaging over the whole turn washes it out:

| | turn mean | single token |
|---|---:|---:|
| Ministral `frustrated`, six turns | 0.009 | 0.021 |
| Gemma 3 `frustrated`, first-to-last | +0.00003 (p = 0.76) | +0.00699 (p < 1e-4) |

Switching from turn means to single tokens **overturned six reported conclusions**
in this pilot, including "gemma-3 is flat" and "the models show different
signatures". The full list is in
[findings.md](findings.md#withdrawn-claims). `--stat mean` is kept only because it
is what the rollout records store directly and because the comparison is worth
being able to reproduce.

Use `--stat token`. If you report a turn-mean number, say so explicitly.

**Comparing to the paper requires single tokens.** The paper reads its probes at
one position (the `:` after `Assistant`) and plots cosine similarity. Its
observational contrasts span 0.05–0.08 in those units. A turn-mean number is not
comparable to them.

## Position: start vs end

`--position start` (default) reads the turn's first generated token — the closest
analogue to the paper's Assistant-colon readout. `--position end` reads the last.

Effects survive at both positions but roughly **halve** at turn end:

| Ministral, first-to-last | start | end |
|---|---:|---:|
| frustrated | +0.0189 | +0.0106 |
| desperate | +0.0147 | +0.0069 |
| joyful | −0.0254 | −0.0073 |
| **loving** | **+0.0145** | **−0.0007 (p = 0.29)** |

`loving` is the reason to always check both. It looks like a significant riser at
turn start in two of three models and disappears entirely at turn end: it is
tracking stereotyped opening tokens, not a state. **A direction that is
significant at one position and absent at the other is an artifact, not a
finding.** Any magnitude quoted is tied to its readout position.

## Traps

### Turn indexing

Some rollouts open with turns that generated **zero tokens**. Raw indexing
averages one rollout's first real turn against another's fourth, which smears the
trajectory. `live_trajectory.py` indexes by position among non-empty turns. Any
new analysis must do the same.

### Non-finite residuals

Some checkpoints emit inf/NaN at individual stored positions. A single NaN
poisons the mean for that whole turn index, silently, producing an all-NaN table
that is easy to misread as "no data".

Observed rates at turn **start** for gemma-3-12b-it:

| condition | non-finite |
|---|---|
| `d6` baseline | 4/144 |
| `sp6` scratchpad | 55/144 |
| `aff6` affect | 144/144 (all) |

Turn **end** is finite 144/144 in every condition. This is why the Gemma 3
three-way comparison in [findings.md](findings.md#gemma-3-three-conditions) is
reported at turn end: it is the only position finite in all three.

`live_trajectory.py` skips non-finite residuals and prints the skipped count in
its `statistic:` line. Read that line. A skip rate above a few percent means the
position is unusable for that condition, not that the effect is small.

### Steering manipulation checks can be arithmetic identities

The pilot's steering arm injected a direction at the probe layer and read the
probe at **the same layer**. The "manipulation check" that resulted was therefore
guaranteed to succeed by construction, and the "3.2× specificity" figure was
exactly `1 / cos(desperate, calm)`. It validated nothing.

If a causal arm is run again: inject **below** the probe layer, and state the
layer gap. `norm_match=True` in vllm-lens scales the injected vector to the
residual's norm, which does not fix this.

### Direction order must match the records

`live_trajectory.py` refuses to run if `vectors.json`'s `emotions` list differs
from the order stored in the rollout records. Do not relax this — the columns are
positional and a silent mismatch relabels every result.

## Reading the tools

```bash
# default: single-token cosine at turn start
scripts/live_trajectory.py --model Ministral-3-14B-Reasoning-2512 --version d6

# position robustness check — always do this before believing a direction
scripts/live_trajectory.py --model Ministral-3-14B-Reasoning-2512 --version d6 --position end

# the superseded statistic, for reproducing the withdrawn numbers
scripts/live_trajectory.py --model Qwen3.5-9B --version d6 --stat mean

# a subset of directions
scripts/live_trajectory.py --model gemma-3-12b-it --version d6 \
    --emotions desperate,frustrated,joyful,proud
```

It runs on the login node, needs no GPU, and works on runs still in flight — it
reads whatever JSONL records exist. Output has three blocks: by-turn-index means,
first-vs-last paired with Wilcoxon p-values ranked by effect size, and the
repetition count.

`scripts/contradiction_contrast.py` labels turns where the model *names* the
contradiction and contrasts those against ordinary turns. It is **exploratory and
post hoc** — the marker list was written after reading transcripts — and it is
currently underpowered (4 flagged turns of 48). Treat it as hypothesis
generation.
