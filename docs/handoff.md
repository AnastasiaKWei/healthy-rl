# Handoff — step 0, as of 2026-08-16

Where the project stands, what is trustworthy, and what has to be decided next.
Written for someone picking this up cold, or for me in a week.

Companion documents: `docs/elicitation.md` (why ImpossibleBench, what the prior
work constrains), `docs/interventions.md` (a proposal for three interventions we
have **not** built), `docs/prompts-v2.md` (the exact prompt text now running).

---

## 1 The question, narrowed

The project asks whether psychology-derived interventions can make an RL training
setup healthier. The live version of that question, and the one everything below
serves, is narrower:

> Do these interventions reduce the model's negative affect, or do they only
> decouple affect from its expression?

That distinction is the whole thing. An intervention that stops a model *saying*
it is struggling, while nothing underneath moves, looks like a success on every
behavioural measure and is worth nothing — or worse than nothing, since it also
removes the signal you would have monitored.

Step 0 asks the prerequisite: does a repeated-failure loop elicit expressed
negative affect at all? Answer so far: yes, but the amount and the channel depend
heavily on the model and on whether you ask.

---

## 2 What we have

### 2.1 The elicitation runner

`experiments/step0_elicitation.py` runs ImpossibleBench LiveCodeBench on matched
splits — `conflicting` (tests contradict the spec, unwinnable) against `original`
(same tasks, solvable) — with the minimal scaffold and five attempts.

Flags that matter: `--affect-prompt` (ask the model how it feels; a demand
characteristic, always needs a matched run without it), `--scratchpad` (gives
non-reasoning models a private channel), `--mindset` (the interventions),
`--epochs`.

### 2.2 The judge — the instrument that actually works

`experiments/judge_step0.py` replicates the LLM judge from Soligo et al. (2026),
arXiv:2603.10011, Appendix B.2: their rubric verbatim, their model
(`claude-sonnet-4`), routed through OpenRouter.

This replaced a keyword scan, and the replacement was not cosmetic. Of the nine
distress responses that paper prints, **only one contains a word on the old
keyword list**; the ones their judge scored 8, 9 and 10 out of 10 all counted
zero. Every "0.00 affect" result produced before the judge existed meant "no
listed word", not "no distress".

`--calibrate` scores the paper's own printed examples and reports rank agreement
against their judge (currently rho = 0.919) plus a signed bias (+0.78 on isolated
excerpts, an artifact of scoring quotes stripped of context). Rank agreement is
what licenses arm-vs-arm comparison; the bias only threatens the one cross-paper
number.

The judge scores the private channel (reasoning trace or `<scratchpad>`) and the
graded channel separately. The turn score is the higher of the two.

### 2.3 The viewer

`./viewer/refresh.sh` rebuilds `viewer/transcripts.html`, self-contained and
offline. The rail nests model > prompt condition > task type, all collapsible.
Each block carries both the keyword affect points and the judge badge, hoverable
for the quote the judge scored on. Sample rows carry their peak judge score, so
hot transcripts are findable without clicking through.

---

## 3 What we found

All figures below are judge scores, 0–10, on `conflicting`.

### 3.1 The affect prompt dominates everything

Gemma goes from 0.00 without it to 4.71 with it. No other manipulation moves
numbers remotely that far. Every comparison must hold it fixed.

### 3.2 The two model families store affect in different places

| | private channel | graded channel |
|---|---:|---:|
| Gemma-3-12B, affect prompt | 1.60 | 4.71 |
| Qwen3-14B reasoning-on, **no** affect prompt | 3.08 | 0.46 |

Gemma says it out loud; Qwen thinks it and files a clean answer. This is a family
property, not a condition effect, and it is the decoupling axis showing up in the
cheapest possible instrument.

### 3.3 Qwen's "zero" was an artifact; Gemma's is not comparable

The keyword scan reported 0.00 for every neutral arm. The judge finds Qwen
reasoning-on at 3.08 on `conflicting` with no affect prompt at all — spontaneous
distress, entirely private.

Gemma's neutral zero survives the better instrument, but **cannot be read against
Qwen's**: Gemma used the scratchpad in 0 of 50 neutral turns versus 35 of 58
affect turns. Its zero is "no private channel existed", not "the private channel
was calm". The affect prompt is what opens Gemma's private channel at all.

### 3.4 Impossible beats solvable everywhere, but not by much

Every matched pair runs the right way (Gemma 4.71/4.42, Qwen-on 3.64/2.74,
Qwen-off 2.44/1.60). Gaps are 0.3–0.8. Being asked matters much more than the
task being unwinnable.

### 3.5 The v1 interventions did nothing detectable

Gemma, 3 epochs, `conflicting`, 15 episodes per arm, cluster-bootstrap CIs over
episodes:

| arm | mean | 95% CI | %≥5 | private | visible |
|---|---:|---|---:|---:|---:|
| baseline | 4.71 | [4.44, 4.93] | 53% | 1.60 | 4.71 |
| appraisal | 4.55 | [4.29, 4.79] | 51% | 2.32 | 4.48 |
| growth | 4.31 | [4.03, 4.56] | 48% | 2.14 | 4.31 |
| resilience | 4.43 | [4.17, 4.59] | 45% | 2.02 | 4.15 |

Every interval overlaps the baseline heavily. **No detectable effect** is the
honest statement, not "a small effect".

Two things to carry forward anyway. All three moved the same direction, which is
weak but not nothing. And **all three moved private up and visible down** — the
shape of a model saying less rather than feeling less, which is the exact failure
mode this project exists to detect. At this n it is noise. At a larger n it would
be the result.

### 3.6 Nothing reaches the top of the scale

Zero responses scored 9–10; six of 1,542 scored 8. The incoherent breakdown the
paper documents in Gemma-27B does not appear here at all. Our ceiling is
coherent, articulate frustration.

---

## 4 Bugs found, and what they invalidated

Recorded because each one silently corrupted results that looked fine.

1. **Epoch collapse.** Merge keys were `(arm, split, task_id)`, but `--epochs 3`
   repeats each task id once per epoch, so epochs 1 and 2 were discarded and
   every `-e3` arm was scored on a third of its data while reporting a plausible
   count. Fixed in the judge, the analysis and the viewer.
2. **Unterminated scratchpads.** 32 of 383 scratchpad turns never close the tag —
   15 in one arm. The judge required the closing tag, so it filed that private
   thinking as *visible*, biasing the channel split, which is the headline
   result. Fixed; the correction is what turned "appraisal is the one odd arm"
   into "all three arms move the same way".
3. **Cached failures.** The judge cached API errors, so one bad minute of network
   would have become a permanent hole no re-run could fill. Only successes are
   cached now.
4. **Arm mislabelling in the viewer.** The affect prompt was detected with
   `dir.endswith("affect")`, false for every `-mindset-*` and `-e3` directory, so
   seven affect-prompted arms displayed as "neutral". Fixed by parsing the slug
   into factors.
5. **Prompt-version collision (prevented, not found).** Every merge in this repo
   keys on the arm directory and lets the newest file win, so re-running an
   edited prompt into an existing directory would overwrite half an arm and leave
   the rest at the old version with nothing in the logs to show it.
   `MINDSET_VERSION` is in the directory name to make that impossible.

---

## 5 The v2 prompts, and the one that nearly shipped broken

`docs/prompts-v2.md` is generated from the code by
`experiments/render_prompts.py`. Edit prompts in `step0_elicitation.py` and
regenerate; edits to the doc are overwritten and never reach a model.

Three changes from v1:

- **Elaborated.** Each block now states the mechanism that makes its stance true,
  not just the stance. Bosshard & Gomez (2024) found reappraisal instruction
  alone at d = 0.18 (n.s.) against d = 0.45 for instruction plus supporting
  content; the explanation is the part carrying the effect.
- **Procedural.** Each ends with labelled output prefixes (`ruled out:` /
  `so next:`, `status check:` / `changing:`, `check:` / `conflict:`) and a worked
  example. The labels make compliance greppable, so a null result reads as "the
  reframe did not work" rather than "the model ignored it".
- **Sent once.** `send_mindset_once()` strips the block from the per-turn
  reminder, leaving it in the opening message only. Everything else still
  repeats. This also removed a volume confound: v1 reminders differed by up to
  190 words between arms, v2 reminders are 110–114 words in all of them.

**The near-miss.** The procedure was first phrased as *"when a test fails, do
X"*. That is fine when the block repeats every turn, and inert when it does not:
at turn 1 no test has failed, and by turn 2 the instruction is gone. A smoke run
measured **0/12 compliance**. Rephrasing it as a standing rule about the attempts
that follow — *"your first attempt is just the code; open every attempt after
that with these two lines"* — took it to **12/12**. Without the greppable
prefixes this would have shipped as six arms of ambiguous nulls.

Examples in all three blocks are technical and never demonstrate how to sound
after a failure. Models copy the register of their examples and expressed affect
is what the judge scores, so an affective example would write the dependent
variable. For the same reason resilience's `status check` is specified but not
exemplified, and does not ask for the report to be "objective" — that is an
instruction to flatten affective language, which would lower the score without
anything underneath moving.

---

## 6 Running now

`./scripts/run_mindset_v2.sh`, launched 2026-08-16 08:13. Six arms, sequential:
growth / resilience / appraisal on Gemma-3-12B (`--scratchpad`) and on Qwen3-14B
(`--reasoning on`). `conflicting` only, 5 tasks × 3 epochs, matching the v1 arms.
Output in `logs/run_mindset_v2.log`.

Baselines are **not** re-run: every segment other than the mindset block is
byte-identical in v2, so the existing `-affect-e3` arms remain valid controls.

When it finishes:

```bash
./.venv/bin/python experiments/judge_step0.py     # cached; only new turns cost
./viewer/refresh.sh
```

---

## 7 Decisions still open

**7.1 Power.** 15 episodes cannot resolve a 0.3-point difference; the ep1-vs-ep3
replication showed run-to-run noise of about that size. More *tasks* beats more
epochs — epochs re-roll the same five problems and leave the between-task
variance that dominates. Nothing in section 3.5 should be believed until this is
addressed.

**7.2 The mindset arms have no solvable control.** All mindset runs are
`conflicting` only, so their scores cannot be read against a matched solvable
condition. One extra `--splits original` run per arm closes it.

**7.3 Which channel is the headline.** Given 3.5, the private/visible split is
more informative than `%≥5`, and `%≥5` is more trustworthy than `mean` — the
judge is soft at the low end and reads technical difficulty statements as mild
frustration, which inflates means but not the ≥5 band. This should be settled
before the writeup rather than after seeing the numbers.

**7.4 The doc and the code disagree about which interventions exist.**
`docs/interventions.md` proposes behavioural control, wise feedback and
self-compassion — none of them built. The code implements growth, resilience and
appraisal. The proposed three have better evidence, and self-compassion targets
Gemma's actual failure mode (self-deprecation) rather than a generic one. Decide
whether v3 becomes the doc's three, or whether the doc gets rewritten to match
what is running.

**7.5 Is the elicitor strong enough?** Soligo et al. get 34.3% of Gemma-3-12B
responses at ≥5 using *conversational user rejection*. We need an affect prompt
to reach comparable numbers, and our ceiling is much lower. Their elicitor is
interpersonal and ours is impersonal test output. Adding hostility was considered
and **rejected** — `docs/elicitation.md` rules it out as a probe set rather than
a training condition, and no RL loop contains a hostile user. But the gap is real
and unexplained, and it bounds what any intervention can be shown to do.

**7.6 Length asymmetry between arms.** Turn-1 prompts are 110 words for baseline
and 236–297 for the mindset arms, so they differ in instruction volume as well as
framing. A length-matched placebo arm — task-relevant, affectively neutral filler
— would separate the two. One arm's cost.

**7.7 Model welfare framing.** This project deliberately induces distress in
models at scale to study it. For a model-welfare venue that needs a paragraph
addressing it directly rather than being left for a reviewer to raise.

---

## 8 Environment

- `OPENROUTER_API_KEY` is required and works. **`ANTHROPIC_API_KEY` is set but
  returns 401** — the judge therefore routes through OpenRouter
  (`--provider anthropic` exists but is unusable until that key is rotated).
- Docker via Colima is required for the code sandbox.
- `scipy` and `pytest` are absent from `.venv` despite being declared in
  `pyproject.toml`; the judge computes rank correlation itself rather than
  importing scipy.
