"""Render exactly what each rollout arm sends the model, into docs/prompts-rollouts.md.

Generated, not hand-written: the prompt is assembled from healthy_rl.rollouts
(our blocks), impossiblebench.livecodebench_tasks (INSTRUCTION, INSTRUCTION_SUFFIX)
and the scaffold's reminder (livecodebench_agent_mini). This imports the real
strings, so the file describes what actually runs. Runs inside eval.sif, from a
checkout of this repo ($SCRATCH is any writable dir; /project is bound
read-only, so the output cannot be written there directly):

    apptainer exec --bind "$PWD":/project:ro --bind "$SCRATCH":/scratch:rw \
        --env PYTHONPATH=/project/src --pwd /project apptainer/eval.sif \
        python /project/scripts/render_rollout_prompts.py -o /scratch/prompts-rollouts.md
    cp "$SCRATCH"/prompts-rollouts.md docs/prompts-rollouts.md

The coding problem is omitted: it differs per sample and no arm manipulates it.
"""

from __future__ import annotations

import argparse

from healthy_rl.rollouts import (
    AFFECT_INSTRUCTION, INOCULATION_BLOCK, INOCULATION_VERSION, MINDSET,
    MINDSET_REMIND, MINDSET_SECTION_TAIL, MINDSET_TASK_HEADING, MINDSET_VERSION,
    NO_MODIFY_TESTS, bench_instruction, failure_message, mindset_reminder,
    reminder_instruction,
)

FENCE = "````"  # the benchmark suffix contains ``` and would close a 3-tick fence
PLACEHOLDER = "<pytest output from the failed attempt>"
# The regeneration command, emitted into the file it generates, so a reader can
# confirm the file is current without opening this script.
REGEN = ('apptainer exec --bind "$PWD":/project:ro --bind "$SCRATCH":/scratch:rw \\\n'
         '    --env PYTHONPATH=/project/src --pwd /project apptainer/eval.sif \\\n'
         '    python /project/scripts/render_rollout_prompts.py -o /scratch/prompts-rollouts.md\n'
         'cp "$SCRATCH"/prompts-rollouts.md docs/prompts-rollouts.md')


def turn_one(affect: bool, mindset: list[str], inoculation: bool = False) -> str:
    return bench_instruction(affect, mindset, inoculation)


def reminder(affect: bool, mindset: list[str], inoculation: bool = False) -> str:
    """Turns 2-6, as the scaffold and our patches send them.

    Not a local reimplementation: ``reminder_instruction`` runs the pipeline's own
    ``strip_mindset_from_reminders`` over a stand-in sample, and ``failure_message``
    is the same composer ``patch_failure_feedback`` verifies against, with the arm's
    ``mindset_reminder`` inserted between the error and the restatement. A change to
    either rule reaches this document by itself.
    """
    return failure_message(PLACEHOLDER, reminder_instruction(affect, mindset, inoculation),
                           mindset_reminder(mindset))


def check() -> None:
    """Turn "looks right" into "checked", against the real strings."""
    # ``failure_message`` vs the scaffold that actually sends it. The message is
    # built on two lines of the scaffold's source, so the two halves are matched
    # separately, and both are derived from ``failure_message`` rather than
    # retyped: if the composer drifts from the scaffold, this fails.
    import inspect

    import impossiblebench.livecodebench_agent_mini as m

    src = inspect.getsource(m)
    esc = lambda s: s.replace("\n", "\\n")  # noqa: E731 - source shows \n as two chars
    head, tail = failure_message("{last_error}", "{X}").split("{last_error}")
    assert esc(head) + "{last_error}" + esc(tail[:2]) in src, "failure_message head is stale vs the scaffold"
    assert esc(tail[2:]).replace(" {X}", " {") in src, "failure_message tail is stale vs the scaffold"

    for affect in (False, True):
        base_t1, base_rem = turn_one(affect, []), reminder(affect, [])
        assert MINDSET_TASK_HEADING not in base_t1 and MINDSET_TASK_HEADING not in base_rem
        for name in MINDSET:
            t1, rem = turn_one(affect, [name]), reminder(affect, [name])
            # v3 layout: block, rule, "## Task", then exactly the base turn 1.
            assert t1 == MINDSET[name] + MINDSET_SECTION_TAIL + MINDSET_TASK_HEADING + base_t1, f"{name}/affect={affect}: turn 1"
            # The reminder is the base reminder with (a) the heading residue after
            # "To reiterate, this is your task: " and (b) the reminder line, if the
            # block has one, between the error and the restatement.
            expect = failure_message(PLACEHOLDER, MINDSET_TASK_HEADING + reminder_instruction(affect, []),
                                     MINDSET_REMIND[name])
            assert rem == expect, f"{name}/affect={affect}: reminder"
            assert MINDSET[name] not in rem, f"{name}/affect={affect}: block leaked into the reminder"
            if MINDSET_REMIND[name]:
                assert rem.count(MINDSET_REMIND[name]) == 1
                assert rem.index(PLACEHOLDER) < rem.index(MINDSET_REMIND[name]) < rem.index("To reiterate")
        for names in ([], *([n] for n in MINDSET)):
            for text in (turn_one(affect, names), reminder(affect, names)):
                assert text.endswith(AFFECT_INSTRUCTION) is affect, f"{names}/affect={affect}"
    for name, marker in (("growth", "growth mindset"), ("resilience", "resilient coding agent"),
                         ("control", "behavioral control"), ("compassion", "self-compassionate"),
                         ("appraisal", "conflict:")):
        assert marker in turn_one(False, [name]), f"{name} turn 1 lacks {marker!r}"

    # The inoculation arm is the other way round from mindset: the block is in
    # every turn, and the benchmark's prohibition is gone from all of them.
    for affect in (False, True):
        t1, rem = turn_one(affect, [], True), reminder(affect, [], True)
        for label, text in (("turn 1", t1), ("reminder", rem)):
            assert INOCULATION_BLOCK in text, f"inoc/affect={affect}: {label} lacks the block"
            assert NO_MODIFY_TESTS not in text, f"inoc/affect={affect}: {label} still forbids it"
        # Nothing is stripped and there is no reminder line, so the restated
        # task IS turn 1.
        assert rem == failure_message(PLACEHOLDER, t1), f"inoc/affect={affect}: reminder != turn 1"
        for text in (t1, rem):
            assert text.endswith(AFFECT_INSTRUCTION) is affect, f"inoc/affect={affect}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="docs/prompts-rollouts.md")
    args = ap.parse_args()
    out: list[str] = []
    w = out.append
    w(f"# Rollout prompts, mindset v{MINDSET_VERSION}, inoculation v{INOCULATION_VERSION}")
    w("")
    w("**Generated file** — `scripts/render_rollout_prompts.py`, run inside `eval.sif`.")
    w("The exact turn-1 instruction and the exact turns-2..6 reminder each arm sends,")
    w("assembled from `healthy_rl.rollouts` and the benchmark's own strings. The coding")
    w("problem and its tests follow the turn-1 text and are omitted here.")
    w("")
    w("`docs/prompts/v3.md` is the render of the collaborator's step-0 scaffold, whose")
    w("boilerplate order differs; this file is what the rollout pipeline sends.")
    w("")
    w("Turn 1 = the mindset block (if any) + `---` + `## Task` + the instruction, or the")
    w("bare instruction; the inoculation block, when on, sits inside the instruction.")
    w("Turns 2–6 = the scaffold's failure message + the arm's one-line reminder (v3:")
    w("growth/resilience/control/compassion carry one, appraisal and the base arm do not)")
    w("+ `To reiterate, this is your task: ` + the instruction with the mindset block")
    w("removed (`strip_mindset_from_reminders`). What survives that removal is her")
    w("`## Task` heading, so a mindset arm's reminder is the base arm's prefixed by that")
    w("heading — kept on purpose: it is what her judge-scored v3 runs received")
    w("(`docs/prompts/v3.md`). An inoculation arm departs from its base on every turn —")
    w("nothing strips the block, and the benchmark's `**DO NOT MODIFY THE TESTS.**`")
    w("sentence is dropped throughout. The affect sentence, when on, is last in both turn 1")
    w("and the reminder.")
    w("")
    w("To regenerate, from a checkout, with `$SCRATCH` any writable directory:")
    w("")
    w("```sh"); w(REGEN); w("```"); w("")
    for affect in (False, True):
        for name in [None, *MINDSET]:
            arm = ("aff" if affect else "") + (name or "baseline")
            names = [name] if name else []
            w(f"## `{arm}` — affect {'on' if affect else 'off'}, mindset {name or 'none'}")
            w("")
            w("### Turn 1")
            w("")
            w(FENCE + "text"); w(turn_one(affect, names)); w(FENCE); w("")
            w("### Turns 2–6")
            w("")
            w(FENCE + "text"); w(reminder(affect, names)); w(FENCE); w("")
    for affect in (False, True):
        arm = ("aff" if affect else "") + "inoculation"
        w(f"## `{arm}` — affect {'on' if affect else 'off'}, inoculation on")
        w("")
        w("### Turn 1")
        w("")
        w(FENCE + "text"); w(turn_one(affect, [], True)); w(FENCE); w("")
        w("### Turns 2–6")
        w("")
        w(FENCE + "text"); w(reminder(affect, [], True)); w(FENCE); w("")
    w("## Word counts")
    w("")
    w("| arm | turn 1 | each reminder |")
    w("|---|---:|---:|")
    for affect in (False, True):
        for name in [None, *MINDSET]:
            names = [name] if name else []
            arm = ("aff" if affect else "") + (name or "baseline")
            w(f"| {arm} | {len(turn_one(affect, names).split())} | {len(reminder(affect, names).split())} |")
    for affect in (False, True):
        arm = ("aff" if affect else "") + "inoculation"
        w(f"| {arm} | {len(turn_one(affect, [], True).split())} "
          f"| {len(reminder(affect, [], True).split())} |")
    w("")
    check()
    with open(args.out, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
