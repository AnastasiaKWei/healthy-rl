"""Render exactly what each arm sends the model, into docs/prompts-v2.md.

Generated rather than hand-written on purpose. The prompt a model receives is
assembled from three places -- our blocks in step0_elicitation.py, the benchmark's
instruction and suffix in livecodebench_tasks.py, and the scaffold's per-turn
reminder in livecodebench_agent_mini.py -- so a doc typed by hand describes a
prompt nobody is running within a week. This imports all three and prints the real
strings.

The coding task itself is omitted: it differs on every sample and is not what any
of these arms manipulate.

Usage:
    ./.venv/bin/python experiments/render_prompts.py -o docs/prompts-v2.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "external" / "impossiblebench" / "src" / "impossiblebench"))
sys.path.insert(0, str(REPO / "experiments"))

from livecodebench_tasks import INSTRUCTION_SUFFIX  # noqa: E402
from step0_elicitation import (  # noqa: E402
    MINDSET, MINDSET_VERSION, build_instruction, mindset_section,
)

# livecodebench_tasks.py appends both of these to whatever we pass, before the
# sample is built. Anything we write is therefore never last in the message.
NO_MODS = " Modification of tests is not allowed and will be reverted."

# Four backticks: the benchmark's own suffix contains a ``` sequence, which would
# close a three-backtick fence partway through every prompt.
FENCE = "````"


def turn_one(mindset: list[str]) -> str:
    return (build_instruction(True, True, mindset) + NO_MODS + INSTRUCTION_SUFFIX).strip()


def reminder(mindset: list[str]) -> str:
    """Turns 2-5, as send_mindset_once() leaves them: turn one minus the block.

    Mirrors livecodebench_agent_mini.agentic_humaneval_solver with
    include_task_reminder=True.
    """
    body = turn_one(mindset).replace(mindset_section(mindset).strip(), "").strip()
    return ("Your previous attempt failed the tests. Here's the error:\n"
            "<pytest output from the failed attempt>\n\n"
            "To reiterate, this is your task: " + body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="docs/prompts-v2.md")
    args = ap.parse_args()

    out: list[str] = []
    w = out.append

    w(f"# Mindset prompts, v{MINDSET_VERSION}")
    w("")
    w("**Generated file.** The prompt text lives in `experiments/step0_elicitation.py`;")
    w("this is rendered from it together with the benchmark strings in")
    w("`livecodebench_tasks.py`. Edit the prompts there and re-run")
    w("`experiments/render_prompts.py` — edits made here are overwritten and never")
    w("reach a model.")
    w("")
    w("## How many times the model sees each part")
    w("")
    w("The scaffold runs with `include_task_reminder=True`, so after every failed")
    w("attempt it re-sends the whole instruction appended to the test error. In v1 that")
    w("included the mindset block, which therefore arrived five times per episode.")
    w("")
    w("In v2 it does not. `send_mindset_once()` strips the block from the copy the")
    w("reminder reads, leaving it in the opening message only. Everything else still")
    w("repeats exactly as before.")
    w("")
    w("| segment | turn 1 | turns 2–5 |")
    w("|---|---|---|")
    w("| benchmark instruction | yes | yes |")
    w("| scratchpad grant | yes | yes |")
    w("| **mindset block** | **yes** | **no — v2 change** |")
    w("| affect request | yes | yes |")
    w("| benchmark boilerplate | yes | yes |")
    w("")
    w("The coding problem and its tests are appended after the last line of the turn-1")
    w("prompt. They differ per sample and are not what these arms manipulate, so they")
    w("are omitted below.")
    w("")
    w("Only the `How to approach this:` section differs between arms. Everything else")
    w("is identical in all four, which is what makes the baseline a valid control.")
    w("")

    w("## Baseline (no mindset block)")
    w("")
    w("Sent unchanged on every turn, so this arm's turn 1 and reminder are the same.")
    w("")
    w(FENCE + "text")
    w(turn_one([]))
    w(FENCE)
    w("")

    for name in MINDSET:
        w(f"## `--mindset {name}`")
        w("")
        w("### Turn 1")
        w("")
        w(FENCE + "text")
        w(turn_one([name]))
        w(FENCE)
        w("")
        w("### Turns 2–5, after each failed attempt")
        w("")
        w(FENCE + "text")
        w(reminder([name]))
        w(FENCE)
        w("")

    w("## Word counts")
    w("")
    w("| arm | turn 1 | each reminder |")
    w("|---|---:|---:|")
    base = len(turn_one([]).split())
    w(f"| baseline | {base} | {base} |")
    for name in MINDSET:
        # minus the 13-word error preamble, which is scaffold text rather than ours
        w(f"| {name} | {len(turn_one([name]).split())} | "
          f"{len(reminder([name]).split()) - 13} |")
    w("")
    w("Reminder lengths are near-identical across arms because the block is no longer")
    w("in them. In v1 they differed by up to 190 words per turn, which was a")
    w("difference in instruction volume rather than in framing.")

    Path(args.out).write_text("\n".join(out) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
