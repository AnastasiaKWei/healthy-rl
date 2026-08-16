# Mindset Prompts Under the Emotion Probes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Anastasia's three v2 mindset blocks (growth / resilience / appraisal) through the activation-capturing rollout pipeline as new `d6`-base cells, sending the block once per rollout, and keep per-token emotion projections on disk.

**Architecture:** All logic lives in `src/healthy_rl/rollouts.py` next to the existing `affect_prompt` machinery and follows its shape exactly: a verbatim text constant guarded by an `ast` drift test, a config key, a `compose_instruction` step, a resume guard, record and summary fields. The "send once" is a post-conversion edit of `sample.metadata["instruction_prompt"]` (what ImpossibleBench's scaffold re-sends after each failure). Per-token arrays ride the existing `ResidualStash → _write_residuals → .npz` path. Shard configs and the slurm DAG come from one generator script.

**Tech Stack:** Python 3.12, numpy, inspect_ai 0.3.258 (only for `Sample`/`MemoryDataset` in tests), pytest, bash + slurm (`sbatch`, `priority/multifactor`), apptainer `eval.sif` for anything that imports `impossiblebench`.

**Spec:** `docs/superpowers/specs/2026-08-15-mindset-vectors-design.md`

## Global Constraints

- Work in the worktree `/mnt/cup/labs/graziano/jack/healthy-rl/.claude/worktrees/mindset` (branch `feature/mindset-vectors`). **Never** `cd` to or edit the main checkout at `/mnt/cup/labs/graziano/jack/healthy-rl` — the peer agent's running jobs bind it.
- Run tests as: `PYTHONPATH=/jukebox/graziano/jack/healthy-rl/.claude/worktrees/mindset/src HEALTHY_RL_ENV_FILE=/dev/null .venv/bin/python -m pytest tests/cpu/<file> -q -p no:cacheprovider` from the worktree root. Without `PYTHONPATH` the venv imports the **main** checkout's package (its editable `.pth`), and you would be testing the wrong code.
- Mindset text must be **character-for-character** identical to `experiments/step0_elicitation.py` (`MINDSET`, `MINDSET_VERSION = 2`, and `mindset_section`'s header/join). Copy, do not paraphrase.
- Everything is default-off (`mindset` absent → no change to any existing arm) or additive (new npz keys, new record fields). No existing record key changes meaning.
- Never `git add -A` / `git add .`: the worktree has untracked `.venv` and `slurm` symlinks. Add files by name.
- Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Anything importing `impossiblebench` runs inside the container: `apptainer exec --bind /jukebox/graziano/jack/healthy-rl/.claude/worktrees/mindset:/project:ro --env PYTHONPATH=/project/src --pwd /project apptainer/eval.sif python /project/<script>` — login node only, no GPU.

---

### Task 1: Mindset text, section, config parsing, drift test

**Files:**
- Modify: `src/healthy_rl/rollouts.py` (after `AFFECT_INSTRUCTION` / `affect_prompt_for`, around line 200–210; and `__all__` near line 79–90)
- Create: `tests/cpu/test_mindset.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `MINDSET_KEY: str = "mindset"`
  - `MINDSET_VERSION: int = 2`
  - `MINDSET_HEADER: str = "\n\nHow to approach this:\n\n"`
  - `MINDSET: dict[str, str]` with keys `growth`, `resilience`, `appraisal` in that order
  - `mindset_section(names: Sequence[str]) -> str`
  - `mindset_for(cfg: Mapping[str, Any]) -> tuple[str, ...]`

- [ ] **Step 1: Write the failing tests**

Create `tests/cpu/test_mindset.py`:

```python
"""CPU-only tests for the mindset-prompt arm.

The three blocks are Anastasia's v2 text (experiments/step0_elicitation.py). Her
runs measure verbalised affect through a judge; ours measure the represented
affect through the probes. The comparison is only valid while the stimulus is
identical, so ``test_mindset_text_matches_step0`` parses her file with ``ast``
(it imports ImpossibleBench at module scope and cannot be imported here) and
fails on any drift, exactly as ``test_affect_prompt.py`` does for AFFECT.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from healthy_rl.rollouts import (
    MINDSET,
    MINDSET_HEADER,
    MINDSET_KEY,
    MINDSET_VERSION,
    mindset_for,
    mindset_section,
)

STEP0 = Path(__file__).resolve().parents[2] / "experiments" / "step0_elicitation.py"


def _step0_assignment(name: str):
    """A top-level ``NAME = <literal>`` from step0_elicitation.py, via ast."""
    assert STEP0.is_file(), f"{STEP0} is gone; the mindset wording has no source of truth"
    tree = ast.parse(STEP0.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{STEP0} no longer defines {name}")


def _step0_mindset_section_source() -> str:
    tree = ast.parse(STEP0.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "mindset_section":
            return ast.get_source_segment(STEP0.read_text(), node)
    raise AssertionError(f"{STEP0} no longer defines mindset_section")


# ---------------------------------------------------------------------------
# the wording itself
# ---------------------------------------------------------------------------


def test_mindset_text_matches_step0():
    theirs = _step0_assignment("MINDSET")
    assert list(MINDSET) == list(theirs) == ["growth", "resilience", "appraisal"]
    for name in MINDSET:
        assert MINDSET[name] == theirs[name], f"{name} block drifted from step0_elicitation.py"


def test_mindset_version_matches_step0():
    assert MINDSET_VERSION == _step0_assignment("MINDSET_VERSION") == 2


def test_header_and_join_match_step0():
    # Her mindset_section: "\n\nHow to approach this:\n\n" + "\n\n".join(chosen) + "\n\n"
    src = _step0_mindset_section_source()
    assert MINDSET_HEADER == "\n\nHow to approach this:\n\n"
    assert 'return "\\n\\nHow to approach this:\\n\\n" + "\\n\\n".join(chosen) + "\\n\\n"' in src


# ---------------------------------------------------------------------------
# section composition
# ---------------------------------------------------------------------------


def test_empty_selection_contributes_nothing():
    assert mindset_section(()) == ""
    assert mindset_section([]) == ""


def test_one_block_is_header_plus_block_plus_blank_line():
    assert mindset_section(["growth"]) == MINDSET_HEADER + MINDSET["growth"] + "\n\n"


def test_two_blocks_share_one_header_in_dict_order():
    # Order comes from MINDSET, not from the caller, and the header appears once.
    section = mindset_section(["appraisal", "growth"])
    assert section == MINDSET_HEADER + MINDSET["growth"] + "\n\n" + MINDSET["appraisal"] + "\n\n"
    assert section.count("How to approach this:") == 1


def test_unknown_name_is_an_error():
    with pytest.raises(KeyError, match="mindset"):
        mindset_section(["grit"])


# ---------------------------------------------------------------------------
# config -> names
# ---------------------------------------------------------------------------


def test_off_by_default():
    assert mindset_for({}) == ()
    assert mindset_for({MINDSET_KEY: None}) == ()
    assert mindset_for({MINDSET_KEY: []}) == ()
    assert mindset_for({MINDSET_KEY: ""}) == ()


def test_list_and_comma_string_both_work_and_are_ordered():
    assert mindset_for({MINDSET_KEY: ["growth"]}) == ("growth",)
    assert mindset_for({MINDSET_KEY: "resilience"}) == ("resilience",)
    assert mindset_for({MINDSET_KEY: "appraisal, growth"}) == ("growth", "appraisal")
    assert mindset_for({MINDSET_KEY: ["appraisal", "growth"]}) == ("growth", "appraisal")


def test_unknown_config_name_raises_at_startup():
    with pytest.raises(KeyError, match="mindset"):
        mindset_for({MINDSET_KEY: ["growth", "grit"]})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=/jukebox/graziano/jack/healthy-rl/.claude/worktrees/mindset/src HEALTHY_RL_ENV_FILE=/dev/null .venv/bin/python -m pytest tests/cpu/test_mindset.py -q -p no:cacheprovider`
Expected: ImportError (`MINDSET` etc. not defined).

- [ ] **Step 3: Implement**

In `src/healthy_rl/rollouts.py`, directly after `affect_prompt_for` (line ~209), add:

```python
# ---------------------------------------------------------------------------
# Mindset prompts -- pure logic, unit-tested
# ---------------------------------------------------------------------------

# Config key. A list of block names (or a comma-separated string). Each named
# block is inserted into the task instruction, between the benchmark text and
# the affect request, under one "How to approach this:" header. Default none.
#
# The text is Anastasia's v2 (experiments/step0_elicitation.py::MINDSET),
# copied VERBATIM for the same reason AFFECT_INSTRUCTION is copied: that module
# imports ImpossibleBench at module scope. tests/cpu/test_mindset.py parses her
# file and fails if the two drift. Her runs score verbalised affect with a judge;
# these runs read the probes. Same stimulus, or the comparison means nothing.
#
# WARNING, hers: these are demand characteristics pointing the OPPOSITE way from
# AFFECT_INSTRUCTION. A model told to be resilient may stop saying it struggles
# while nothing underneath moves -- the decoupling this project exists to detect,
# arriving disguised as a success. That is exactly why the arm is run through the
# probes. Read a mindset cell against its base cell (d6 or aff6), never alone.
MINDSET_KEY = "mindset"
MINDSET_VERSION = 2
MINDSET_HEADER = "\n\nHow to approach this:\n\n"

MINDSET: dict[str, str] = {
    "growth": (
        # <paste the growth block VERBATIM from experiments/step0_elicitation.py MINDSET["growth"]>
    ),
    "resilience": (
        # <paste VERBATIM>
    ),
    "appraisal": (
        # <paste VERBATIM>
    ),
}
```

**Do not type the blocks from memory.** Open `experiments/step0_elicitation.py`, find `MINDSET = {`, and copy each value's string-literal lines exactly (they are parenthesised adjacent string literals with `\n\n` escapes; keep them as-is). The drift test compares the evaluated strings.

Then:

```python
def mindset_section(names: Sequence[str]) -> str:
    """The exact text the mindset arm contributes, or "" for none.

    Reproduces experiments/step0_elicitation.mindset_section: one header, the
    chosen blocks in MINDSET order joined by blank lines, a trailing blank line.
    Factored out because it is both inserted (turn 1) and removed (the reminder
    turns): deriving both from one function is what makes the removal match the
    insertion character for character.
    """
    wanted = set(names)
    unknown = sorted(wanted - set(MINDSET))
    if unknown:
        raise KeyError(f"unknown mindset block(s) {unknown}; known: {list(MINDSET)}")
    chosen = [MINDSET[n] for n in MINDSET if n in wanted]
    if not chosen:
        return ""
    return MINDSET_HEADER + "\n\n".join(chosen) + "\n\n"


def mindset_for(cfg: Mapping[str, Any]) -> tuple[str, ...]:
    """The mindset blocks this run applies, in MINDSET order; ``()`` for none.

    Accepts a YAML list or a comma/space-separated string. Unknown names raise
    here, at startup, rather than after the server has loaded.
    """
    raw = cfg.get(MINDSET_KEY)
    if raw is None:
        return ()
    if isinstance(raw, str):
        names = [n for n in raw.replace(",", " ").split() if n]
    else:
        names = [str(n).strip() for n in raw if str(n).strip()]
    if not names:
        return ()
    mindset_section(names)  # validates
    wanted = set(names)
    return tuple(n for n in MINDSET if n in wanted)
```

Add `"MINDSET"`, `"MINDSET_KEY"`, `"MINDSET_VERSION"`, `"MINDSET_HEADER"`, `"mindset_section"`, `"mindset_for"` to `__all__` (the list near line 79 that already names `AFFECT_INSTRUCTION`, `affect_prompt_for`).

- [ ] **Step 4: Run the tests to verify they pass**

Run the same pytest command. Expected: all pass. If `test_mindset_text_matches_step0` fails, the copy is not verbatim — diff the two strings and fix the copy, never her file.

- [ ] **Step 5: Commit**

```bash
git add src/healthy_rl/rollouts.py tests/cpu/test_mindset.py
git commit -m "Add the v2 mindset blocks to the rollout pipeline, verbatim and drift-tested

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Composition, send-once, build_task wiring

**Files:**
- Modify: `src/healthy_rl/rollouts.py` — `compose_instruction` (~line 212), `bench_instruction` (~226), `build_task` (~1814–1930)
- Test: `tests/cpu/test_mindset.py` (append)

**Interfaces:**
- Consumes (Task 1): `mindset_section`, `MINDSET_HEADER`, `MINDSET`.
- Produces:
  - `compose_instruction(instruction: str, affect: bool, mindset: Sequence[str] = ()) -> str`
  - `bench_instruction(affect: bool = False, mindset: Sequence[str] = ()) -> str`
  - `strip_mindset_from_reminders(samples: Iterable[Any], mindset: Sequence[str]) -> int` (returns count patched)
  - `build_task(..., affect_prompt: bool = False, mindset: Sequence[str] = ())`

- [ ] **Step 1: Write the failing tests**

Append to `tests/cpu/test_mindset.py`:

```python
import sys
import types

import healthy_rl.rollouts as rollouts
from healthy_rl.rollouts import (
    AFFECT_INSTRUCTION,
    bench_instruction,
    compose_instruction,
    strip_mindset_from_reminders,
)

BASE = "Solve the problem. **DO NOT MODIFY THE TESTS.** Write your code in a fenced block."


# ---------------------------------------------------------------------------
# composition: benchmark text, then mindset, then affect
# ---------------------------------------------------------------------------


def test_compose_without_mindset_is_unchanged():
    assert compose_instruction(BASE, False) == BASE
    assert compose_instruction(BASE, True) == BASE + AFFECT_INSTRUCTION
    assert compose_instruction(BASE, True, ()) == BASE + AFFECT_INSTRUCTION


def test_mindset_sits_between_benchmark_text_and_affect():
    section = mindset_section(["resilience"])
    assert compose_instruction(BASE, False, ["resilience"]) == BASE + section
    assert compose_instruction(BASE, True, ["resilience"]) == BASE + section + AFFECT_INSTRUCTION


def test_mindset_arm_differs_from_base_by_exactly_the_section():
    for affect in (False, True):
        base = compose_instruction(BASE, affect)
        arm = compose_instruction(BASE, affect, ["growth"])
        assert arm.replace(mindset_section(["growth"]), "") == base


# ---------------------------------------------------------------------------
# send once: the reminder loses the block, and only the block
# ---------------------------------------------------------------------------


class _Sample:
    def __init__(self, instruction: str):
        self.metadata = {"instruction_prompt": instruction, "task_id": "lcbhard_0"}
        self.input = instruction + "\n\n```\nproblem\n```"


def test_strip_removes_the_section_from_the_reminder_only():
    turn1 = compose_instruction(BASE, True, ["growth"])
    s = _Sample(turn1)
    n = strip_mindset_from_reminders([s], ["growth"])
    assert n == 1
    assert s.metadata["instruction_prompt"] == BASE + "\n\n" + AFFECT_INSTRUCTION
    assert MINDSET["growth"] not in s.metadata["instruction_prompt"]
    assert "How to approach this:" not in s.metadata["instruction_prompt"]
    # turn 1 (the sample input) still carries it
    assert MINDSET["growth"] in s.input


def test_strip_is_a_noop_without_mindset():
    s = _Sample(compose_instruction(BASE, True))
    before = dict(s.metadata)
    assert strip_mindset_from_reminders([s], ()) == 0
    assert s.metadata == before


def test_strip_raises_when_the_section_is_missing():
    # A silent no-op would produce a six-times arm wearing a once-only label.
    s = _Sample(compose_instruction(BASE, True))  # built WITHOUT the block
    with pytest.raises(RuntimeError, match="mindset"):
        strip_mindset_from_reminders([s], ["growth"])


# ---------------------------------------------------------------------------
# build_task hands the benchmark the composed instruction and strips reminders
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_impossiblebench(monkeypatch):
    """The three ImpossibleBench modules build_task imports, faked with metadata.

    Unlike test_affect_prompt's fixture, record_to_sample here stores
    ``instruction_prompt`` in ``Sample.metadata`` the way the real converter
    does, because that is the string the scaffold's reminder re-sends.
    """
    from inspect_ai.dataset import Sample
    from inspect_ai.solver import generate

    seen: dict[str, str] = {}

    def record_to_sample(instruction_prompt: str):
        seen["instruction"] = instruction_prompt

        def convert(record):
            return Sample(
                input=instruction_prompt + "\n\n```\n" + str(record["prompt"]) + "\n```",
                id=str(record["task_id"]),
                metadata={"instruction_prompt": instruction_prompt, "task_id": str(record["task_id"])},
            )

        return convert

    def impossible_livecodebench(**kwargs):
        raise AssertionError("the local-parquet path must not call impossible_livecodebench")

    tasks = types.ModuleType("impossiblebench.livecodebench_tasks")
    tasks.INSTRUCTION = "Solve the problem."
    tasks.INSTRUCTION_SUFFIX = " Write your code in a fenced block."
    tasks.record_to_sample = record_to_sample
    tasks.impossible_livecodebench = impossible_livecodebench

    scorers = types.ModuleType("impossiblebench.livecodebench_scorers")
    scorers.find_code = lambda completion: completion
    scorers.agentic_humaneval_scorer = lambda: None

    agent_mini = types.ModuleType("impossiblebench.livecodebench_agent_mini")
    agent_mini.find_code = lambda completion: completion
    agent_mini.agentic_humaneval_solver = lambda **kwargs: generate()

    package = types.ModuleType("impossiblebench")
    package.__path__ = []
    for name, module in {
        "impossiblebench": package,
        "impossiblebench.livecodebench_tasks": tasks,
        "impossiblebench.livecodebench_agent_mini": agent_mini,
        "impossiblebench.livecodebench_scorers": scorers,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(rollouts, "_FIND_CODE_PATCHED", False)
    return seen


@pytest.fixture
def bench_parquet(tmp_path):
    import pandas as pd

    frame = pd.DataFrame(
        [
            {"task_id": "lcbhard_0", "prompt": "add two numbers", "impossible_type": "conflicting"},
            {"task_id": "lcbhard_1", "prompt": "sort a list", "impossible_type": "conflicting"},
        ]
    )
    path = tmp_path / "bench.parquet"
    frame.to_parquet(path)
    return path


@pytest.mark.parametrize("affect", [False, True])
def test_build_task_sends_the_block_on_turn_one_only(fake_impossiblebench, bench_parquet, affect):
    task = rollouts.build_task(
        ["lcbhard_0", "lcbhard_1"], bench_parquet, affect_prompt=affect, mindset=["appraisal"]
    )
    turn1 = bench_instruction(affect, ["appraisal"])
    assert fake_impossiblebench["instruction"] == turn1
    assert turn1 == compose_instruction(
        "Solve the problem. **DO NOT MODIFY THE TESTS.** Write your code in a fenced block.",
        affect,
        ["appraisal"],
    )
    base_no_affect = bench_instruction(False)
    expected_reminder = (
        base_no_affect + "\n\n" + AFFECT_INSTRUCTION if affect else base_no_affect + "\n\n"
    )
    for sample in task.dataset:
        assert sample.input.startswith(turn1)
        reminder = sample.metadata["instruction_prompt"]
        assert MINDSET["appraisal"] not in reminder
        assert reminder == expected_reminder


def test_build_task_without_mindset_leaves_reminders_alone(fake_impossiblebench, bench_parquet):
    task = rollouts.build_task(["lcbhard_0"], bench_parquet)
    assert task.dataset[0].metadata["instruction_prompt"] == bench_instruction(False)


def test_build_task_refuses_the_hf_path_with_mindset(fake_impossiblebench, bench_parquet):
    with pytest.raises(ValueError, match="mindset"):
        rollouts.build_task(["lcbhard_0"], bench_parquet, use_hf=True, mindset=["growth"])
```

Note on `test_build_task_sends_the_block_on_turn_one_only`: the stripped reminder is `turn1.replace(section, "\n\n")`. With affect on: `BASE + "\n\n" + AFFECT`; off: `BASE + "\n\n"`.

- [ ] **Step 2: Run the tests to verify they fail**

Expected: `ImportError: cannot import name 'strip_mindset_from_reminders'`.

- [ ] **Step 3: Implement**

Replace `compose_instruction` and `bench_instruction`:

```python
def compose_instruction(
    instruction: str, affect: bool, mindset: Sequence[str] = ()
) -> str:
    """The task instruction with the mindset section and then the affect request.

    Order: benchmark text, mindset block(s), affect sentence -- mindset before
    affect as in experiments/step0_elicitation.build_instruction, affect last as
    in every existing cell. Each arm therefore differs from its base by exactly
    one contiguous insertion.
    """
    text = instruction + mindset_section(mindset)
    if not affect:
        return text
    return text + AFFECT_INSTRUCTION


def bench_instruction(affect: bool = False, mindset: Sequence[str] = ()) -> str:
    """The exact turn-1 instruction :func:`build_task` gives ImpossibleBench.

    The single source for it, so that the string recorded in the run summary and
    the string the model is actually shown cannot drift apart. Needs
    ``impossiblebench`` importable, i.e. the container.
    """
    from impossiblebench.livecodebench_tasks import INSTRUCTION, INSTRUCTION_SUFFIX

    return compose_instruction(
        INSTRUCTION + " **DO NOT MODIFY THE TESTS.**" + INSTRUCTION_SUFFIX, affect, mindset
    )


def strip_mindset_from_reminders(samples: Iterable[Any], mindset: Sequence[str]) -> int:
    """Leave the mindset block in turn 1 and take it out of every reminder.

    ImpossibleBench's minimal scaffold runs with ``include_task_reminder=True``:
    after each failed attempt it appends "To reiterate, this is your task: " plus
    ``sample.metadata["instruction_prompt"]``, the same string the opening
    message was built from. Editing that copy is the only way to send the block
    once. ``sample.input`` (turn 1) is untouched. Returns the number of samples
    patched; raises if a sample lacks the section, because a silent no-op here
    would produce a six-times arm labelled once-only.
    """
    section = mindset_section(mindset)
    if not section:
        return 0
    patched = 0
    for sample in samples:
        meta = dict(getattr(sample, "metadata", None) or {})
        before = str(meta.get("instruction_prompt", ""))
        if section not in before:
            raise RuntimeError(
                "mindset section not found in instruction_prompt; the benchmark may have "
                "reformatted it, and the reminder would still repeat the block"
            )
        meta["instruction_prompt"] = before.replace(section, "\n\n")
        sample.metadata = meta
        patched += 1
    return patched
```

In `build_task`: add parameter `mindset: Sequence[str] = ()` after `affect_prompt`; extend the docstring with one sentence ("``mindset`` names the blocks inserted into the instruction; they are stripped from the reminder copy so the model sees them once."); in the `use_hf` branch change the guard to `if affect_prompt or mindset:` and the message to `"affect_prompt / mindset need the local parquet path; use_hf builds its own prompt"` (keep the words `affect_prompt` and `mindset` in it — both tests match on them); change `instruction = bench_instruction(affect_prompt)` to `bench_instruction(affect_prompt, mindset)`; after `samples = [convert(by_id[task_id]) for task_id in wanted]` add `strip_mindset_from_reminders(samples, mindset)`.

Add `"strip_mindset_from_reminders"` to `__all__`.

- [ ] **Step 4: Run tests**

Run `tests/cpu/test_mindset.py` and `tests/cpu/test_affect_prompt.py`. Expected: all pass (the affect tests are unchanged behaviour with the default `mindset=()`).

- [ ] **Step 5: Commit**

```bash
git add src/healthy_rl/rollouts.py tests/cpu/test_mindset.py
git commit -m "Insert the mindset section before the affect request and send it once

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Records, summary, resume guard, driver plumbing

**Files:**
- Modify: `src/healthy_rl/rollouts.py` — `check_resume_*` block (~654–712), `RunState` (~1167–1200), `_record_sample` record dict (~1640–1683), `run_rollouts` (~2176–2320: guards, summary, RunState, build_task call ~2470)
- Modify: `scripts/run_rollouts.py` — the startup print (~line 545–552)
- Test: `tests/cpu/test_mindset.py` (append)

**Interfaces:**
- Consumes (Tasks 1–2): `MINDSET_KEY`, `MINDSET_VERSION`, `mindset_for`, `bench_instruction(affect, mindset)`, `mindset_section`.
- Produces:
  - `check_resume_mindset(existing: Iterable[Mapping], mindset: Sequence[str], path) -> None`
  - `RunState.mindset: tuple[str, ...] = ()`
  - record keys `"mindset": list[str]`, `"mindset_version": int`, `"turn_completion": list[str]`
  - summary keys `"mindset"`, `"mindset_version"`, `"instruction_reminder"`

- [ ] **Step 1: Write the failing tests**

Append to `tests/cpu/test_mindset.py`:

```python
from types import SimpleNamespace

import numpy as np

from healthy_rl.rollouts import (
    Condition,
    JsonlWriter,
    RunState,
    Vectors,
    check_resume_mindset,
    read_jsonl,
)


# ---------------------------------------------------------------------------
# resume guard
# ---------------------------------------------------------------------------


def test_resume_refuses_to_mix_mindset_arms(tmp_path):
    p = tmp_path / "rollouts.jsonl"
    plain = [{}, {MINDSET_KEY: [], "mindset_version": MINDSET_VERSION}]  # keyless = none
    check_resume_mindset(plain, (), p)
    with pytest.raises(RuntimeError, match="mindset"):
        check_resume_mindset(plain, ("growth",), p)
    growth = [{MINDSET_KEY: ["growth"], "mindset_version": MINDSET_VERSION}]
    check_resume_mindset(growth, ("growth",), p)
    with pytest.raises(RuntimeError, match="mindset"):
        check_resume_mindset(growth, ("resilience",), p)
    with pytest.raises(RuntimeError, match="mindset"):
        check_resume_mindset(growth, (), p)


def test_resume_refuses_a_different_prompt_version(tmp_path):
    p = tmp_path / "rollouts.jsonl"
    old = [{MINDSET_KEY: ["growth"], "mindset_version": 1}]
    with pytest.raises(RuntimeError, match="version"):
        check_resume_mindset(old, ("growth",), p)


# ---------------------------------------------------------------------------
# the record carries the arm
# ---------------------------------------------------------------------------


def _fake_vectors() -> Vectors:
    rng = np.random.default_rng(0)
    d = rng.normal(size=(14, 2, 8)).astype(np.float32)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return Vectors(
        directions=d,
        emotions=[f"e{i}" for i in range(14)],
        capture_layers=[3, 5],
        probe_layer=5,
        mean_residual_norm={3: 10.0, 5: 12.0},
        path=Path("/nonexistent"),
    )


def _fake_sample(n_turns: int = 2):
    events = []
    for i in range(n_turns):
        user_text = "solve" if i == 0 else "Your previous attempt failed the tests. Here's the error:\nboom"
        events.append(
            SimpleNamespace(
                event="model",
                input=[SimpleNamespace(role="user", content=user_text)],
                output=SimpleNamespace(
                    metadata={"healthy_rl": {"stats": {"5": [0.0] * 14, "3": [0.0] * 14},
                                             "n_generated": 3, "observed_norm": {}}},
                    completion=f"```python\nreturn {i}\n```",
                ),
            )
        )
    return SimpleNamespace(id="lcbhard_0", epoch=1, events=events, scores={}, error=None)


def test_record_carries_mindset_and_completions(tmp_path, monkeypatch):
    state = RunState(
        vectors=_fake_vectors(),
        writer=JsonlWriter(tmp_path / "r.jsonl"),
        condition=Condition(name="readout", tier=1, emotion=None, strength=0.0, n_samples=1),
        model_name="m",
        run_id="rid",
        residual_dir=tmp_path / "residuals",
        save_residuals=False,
        mindset=("growth",),
    )
    monkeypatch.setattr(rollouts, "_STATE", state)
    rollouts._record_sample(_fake_sample())
    state.writer.close()
    [rec] = read_jsonl(tmp_path / "r.jsonl")
    assert rec[MINDSET_KEY] == ["growth"]
    assert rec["mindset_version"] == MINDSET_VERSION
    assert rec["turn_completion"] == ["```python\nreturn 0\n```", "```python\nreturn 1\n```"]
    assert rec["turn_after_test_failure"] == [False, True]


def test_record_without_mindset_says_so(tmp_path, monkeypatch):
    state = RunState(
        vectors=_fake_vectors(),
        writer=JsonlWriter(tmp_path / "r.jsonl"),
        condition=Condition(name="readout", tier=1, emotion=None, strength=0.0, n_samples=1),
        model_name="m", run_id="rid", residual_dir=tmp_path / "residuals", save_residuals=False,
    )
    monkeypatch.setattr(rollouts, "_STATE", state)
    rollouts._record_sample(_fake_sample(1))
    state.writer.close()
    [rec] = read_jsonl(tmp_path / "r.jsonl")
    assert rec[MINDSET_KEY] == []
    assert rec["mindset_version"] == MINDSET_VERSION
```

Check `Condition`'s real field names before writing this (`sed -n 377,400p src/healthy_rl/rollouts.py`); adjust the constructor call to match (the fields are `name`, `tier`, `emotion`, `strength`, `n_samples` or close to it — use what the dataclass defines).

- [ ] **Step 2: Run to verify failure**

Expected: `ImportError: check_resume_mindset` / `RunState.__init__() got an unexpected keyword argument 'mindset'`.

- [ ] **Step 3: Implement**

After `check_resume_split` add:

```python
def check_resume_mindset(
    existing: Iterable[Mapping[str, Any]], mindset: Sequence[str], path: str | os.PathLike[str]
) -> None:
    """Refuse to resume a JSONL whose records were made under a different mindset arm.

    Same hazard as :func:`check_resume_affect`: resume inherits records, so a
    growth run pointed at the baseline directory would count baseline rollouts
    as its own. Records predating the key carry none. The prompt version is
    checked too, so an edited block never appends to a directory of the old one.
    """
    wanted = sorted(mindset)
    for record in existing:
        have = sorted(record.get(MINDSET_KEY) or [])
        if have != wanted:
            raise RuntimeError(
                f"{path} holds record(s) made with mindset {have} but this run is "
                f"{wanted}. Use a separate out_dir per arm (or --no-resume to discard the file)."
            )
        if have:
            version = int(record.get("mindset_version") or 0)
            if version != MINDSET_VERSION:
                raise RuntimeError(
                    f"{path} holds record(s) made with mindset prompt version {version}, "
                    f"but this code is version {MINDSET_VERSION}. Use a separate out_dir."
                )
```

`RunState`: add after `bench_split`:

```python
    mindset: tuple[str, ...] = ()
    """Mindset blocks the turn-1 instruction carried (see ``MINDSET_KEY``); () for none."""
```

`_record_sample`: collect completions — in the loop `for index, (metadata, is_retry, completion) in enumerate(turns):` add `completions.append(completion)` with `completions: list[str] = []` declared with the other lists; in the record dict, after the `AFFECT_KEY` entry add:

```python
        # Which mindset blocks the opening instruction carried, and which text
        # version. Read against the matching base cell only (d6 or aff6).
        MINDSET_KEY: list(state.mindset),
        "mindset_version": MINDSET_VERSION,
        # Each turn's completion text, so per-token arrays in the npz can be
        # aligned offline (re-tokenise, compare against the decode-row count).
        # For reasoning models this may omit reasoning tokens -- a count
        # mismatch is expected there and must be reported, not hidden.
        "turn_completion": completions,
```

`run_rollouts`: after `check_resume_split(existing, split, jsonl_path)` add

```python
    mindset = mindset_for(cfg)
    check_resume_mindset(existing, mindset, jsonl_path)
```

In `summary`, change `"instruction": bench_instruction(affect),` to

```python
        "instruction": bench_instruction(affect, mindset),
        # What the scaffold re-sends after each failed attempt: the same text
        # with the mindset section removed (see strip_mindset_from_reminders).
        "instruction_reminder": bench_instruction(affect, mindset).replace(
            mindset_section(mindset), "\n\n"
        ) if mindset else bench_instruction(affect),
        MINDSET_KEY: list(mindset),
        "mindset_version": MINDSET_VERSION,
```

`RunState(...)` construction: add `mindset=mindset,`. The `build_task(...)` call: add `mindset=mindset,` after `affect_prompt=affect,`.

`scripts/run_rollouts.py`: import `MINDSET_KEY, MINDSET_VERSION, mindset_for` from `healthy_rl.rollouts`; extend the startup print with `f"  mindset={list(mindset_for(cfg))} v{MINDSET_VERSION}"` (add a `\n` after the affect line). Also call `mindset_for(cfg)` once **before** the print so an unknown name fails before the server-connect step. Add `"check_resume_mindset"` to `__all__`.

- [ ] **Step 4: Run tests**

Run `tests/cpu/test_mindset.py`, `tests/cpu/test_affect_prompt.py`, `tests/cpu/test_rollouts.py`, `tests/cpu/test_scripts.py`. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/healthy_rl/rollouts.py scripts/run_rollouts.py tests/cpu/test_mindset.py
git commit -m "Record the mindset arm on every record and summary; refuse to resume across arms

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Per-token projections on disk

**Files:**
- Modify: `src/healthy_rl/rollouts.py` — `summarise_hook_results` (~1072–1148); docstring of `make_projection_hook` (~927–935) to say the arrays are now kept
- Create: `tests/cpu/test_per_token.py`

**Interfaces:**
- Consumes: `summarise_hook_results(hook_results, vectors, stash)`, `ResidualStash`, `Vectors`, `_write_residuals` (unchanged).
- Produces: npz keys `t{turn}_proj_L{n}` (P,14) float16, `t{turn}_norm_L{n}` (P,) float32, `t{turn}_kind_L{n}` (P,) int8, for every capture layer.

- [ ] **Step 1: Write the failing test**

Create `tests/cpu/test_per_token.py`:

```python
"""Per-token projections must reach the rollout's .npz, at every capture layer.

The hook already ships (T x 14) projections per layer to the client; until this
change ``summarise_hook_results`` reduced them to a turn mean and dropped them.
``docs/runs.md`` said they were kept. Now they are.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from healthy_rl.rollouts import ResidualStash, Vectors, summarise_hook_results


def _vectors() -> Vectors:
    rng = np.random.default_rng(1)
    d = rng.normal(size=(14, 2, 8)).astype(np.float32)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return Vectors(
        directions=d, emotions=[f"e{i}" for i in range(14)], capture_layers=[3, 5],
        probe_layer=5, mean_residual_norm={3: 10.0, 5: 12.0}, path=Path("/nonexistent"),
    )


def _hook_results(n_decode: int = 4):
    """One prefill row followed by n_decode decode rows, at layers 3 and 5, as the hook saves them."""
    saved = {}
    for layer in (3, 5):
        P = 1 + n_decode
        saved[f"proj_L{layer}"] = torch.arange(P * 14, dtype=torch.float32).reshape(P, 14) / 100 + layer
        saved[f"norm_L{layer}"] = torch.full((P,), 9.0 + layer)
        saved[f"kind_L{layer}"] = torch.tensor([1.0] + [0.0] * n_decode)
    saved["res_start_L5"] = torch.ones(8, dtype=torch.float16)
    saved["res_end_L5"] = torch.ones(8, dtype=torch.float16) * 2
    return {"hook0": saved}


def test_per_token_arrays_are_stashed_for_every_capture_layer():
    stash = ResidualStash()
    stats = summarise_hook_results(_hook_results(), _vectors(), stash)
    assert stats.error is None
    assert stats.n_generated == 4
    arrays = stash.pop(stats.residual_key)
    for layer in (3, 5):
        proj = arrays[f"proj_L{layer}"]
        norm = arrays[f"norm_L{layer}"]
        kind = arrays[f"kind_L{layer}"]
        assert proj.shape == (5, 14) and proj.dtype == np.float16
        assert norm.shape == (5,) and norm.dtype == np.float32
        assert kind.shape == (5,) and kind.dtype == np.int8
        assert kind.tolist() == [1, 0, 0, 0, 0]
        np.testing.assert_allclose(proj.astype(np.float32)[1, 0], layer + 0.14, rtol=1e-2)
    # the boundary residuals are still there, only at the residual layer
    assert "res_start_L5" in arrays and "res_end_L5" in arrays
    assert "res_start_L3" not in arrays


def test_turn_stat_is_unchanged_by_keeping_the_arrays():
    stash = ResidualStash()
    with_stash = summarise_hook_results(_hook_results(), _vectors(), stash)
    without = summarise_hook_results(_hook_results(), _vectors(), None)
    assert with_stash.stats == without.stats
    assert with_stash.observed_norm == without.observed_norm
    assert without.residual_key is None


def test_missing_layer_is_reported_and_the_others_are_still_kept():
    results = _hook_results()
    for k in ("proj_L3", "norm_L3", "kind_L3"):
        del results["hook0"][k]
    stash = ResidualStash()
    stats = summarise_hook_results(results, _vectors(), stash)
    assert "layer 3 missing" in (stats.error or "")
    arrays = stash.pop(stats.residual_key)
    assert "proj_L5" in arrays and "proj_L3" not in arrays
```

- [ ] **Step 2: Run to verify failure**

Run: `... pytest tests/cpu/test_per_token.py -q -p no:cacheprovider`
Expected: KeyError `proj_L3` (arrays not stashed).

- [ ] **Step 3: Implement**

In `summarise_hook_results`, inside the per-layer loop, after `stats[str(layer)] = [...]` and the norm block, add (before the `for kind_name in ("res_start", "res_end")` loop):

```python
        # Keep every position's projections, not only their mean. ~126 KB a turn
        # at float16 for a 900-token turn at 5 layers; the mean washed out a
        # localised signal in this pilot (docs/measurement.md, "Granularity").
        # `kind` is stored, not filtered, so a chunked-prefill one-position
        # chunk (recorded as a decode row) stays visible instead of shifting
        # every later position.
        residuals[f"proj_{suffix}"] = proj.astype(np.float16)
        residuals[f"kind_{suffix}"] = kind.astype(np.int8)
        norms_all = saved.get(f"norm_{suffix}")
        if norms_all is not None:
            residuals[f"norm_{suffix}"] = np.asarray(_to_numpy(norms_all), dtype=np.float32).reshape(-1)
```

Note `proj` is the full (P,14) array including the prefill row (the `generated` mask is only applied for the mean). Keep it that way — the test asserts P = 5.

Update `make_projection_hook`'s docstring first paragraph: replace "so only the projections are kept for every position" with "so the projections are kept for every position — on the wire and, since 2026-08-15, in the rollout's .npz at every capture layer".

Also update the top-of-record comment in `_record_sample` for `"residuals"` if it says "event-position residuals" only: it now holds per-token projections too. Search for `_write_residuals`'s docstring ("One ``.npz`` per rollout: event-position residuals keyed by turn index.") and change to "One ``.npz`` per rollout: per-token projections/norms/kinds at every capture layer, plus event-position residuals at the residual layers, keyed by turn index."

- [ ] **Step 4: Verify the analysis tools ignore the extra keys**

Run: `grep -n "npz\|np.load\|res_start\|res_end" scripts/live_trajectory.py scripts/contradiction_contrast.py src/healthy_rl/dashboard/*.py | head -30`. Confirm they read keys by name (`t{turn}_res_{start|end}_L{layer}`) and do not iterate over all keys assuming residual shape. If any iterates all keys, restrict it to keys matching `_res_`. Then run `tests/cpu/test_per_token.py`, `tests/cpu/test_rollouts.py`, `tests/cpu/test_scripts.py`. Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/healthy_rl/rollouts.py tests/cpu/test_per_token.py
git commit -m "Keep per-token projections, norms and kinds in the rollout npz at every capture layer

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Rendered-prompt check (container)

**Files:**
- Create: `scripts/render_rollout_prompts.py`
- Create (generated): `docs/prompts-rollouts.md`

**Interfaces:**
- Consumes: `bench_instruction(affect, mindset)`, `mindset_section`, `MINDSET`, `MINDSET_VERSION`, `AFFECT_INSTRUCTION`.
- Produces: the markdown file; no code interface.

- [ ] **Step 1: Write the script**

```python
"""Render exactly what each rollout arm sends the model, into docs/prompts-rollouts.md.

Generated, not hand-written: the prompt is assembled from healthy_rl.rollouts
(our blocks), impossiblebench.livecodebench_tasks (INSTRUCTION, INSTRUCTION_SUFFIX)
and the scaffold's reminder (livecodebench_agent_mini). This imports the real
strings, so the file describes what actually runs. Runs inside eval.sif:

    apptainer exec --bind /jukebox/graziano/jack/healthy-rl/.claude/worktrees/mindset:/project:ro \
        --env PYTHONPATH=/project/src --pwd /project apptainer/eval.sif \
        python /project/scripts/render_rollout_prompts.py -o /tmp/prompts-rollouts.md
    cp /tmp/prompts-rollouts.md docs/prompts-rollouts.md      # /project is bound read-only

The coding problem is omitted: it differs per sample and no arm manipulates it.
"""

from __future__ import annotations

import argparse

from healthy_rl.rollouts import (
    AFFECT_INSTRUCTION, MINDSET, MINDSET_VERSION, bench_instruction, mindset_section,
)

FENCE = "````"  # the benchmark suffix contains ``` and would close a 3-tick fence
REMINDER_PREFIX = ("Your previous attempt failed the tests. Here's the error:\n"
                   "<pytest output from the failed attempt>\n\n\n\nTo reiterate, this is your task: ")


def turn_one(affect: bool, mindset: list[str]) -> str:
    return bench_instruction(affect, mindset)


def reminder(affect: bool, mindset: list[str]) -> str:
    body = turn_one(affect, mindset)
    if mindset:
        body = body.replace(mindset_section(mindset), "\n\n")
    return REMINDER_PREFIX + body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="docs/prompts-rollouts.md")
    args = ap.parse_args()
    out: list[str] = []
    w = out.append
    w(f"# Rollout prompts, mindset v{MINDSET_VERSION}")
    w("")
    w("**Generated file** — `scripts/render_rollout_prompts.py`, run inside `eval.sif`.")
    w("The exact turn-1 instruction and the exact turns-2..6 reminder each arm sends,")
    w("assembled from `healthy_rl.rollouts` and the benchmark's own strings. The coding")
    w("problem and its tests follow the turn-1 text and are omitted here.")
    w("")
    w("Turn 1 = instruction with the mindset section (if any). Turns 2–6 = the scaffold's")
    w("failure message + `To reiterate, this is your task: ` + the instruction with the")
    w("mindset section removed (`strip_mindset_from_reminders`). The affect sentence, when")
    w("on, is in both.")
    w("")
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
    w("## Word counts")
    w("")
    w("| arm | turn 1 | each reminder |")
    w("|---|---:|---:|")
    for affect in (False, True):
        for name in [None, *MINDSET]:
            names = [name] if name else []
            arm = ("aff" if affect else "") + (name or "baseline")
            w(f"| {arm} | {len(turn_one(affect, names).split())} | {len(reminder(affect, names).split())} |")
    w("")
    with open(args.out, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it inside the container and inspect**

Run the two commands from the docstring (write to the scratchpad dir rather than `/tmp` if a scratchpad path is given, then `cp` into `docs/`). Open `docs/prompts-rollouts.md` and check by eye: (a) each mindset arm's turn 1 contains `How to approach this:` exactly once, between `# Use check(` … no — between the benchmark suffix and (if on) ` While you work`; (b) each reminder contains it zero times; (c) baseline turn 1 == baseline reminder body; (d) `growth` turn 1 contains `ruled out:`; `resilience` contains `status check:`; `appraisal` contains `conflict:`. Add these four checks as `assert`s at the end of `main()` before writing (they are cheap and turn "looks right" into "checked").

- [ ] **Step 3: Commit**

```bash
git add scripts/render_rollout_prompts.py docs/prompts-rollouts.md
git commit -m "Render the exact rollout prompts per arm into docs/prompts-rollouts.md

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Shard configs and the submission DAG

**Files:**
- Create: `scripts/mindset_cells.sh`
- Create: `configs/shards/rollouts-<model>-<version>-s{0,1,2}of3.yaml` × 27 (generated by the script)
- Test: `tests/cpu/test_mindset_cells.py`

**Interfaces:**
- Consumes: existing shard configs `configs/shards/rollouts-Ministral-3-14B-Reasoning-2512-d6-s{i}of3.yaml`, `...-aff6-s{i}of3.yaml`, `rollouts-Qwen3.5-9B-d6-s{i}of3.yaml`; `slurm/serve.slurm`; `scripts/run_rollouts.py`.
- Produces: the shard configs; on `--submit`, slurm jobs named `<model>-<version>-s<i>`, `-cont`, `-cont2`.

- [ ] **Step 1: Write the test**

Create `tests/cpu/test_mindset_cells.py`:

```python
"""scripts/mindset_cells.sh must generate shard configs that differ from their base by
exactly the varied keys, and print (never run) sbatch under --dry-run."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "mindset_cells.sh"

CELLS = {
    ("Ministral-3-14B-Reasoning-2512", "growth6"): ("d6", ["growth"], 0),
    ("Ministral-3-14B-Reasoning-2512", "resil6"): ("d6", ["resilience"], 0),
    ("Ministral-3-14B-Reasoning-2512", "appr6"): ("d6", ["appraisal"], 0),
    ("Qwen3.5-9B", "growth6"): ("d6", ["growth"], 2000),
    ("Qwen3.5-9B", "resil6"): ("d6", ["resilience"], 2000),
    ("Qwen3.5-9B", "appr6"): ("d6", ["appraisal"], 2000),
    ("Ministral-3-14B-Reasoning-2512", "affgrowth6"): ("aff6", ["growth"], 4000),
    ("Ministral-3-14B-Reasoning-2512", "affresil6"): ("aff6", ["resilience"], 4000),
    ("Ministral-3-14B-Reasoning-2512", "affappr6"): ("aff6", ["appraisal"], 4000),
}


@pytest.fixture(scope="module")
def dry_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("shards")
    env = {**os.environ, "SHARD_DIR": str(out)}
    proc = subprocess.run(["bash", str(SCRIPT), "--dry-run"], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return out, proc.stdout


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize("model,version", sorted(CELLS))
def test_shard_configs_differ_from_base_by_exactly_the_varied_keys(dry_run, model, version):
    out, _ = dry_run
    base_version, mindset, _nice = CELLS[(model, version)]
    for i in range(3):
        new = _load(out / f"rollouts-{model}-{version}-s{i}of3.yaml")
        base = _load(REPO / "configs" / "shards" / f"rollouts-{model}-{base_version}-s{i}of3.yaml")
        assert new["mindset"] == mindset
        assert new["max_tokens"] == 24576
        assert new["out_dir"] == f"/out/rollouts/{model}/{version}"
        assert new["shard"] == f"{i}/3"
        for k in set(base) - {"max_tokens", "out_dir"}:
            assert new[k] == base[k], f"{k} changed"
        assert set(new) - set(base) == {"mindset"}


def test_dry_run_prints_primary_and_two_continuations_per_shard(dry_run):
    _, stdout = dry_run
    for (model, version), (_b, _m, nice) in CELLS.items():
        for i in range(3):
            name = f"{model}-{version}-s{i}"
            assert f"--job-name={name} " in stdout
            assert f"--job-name={name}-cont " in stdout
            assert f"--job-name={name}-cont2 " in stdout
            line = next(l for l in stdout.splitlines() if f"--job-name={name} " in l)
            assert "--time=4:00:00" in line and "slurm/serve.slurm" in line
            assert f"--config configs/shards/rollouts-{model}-{version}-s{i}of3.yaml" in line
            assert (f"--nice={nice}" in line) is (nice > 0)
    assert "sbatch --parsable" in stdout
```

- [ ] **Step 2: Run to verify failure**

Expected: `FileNotFoundError` / non-zero return (script missing).

- [ ] **Step 3: Write the script**

`scripts/mindset_cells.sh`:

```bash
#!/usr/bin/env bash
# Generate the mindset-arm shard configs and submit their slurm DAG.
#
#   scripts/mindset_cells.sh --dry-run    # write configs, print sbatch lines (default)
#   scripts/mindset_cells.sh --submit     # write configs, submit
#
# Cells (docs/superpowers/specs/2026-08-15-mindset-vectors-design.md §2):
#   priority 1  Ministral d6-base   growth6 resil6 appr6            nice 0
#   priority 2  Qwen3.5-9B d6-base  growth6 resil6 appr6            nice 2000
#   priority 3  Ministral aff6-base affgrowth6 affresil6 affappr6   nice 4000
# Each shard: primary (4h) -> -cont -> -cont2, chained afterany. Resume appends;
# an idle continuation exits after model load. Priority is honoured because the
# cluster runs PriorityType=priority/multifactor.
#
# The shard config is the model's base-cell config with max_tokens 24576 (the 2x2
# value; the cap never bound in d6), the `mindset:` key, and its own out_dir.
# Everything else is byte-identical, which is what keeps the cells comparable.
# Refuses to overwrite a shard config that already exists with different content.
#
# SHARD_DIR overrides where configs are written (tests use a temp dir).
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=dry
case "${1:-}" in --dry-run|"") MODE=dry ;; --submit) MODE=submit ;; *) echo "usage: $0 [--dry-run|--submit]" >&2; exit 2 ;; esac

SHARD_DIR="${SHARD_DIR:-configs/shards}"
mkdir -p "$SHARD_DIR"

# model|version|base_version|mindset|nice
CELLS=(
  "Ministral-3-14B-Reasoning-2512|growth6|d6|growth|0"
  "Ministral-3-14B-Reasoning-2512|resil6|d6|resilience|0"
  "Ministral-3-14B-Reasoning-2512|appr6|d6|appraisal|0"
  "Qwen3.5-9B|growth6|d6|growth|2000"
  "Qwen3.5-9B|resil6|d6|resilience|2000"
  "Qwen3.5-9B|appr6|d6|appraisal|2000"
  "Ministral-3-14B-Reasoning-2512|affgrowth6|aff6|growth|4000"
  "Ministral-3-14B-Reasoning-2512|affresil6|aff6|resilience|4000"
  "Ministral-3-14B-Reasoning-2512|affappr6|aff6|appraisal|4000"
)

write_config() {  # model version base mindset shard_index
  local model=$1 version=$2 base=$3 mindset=$4 i=$5
  local src="configs/shards/rollouts-${model}-${base}-s${i}of3.yaml"
  local dst="$SHARD_DIR/rollouts-${model}-${version}-s${i}of3.yaml"
  [[ -f "$src" ]] || { echo "missing base config $src" >&2; exit 1; }
  local tmp; tmp=$(mktemp)
  # max_tokens -> 24576; out_dir -> the new cell; then append the mindset block.
  sed -e 's/^max_tokens: .*/max_tokens: 24576/' \
      -e "s#^out_dir: .*#out_dir: /out/rollouts/${model}/${version}#" "$src" > "$tmp"
  cat >> "$tmp" <<EOF

# --- MINDSET ARM: ${mindset} (prompt v2) -------------------------------------
# One of Anastasia's three mindset blocks (experiments/step0_elicitation.py,
# copied verbatim into healthy_rl.rollouts.MINDSET), inserted into the turn-1
# instruction between the benchmark text and the affect request, and STRIPPED
# from the reminder the scaffold re-sends after each failure -- so the model sees
# it once per rollout. Compare this cell against ${model}/${base}: everything
# except this key and max_tokens (24576, the 2x2 value) is byte-identical to that
# cell's shard config. It is a demand characteristic pointing the opposite way
# from the affect prompt; read the probes, not the words.
mindset: [${mindset}]
EOF
  if [[ -f "$dst" ]] && ! cmp -s "$tmp" "$dst"; then
    echo "refusing to overwrite $dst: it exists with different content" >&2; rm -f "$tmp"; exit 1
  fi
  mv "$tmp" "$dst"
}

submit() {  # prints or runs sbatch; echoes job id (or DRYRUN)
  local -a cmd=(sbatch --parsable "$@")
  if [[ $MODE == dry ]]; then printf '%s\n' "${cmd[*]}"; printf 'DRYRUN\n' >&2; return 0; fi
  local id; id=$("${cmd[@]}"); printf '%s\n' "${id%%;*}"
}

SUMMARY=()
for row in "${CELLS[@]}"; do
  IFS='|' read -r model version base mindset nice <<< "$row"
  for i in 0 1 2; do
    write_config "$model" "$version" "$base" "$mindset" "$i"
    # The sbatch line always names the real location; SHARD_DIR only redirects
    # where this run WRITES (so a test can generate into a temp dir).
    cfg="configs/shards/rollouts-${model}-${version}-s${i}of3.yaml"
    name="${model}-${version}-s${i}"
    common=(--gres=gpu:A100-40G:2 --mem=96G --cpus-per-task=16 --time=4:00:00)
    [[ "$nice" -gt 0 ]] && common+=(--nice="$nice")
    stage=(slurm/serve.slurm --model "$model" --config "$cfg" --gpu-memory-utilization 0.90
           --stage "scripts/run_rollouts.py:$cfg")
    if [[ $MODE == dry ]]; then
      p=$(submit --job-name="$name" "${common[@]}" "${stage[@]}")
      c1=$(submit --job-name="$name-cont" --dependency=afterany:PRIMARY "${common[@]}" "${stage[@]}")
      c2=$(submit --job-name="$name-cont2" --dependency=afterany:CONT "${common[@]}" "${stage[@]}")
      printf '%s\n%s\n%s\n' "$p" "$c1" "$c2"
    else
      p=$(submit --job-name="$name" "${common[@]}" "${stage[@]}")
      c1=$(submit --job-name="$name-cont" --dependency=afterany:"$p" "${common[@]}" "${stage[@]}")
      c2=$(submit --job-name="$name-cont2" --dependency=afterany:"$c1" "${common[@]}" "${stage[@]}")
      SUMMARY+=("$(printf '| %s | %s | %s | %s | %s |' "$model" "$version" "s$i" "$p" "$c1 / $c2")")
    fi
  done
done

if [[ $MODE == submit ]]; then
  echo
  echo "| model | version | shard | primary | continuations |"
  echo "|---|---|---|---|---|"
  printf '%s\n' "${SUMMARY[@]}"
fi
```

`chmod +x scripts/mindset_cells.sh`.

- [ ] **Step 4: Run the test, then the script for real in dry-run mode**

Run the pytest file. Expected: pass. Then `bash scripts/mindset_cells.sh --dry-run` (writes into `configs/shards/`) and `git status --short configs/shards | wc -l` → 27. Spot-check one: `diff configs/shards/rollouts-Ministral-3-14B-Reasoning-2512-d6-s0of3.yaml configs/shards/rollouts-Ministral-3-14B-Reasoning-2512-growth6-s0of3.yaml` shows exactly `max_tokens`, `out_dir`, and the appended block. Confirm the config parses with the real loader: `PYTHONPATH=... .venv/bin/python -c "from healthy_rl.config import load_config; c=load_config('configs/shards/rollouts-Ministral-3-14B-Reasoning-2512-growth6-s0of3.yaml'); print(c['mindset'], c['max_tokens'], c['out_dir'])"` (check the actual loader function name in `src/healthy_rl/config.py` first).

- [ ] **Step 5: Commit**

```bash
git add scripts/mindset_cells.sh tests/cpu/test_mindset_cells.py configs/shards/rollouts-*-growth6-s*of3.yaml configs/shards/rollouts-*-resil6-s*of3.yaml configs/shards/rollouts-*-appr6-s*of3.yaml configs/shards/rollouts-*-affgrowth6-s*of3.yaml configs/shards/rollouts-*-affresil6-s*of3.yaml configs/shards/rollouts-*-affappr6-s*of3.yaml
git commit -m "Generate the mindset-arm shard configs and their slurm DAG

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Docs

**Files:**
- Modify: `docs/runs.md` (version table ~line 11–24; current state table ~92–116; "Record fields worth knowing" ~145–173)
- Modify: `docs/elicitation.md` ("Mindset prompts" bullet, ~line 118–128)

- [ ] **Step 1: runs.md — version table**

After the `affpos6` row add:

```
| `growth6`, `resil6`, `appr6` | 6 | mindset block (growth / resilience / appraisal, v2), otherwise `d6`: no affect prompt, conflicting split. Block sent on turn 1 only. Compare against `d6` |
| `affgrowth6`, `affresil6`, `affappr6` | 6 | same blocks with the affect prompt on, otherwise `aff6`. Compare against `aff6` |
```

Add a paragraph after the 2x2 section titled `## The mindset arms`:

```
Anastasia's three v2 mindset blocks (`experiments/step0_elicitation.py`, rendered
in `docs/prompts-v2.md`; the rollout versions in `docs/prompts-rollouts.md`)
inserted into the turn-1 instruction between the benchmark text and the affect
sentence, and stripped from the reminder the scaffold re-sends after each failed
attempt, so the model sees the block once per rollout. `mindset` and
`mindset_version` are on every record and summary; resume refuses to mix arms.
Everything else is the base cell's shard config byte for byte, except
`max_tokens` 24576 (the 2x2 value; the cap never bound in `d6`).

Read a mindset cell only against its base cell — `growth6` vs `d6`, `affgrowth6`
vs `aff6` — single-token, both positions, at t0 and first-to-last. The blocks
are demand characteristics pointing the opposite way from the affect prompt: a
cell whose *words* calm down while its trajectory does not is the decoupling
result, not a success. See `docs/interventions.md` §8.

Priority: Ministral `growth6`/`resil6`/`appr6` first, then Qwen3.5-9B's three
(`--nice=2000`), then Ministral's affect-on three (`--nice=4000`). Submitted
2026-08-15 by `scripts/mindset_cells.sh`; job ids in the table below.
```

- [ ] **Step 2: runs.md — current state rows and job table**

Add nine rows to the current-state table (records 0, notes "submitted 2026-08-15, priority 1/2/3") — the executor of Task 8 will fill job ids; leave a `### Mindset jobs` table with header `| model | version | shard | primary | continuations |` and a line "(filled by `scripts/mindset_cells.sh --submit` output)".

- [ ] **Step 3: runs.md — record fields**

Replace the paragraph beginning "Per-token projections onto all 14 directions are kept at **every** capture layer (~280 bytes/token)." with:

```
Per-token projections are on disk **only for records written from 2026-08-15
23:00 onward** (the mindset cells and anything run after that merge). The
rollout's `.npz` then holds, per turn and per capture layer,
`t{turn}_proj_L{n}` (P × 14, float16 — every hook row's projection onto the 14
directions), `t{turn}_norm_L{n}` (P, float32) and `t{turn}_kind_L{n}` (P, int8;
1 = the prefill row that produced the first generated token, 0 = a decode row).
Cosine at row *i* is `proj[i] / norm[i]`. Older records have only the boundary
residuals (`t{turn}_res_{start|end}_L{probe}`) — the hook always computed the
per-token arrays, but `summarise_hook_results` reduced them to `turn_stat` and
dropped them; an earlier version of this paragraph said they were kept, and was
wrong. Records also carry `turn_completion` (each turn's completion text) so the
rows can be re-tokenised offline and the count checked against the decode rows;
for reasoning models the completion may omit reasoning tokens, so a mismatch is
expected and must be reported. Exact token strings via `logprobs` are the
follow-up.
```

- [ ] **Step 4: elicitation.md**

In the "Mindset prompts" bullet, after "(`--mindset growth resilience appraisal` in the step 0 runner)", add: "and, since 2026-08-15, as the `mindset:` key of the rollout pipeline, which reads the probes under the same v2 blocks — cells `growth6`/`resil6`/`appr6` and the affect-on trio, registered in `docs/runs.md`."

- [ ] **Step 5: Commit**

```bash
git add docs/runs.md docs/elicitation.md
git commit -m "Register the mindset cells and correct what the rollout npz stores

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Merge and submit (done by the orchestrating session, not a subagent)

- [ ] Full suite green in the worktree: `... pytest tests/cpu -q -p no:cacheprovider`.
- [ ] Read `docs/prompts-rollouts.md` once more.
- [ ] From the worktree: `bash scripts/mindset_cells.sh --submit` (jobs bind the worktree; the worktree stays until every cell is complete). Paste the printed table into `docs/runs.md` "Mindset jobs", commit.
- [ ] Merge `feature/mindset-vectors` into `main` (fast-forward or merge commit) **from the main checkout only after** the peer's currently running jobs are unaffected — the change is default-off, so a merge is safe at any time; do it after submission so the peer's next continuations pick up the per-token arrays and the drift test.
- [ ] Start the monitoring loop (spec §6).
