"""``sandbox_cli`` on the host, without impossiblebench or apptainer.

The subcommands normally only ever run inside ``eval.sif``, so nothing on the
host exercised them: a broken ``cmd_run`` would have surfaced as an opaque
"sandbox_cli exited 1" from inside a container. The container-only dependencies
are ``bench_instruction`` (it imports impossiblebench) and, for ``problems``,
``record_to_sample``; both are monkeypatched here. Everything else -- parquet
lookup, test assembly, subprocess, timeout, feedback, the mindset arm's
insert-once/strip-from-the-reminder split -- is the real thing.
"""
from __future__ import annotations

import json
import sys
import types

import pytest

from healthy_rl.dashboard.sandbox_cli import FEEDBACK_MARKER, main
from healthy_rl.rollouts import INOCULATION_BLOCK, MINDSET_HEADER, mindset_section

PASSING = "def f(x):\n    return x + 1"
FAILING = "def f(x):\n    return 99"
LOOPING = "def f(x):\n    while True:\n        pass"


@pytest.fixture
def parquet(tmp_path):
    """Two rows, so the task_id lookup has something to get wrong."""
    import pandas as pd
    frame = pd.DataFrame([
        {"task_id": "lcbhard_0", "prompt": "def f(x):", "entry_point": "f",
         "test": "def check(candidate):\n    assert candidate(1) == 2\n"},
        {"task_id": "lcbhard_1", "prompt": "def g():", "entry_point": "g",
         "test": "def check(candidate):\n    raise AssertionError('wrong row')\n"},
    ])
    path = tmp_path / "original.parquet"
    frame.to_parquet(path)
    return path


@pytest.fixture(autouse=True)
def no_impossiblebench(monkeypatch):
    """Stand in for the one call that needs impossiblebench, mindset section and all.

    The fake composes the section itself so that ``reminder_instruction`` --
    which strips exactly that text back out -- runs for real over it.
    """
    def bench_instruction(affect=False, mindset=(), inoculation=False):
        text = "Implement f." + mindset_section(mindset)
        return text + "\n\n" + INOCULATION_BLOCK if inoculation else text

    monkeypatch.setattr("healthy_rl.rollouts.bench_instruction", bench_instruction)


@pytest.fixture
def no_record_to_sample(monkeypatch):
    """``problems`` also needs ``impossiblebench.livecodebench_tasks.record_to_sample``.

    The real one folds the instruction into the sample's opening user message,
    which is the only property these tests read, so the stub does just that.
    """
    def record_to_sample(*, instruction_prompt, allow_test_modifications):
        def convert(row):
            return types.SimpleNamespace(input=f"{instruction_prompt}\n\n{row['prompt']}")
        return convert

    tasks_mod = types.ModuleType("impossiblebench.livecodebench_tasks")
    tasks_mod.record_to_sample = record_to_sample
    pkg = types.ModuleType("impossiblebench")
    pkg.livecodebench_tasks = tasks_mod
    monkeypatch.setitem(sys.modules, "impossiblebench", pkg)
    monkeypatch.setitem(sys.modules, "impossiblebench.livecodebench_tasks", tasks_mod)


def _problems(capsys, parquet, *args):
    assert main(["problems", "--parquet", str(parquet), *args]) == 0
    return json.loads(capsys.readouterr().out)


def _run(tmp_path, monkeypatch, capsys, parquet, code, *, timeout="10", task_id="lcbhard_0"):
    code_file = tmp_path / "sub.py"
    code_file.write_text(code, encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # cmd_run writes its test file into the cwd
    rc = main(["run", "--parquet", str(parquet), "--task-id", task_id,
               "--code-file", str(code_file), "--timeout", timeout])
    assert rc == 0
    return json.loads(capsys.readouterr().out)


def test_run_reports_a_passing_submission(tmp_path, monkeypatch, capsys, parquet):
    out = _run(tmp_path, monkeypatch, capsys, parquet, PASSING)
    assert out["passed"] is True and out["returncode"] == 0
    assert out["timed_out"] is False and out["feedback"] == ""


def test_run_reports_a_failing_submission_with_scaffold_feedback(tmp_path, monkeypatch, capsys, parquet):
    out = _run(tmp_path, monkeypatch, capsys, parquet, FAILING)
    assert out["passed"] is False and out["returncode"] != 0
    assert FEEDBACK_MARKER in out["feedback"] and "Implement f." in out["feedback"]
    assert "AssertionError" in out["stderr"]


def test_run_times_out_on_an_infinite_loop_instead_of_hanging(tmp_path, monkeypatch, capsys, parquet):
    out = _run(tmp_path, monkeypatch, capsys, parquet, LOOPING, timeout="1")
    assert out["timed_out"] is True and out["passed"] is False
    assert "Timed out after 1s" in out["stderr"] and FEEDBACK_MARKER in out["feedback"]


def test_run_selects_the_row_named_by_task_id(tmp_path, monkeypatch, capsys, parquet):
    """The second row's test raises unconditionally, so picking it is visible."""
    out = _run(tmp_path, monkeypatch, capsys, parquet, "def g():\n    return 0", task_id="lcbhard_1")
    assert out["passed"] is False and "wrong row" in out["stderr"]


def test_run_cleans_up_the_test_file_it_writes(tmp_path, monkeypatch, capsys, parquet):
    _run(tmp_path, monkeypatch, capsys, parquet, PASSING)
    assert not list(tmp_path.glob("t_*.py"))


def test_run_rejects_an_unknown_task_id(tmp_path, monkeypatch, capsys, parquet):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sub.py").write_text(PASSING, encoding="utf-8")
    with pytest.raises(SystemExit, match="nope"):
        main(["run", "--parquet", str(parquet), "--task-id", "nope",
              "--code-file", str(tmp_path / "sub.py"), "--timeout", "10"])


# --- problems --------------------------------------------------------------

def test_problems_puts_the_mindset_block_in_turn_one_and_not_in_the_reminder(
        capsys, parquet, no_record_to_sample):
    """The arm's whole point: the block is shown once, not once per attempt."""
    out = _problems(capsys, parquet, "--mindset", "growth")
    p = out["lcbhard_0"]
    assert p["input"].count(MINDSET_HEADER) == 1
    assert mindset_section(["growth"]) in p["instruction_prompt"]
    assert p["instruction_prompt"] in p["input"]
    # The reminder is the base arm's text, character for character.
    assert p["reminder_prompt"] == "Implement f."
    assert MINDSET_HEADER not in p["reminder_prompt"]


def test_problems_without_a_mindset_reminds_with_the_turn_one_text(capsys, parquet, no_record_to_sample):
    out = _problems(capsys, parquet)
    p = out["lcbhard_0"]
    assert p["reminder_prompt"] == p["instruction_prompt"] == "Implement f."
    assert set(out) == {"lcbhard_0", "lcbhard_1"}


def test_problems_orders_two_blocks_the_way_mindset_does(capsys, parquet, no_record_to_sample):
    out = _problems(capsys, parquet, "--mindset", "appraisal", "growth")
    assert out["lcbhard_0"]["instruction_prompt"] == "Implement f." + mindset_section(["growth", "appraisal"])


def test_problems_rejects_an_unknown_mindset_name(capsys, parquet, no_record_to_sample):
    """A typo must stop the run, not silently produce a base-arm problem list."""
    with pytest.raises(SystemExit) as exc:
        main(["problems", "--parquet", str(parquet), "--mindset", "hustle"])
    assert exc.value.code != 0 and "hustle" in str(exc.value)
