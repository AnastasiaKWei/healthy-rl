"""``sandbox_cli run`` on the host, without impossiblebench or apptainer.

The subcommand normally only ever runs inside ``eval.sif``, so nothing on the
host exercised it: a broken ``cmd_run`` would have surfaced as an opaque
"sandbox_cli exited 1" from inside a container. The only container-only
dependency is ``bench_instruction`` (it imports impossiblebench), which is
monkeypatched here; everything else -- parquet lookup, test assembly,
subprocess, timeout, feedback -- is the real thing.
"""
from __future__ import annotations

import json

import pytest

from healthy_rl.dashboard.sandbox_cli import FEEDBACK_MARKER, main

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
    monkeypatch.setattr("healthy_rl.rollouts.bench_instruction", lambda affect=False: "Implement f.")


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
