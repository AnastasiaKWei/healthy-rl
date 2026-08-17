from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from healthy_rl.dashboard.sandbox import Sandbox, SandboxResult
from healthy_rl.dashboard.sandbox_cli import FEEDBACK_MARKER, assemble_test_code, feedback_message


def test_assemble_test_code_matches_solver_order():
    code = assemble_test_code("def f(x):\n    return x", "def check(c):\n    assert c(1) == 1", "def f(x):\n    return 1", "f")
    assert code == "def f(x):\n    return x\n    pass\ndef check(c):\n    assert c(1) == 1\ndef f(x):\n    return 1\ncheck(f)"


def test_feedback_message_is_the_scaffolds():
    m = feedback_message("AssertionError", "Do the thing")
    assert m.startswith(f"\n{FEEDBACK_MARKER}. Here's the error:\nAssertionError\n\n")
    assert m.endswith("\n\nTo reiterate, this is your task: Do the thing")
    assert "[unknown error]" in feedback_message("", "x")
    # One composer: the dashboard cannot drift from what the pipeline sends.
    from healthy_rl.rollouts import failure_message
    assert feedback_message("AssertionError", "Do the thing") == failure_message("AssertionError", "Do the thing")
    assert feedback_message("AssertionError", "Do the thing", "R.") == failure_message("AssertionError", "Do the thing", "R.")


def _sandbox(tmp_path, runner):
    return Sandbox(sif=tmp_path / "eval.sif", project_dir=tmp_path / "proj", bench_dir=tmp_path / "bench",
                   scratch_dir=tmp_path / "scratch", timeout_s=5, runner=runner)


def test_feedback_marker_is_the_rollout_scaffolds_marker():
    """The dashboard and the rollout scaffold must agree on the marker verbatim.

    Two copies of the same string is how a feedback message stops matching the
    one the pilot recorded, silently.
    """
    from healthy_rl.rollouts import TEST_FAILURE_MARKER
    assert FEEDBACK_MARKER == TEST_FAILURE_MARKER


def test_command_binds_project_bench_scratch_readonly_where_it_should(tmp_path):
    sb = _sandbox(tmp_path, runner=None)
    cmd = sb.command("problems", "--parquet", "/bench/orig1/original.parquet")
    assert cmd[:3] == ["apptainer", "exec", "--contain"]
    joined = " ".join(cmd)
    # Empty network namespace: model-generated code gets no network at all.
    assert cmd[cmd.index("--net") + 1:cmd.index("--net") + 3] == ["--network", "none"]
    # Only src is bound, so .env (HF_TOKEN) is never inside the container.
    assert f"{tmp_path/'proj'}/src:/project/src:ro" in joined
    assert f"{tmp_path/'proj'}:/project:ro" not in joined
    assert f"{tmp_path/'bench'}:/bench:ro" in joined
    assert f"{tmp_path/'scratch'}:/scratch:rw" in joined and "--pwd /scratch" in joined
    assert "PYTHONPATH=/project/src" in joined
    assert "HOME=" not in joined  # apptainer refuses APPTAINERENV_HOME; the image's /work is already a tmpfs
    assert cmd[cmd.index(str(tmp_path / "eval.sif")) + 1:] == ["python", "-m", "healthy_rl.dashboard.sandbox_cli", "problems", "--parquet", "/bench/orig1/original.parquet"]


def test_problems_parses_json(tmp_path):
    payload = {"lcbhard_0": {"input": "x", "prompt": "p", "test": "t", "entry_point": "f", "impossible_type": "original", "instruction_prompt": "i"}}
    def runner(cmd, **kw):
        assert "/bench/orig1/original.parquet" in cmd and "--affect" not in cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")
    assert _sandbox(tmp_path, runner).problems("original") == payload


def test_each_split_resolves_to_its_own_bench_subdirectory(tmp_path):
    seen = []
    def runner(cmd, **kw):
        seen.append(cmd[cmd.index("--parquet") + 1])
        return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")
    sb = _sandbox(tmp_path, runner)
    sb.problems("conflicting")
    sb.problems("original")
    assert seen == ["/bench/v1/conflicting.parquet", "/bench/orig1/original.parquet"]
    assert sb.parquet_for("conflicting") == "/bench/v1/conflicting.parquet"


def test_unknown_split_raises_before_touching_the_runner(tmp_path):
    def never(cmd, **kw):
        raise AssertionError("runner must not be invoked for an unknown split")
    sb = _sandbox(tmp_path, never)
    with pytest.raises(ValueError, match="unknown split 'nope'"):
        sb.parquet_for("nope")
    with pytest.raises(ValueError, match="unknown split 'nope'"):
        sb.problems("nope")
    with pytest.raises(ValueError, match="unknown split 'nope'"):
        sb.run("nope", "lcbhard_0", "x")
    assert not list((tmp_path / "scratch").glob("*.py"))


def test_run_writes_code_file_passes_container_path_and_cleans_up(tmp_path):
    seen = {}
    def runner(cmd, **kw):
        i = cmd.index("--code-file"); seen["path"] = cmd[i + 1]
        host = tmp_path / "scratch" / Path(cmd[i + 1]).name
        seen["content"] = host.read_text()
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"passed": False, "returncode": 1, "stdout": "", "stderr": "AssertionError", "feedback": "fb", "timed_out": False}), stderr="")
    r = _sandbox(tmp_path, runner).run("conflicting", "lcbhard_2", "def f(): pass")
    assert isinstance(r, SandboxResult) and r.passed is False and r.feedback == "fb" and r.stderr == "AssertionError"
    assert seen["path"].startswith("/scratch/") and seen["content"] == "def f(): pass"
    assert not list((tmp_path / "scratch").glob("*.py"))


def test_run_forwards_affect_and_mindset_to_the_cli(tmp_path):
    seen = {}
    def runner(cmd, **kw):
        seen["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"passed": False, "returncode": 1, "stdout": "", "stderr": "E", "feedback": "fb", "timed_out": False}), stderr="")
    _sandbox(tmp_path, runner).run("conflicting", "lcbhard_2", "def f(): pass", affect=True, mindset=("growth",))
    tail = seen["cmd"][seen["cmd"].index("run"):]
    assert "--affect" in tail
    assert tail[tail.index("--mindset") + 1] == "growth"
    _sandbox(tmp_path, runner).run("conflicting", "lcbhard_2", "def f(): pass")
    assert "--mindset" not in seen["cmd"] and "--affect" not in seen["cmd"]


def test_run_timeout_and_garbage_are_errors_not_exceptions(tmp_path):
    def slow(cmd, **kw): raise subprocess.TimeoutExpired(cmd, kw.get("timeout", 0))
    r = _sandbox(tmp_path, slow).run("original", "lcbhard_0", "x")
    assert r.timed_out and r.passed is False and r.error
    def garbage(cmd, **kw): return subprocess.CompletedProcess(cmd, 1, stdout="not json", stderr="boom")
    r2 = _sandbox(tmp_path, garbage).run("original", "lcbhard_0", "x")
    assert r2.passed is False and "boom" in r2.error


def test_valid_json_of_the_wrong_shape_is_an_error_not_an_exception(tmp_path):
    for payload in ("{}", "null", "[]", '{"lcbhard_0": {"prompt": "p"}}'):
        def runner(cmd, __p=payload, **kw):
            return subprocess.CompletedProcess(cmd, 0, stdout=__p, stderr="")
        r = _sandbox(tmp_path, runner).run("original", "lcbhard_0", "x")
        assert r.passed is False and r.error and "unexpected payload" in r.error


def test_wrapper_timeout_sweeps_the_containers_orphaned_test_file(tmp_path):
    def slow(cmd, **kw):
        (tmp_path / "scratch" / "t_abc.py").write_text("orphan")
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout", 0))
    r = _sandbox(tmp_path, slow).run("original", "lcbhard_0", "x")
    assert r.timed_out and not (tmp_path / "scratch" / "t_abc.py").exists()
    assert not list((tmp_path / "scratch").glob("*.py"))
