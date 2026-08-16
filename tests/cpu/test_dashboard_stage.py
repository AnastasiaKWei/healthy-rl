"""Pure helpers of the ``scripts/dashboard.py`` serve stage.

Only the parts that run without a server: where the endpoint file lands, that a
missing vectors artifact stops the stage with a message naming the path (rather
than a traceback from deep inside ``load_vectors``), and that the tunnel command
the job prints is the one a user can paste.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from healthy_rl.config import repo_root

_spec = importlib.util.spec_from_file_location("_script_dashboard", repo_root() / "scripts" / "dashboard.py")
stage = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = stage
_spec.loader.exec_module(stage)


def test_write_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    p = stage.write_endpoint("m", "123", "node07", 41000)
    assert p == tmp_path / "serve" / "m" / "123" / "dashboard-endpoint"
    assert p.read_text().strip() == "node07:41000"


def test_startup_checks_names_missing_vectors(tmp_path):
    missing = tmp_path / "vectors" / "m" / "v1"
    with pytest.raises(SystemExit) as e:
        stage.startup_checks(missing)
    assert str(missing) in str(e.value)


def test_job_info_has_tunnel_cmd(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "77")
    monkeypatch.setenv("HEALTHY_RL_LOGIN_HOST", "login.example")
    info = stage.job_info(port=5000, node="della-l06g2")
    assert info["id"] == "77"
    assert info["tunnel_cmd"] == "ssh -L 5000:della-l06g2:5000 login.example"


def test_job_info_without_port_has_node_but_no_tunnel(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "78")
    info = stage.job_info()
    assert info["id"] == "78" and info["tunnel_cmd"] is None and info["node"]


# ---------------------------------------------------------------------------
# scripts/dashboard_tunnel.sh
# ---------------------------------------------------------------------------

TUNNEL = repo_root() / "scripts" / "dashboard_tunnel.sh"


def _tunnel(artifact_dir: Path | None, *args: str, env_file: Path | None = None,
            login_host: str | None = "login.example", squeue: str | None = None):
    """Run the helper with its ``.env`` pinned, so the repo's own never leaks in."""
    import subprocess

    env = {k: v for k, v in os.environ.items() if k not in ("ARTIFACT_DIR", "HEALTHY_RL_LOGIN_HOST")}
    env["HEALTHY_RL_ENV_FILE"] = str(env_file) if env_file else os.devnull
    if artifact_dir is not None:
        env["ARTIFACT_DIR"] = str(artifact_dir)
    if login_host is not None:
        env["HEALTHY_RL_LOGIN_HOST"] = login_host
    if squeue is not None:
        env["PATH"] = f"{squeue}:{env['PATH']}"
    return subprocess.run([str(TUNNEL), *args], capture_output=True, text=True, env=env)


def _stub_squeue(tmp_path: Path, *, queued: str = "") -> str:
    """A ``squeue`` on PATH that reports ``queued`` as the only live job id."""
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    stub = d / "squeue"
    stub.write_text(f'#!/bin/bash\nfor a in "$@"; do [[ "$a" == "{queued}" ]] && echo "{queued}"; done\nexit 0\n')
    stub.chmod(0o755)
    return str(d)


def _endpoint(artifact_dir: Path, model: str, job: str, text: str) -> Path:
    d = artifact_dir / "serve" / model / job
    d.mkdir(parents=True)
    (d / "dashboard-endpoint").write_text(text)
    return d / "dashboard-endpoint"


def test_tunnel_explains_when_no_endpoint_exists(tmp_path):
    """The no-match ``ls`` must not end the script before its explanation.

    Under ``set -euo pipefail`` an unguarded ``EP=$(ls ... | head -1)`` exits
    right there, with ls's status and nothing on stderr.
    """
    r = _tunnel(tmp_path)
    assert r.returncode == 1
    assert "no dashboard-endpoint found" in r.stderr


def test_tunnel_prints_the_command_for_a_job(tmp_path):
    _endpoint(tmp_path, "m", "77", "spock-01:41000\n")
    _endpoint(tmp_path, "m", "78", "spock-02:41001\n")
    r = _tunnel(tmp_path, "77", squeue=_stub_squeue(tmp_path, queued="77"))
    assert r.returncode == 0, r.stderr
    assert "ssh -L 41000:spock-01:41000 login.example" in r.stdout
    assert "http://localhost:41000" in r.stdout
    assert "stale" not in r.stderr


def test_tunnel_prefers_the_environment_over_the_env_file(tmp_path):
    """Both overrides documented at the top of the script, not just ARTIFACT_DIR.

    ``set -a; . .env`` clobbers whatever the caller exported, so each variable
    the script promises to honour has to be saved and restored by name.
    """
    env_file = tmp_path / "dot.env"
    env_file.write_text(f"ARTIFACT_DIR={tmp_path / 'wrong'}\nHEALTHY_RL_LOGIN_HOST=from-dot-env\n")
    _endpoint(tmp_path, "m", "77", "spock-01:41000\n")
    r = _tunnel(tmp_path, "77", env_file=env_file, squeue=_stub_squeue(tmp_path, queued="77"))
    assert r.returncode == 0, r.stderr
    assert "login.example" in r.stdout and "from-dot-env" not in r.stdout


def test_tunnel_falls_back_to_the_env_file(tmp_path):
    """...and still reads the file when the caller exported nothing."""
    env_file = tmp_path / "dot.env"
    env_file.write_text(f"ARTIFACT_DIR={tmp_path}\nHEALTHY_RL_LOGIN_HOST=from-dot-env\n")
    _endpoint(tmp_path, "m", "77", "spock-01:41000\n")
    r = _tunnel(None, "77", env_file=env_file, login_host=None, squeue=_stub_squeue(tmp_path, queued="77"))
    assert r.returncode == 0, r.stderr
    assert "ssh -L 41000:spock-01:41000 from-dot-env" in r.stdout


def test_tunnel_warns_that_a_dead_job_s_endpoint_is_stale(tmp_path):
    """An endpoint file outlives its job; without this you ssh into nothing."""
    _endpoint(tmp_path, "m", "77", "spock-01:41000\n")
    r = _tunnel(tmp_path, squeue=_stub_squeue(tmp_path, queued="99"))
    assert r.returncode == 0, r.stderr
    assert "ssh -L 41000:spock-01:41000 login.example" in r.stdout
    assert "77" in r.stderr and "stale" in r.stderr


# ---------------------------------------------------------------------------
# smoke() itself, driven by the fakes: everything but the GPU
# ---------------------------------------------------------------------------


def test_smoke_reports_ok_over_the_fakes(tmp_path, capsys):
    """The gate's own logic -- routes, record count, readout -- without a server.

    The GPU run can only tell you *that* the gate failed; this pins what a pass
    looks like, so a red gate on the node means the model/hook, not this script.
    """
    import json

    from healthy_rl.dashboard.app import AppState
    from healthy_rl.dashboard.fake import FakeEngine, FakeSandbox
    from healthy_rl.dashboard.store import SessionStore
    from healthy_rl.dashboard.__main__ import session_meta

    engine = FakeEngine()
    store = SessionStore.create(tmp_path / "s", session_meta(engine.vectors, "fake-model"))
    state = AppState(engine=engine, sandbox=FakeSandbox(pass_on_attempt=2), store=store,
                     vectors=engine.vectors, cfg={"max_tokens": 16, "max_attempts": 2, "temperature": 0.0})
    rc = stage.smoke(state)
    store.close()
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0, out
    assert out["smoke_ok"] is True and out["chat_turn_event"] and out["task_done_event"]
    assert out["n_records"] >= 2 and out["misaligned"] == []
    assert isinstance(out["first_start_readout"], float)


def test_smoke_still_prints_its_summary_when_nothing_was_recorded(tmp_path, capsys):
    """A gate that dies on ``recs[0]`` tells you nothing; it must report instead."""
    import json

    from healthy_rl.dashboard.app import AppState
    from healthy_rl.dashboard.fake import FakeEngine, FakeSandbox
    from healthy_rl.dashboard.store import SessionStore
    from healthy_rl.dashboard.__main__ import session_meta

    engine = FakeEngine()
    store = SessionStore.create(tmp_path / "s", session_meta(engine.vectors, "fake-model"))
    state = AppState(engine=engine, sandbox=FakeSandbox(pass_on_attempt=2), store=store,
                     vectors=engine.vectors, cfg={"max_tokens": 16, "max_attempts": 2, "temperature": 0.0},
                     read_only=True)  # every write is refused, so no record is ever written
    rc = stage.smoke(state)
    store.close()
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 1 and out["smoke_ok"] is False and out["n_records"] == 0
    assert out["first_start_readout"] is None


def _fake_state(tmp_path, engine=None, sandbox=None, **kw):
    from healthy_rl.dashboard.app import AppState
    from healthy_rl.dashboard.fake import FakeEngine, FakeSandbox
    from healthy_rl.dashboard.store import SessionStore
    from healthy_rl.dashboard.__main__ import session_meta

    engine = engine or FakeEngine()
    store = SessionStore.create(tmp_path / "s", session_meta(engine.vectors, "fake-model"))
    return AppState(engine=engine, sandbox=sandbox if sandbox is not None else FakeSandbox(pass_on_attempt=2),
                    store=store, vectors=engine.vectors,
                    cfg={"max_tokens": 16, "max_attempts": 2, "temperature": 0.0}, **kw)


def _smoke_json(capsys) -> dict:
    import json

    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_smoke_reports_a_sandbox_failure_instead_of_dying(tmp_path, capsys):
    """``Sandbox.problems`` raises RuntimeError on any apptainer/parquet trouble.

    ``_problems`` only converts ValueError, and TestClient re-raises server
    exceptions, so that RuntimeError lands in ``smoke()`` -- the gate's most
    likely first-contact failure, since the bench mapping has only ever run
    against the fakes. It has to become a verdict, not a traceback.
    """
    from healthy_rl.dashboard.fake import FakeSandbox

    class Broken(FakeSandbox):
        def problems(self, split, affect=False, mindset=()):
            raise RuntimeError(f"sandbox problems({split}) failed: apptainer: bind /bench: no such file")

    state = _fake_state(tmp_path, sandbox=Broken())
    rc = stage.smoke(state)
    state.store.close()
    out = _smoke_json(capsys)
    assert rc == 1 and out["smoke_ok"] is False
    assert "RuntimeError" in out["problems_error"] and "/bench" in out["problems_error"]
    assert out["task_done_event"] is False


def test_smoke_fails_when_a_turn_recorded_an_error(tmp_path, capsys):
    """A turn whose generation errored is a red gate, not a green one with a note."""
    from healthy_rl.dashboard.engine import _error_generation
    from healthy_rl.dashboard.fake import FakeEngine

    class Erroring(FakeEngine):
        """Healthy first turn, then errors: everything else about the run looks fine.

        Without folding ``errors`` into the verdict this run is a green gate --
        three records, none misaligned, a finite first readout -- over a model
        that stopped answering after the first turn.
        """

        def generate(self, messages, *, max_tokens, temperature):
            if not self.calls:
                return super().generate(messages, max_tokens=max_tokens, temperature=temperature)
            return _error_generation("RuntimeError: server said 500", self.vectors, 0.1)

    state = _fake_state(tmp_path, engine=Erroring())
    rc = stage.smoke(state)
    state.store.close()
    out = _smoke_json(capsys)
    assert out["n_records"] >= 2 and out["misaligned"] == []
    assert isinstance(out["first_start_readout"], float)  # the healthy first turn
    assert any("server said 500" in e for e in out["errors"])
    assert rc == 1 and out["smoke_ok"] is False


def test_smoke_reports_an_unreadable_conversation_instead_of_dying(tmp_path, capsys):
    """Same rule for the last step: a broken readout is a verdict, not a traceback."""
    state = _fake_state(tmp_path)
    broken = tmp_path / "s" / "arrays"

    class NoArrays:
        """A store whose rows are readable but whose arrays are gone (npz deleted)."""

        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def arrays(self, record_id):
            raise FileNotFoundError(f"{broken}/{record_id}.npz")

    state.store = NoArrays(state.store)
    rc = stage.smoke(state)
    state.store.close()
    out = _smoke_json(capsys)
    assert rc == 1 and out["smoke_ok"] is False
    assert "FileNotFoundError" in out["readout_error"]
    assert out["n_records"] >= 2  # the run itself was fine; only the read-back broke
