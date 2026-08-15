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


def _tunnel(artifact_dir: Path, *args: str):
    import subprocess

    env = {**os.environ, "ARTIFACT_DIR": str(artifact_dir), "HEALTHY_RL_LOGIN_HOST": "login.example"}
    return subprocess.run([str(TUNNEL), *args], capture_output=True, text=True, env=env)


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
    r = _tunnel(tmp_path, "77")
    assert r.returncode == 0, r.stderr
    assert "ssh -L 41000:spock-01:41000 login.example" in r.stdout
    assert "http://localhost:41000" in r.stdout


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
