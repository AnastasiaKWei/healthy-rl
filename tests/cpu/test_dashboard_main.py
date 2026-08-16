from __future__ import annotations

from fastapi.testclient import TestClient

from healthy_rl.dashboard.__main__ import build_state
from healthy_rl.dashboard.app import create_app


def test_fake_state_serves_and_records(tmp_path):
    state = build_state(fake=True, replay=None, session_dir=str(tmp_path / "s"), vectors_dir=None, cfg={"max_tokens": 6})
    c = TestClient(create_app(state))
    assert c.get("/api/session").json()["session"]["model"] == "fake-model"
    with c.stream("POST", "/api/chat/new/send", json={"text": "hi"}) as r:
        assert "event: turn" in r.read().decode()


def test_replay_state_is_read_only_and_reads_old_records(tmp_path):
    live = build_state(fake=True, replay=None, session_dir=str(tmp_path / "s"), vectors_dir=None, cfg={"max_tokens": 6})
    c = TestClient(create_app(live))
    with c.stream("POST", "/api/chat/new/send", json={"text": "hi"}) as r:
        r.read()
    live.store.close()
    replay = build_state(fake=False, replay=str(tmp_path / "s"), session_dir=None, vectors_dir=None, cfg={})
    rc = TestClient(create_app(replay))
    assert rc.get("/api/session").json()["read_only"] is True
    convs = rc.get("/api/conversations").json()["conversations"]
    assert len(convs) == 1
    cid = convs[0]["conversation_id"]
    assert rc.get(f"/api/conversations/{cid}").json()["turns"][0]["readouts"]
    assert rc.post("/api/chat/new/send", json={"text": "x"}).status_code == 409


def test_rollouts_state_opens_cells_read_only(tmp_path, capsys, monkeypatch):
    from rollout_cell import make_cell
    from healthy_rl.dashboard.__main__ import startup_report
    # build_state uses the default loaders, which stat $MODEL_DIR/$ARTIFACT_DIR: no login-node
    # test may read the real ones (Global Constraint), so they point at the tmp tree here.
    monkeypatch.setenv("MODEL_DIR", str(tmp_path)); monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    make_cell(tmp_path / "r", "m-a", "appr6", rows=[{"task_id": "lcbhard_0", "sample": 0, "completions": ["a b"], "passed": False}])
    st = build_state(fake=False, replay=None, session_dir=None, vectors_dir=None, cfg={}, rollouts=[str(tmp_path / "r")])
    assert st.mode == "rollouts" and st.read_only and st.vectors is None
    rep = startup_report(st.store)
    assert "m-a" in rep and "appr6" in rep and "tokenizer" in rep
    c = TestClient(create_app(st))
    assert c.get("/api/session").json()["mode"] == "rollouts"
    assert c.post("/api/chat/new/send", json={"text": "x"}).status_code == 409


def test_rollouts_state_with_no_cells_exits(tmp_path):
    import pytest
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit) as e:
        build_state(fake=False, replay=None, session_dir=None, vectors_dir=None, cfg={}, rollouts=[str(tmp_path / "empty")])
    assert e.value.code == 2


def test_startup_report_summarises_a_long_ignored_list(tmp_path):
    """The real rollouts root has ~200 scratch directories: one line per name is a wall of text."""
    from rollout_cell import make_cell
    from healthy_rl.dashboard.__main__ import startup_report
    make_cell(tmp_path / "r", "m-a", "appr6", rows=[{"task_id": "lcbhard_0", "sample": 0, "completions": ["a b"]}])
    for i in range(12):
        (tmp_path / "r" / f"empty{i:02d}").mkdir()
    st = build_state(fake=False, replay=None, session_dir=None, vectors_dir=None, cfg={}, rollouts=[str(tmp_path / "r")])
    assert len(st.store.session["ignored"]) == 12
    line = next(l for l in startup_report(st.store).splitlines() if l.startswith("ignored"))
    assert "empty00" in line and "empty11" not in line and "(+4 more)" in line
