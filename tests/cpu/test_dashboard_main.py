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
