from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from healthy_rl.dashboard.app import AppState, HealthMonitor, create_app
from healthy_rl.dashboard.fake import FakeEngine, FakeSandbox
from healthy_rl.dashboard.store import SessionStore


def _state(tmp_path, **kw):
    eng = FakeEngine()
    store = SessionStore.create(tmp_path / "s", {"model": "fake", "emotions": eng.vectors.emotions, "probe_layer": 20})
    return AppState(engine=eng, sandbox=FakeSandbox(pass_on_attempt=2), store=store, vectors=eng.vectors,
                    cfg={"max_tokens": 8, "max_attempts": 3, "temperature": 0.0}, **kw)


@pytest.fixture
def state(tmp_path):
    return _state(tmp_path)


@pytest.fixture
def client(state):
    return TestClient(create_app(state))


def _sse(resp):
    events, name = [], None
    for line in resp.iter_lines():
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            events.append((name, json.loads(line[6:])))
    return events


def test_index_and_session(client):
    assert client.get("/").status_code == 200 and "text/html" in client.get("/").headers["content-type"]
    s = client.get("/api/session").json()
    assert s["session"]["model"] == "fake" and "health" in s and s["read_only"] is False
    assert s["emotions"] == ["desperate", "frustrated", "joyful"] and s["probe_layer"] == 20


def test_chat_roundtrip_and_conversation_readouts(client):
    with client.stream("POST", "/api/chat/new/send", json={"text": "hello", "title": "Hi"}) as r:
        ev = _sse(r)
    assert ev[0][0] == "queued" and ev[0][1]["conversation_id"].startswith("chat-")
    assert ev[-1][0] == "turn"
    cid = ev[-1][1]["record"]["conversation_id"]
    convs = client.get("/api/conversations").json()["conversations"]
    assert convs[0]["conversation_id"] == cid and convs[0]["title"] == "Hi"
    conv = client.get(f"/api/conversations/{cid}").json()
    t = conv["turns"][0]
    assert set(t["readouts"]) == {"desperate", "frustrated", "joyful"}
    assert set(t["readouts"]["desperate"]) == {"start", "think_end", "answer_start", "end"}
    assert isinstance(t["readouts"]["desperate"]["start"], float)
    with client.stream("POST", f"/api/chat/{cid}/send", json={"text": "more"}) as r:
        assert _sse(r)[-1][0] == "turn"
    assert len(client.get(f"/api/conversations/{cid}").json()["turns"]) == 2


def test_task_start_pauses_then_continue_then_done(client):
    with client.stream("POST", "/api/task/start", json={"split": "original", "task_id": "lcbhard_0", "attempts": 3}) as r:
        ev = _sse(r)
    names = [n for n, _ in ev]
    assert "turn" in names and "tests" in names and names[-1] == "awaiting_user"
    cid = [d for n, d in ev if n == "turn"][0]["record"]["conversation_id"]
    with client.stream("POST", f"/api/task/{cid}/continue", json={"intervention": None}) as r:
        ev2 = _sse(r)
    assert ev2[-1][0] == "done" and ev2[-1][1]["reason"] == "passed"
    conv = client.get(f"/api/conversations/{cid}").json()
    assert conv["conversation"]["passed"] is True and len(conv["turns"]) == 2


def test_task_auto_continue_streams_to_done_and_stop_endpoint_exists(client):
    with client.stream("POST", "/api/task/start", json={"split": "original", "task_id": "lcbhard_1", "attempts": 2, "auto_continue": True}) as r:
        ev = _sse(r)
    assert ev[-1][0] == "done"
    cid = [d for n, d in ev if n == "turn"][0]["record"]["conversation_id"]
    assert client.post(f"/api/task/{cid}/stop").status_code == 200


def test_tokens_endpoint(client):
    with client.stream("POST", "/api/chat/new/send", json={"text": "please think"}) as r:
        rid = _sse(r)[-1][1]["record"]["record_id"]
    t = client.get(f"/api/records/{rid}/tokens", params={"layer": 20}).json()
    assert len(t["tokens"]) == len(t["cosine"]) == len(t["token_kind"]) and len(t["cosine"][0]) == 3
    assert t["markers"]["think_end"] is not None and t["layer"] == 20
    smoothed = client.get(f"/api/records/{rid}/tokens", params={"layer": 20, "smooth": 3}).json()
    assert len(smoothed["cosine"]) == len(t["cosine"]) and smoothed["cosine"] != t["cosine"]
    assert client.get(f"/api/records/{rid}/tokens", params={"layer": 99}).status_code == 400
    assert client.get("/api/records/nope/tokens").status_code == 404


def test_aggregate_shapes_and_split_guard(client):
    for tid in ("lcbhard_0", "lcbhard_1"):
        with client.stream("POST", "/api/task/start", json={"split": "original", "task_id": tid, "attempts": 2, "auto_continue": True}) as r:
            _sse(r)
    a = client.get("/api/aggregate", params={"source": "task", "split": "original", "position": "start", "stat": "token", "segment": "all"}).json()
    assert a["emotions"] == ["desperate", "frustrated", "joyful"] and a["n_conversations"] == 2
    assert len(a["by_turn"]["mean"]) >= 1 and len(a["by_turn"]["mean"][0]) == 3 and a["delta"]["n"] == 2
    m = client.get("/api/aggregate", params={"source": "task", "split": "original", "position": "end", "stat": "mean", "segment": "answer"}).json()
    assert m["delta"]["n"] == 2
    with client.stream("POST", "/api/task/start", json={"split": "conflicting", "task_id": "lcbhard_0", "attempts": 1, "auto_continue": True}) as r:
        _sse(r)
    assert client.get("/api/aggregate", params={"source": "task"}).status_code == 400


def test_problems_and_health(client):
    p = client.get("/api/problems", params={"split": "original"}).json()
    assert p["split"] == "original" and p["problems"][0]["task_id"] == "lcbhard_0"
    assert "ok" in client.get("/api/health").json()


# --- guards and error paths ------------------------------------------------

def test_at_cap_turns_are_excluded_from_the_end_readout_unless_asked(client):
    """FakeEngine stops at the token cap, so every turn is ``at_cap``."""
    with client.stream("POST", "/api/task/start", json={"split": "original", "task_id": "lcbhard_0", "attempts": 2, "auto_continue": True}) as r:
        _sse(r)
    p = {"source": "task", "split": "original", "position": "end", "stat": "token"}
    excluded = client.get("/api/aggregate", params=p).json()
    assert excluded["excluded_cap"] == 2 and excluded["delta"]["n"] == 0
    assert len(excluded["delta"]["mean"]) == 3  # width held even with nothing to average
    kept = client.get("/api/aggregate", params={**p, "include_cap": "true"}).json()
    assert kept["excluded_cap"] == 0 and kept["delta"]["n"] == 1


def test_aggregate_rejects_bad_params(client):
    for bad in ({"position": "middle"}, {"stat": "median"}, {"segment": "preamble"}, {"layer": 99}):
        assert client.get("/api/aggregate", params={"source": "task", **bad}).status_code == 400


def test_conversation_selection_params_are_validated_and_echoed(client):
    with client.stream("POST", "/api/chat/new/send", json={"text": "hello"}) as r:
        cid = _sse(r)[-1][1]["record"]["conversation_id"]
    default = client.get(f"/api/conversations/{cid}").json()
    assert default["emotion"] == "desperate" and default["readout"] == "start"
    picked = client.get(f"/api/conversations/{cid}", params={"emotion": "joyful", "readout": "end"}).json()
    assert picked["emotion"] == "joyful" and picked["readout"] == "end"
    assert client.get(f"/api/conversations/{cid}", params={"emotion": "smug"}).status_code == 400
    assert client.get(f"/api/conversations/{cid}", params={"readout": "middle"}).status_code == 400
    assert client.get("/api/conversations/nope").status_code == 404


def test_task_continue_and_stop_require_a_live_run(client):
    assert client.post("/api/task/nope/continue", json={"intervention": None}).status_code == 404
    assert client.post("/api/task/nope/stop").status_code == 404
    with client.stream("POST", "/api/task/start", json={"split": "original", "task_id": "lcbhard_1", "attempts": 2, "auto_continue": True}) as r:
        cid = [d for n, d in _sse(r) if n == "turn"][0]["record"]["conversation_id"]
    r = client.post(f"/api/task/{cid}/continue", json={"intervention": None})
    assert r.status_code == 409


def test_task_start_404s_on_an_unknown_task_and_chat_400s_on_empty_text(client):
    assert client.post("/api/task/start", json={"split": "original", "task_id": "nope"}).status_code == 404
    assert client.post("/api/chat/new/send", json={"text": "   "}).status_code == 400


def test_read_only_sessions_refuse_to_generate(tmp_path):
    client = TestClient(create_app(_state(tmp_path, read_only=True)))
    assert client.get("/api/session").json()["read_only"] is True
    assert client.post("/api/chat/new/send", json={"text": "hi"}).status_code == 409
    assert client.post("/api/task/start", json={"split": "original", "task_id": "lcbhard_0"}).status_code == 409


def test_a_misaligned_record_is_unreadable_not_a_500(client, state):
    """``token_kind`` shorter than the decode run is ``Generation.misaligned``."""
    with client.stream("POST", "/api/chat/new/send", json={"text": "hello"}) as r:
        rec = _sse(r)[-1][1]["record"]
    arrays = state.store.arrays(rec["record_id"])
    bad = {k: v for k, v in rec.items() if k != "record_id"}
    bad["token_kind"] = bad["token_kind"][:-1]
    bad["turn_index"] = 1
    state.store.append(bad, arrays)
    conv = client.get(f"/api/conversations/{rec['conversation_id']}").json()
    assert conv["turns"][1]["readouts"]["desperate"]["end"] is None
    assert isinstance(conv["turns"][1]["readouts"]["desperate"]["start"], float)  # prefill needs no kinds
    a = client.get("/api/aggregate", params={"source": "chat", "position": "end", "stat": "token",
                                             "include_cap": "true"}).json()
    assert a["by_turn"]["skipped"] == [0, 1]


def test_health_monitor_reports_the_failure_it_saw():
    mon = HealthMonitor("http://127.0.0.1:1", interval_s=0.01)
    assert mon.status() == {"ok": False, "last_ok_at": None, "last_error": "not polled yet"}
    mon.poll_once()
    s = mon.status()
    assert s["ok"] is False and s["last_ok_at"] is None and "not polled yet" not in s["last_error"]
