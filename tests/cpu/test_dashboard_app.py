from __future__ import annotations

import json
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from healthy_rl.dashboard.app import AppState, HealthMonitor, _pump, create_app
from healthy_rl.dashboard.fake import FakeEngine, FakeSandbox
from healthy_rl.dashboard.store import SessionStore


def _state(tmp_path, engine=None, max_tokens=8, **kw):
    # max_tokens=8 is under the fake's 12-token ceiling, so its turns are all at_cap;
    # pass something above 12 for a turn that finishes on "stop".
    eng = engine or FakeEngine()
    store = SessionStore.create(tmp_path / "s", {"model": "fake", "emotions": eng.vectors.emotions, "probe_layer": 20})
    return AppState(engine=eng, sandbox=FakeSandbox(pass_on_attempt=2), store=store, vectors=eng.vectors,
                    cfg={"max_tokens": max_tokens, "max_attempts": 3, "temperature": 0.0}, **kw)


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
    assert s["mode"] == "live"


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
    g = a["groups"][0]
    assert a["emotions"] == ["desperate", "frustrated", "joyful"] and g["n_conversations"] == 2
    assert len(g["by_turn"]["mean"]) >= 1 and len(g["by_turn"]["mean"][0]) == 3 and g["delta"]["n"] == 2
    m = client.get("/api/aggregate", params={"source": "task", "split": "original", "position": "end", "stat": "mean", "segment": "answer"}).json()
    assert m["groups"][0]["delta"]["n"] == 2
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
    excluded = client.get("/api/aggregate", params=p).json()["groups"][0]
    assert excluded["excluded_cap"] == 2 and excluded["delta"]["n"] == 0
    assert len(excluded["delta"]["mean"]) == 3  # width held even with nothing to average
    kept = client.get("/api/aggregate", params={**p, "include_cap": "true"}).json()["groups"][0]
    # Asking for them back stops the exclusion, but a capped turn's last token was
    # never fed back through the model, so "end" has no residual row to read either.
    assert kept["excluded_cap"] == 0 and kept["delta"]["n"] == 0
    # The same turns are readable at a position the cap did not eat.
    start = client.get("/api/aggregate", params={**p, "position": "start"}).json()["groups"][0]
    assert start["excluded_cap"] == 0 and start["delta"]["n"] == 1


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


def test_a_misaligned_record_is_unreadable_not_a_500(tmp_path):
    """``token_kind`` shorter than the decode run is ``Generation.misaligned``."""
    # Above the fake's ceiling, so the turn finishes on "stop" and its end row is
    # real: what turn 1 loses is the misalignment, not the token cap.
    state = _state(tmp_path, max_tokens=64)
    client = TestClient(create_app(state))
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
    assert a["groups"][0]["by_turn"]["skipped"] == [0, 1]


def test_health_monitor_reports_the_failure_it_saw():
    mon = HealthMonitor("http://127.0.0.1:1", interval_s=0.01)
    assert mon.status() == {"ok": False, "last_ok_at": None, "last_error": "not polled yet"}
    mon.poll_once()
    s = mon.status()
    assert s["ok"] is False and s["last_ok_at"] is None and "not polled yet" not in s["last_error"]


def test_chat_send_to_an_unknown_or_non_chat_conversation(client):
    """A send must never invent an empty history under someone else's id."""
    assert client.post("/api/chat/nope/send", json={"text": "hi"}).status_code == 404
    with client.stream("POST", "/api/task/start", json={"split": "original", "task_id": "lcbhard_0",
                                                        "attempts": 2, "auto_continue": True}) as r:
        cid = [d for n, d in _sse(r) if n == "turn"][0]["record"]["conversation_id"]
    r = client.post(f"/api/chat/{cid}/send", json={"text": "hi"})
    assert r.status_code == 409 and "not a chat" in r.json()["detail"]


def test_chat_rehydrates_from_the_store_when_the_session_object_is_gone(client, state):
    with client.stream("POST", "/api/chat/new/send", json={"text": "hello", "title": "Hi"}) as r:
        cid = _sse(r)[-1][1]["record"]["conversation_id"]
    state.chats.clear()  # as after a restart, or a replay session opened read-write
    with client.stream("POST", f"/api/chat/{cid}/send", json={"text": "more"}) as r:
        rec = _sse(r)[-1][1]["record"]
    assert rec["turn_index"] == 1 and rec["non_empty_turn_index"] == 1
    roles_and_text = [(m["role"], m["content"]) for m in rec["messages_in"]]
    assert roles_and_text[0] == ("user", "hello")
    assert roles_and_text[1][0] == "assistant" and roles_and_text[-1] == ("user", "more")
    turns = client.get(f"/api/conversations/{cid}").json()["turns"]
    assert len(turns) == 2 and turns[0]["title"] == "Hi"


def test_bad_request_bodies_and_params_are_400_not_500(client):
    assert client.post("/api/task/start", json={"task_id": "lcbhard_0"}).status_code == 400
    assert client.post("/api/task/start", json={"split": "original"}).status_code == 400
    for bad in ({"attempts": "lots"}, {"max_tokens": "many"}, {"temperature": "warm"}):
        body = {"split": "original", "task_id": "lcbhard_0", **bad}
        assert client.post("/api/task/start", json=body).status_code == 400, bad
    assert client.post("/api/task/start", json={"split": "sideways", "task_id": "lcbhard_0"}).status_code == 400
    assert client.get("/api/aggregate", params={"source": "everything"}).status_code == 400
    assert client.get("/api/problems", params={"split": "sideways"}).status_code == 400


def test_sse_content_type(client):
    with client.stream("POST", "/api/chat/new/send", json={"text": "hi"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        _sse(r)


def test_pump_finishes_the_turn_when_the_stream_is_abandoned():
    """``ChatSession.send`` appends the record last, so an abandoned generator loses the turn.

    ``TestClient`` drains a StreamingResponse on close rather than cancelling it,
    so this cannot be provoked through a route -- both implementations pass at
    that level. The property is pinned here on ``_pump`` itself, against the
    bare generator that shows what it is protecting against.
    """
    import time

    def make(appended: list) -> "Iterator[dict]":
        def work():
            yield {"event": "queued", "data": {}}
            time.sleep(0.2)
            appended.append("record")  # ChatSession.send's store.append
            yield {"event": "turn", "data": {}}
        return work

    bare: list = []
    raw = make(bare)()
    assert next(raw)["event"] == "queued"
    raw.close()
    time.sleep(0.4)
    assert bare == [], "a bare generator is expected to lose the record; the contrast is the point"

    pumped: list = []
    stream = _pump(make(pumped))
    assert next(stream)["event"] == "queued"
    stream.close()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not pumped:
        time.sleep(0.02)
    assert pumped == ["record"]


def test_replay_sessions_answer_409_for_problem_lists_not_500(tmp_path):
    """A replay session has no sandbox, and the problem list lives inside one.

    Without the guard the ``None.problems`` attribute error surfaces as a 500,
    which reads as a broken dashboard rather than a read-only one.
    """
    state = _state(tmp_path, read_only=True)
    state.sandbox = None
    client = TestClient(create_app(state))
    r = client.get("/api/problems", params={"split": "original"})
    assert r.status_code == 409 and "read-only" in r.json()["detail"]
    r2 = client.post("/api/task/start", json={"split": "original", "task_id": "lcbhard_0"})
    assert r2.status_code == 409 and "read-only" in r2.json()["detail"]


def test_a_record_written_under_a_different_emotion_order_is_refused_not_relabelled(client, state):
    """Reordered directions would draw joy in despair's column, silently.

    docs/runs.md says the dashboard checks ``emotions`` the way the rollout
    analysis does; this is that check.
    """
    with client.stream("POST", "/api/chat/new/send", json={"text": "hello"}) as r:
        rec = _sse(r)[-1][1]["record"]
    arrays = state.store.arrays(rec["record_id"])
    bad = {k: v for k, v in rec.items() if k != "record_id"}
    bad["emotions"] = list(reversed(rec["emotions"]))
    bad["turn_index"] = 1
    state.store.append(bad, arrays)
    turns = client.get(f"/api/conversations/{rec['conversation_id']}").json()["turns"]
    assert turns[0]["emotion_order_mismatch"] is False
    assert turns[1]["emotion_order_mismatch"] is True
    assert all(v is None for v in turns[1]["readouts"]["desperate"].values())
    # And it is skipped-and-counted in the aggregate, never quietly relabelled.
    a = client.get("/api/aggregate", params={"source": "chat", "position": "start"}).json()
    assert a["groups"][0]["by_turn"]["skipped"] == [0, 1]


def test_rehydrated_chat_replays_the_answer_when_the_server_parsed_the_reasoning(client, state):
    """The stored ``text`` is reasoning + answer for display; only the answer goes back."""
    with client.stream("POST", "/api/chat/new/send", json={"text": "hello"}) as r:
        rec = _sse(r)[-1][1]["record"]
    cid = rec["conversation_id"]
    arrays = state.store.arrays(rec["record_id"])
    parsed = {k: v for k, v in rec.items() if k != "record_id"}
    parsed.update(turn_index=1, reasoning_from_parser=True, answer="the answer alone",
                  text="secret reasoning\n\nthe answer alone")
    state.store.append(parsed, arrays)
    state.chats.clear()
    with client.stream("POST", f"/api/chat/{cid}/send", json={"text": "more"}) as r:
        nxt = _sse(r)[-1][1]["record"]
    assistant = [m for m in nxt["messages_in"] if m["role"] == "assistant"]
    assert assistant == [{"role": "assistant", "content": "the answer alone"}]


def test_problems_pass_the_mindset_arm_through_and_cache_per_arm(tmp_path):
    """Two arms are two different problem lists; one cache entry would serve the wrong text."""
    state = _state(tmp_path)
    calls = []
    real = state.sandbox.problems

    def spy(split, affect=False, mindset=()):
        calls.append((split, affect, tuple(mindset)))
        return real(split, affect=affect, mindset=mindset)

    state.sandbox.problems = spy
    client = TestClient(create_app(state))
    p = client.get("/api/problems", params={"split": "original", "mindset": "growth"}).json()
    assert p["mindset"] == ["growth"] and p["problems"][0]["task_id"] == "lcbhard_0"
    client.get("/api/problems", params={"split": "original", "mindset": "growth"})
    base = client.get("/api/problems", params={"split": "original"}).json()
    assert base["mindset"] == []
    assert calls == [("original", False, ("growth",)), ("original", False, ())]


def test_an_unknown_mindset_name_is_a_400_not_a_500(client):
    """The container would exit nonzero on it, which surfaces as a broken dashboard."""
    r = client.post("/api/task/start", json={"split": "original", "task_id": "lcbhard_0", "mindset": ["hustle"]})
    assert r.status_code == 400 and "hustle" in r.json()["detail"]
    assert client.get("/api/problems", params={"split": "original", "mindset": "hustle"}).status_code == 400


def test_task_start_carries_the_mindset_into_the_condition(client):
    from healthy_rl.rollouts import MINDSET_VERSION
    body = {"split": "original", "task_id": "lcbhard_0", "attempts": 1, "auto_continue": True,
            "mindset": ["growth"]}
    with client.stream("POST", "/api/task/start", json=body) as r:
        rec = [d for n, d in _sse(r) if n == "turn"][0]["record"]
    assert rec["condition"]["mindset"] == ["growth"]
    assert rec["condition"]["mindset_version"] == MINDSET_VERSION


from rollout_cell import EMOTIONS, FakeEvalSamples, WhitespaceTokenizer, make_cell

from healthy_rl.dashboard.rollout_store import RolloutStore

RROWS = [
    {"task_id": "lcbhard_0", "sample": 0, "completions": ["a b c", "[THINK]x y[/THINK] z"], "passed": False},
    {"task_id": "lcbhard_0", "sample": 1, "completions": ["p q", "r s t u"], "passed": True, "bench_split": "original"},
]


def _rollout_client(tmp_path):
    make_cell(tmp_path / "r", "m-a", "appr6", rows=RROWS[:1])
    make_cell(tmp_path / "r", "m-a", "d6", rows=RROWS[:1], token_arrays=False)
    make_cell(tmp_path / "r", "m-b", "appr6", rows=RROWS, capture_layers=(5, 15), probe_layer=15)
    store = RolloutStore.open([tmp_path / "r"], tokenizer_loader=lambda m: WhitespaceTokenizer(),
                              vectors_loader=lambda m: None, eval_loader=FakeEvalSamples({}))
    st = AppState(engine=None, sandbox=None, store=store, vectors=None, cfg={}, read_only=True, mode="rollouts")
    return TestClient(create_app(st))


def test_rollouts_session_and_conversations(tmp_path):
    c = _rollout_client(tmp_path)
    s = c.get("/api/session").json()
    assert s["mode"] == "rollouts" and s["read_only"] is True
    assert set(s["session"]["models"]) == {"m-a", "m-b"} and s["session"]["models"]["m-b"]["probe_layer"] == 15
    assert len(s["session"]["cells"]) == 3 and s["emotions"] == list(EMOTIONS)
    convs = c.get("/api/conversations").json()["conversations"]
    assert len(convs) == 4 and all(x["source"] == "rollout" for x in convs)
    assert len(c.get("/api/conversations", params={"model": "m-b"}).json()["conversations"]) == 2
    assert len(c.get("/api/conversations", params={"model": "m-a", "version": "d6"}).json()["conversations"]) == 1


def test_rollouts_conversation_readouts_at_own_probe_layer(tmp_path):
    c = _rollout_client(tmp_path)
    conv = c.get("/api/conversations/m-b/appr6/lcbhard_0/s0").json()
    t = conv["turns"][1]
    assert t["probe_layer"] == 15 and t["has_token_arrays"] is True and t["misaligned"] is False
    assert isinstance(t["readouts"]["desperate"]["start"], float) and isinstance(t["readouts"]["desperate"]["think_end"], float)
    assert t["tokens"][-1] == "<eos>" and t["emotion_order_mismatch"] is False
    old = c.get("/api/conversations/m-a/d6/lcbhard_0/s0").json()["turns"][0]
    assert old["has_token_arrays"] is False and old["readouts"]["desperate"]["start"] is None   # no vectors loaded
    assert any("vectors" in w for w in old["warnings"])
    assert c.get("/api/conversations/nope").status_code == 404


def test_rollouts_tokens_route_validates_record_layers(tmp_path):
    c = _rollout_client(tmp_path)
    rid = "m-b/appr6/lcbhard_0/s0/t1"
    p = c.get(f"/api/records/{rid}/tokens").json()
    assert p["layer"] == 15 and len(p["tokens"]) == 4 and len(p["cosine"]) == 4 and p["markers"]["think_end"] == 1
    assert c.get(f"/api/records/{rid}/tokens", params={"layer": 5}).status_code == 200
    assert c.get(f"/api/records/{rid}/tokens", params={"layer": 20}).status_code == 400
    assert c.get("/api/records/nope/tokens").status_code == 404


def test_rollouts_mode_refuses_generation(tmp_path):
    c = _rollout_client(tmp_path)
    assert c.post("/api/chat/new/send", json={"text": "x"}).status_code == 409
    assert c.post("/api/task/start", json={"split": "original", "task_id": "lcbhard_0"}).status_code == 409
    assert c.get("/api/problems").status_code == 409


def test_aggregate_live_is_a_single_group(client):
    # Even with nothing recorded yet: the page reads groups[0] unconditionally.
    empty = client.get("/api/aggregate", params={"source": "chat"}).json()
    assert len(empty["groups"]) == 1 and empty["groups"][0]["n_records"] == 0
    with client.stream("POST", "/api/chat/new/send", json={"text": "hello"}) as r:
        r.read()
    a = client.get("/api/aggregate", params={"source": "chat"}).json()
    assert len(a["groups"]) == 1 and a["groups"][0]["n_conversations"] == 1
    g = a["groups"][0]
    assert g["model"] == "fake" and g["version"] is None and g["layer"] == 20
    assert "mean" in g["by_turn"] and "mean" in g["delta"] and a["emotions"] == ["desperate", "frustrated", "joyful"]


def test_aggregate_rollout_groups(tmp_path):
    c = _rollout_client(tmp_path)
    a = c.get("/api/aggregate", params={"source": "rollout", "split": "conflicting"}).json()
    keys = {(g["model"], g["version"]) for g in a["groups"]}
    assert keys == {("m-a", "appr6"), ("m-a", "d6"), ("m-b", "appr6")}
    gb = next(g for g in a["groups"] if g["model"] == "m-b")
    assert gb["layer"] == 15 and gb["n_conversations"] == 1 and gb["bench_split"] == "conflicting"
    ga = next(g for g in a["groups"] if (g["model"], g["version"]) == ("m-a", "appr6"))
    assert ga["layer"] == 20 and len(ga["by_turn"]["mean"]) == 2 and ga["skipped"] == 0
    old = next(g for g in a["groups"] if g["version"] == "d6")
    assert old["skipped"] == 2                                     # no vectors: both turns None, counted
    # filters
    a = c.get("/api/aggregate", params={"source": "rollout", "split": "conflicting", "model": ["m-b"]}).json()
    assert [(g["model"], g["version"]) for g in a["groups"]] == [("m-b", "appr6")]
    a = c.get("/api/aggregate", params={"source": "rollout", "split": "conflicting", "model": ["m-a"], "version": ["d6"]}).json()
    assert [(g["model"], g["version"]) for g in a["groups"]] == [("m-a", "d6")]
    # layer must exist for every selected model
    assert c.get("/api/aggregate", params={"source": "rollout", "split": "conflicting", "layer": 20}).status_code == 400
    assert "m-b" in c.get("/api/aggregate", params={"source": "rollout", "split": "conflicting", "layer": 20}).json()["detail"]
    assert c.get("/api/aggregate", params={"source": "rollout", "split": "conflicting", "model": ["m-a"], "layer": 10}).status_code == 200
    # splits are never pooled
    assert c.get("/api/aggregate", params={"source": "rollout"}).status_code == 400
    assert c.get("/api/aggregate", params={"source": "rollout", "split": "original"}).json()["groups"][0]["model"] == "m-b"
