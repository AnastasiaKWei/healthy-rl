from __future__ import annotations

import queue
import threading
import time

from healthy_rl.dashboard.chat import ChatSession
from healthy_rl.dashboard.fake import FakeEngine, FakeSandbox
from healthy_rl.dashboard.sandbox_cli import FEEDBACK_MARKER
from healthy_rl.dashboard.store import SessionStore
from healthy_rl.dashboard.tasks import TaskConfig, TaskRun


def _drain(q):
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def _setup(tmp_path, **kw):
    eng = FakeEngine()
    store = SessionStore.create(tmp_path / "s", {"model": "fake"})
    sb = FakeSandbox(pass_on_attempt=kw.pop("pass_on", None))
    cfg = TaskConfig(split="original", task_id="lcbhard_0", attempts=kw.pop("attempts", 3), max_tokens=8, **kw)
    run = TaskRun(cfg, sb.problems("original")["lcbhard_0"], eng, sb, store, eng.vectors)
    return run, eng, sb, store


def test_auto_continue_runs_to_exhaustion_and_records_every_attempt(tmp_path):
    run, eng, sb, store = _setup(tmp_path, auto_continue=True)
    run.run()
    names = [e["event"] for e in _drain(run.events)]
    assert names.count("turn") == 3 and names.count("tests") == 3 and names[-1] == "done"
    assert "awaiting_user" not in names
    assert run.state == "done" and run.passed is False
    recs = store.records()
    assert [r["attempt"] for r in recs] == [1, 2, 3] and all(r["bench_split"] == "original" for r in recs)
    assert recs[1]["messages_in"][-1]["role"] == "user" and FEEDBACK_MARKER in recs[1]["messages_in"][-1]["content"]
    assert recs[0]["messages_in"][0]["content"].startswith("Implement f.")
    assert recs[0]["source"] == "task" and recs[0]["condition"]["auto_continue"] is True
    assert recs[0]["non_empty_turn_index"] == 0 and recs[0]["emotions"] == eng.vectors.emotions
    assert recs[0]["warnings"] == [] and recs[0]["probe_layer"] == eng.vectors.probe_layer


def test_stops_early_on_pass(tmp_path):
    run, *_ = _setup(tmp_path, auto_continue=True, pass_on=2)
    run.run()
    ev = _drain(run.events)
    assert run.passed is True and ev[-1]["data"]["reason"] == "passed" and sum(e["event"] == "turn" for e in ev) == 2


def test_manual_mode_waits_and_inserts_intervention(tmp_path):
    run, eng, *_ = _setup(tmp_path, attempts=2)
    t = threading.Thread(target=run.run, daemon=True); t.start()
    deadline = time.time() + 5
    while run.state != "awaiting_user" and time.time() < deadline:
        time.sleep(0.01)
    assert run.state == "awaiting_user"
    run.resume("Please try a different approach.")
    t.join(5)
    assert run.state == "done"
    last_user = [m for m in eng.calls[1] if m["role"] == "user"][-1]["content"]
    assert last_user.startswith("Please try a different approach.") and FEEDBACK_MARKER in last_user
    assert run.store_records()[1]["user_intervention"] == "Please try a different approach."


def test_stop_ends_after_current_step(tmp_path):
    run, *_ = _setup(tmp_path, attempts=4)
    t = threading.Thread(target=run.run, daemon=True); t.start()
    while run.state != "awaiting_user":
        time.sleep(0.01)
    run.stop(); t.join(5)
    assert run.state == "stopped" and _drain(run.events)[-1]["data"]["reason"] == "stopped"


def test_scratchpad_flag_prepends_system_prompt(tmp_path):
    from healthy_rl.rollouts import SCRATCHPAD_SYSTEM_PROMPT
    run, eng, *_ = _setup(tmp_path, attempts=1, auto_continue=True, scratchpad=True)
    run.run()
    assert eng.calls[0][0] == {"role": "system", "content": SCRATCHPAD_SYSTEM_PROMPT}


def test_sandbox_error_pauses_without_retry(tmp_path):
    run, eng, sb, store = _setup(tmp_path, attempts=3)
    def broken(split, task_id, code, affect=False):
        from healthy_rl.dashboard.sandbox import SandboxResult
        return SandboxResult(False, "", "", "", timed_out=True, error="sandbox exceeded 60s")
    sb.run = broken
    t = threading.Thread(target=run.run, daemon=True); t.start()
    while run.state != "awaiting_user":
        time.sleep(0.01)
    ev = _drain(run.events)
    tests = [e for e in ev if e["event"] == "tests"][0]["data"]
    assert tests["timed_out"] and tests["error"] and tests["passed"] is False
    run.stop(); t.join(5)


def test_chat_session_records_and_keeps_history(tmp_path):
    eng = FakeEngine(); store = SessionStore.create(tmp_path / "s", {"model": "fake"})
    chat = ChatSession(eng, store, eng.vectors, title="Hello", max_tokens=6)
    ev = list(chat.send("hi there"))
    assert [e["event"] for e in ev][-1] == "turn" and ev[-1]["data"]["record"]["source"] == "chat"
    list(chat.send("and again"))
    assert [m["role"] for m in chat.messages] == ["user", "assistant", "user", "assistant"]
    recs = store.records()
    assert len(recs) == 2 and recs[1]["turn_index"] == 1 and recs[0]["title"] == "Hello"
    assert "arrays" in recs[0] and "proj" not in ev[-1]["data"]["record"]


def test_engine_error_ends_the_run_after_recording(tmp_path):
    run, eng, sb, store = _setup(tmp_path, auto_continue=True)
    real = eng.generate
    def dead(messages, *, max_tokens, temperature):
        gen = real(messages, max_tokens=max_tokens, temperature=temperature)
        gen.n_generated, gen.error = 0, "engine died"
        return gen
    eng.generate = dead
    run.run()
    ev = _drain(run.events)
    assert [e["event"] for e in ev if e["event"] != "generating"] == ["turn", "error", "done"]
    assert ev[-1]["data"]["reason"] == "error" and run.state == "error"
    assert len(store.records()) == 1 and sb.attempts == {}


def test_generating_heartbeats_while_the_engine_runs(tmp_path, monkeypatch):
    from healthy_rl.dashboard import tasks
    monkeypatch.setattr(tasks, "HEARTBEAT_S", 0.05)
    run, eng, *_ = _setup(tmp_path, attempts=1, auto_continue=True)
    real = eng.generate
    def slow(messages, *, max_tokens, temperature):
        time.sleep(0.4)
        return real(messages, max_tokens=max_tokens, temperature=temperature)
    eng.generate = slow
    run.run()
    beats = [e["data"] for e in _drain(run.events) if e["event"] == "generating"]
    assert len(beats) >= 2 and beats[0] == {"attempt": 1, "elapsed_s": 0.0}
    assert [b["elapsed_s"] for b in beats] == sorted(b["elapsed_s"] for b in beats)
