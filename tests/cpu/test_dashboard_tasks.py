from __future__ import annotations

import queue
import threading
import time

import numpy as np

from healthy_rl.dashboard.chat import ChatSession
from healthy_rl.dashboard.fake import FakeEngine, FakeSandbox
from healthy_rl.dashboard.generation import assemble_generation
from healthy_rl.dashboard.sandbox import SandboxResult
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


def _spin(predicate, what, timeout=5.0):
    """Wait for a state transition with a deadline, so a regression fails instead of hanging."""
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        time.sleep(0.01)
    assert predicate(), f"timed out after {timeout}s waiting for {what}"


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
    assert recs[0]["condition"]["mindset"] == []  # the base arm names itself, rather than being silent
    assert recs[0]["non_empty_turn_index"] == 0 and recs[0]["emotions"] == eng.vectors.emotions
    # _setup caps at 8 tokens, which the fake always hits, so every turn carries the cap warning
    assert recs[0]["warnings"] == ["last token has no residual (generation hit max_tokens)"]
    assert recs[0]["probe_layer"] == eng.vectors.probe_layer


def test_stops_early_on_pass(tmp_path):
    run, *_ = _setup(tmp_path, auto_continue=True, pass_on=2)
    run.run()
    ev = _drain(run.events)
    assert run.passed is True and ev[-1]["data"]["reason"] == "passed" and sum(e["event"] == "turn" for e in ev) == 2


def test_manual_mode_waits_and_inserts_intervention(tmp_path):
    run, eng, *_ = _setup(tmp_path, attempts=2)
    t = threading.Thread(target=run.run, daemon=True); t.start()
    _spin(lambda: run.state == "awaiting_user", "the run to pause")
    assert run.resume("Please try a different approach.") is True
    t.join(5)
    assert run.state == "done"
    last_user = [m for m in eng.calls[1] if m["role"] == "user"][-1]["content"]
    assert last_user.startswith("Please try a different approach.") and FEEDBACK_MARKER in last_user
    assert run.store_records()[1]["user_intervention"] == "Please try a different approach."


def test_stop_ends_after_current_step(tmp_path):
    run, *_ = _setup(tmp_path, attempts=4)
    t = threading.Thread(target=run.run, daemon=True); t.start()
    _spin(lambda: run.state == "awaiting_user", "the run to pause")
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
        return SandboxResult(False, "", "", "", timed_out=True, error="sandbox exceeded 60s")
    sb.run = broken
    t = threading.Thread(target=run.run, daemon=True); t.start()
    _spin(lambda: run.state == "awaiting_user", "the run to pause")
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


def test_auto_continue_still_pauses_on_a_sandbox_error(tmp_path):
    run, eng, sb, store = _setup(tmp_path, attempts=3, auto_continue=True)
    sb.run = lambda split, task_id, code, affect=False: SandboxResult(
        False, "", "", "", timed_out=True, error="apptainer exec failed")
    t = threading.Thread(target=run.run, daemon=True); t.start()
    _spin(lambda: run.state == "awaiting_user", "the run to pause on the harness error")
    assert run.attempt == 1 and [e["event"] for e in _drain(run.events)][-1] == "awaiting_user"
    assert run.resume(None) is True
    _spin(lambda: len(eng.calls) >= 2, "the next attempt to start")
    last_user = [m for m in eng.calls[1] if m["role"] == "user"][-1]["content"]
    assert FEEDBACK_MARKER in last_user and "[harness error: apptainer exec failed]" in last_user
    assert "To reiterate, this is your task: Implement f." in last_user
    run.stop(); t.join(5)
    assert run.state == "stopped"


def test_resume_outside_awaiting_user_is_ignored(tmp_path):
    run, eng, *_ = _setup(tmp_path, attempts=2)
    real, release = eng.generate, threading.Event()
    def gated(messages, *, max_tokens, temperature):
        release.wait(5)
        return real(messages, max_tokens=max_tokens, temperature=temperature)
    eng.generate = gated
    t = threading.Thread(target=run.run, daemon=True); t.start()
    _spin(lambda: run.state == "generating", "the first generation to start")
    assert run.resume("stale click") is False
    release.set()
    _spin(lambda: run.state == "awaiting_user", "the run to pause")
    time.sleep(0.05)
    assert run.state == "awaiting_user" and run.attempt == 1
    assert [r["user_intervention"] for r in run.store_records()] == [None]
    run.stop(); t.join(5)
    assert run.state == "stopped"


def test_engine_exception_ends_the_run_with_an_error(tmp_path):
    run, eng, sb, store = _setup(tmp_path, auto_continue=True)
    def boom(messages, *, max_tokens, temperature):
        raise RuntimeError("connection reset")
    eng.generate = boom
    run.run()
    ev = _drain(run.events)
    assert run.state == "error" and ev[-1]["event"] == "done" and ev[-1]["data"]["reason"] == "error"
    errors = [e for e in ev if e["event"] == "error"]
    assert len(errors) == 1 and "connection reset" in errors[0]["data"]["message"]
    assert store.records() == [] and sb.attempts == {}


def test_chat_send_terminates_when_the_engine_raises(tmp_path):
    eng = FakeEngine(); store = SessionStore.create(tmp_path / "s", {"model": "fake"})
    def boom(messages, *, max_tokens, temperature):
        raise RuntimeError("connection reset")
    eng.generate = boom
    chat = ChatSession(eng, store, eng.vectors, max_tokens=6)
    box = {}
    t = threading.Thread(target=lambda: box.update(ev=list(chat.send("hi"))), daemon=True)
    t.start(); t.join(5)
    assert not t.is_alive(), "send() blocked forever on a raising engine"
    assert box["ev"][-1]["event"] == "error" and "connection reset" in box["ev"][-1]["data"]["message"]
    assert store.records() == []


def test_stop_during_generation_skips_the_pause(tmp_path):
    """stop() sets the resume flag whatever the state; clearing it must not lose the stop."""
    run, eng, *_ = _setup(tmp_path, attempts=3)
    real, release = eng.generate, threading.Event()
    def gated(messages, *, max_tokens, temperature):
        release.wait(5)
        return real(messages, max_tokens=max_tokens, temperature=temperature)
    eng.generate = gated
    t = threading.Thread(target=run.run, daemon=True); t.start()
    _spin(lambda: run.state == "generating", "the first generation to start")
    run.stop()
    release.set()
    t.join(5)
    assert not t.is_alive(), "stop() during generation left the run waiting for a resume"
    names = [e["event"] for e in _drain(run.events)]
    assert "awaiting_user" not in names and names[-1] == "done"
    assert run.state == "stopped" and len(eng.calls) == 1


class ReasoningEngine(FakeEngine):
    """A server with a reasoning parser: the two halves come back as two fields."""

    def generate(self, messages, *, max_tokens, temperature):
        gen = super().generate(messages, max_tokens=max_tokens, temperature=temperature)
        self.calls[-1] = list(messages)
        saved = {}
        for l in self.vectors.capture_layers:
            saved[f"proj_L{l}"] = np.zeros((3, self.vectors.n_emotions), np.float32)
            saved[f"norm_L{l}"] = np.full(3, 10.0, np.float32)
            saved[f"kind_L{l}"] = np.array([1.0, 0.0, 0.0], np.float32)
        return assemble_generation(
            text="```python\ndef f(x):\n    return 2\n```", reasoning_content="the tests disagree",
            tokens=["the tests disagree", "```python\ndef f(x):\n    return 2\n```"],
            finish_reason="stop", hook_saved=saved, capture_layers=self.vectors.capture_layers,
            probe_layer=self.vectors.probe_layer, n_emotions=self.vectors.n_emotions,
            max_tokens=max_tokens, seconds=0.0)


def test_task_loop_feeds_back_the_answer_not_the_chain_of_thought(tmp_path):
    """A reasoning-parser server must not have its own reasoning replayed at it.

    ``text`` joins the two halves for the transcript; feeding that back would put
    the chain of thought into the model's mouth as something it said out loud.
    """
    eng = ReasoningEngine()
    store = SessionStore.create(tmp_path / "s", {"model": "fake"})
    sb = FakeSandbox()
    cfg = TaskConfig(split="original", task_id="lcbhard_0", attempts=2, max_tokens=8, auto_continue=True)
    run = TaskRun(cfg, sb.problems("original")["lcbhard_0"], eng, sb, store, eng.vectors)
    run.run()
    assistant = [m for m in eng.calls[1] if m["role"] == "assistant"]
    assert len(assistant) == 1
    assert assistant[0]["content"] == "```python\ndef f(x):\n    return 2\n```"
    assert "the tests disagree" not in assistant[0]["content"]
    rec = store.records()[0]
    assert rec["reasoning_from_parser"] is True
    assert rec["text"] == "the tests disagree\n\n```python\ndef f(x):\n    return 2\n```"


def test_chat_feeds_back_the_answer_not_the_chain_of_thought(tmp_path):
    eng = ReasoningEngine()
    store = SessionStore.create(tmp_path / "s", {"model": "fake"})
    chat = ChatSession(eng, store, eng.vectors, max_tokens=8)
    list(chat.send("hi"))
    list(chat.send("again"))
    assistant = [m for m in eng.calls[1] if m["role"] == "assistant"]
    assert assistant[0]["content"] == "```python\ndef f(x):\n    return 2\n```"
    assert "the tests disagree" not in assistant[0]["content"]


def test_harness_error_record_carries_the_feedback_the_model_was_actually_sent(tmp_path):
    """A harness error still sends a synthesised message; the record must say so.

    The log is append-only, so a record written with ``feedback: None`` claims
    forever that nothing was fed back, while the next user message carries text.
    """
    run, eng, sb, store = _setup(tmp_path, attempts=3)
    def broken(split, task_id, code, affect=False):
        return SandboxResult(False, "", "", "", timed_out=True, error="sandbox exceeded 60s")
    sb.run = broken
    t = threading.Thread(target=run.run, daemon=True); t.start()
    _spin(lambda: run.state == "awaiting_user", "the run to pause")
    rec = store.records()[0]
    assert rec["feedback"] and "[harness error" in rec["feedback"]
    assert "sandbox exceeded 60s" in rec["feedback"]
    run.resume(None)
    _spin(lambda: len(store.records()) >= 2, "the next attempt to be recorded")
    run.stop(); t.join(5)
    # What the record claims was fed back is what the next prompt actually carried.
    assert store.records()[1]["messages_in"][-1]["content"] == rec["feedback"]


def test_condition_names_the_mindset_arm_and_the_prompt_version(tmp_path):
    """A mindset cell is only readable against its base cell if the record says which it is."""
    from healthy_rl.rollouts import MINDSET_VERSION
    run, eng, sb, store = _setup(tmp_path, attempts=1, auto_continue=True, mindset=("growth",))
    run.run()
    cond = store.records()[0]["condition"]
    assert cond["mindset"] == ["growth"] and cond["mindset_version"] == MINDSET_VERSION


def test_harness_error_feedback_uses_the_reminder_not_the_turn_one_instruction(tmp_path):
    """With a mindset arm the reminder must not carry the block (see strip_mindset_from_reminders).

    The sandbox's own feedback already comes from the reminder text; this is the
    fallback the task loop synthesises when the sandbox never ran.
    """
    run, eng, sb, store = _setup(tmp_path, attempts=3, mindset=("growth",))
    run.problem = dict(run.problem,
                       instruction_prompt="Implement f.\n\nHow to approach this:\n\nGrow.\n\n",
                       reminder_prompt="Implement f.")
    sb.run = lambda split, task_id, code, affect=False: SandboxResult(
        False, "", "", "", timed_out=True, error="apptainer exec failed")
    t = threading.Thread(target=run.run, daemon=True); t.start()
    _spin(lambda: run.state == "awaiting_user", "the run to pause on the harness error")
    feedback = store.records()[0]["feedback"]
    assert "To reiterate, this is your task: Implement f." in feedback
    assert "How to approach this:" not in feedback
    run.stop(); t.join(5)
