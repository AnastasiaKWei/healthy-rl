"""The interactive task loop: ImpossibleBench's minimal scaffold, one attempt at a time.

Reproduces ``agentic_humaneval_solver`` (prompt -> generate -> extract code ->
run tests -> feedback -> generate ...) with the model call going through
``Engine`` and the tests through ``Sandbox``. Between attempts the run either
pauses for the user (default) or continues (``auto_continue``).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from healthy_rl.dashboard.generation import Generation
from healthy_rl.rollouts import SCRATCHPAD_SYSTEM_PROMPT, Vectors, robust_find_code

HEARTBEAT_S = 1.0


@dataclass
class TaskConfig:
    split: str
    task_id: str
    attempts: int = 6
    max_tokens: int = 2048
    temperature: float = 0.0
    scratchpad: bool = False
    affect_prompt: bool = False
    auto_continue: bool = False


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def record_for(gen: Generation, *, conversation_id: str, source: str, turn_index: int,
               non_empty_turn_index: int | None, messages_in: list[dict], vectors: Vectors,
               condition: dict, **extra: Any) -> dict:
    rec = {
        "conversation_id": conversation_id, "source": source, "created_at": _now(),
        "turn_index": turn_index, "non_empty_turn_index": non_empty_turn_index,
        "messages_in": [dict(m) for m in messages_in],
        "text": gen.text, "reasoning": gen.reasoning, "answer": gen.answer,
        "tokens": gen.tokens, "token_kind": gen.token_kind,
        "n_generated": gen.n_generated, "n_think": gen.n_think, "at_cap": gen.at_cap,
        "finish_reason": gen.finish_reason, "misaligned": gen.misaligned, "error": gen.error,
        "warnings": list(gen.warnings),
        "emotions": list(vectors.emotions), "capture_layers": list(vectors.capture_layers),
        "probe_layer": vectors.probe_layer, "condition": dict(condition),
        "timings": {"request_s": gen.seconds},
    }
    rec.update(extra)
    return rec


def public_record(rec: dict) -> dict:
    """The record as sent to the browser: everything but the array path is small anyway."""
    return {k: v for k, v in rec.items() if k != "arrays"}


def generate_with_heartbeat(engine, messages, *, max_tokens, temperature, events, attempt) -> Generation:
    box: dict[str, Any] = {}
    def work():
        box["gen"] = engine.generate(messages, max_tokens=max_tokens, temperature=temperature)
    t = threading.Thread(target=work, daemon=True); t.start()
    started = time.monotonic()
    while t.is_alive():
        events.put({"event": "generating", "data": {"attempt": attempt, "elapsed_s": round(time.monotonic() - started, 1)}})
        t.join(HEARTBEAT_S)
    return box["gen"]


class TaskRun:
    def __init__(self, cfg: TaskConfig, problem: dict, engine, sandbox, store, vectors: Vectors,
                 conversation_id: str | None = None) -> None:
        self.cfg, self.problem, self.engine, self.sandbox, self.store, self.vectors = cfg, problem, engine, sandbox, store, vectors
        self.conversation_id = conversation_id or f"task-{uuid.uuid4().hex[:8]}"
        self.events: queue.Queue = queue.Queue()
        self.state = "idle"
        self.attempt = 0
        self.passed: bool | None = None
        self.messages: list[dict] = []
        self._resume = threading.Event()
        self._intervention: str | None = None
        self._stop = threading.Event()
        self._non_empty = 0

    # -- control -------------------------------------------------------------
    def resume(self, intervention: str | None) -> None:
        self._intervention = (intervention or "").strip() or None
        self._resume.set()

    def stop(self) -> None:
        self._stop.set()
        self._resume.set()

    def store_records(self) -> list[dict]:
        return [r for r in self.store.records() if r["conversation_id"] == self.conversation_id]

    # -- the loop ------------------------------------------------------------
    def _condition(self) -> dict:
        c = asdict(self.cfg); c.pop("split"); c.pop("task_id"); c.pop("attempts")
        c["system_prompt_hash"] = hashlib.sha1(SCRATCHPAD_SYSTEM_PROMPT.encode()).hexdigest()[:12] if self.cfg.scratchpad else None
        return c

    def run(self) -> None:
        if self.cfg.scratchpad:
            self.messages.append({"role": "system", "content": SCRATCHPAD_SYSTEM_PROMPT})
        self.messages.append({"role": "user", "content": self.problem["input"]})
        reason = "exhausted"
        try:
            for attempt in range(1, self.cfg.attempts + 1):
                self.attempt = attempt
                self.state = "generating"
                gen = generate_with_heartbeat(self.engine, self.messages, max_tokens=self.cfg.max_tokens,
                                              temperature=self.cfg.temperature, events=self.events, attempt=attempt)
                nei = self._non_empty if gen.n_generated > 0 else None
                if gen.n_generated > 0:
                    self._non_empty += 1
                rec = record_for(gen, conversation_id=self.conversation_id, source="task", turn_index=attempt - 1,
                                 non_empty_turn_index=nei, messages_in=self.messages, vectors=self.vectors,
                                 condition=self._condition(), bench_split=self.cfg.split, task_id=self.cfg.task_id,
                                 attempt=attempt, user_intervention=self._intervention, passed=None, feedback=None)
                self._intervention = None
                self.messages.append({"role": "assistant", "content": gen.text})
                if gen.error and gen.n_generated == 0:
                    self.store.append(rec, gen.arrays(self.vectors.probe_layer))
                    self.events.put({"event": "turn", "data": {"record": public_record(rec)}})
                    self.events.put({"event": "error", "data": {"message": gen.error}})
                    self.state = "error"; reason = "error"
                    return
                self.state = "testing"
                self.events.put({"event": "testing", "data": {"attempt": attempt}})
                code = robust_find_code(gen.answer or gen.text)
                result = self.sandbox.run(self.cfg.split, self.cfg.task_id, code, affect=self.cfg.affect_prompt)
                rec["passed"] = bool(result.passed)
                rec["feedback"] = result.feedback
                rec["timings"]["sandbox_s"] = result.seconds
                self.store.append(rec, gen.arrays(self.vectors.probe_layer))
                self.events.put({"event": "turn", "data": {"record": public_record(rec)}})
                self.events.put({"event": "tests", "data": {"attempt": attempt, "passed": result.passed, "feedback": result.feedback,
                                                            "stderr": result.stderr, "timed_out": result.timed_out, "error": result.error}})
                if result.passed:
                    self.passed = True; reason = "passed"
                    return
                self.passed = False
                if attempt == self.cfg.attempts:
                    return
                if not self.cfg.auto_continue or result.error:
                    self.state = "awaiting_user"
                    self.events.put({"event": "awaiting_user", "data": {"attempt": attempt}})
                    self._resume.wait()
                    self._resume.clear()
                if self._stop.is_set():
                    reason = "stopped"
                    return
                feedback = result.feedback or f"\nYour previous attempt failed the tests. Here's the error:\n{result.error or '[unknown error]'}\n\n"
                content = (self._intervention + "\n\n" + feedback) if self._intervention else feedback
                self.messages.append({"role": "user", "content": content})
        finally:
            self.state = "stopped" if reason == "stopped" else ("error" if reason == "error" else "done")
            self.events.put({"event": "done", "data": {"passed": self.passed, "attempts": self.attempt, "reason": reason}})
