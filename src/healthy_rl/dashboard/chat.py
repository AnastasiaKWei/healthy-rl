"""Free-form chat with the served model, one recorded generation per send."""
from __future__ import annotations

import queue
import threading
import uuid
from typing import Any, Iterator

from healthy_rl.dashboard.tasks import generate_with_heartbeat, public_record, record_for
from healthy_rl.rollouts import Vectors


class ChatSession:
    def __init__(self, engine, store, vectors: Vectors, *, conversation_id: str | None = None, title: str | None = None,
                 system_prompt: str | None = None, max_tokens: int = 2048, temperature: float = 0.0) -> None:
        self.engine, self.store, self.vectors = engine, store, vectors
        self.conversation_id = conversation_id or f"chat-{uuid.uuid4().hex[:8]}"
        self.title = title
        self.max_tokens, self.temperature = max_tokens, temperature
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}] if system_prompt else []
        self.turn = 0
        self._non_empty = 0

    def send(self, text: str) -> Iterator[dict]:
        self.messages.append({"role": "user", "content": text})
        events: queue.Queue = queue.Queue()
        box: dict[str, Any] = {}
        def work():
            box["gen"] = generate_with_heartbeat(self.engine, self.messages, max_tokens=self.max_tokens,
                                                 temperature=self.temperature, events=events, attempt=self.turn + 1)
            events.put(None)
        threading.Thread(target=work, daemon=True).start()
        while True:
            ev = events.get()
            if ev is None:
                break
            yield ev
        gen = box["gen"]
        nei = self._non_empty if gen.n_generated > 0 else None
        if gen.n_generated > 0:
            self._non_empty += 1
        rec = record_for(gen, conversation_id=self.conversation_id, source="chat", turn_index=self.turn,
                         non_empty_turn_index=nei, messages_in=self.messages, vectors=self.vectors,
                         condition={"max_tokens": self.max_tokens, "temperature": self.temperature},
                         title=self.title if self.turn == 0 else None)
        self.store.append(rec, gen.arrays(self.vectors.probe_layer))
        self.messages.append({"role": "assistant", "content": gen.text})
        self.turn += 1
        if gen.error and gen.n_generated == 0:
            yield {"event": "error", "data": {"message": gen.error}}
        yield {"event": "turn", "data": {"record": public_record(rec)}}
