"""One chat request through vllm-lens with the projection hook, returned as a Generation."""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from healthy_rl.dashboard.generation import Generation, assemble_generation, merge_hook_results
from healthy_rl.rollouts import Vectors


def _default_hook_factory(vectors: Vectors) -> Callable[[], Any]:
    def make():
        from healthy_rl.rollouts import make_projection_hook  # imports vllm_lens lazily
        return make_projection_hook(vectors.directions, vectors.capture_layers, [vectors.probe_layer])
    return make


class Engine:
    """Non-streaming ``/v1/chat/completions`` + per-request hook, one turn per call.

    ``client`` is a ``healthy_rl.server.LensClient`` (or anything with the same
    ``chat`` signature). The hook is rebuilt per request via ``hook_factory``
    (a closure serialised by value; cheap).
    """

    def __init__(self, client: Any, vectors: Vectors, *, hook_factory: Callable[[], Any] | None = None) -> None:
        self.client = client
        self.vectors = vectors
        self._hook_factory = hook_factory or _default_hook_factory(vectors)

    @property
    def model_name(self) -> str:
        return str(self.client.model)

    def generate(self, messages: list[dict], *, max_tokens: int, temperature: float) -> Generation:
        started = time.monotonic()
        try:
            hook = self._hook_factory()
            out = self.client.chat(
                messages, max_tokens=max_tokens, temperature=temperature,
                hooks=[hook] if hook is not None else None, logprobs=True,
            )
            choice = ((out.raw or {}).get("choices") or [{}])[0]
            message = choice.get("message") or {}
            tokens = [c.get("token", "") for c in ((out.logprobs or {}).get("content") or [])]
            hook_saved = merge_hook_results(out.hook_results)
        except Exception as exc:  # recorded, never raised: the turn must land in the store
            return _error_generation(f"{type(exc).__name__}: {exc}", self.vectors, time.monotonic() - started)
        return assemble_generation(
            text=out.text or "", reasoning_content=message.get("reasoning_content"), tokens=tokens,
            finish_reason=choice.get("finish_reason"), hook_saved=hook_saved,
            capture_layers=self.vectors.capture_layers, probe_layer=self.vectors.probe_layer,
            n_emotions=self.vectors.n_emotions, max_tokens=max_tokens, seconds=time.monotonic() - started,
        )


def _error_generation(error: str, vectors: Vectors, seconds: float) -> Generation:
    L, E = len(vectors.capture_layers), vectors.n_emotions
    return Generation(text="", reasoning=None, answer="", tokens=[], token_kind=[],
                      proj=np.zeros((0, L, E), np.float32), norm=np.zeros((0, L), np.float32),
                      proj_prefill=np.full((L, E), np.nan, np.float32), norm_prefill=np.full((L,), np.nan, np.float32),
                      res_start=None, res_end=None, n_generated=0, n_think=0, at_cap=False,
                      finish_reason=None, misaligned=False, error=error, seconds=seconds)
