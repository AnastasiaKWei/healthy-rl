from __future__ import annotations

import numpy as np

from healthy_rl.dashboard.engine import Engine
from healthy_rl.dashboard.fake import FakeEngine, FakeSandbox
from healthy_rl.dashboard.generation import Generation


class _Out:
    def __init__(self, text, saved, tokens, finish="stop", reasoning=None):
        self.text = text
        self.hook_results = {"0": saved}
        self.logprobs = {"content": [{"token": t, "logprob": -0.1} for t in tokens]}
        choice = {"finish_reason": finish, "message": {"content": text}}
        if reasoning is not None:
            choice["message"]["reasoning_content"] = reasoning
        self.raw = {"choices": [choice], "usage": {"completion_tokens": len(tokens)}}


class _Client:
    model = "fake-model"
    def __init__(self, out): self.out, self.calls = out, []
    def chat(self, messages, **kw): self.calls.append((messages, kw)); return self.out


def _saved(vectors, n=3):
    saved = {}
    E, L = vectors.n_emotions, len(vectors.capture_layers)
    for l in vectors.capture_layers:
        saved[f"proj_L{l}"] = np.ones((n + 1, E), np.float32)
        saved[f"norm_L{l}"] = np.ones(n + 1, np.float32)
        saved[f"kind_L{l}"] = np.array([1.0] + [0.0] * n, np.float32)
    return saved


def test_engine_sends_hook_and_logprobs_and_returns_generation():
    vectors = FakeEngine.default_vectors()
    client = _Client(_Out("hello", _saved(vectors), ["hel", "lo", "!"]))
    eng = Engine(client, vectors, hook_factory=lambda: "HOOK")
    g = eng.generate([{"role": "user", "content": "hi"}], max_tokens=16, temperature=0.0)
    assert isinstance(g, Generation) and g.n_generated == 3 and not g.misaligned
    messages, kw = client.calls[0]
    assert kw["hooks"] == ["HOOK"] and kw["logprobs"] is True and kw["max_tokens"] == 16
    assert eng.model_name == "fake-model"


def test_engine_reads_reasoning_content_and_finish_reason():
    vectors = FakeEngine.default_vectors()
    client = _Client(_Out("ans", _saved(vectors, 3), ["th", "ink", "ans"], finish="length", reasoning="think"))
    g = Engine(client, vectors, hook_factory=lambda: None).generate([], max_tokens=3, temperature=0.0)
    assert g.reasoning == "think" and g.at_cap and g.finish_reason == "length"


def test_engine_error_becomes_generation_error():
    class Boom(_Client):
        def chat(self, *a, **k): raise RuntimeError("server said no")
    g = Engine(Boom(None), FakeEngine.default_vectors(), hook_factory=lambda: None).generate([], max_tokens=4, temperature=0.0)
    assert g.error and "server said no" in g.error and g.n_generated == 0 and g.tokens == []


def test_fake_engine_is_deterministic_and_shaped():
    fe = FakeEngine()
    g1 = fe.generate([{"role": "user", "content": "please think"}], max_tokens=8, temperature=0)
    g2 = FakeEngine().generate([{"role": "user", "content": "please think"}], max_tokens=8, temperature=0)
    assert g1.text == g2.text and np.allclose(g1.proj, g2.proj)
    assert g1.n_think > 0 and g1.proj.shape == (g1.n_generated, 2, 3) and g1.norm_prefill.shape == (2,)
    assert len(fe.calls) == 1


def test_fake_sandbox_fails_then_passes():
    sb = FakeSandbox(pass_on_attempt=2)
    probs = sb.problems("original", affect=False)
    assert "lcbhard_0" in probs and "input" in probs["lcbhard_0"]
    r1 = sb.run("original", "lcbhard_0", "def f(): pass", affect=False)
    r2 = sb.run("original", "lcbhard_0", "def f(): pass", affect=False)
    assert r1.passed is False and "Your previous attempt failed the tests" in r1.feedback
    assert r2.passed is True
