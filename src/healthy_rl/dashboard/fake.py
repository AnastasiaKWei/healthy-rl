"""Deterministic stand-ins so the whole app runs on the login node without a GPU or apptainer."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from healthy_rl.dashboard.generation import Generation, assemble_generation
from healthy_rl.dashboard.sandbox import SandboxResult
from healthy_rl.dashboard.sandbox_cli import feedback_message
from healthy_rl.rollouts import Vectors

_WORDS = "let me look at this again the tests disagree so I cannot make both pass I will try once more".split()


class FakeEngine:
    """Same ``generate`` contract as ``Engine``, with canned text and shaped arrays."""

    def __init__(self, vectors: Vectors | None = None, *, seed: int = 0, seconds: float = 0.0) -> None:
        self.vectors = vectors or self.default_vectors()
        self.seed = seed
        self.seconds = seconds
        self.calls: list[list[dict]] = []

    @staticmethod
    def default_vectors() -> Vectors:
        d = np.zeros((3, 2, 8), np.float32); d[:, :, 0] = 1.0
        return Vectors(directions=d, emotions=["desperate", "frustrated", "joyful"], capture_layers=[10, 20],
                       probe_layer=20, mean_residual_norm={10: 1.0, 20: 1.0}, path=Path("fake"))

    model_name = "fake-model"

    def generate(self, messages: list[dict], *, max_tokens: int, temperature: float) -> Generation:
        self.calls.append(list(messages))
        rng = np.random.default_rng(self.seed + len(self.calls))
        prompt = " ".join(str(m.get("content", "")) for m in messages)
        n = int(min(max_tokens, 12))
        words = [_WORDS[i % len(_WORDS)] + " " for i in range(n)]
        if "think" in prompt.lower() and n >= 4:
            words = ["<think>"] + words[1:3] + ["</think>"] + words[4:]
        text = "".join(words)
        E = self.vectors.n_emotions
        capped = n == max_tokens
        # Mirror the server's row count: a capped generation never feeds its last
        # token back, so that token has no residual row and there are n - 1 decode
        # rows. res_end is NOT mirrored -- it is written unconditionally below,
        # where the server's hook would leave it at the last row it did see.
        rows = n - 1 if capped else n
        saved = {}
        for l in self.vectors.capture_layers:
            saved[f"proj_L{l}"] = rng.normal(scale=0.03, size=(rows + 1, E)).astype(np.float32) * 10
            saved[f"norm_L{l}"] = np.full(rows + 1, 10.0, np.float32)
            saved[f"kind_L{l}"] = np.array([1.0] + [0.0] * rows, np.float32)
        saved[f"res_start_L{self.vectors.probe_layer}"] = np.ones(8, np.float16)
        saved[f"res_end_L{self.vectors.probe_layer}"] = np.ones(8, np.float16)
        return assemble_generation(text=text, reasoning_content=None, tokens=words, finish_reason="length" if capped else "stop",
                                   hook_saved=saved, capture_layers=self.vectors.capture_layers, probe_layer=self.vectors.probe_layer,
                                   n_emotions=E, max_tokens=max_tokens, seconds=self.seconds)


class FakeSandbox:
    """Two canned problems that fail with a canned assertion until ``pass_on_attempt``."""

    def __init__(self, pass_on_attempt: int | None = None) -> None:
        self.pass_on_attempt = pass_on_attempt
        self.attempts: dict[str, int] = {}

    def problems(self, split: str, affect: bool = False, mindset: Sequence[str] = ()) -> dict:
        # ``mindset`` only changes the text the real sandbox builds, so the fake
        # takes it and ignores it; the reminder equals the instruction here
        # because the fake never carries a mindset block.
        return {
            "lcbhard_0": {"input": "Implement f.\n\n```\ndef f(x):\n    pass\n\ndef check(candidate):\n    assert candidate(1) == 2\n```",
                          "prompt": "def f(x):\n    pass", "test": "def check(candidate):\n    assert candidate(1) == 2",
                          "entry_point": "f", "impossible_type": split, "instruction_prompt": "Implement f.",
                          "reminder_prompt": "Implement f."},
            "lcbhard_1": {"input": "Implement g.", "prompt": "def g():\n    pass", "test": "def check(candidate):\n    assert candidate() == 0",
                          "entry_point": "g", "impossible_type": split, "instruction_prompt": "Implement g.",
                          "reminder_prompt": "Implement g."},
        }

    def run(self, split: str, task_id: str, code: str, affect: bool = False,
            mindset: Sequence[str] = ()) -> SandboxResult:
        # Taken and ignored, as in ``problems``: this fake never carries a mindset
        # block, so it has no reminder line to attach either.
        k = self.attempts.get(task_id, 0) + 1
        self.attempts[task_id] = k
        if self.pass_on_attempt is not None and k >= self.pass_on_attempt:
            return SandboxResult(True, "", "", "")
        err = "Traceback (most recent call last):\n  File \"t.py\", line 7, in <module>\nAssertionError"
        return SandboxResult(False, "", err, feedback_message(err, "Implement f."))
