"""A rollout's request timeout must outlast a full-length turn.

THE BUG THIS GUARDS. `request_timeout_s` was 600 s and, separately, never
reached the eval's model at all, so generations ran under the OpenAI SDK's own
600 s default. Measured on Qwen3.5-9B (2 x A100-40G, tp=2, 8 concurrent requests
carrying the projection hook): 12288 tokens took 761 s, i.e. 16.1 tok/s per
request. Anything over ~9700 tokens therefore exceeded the timeout, and the
client abandoned a request the server was still serving and retried it from
scratch -- forever.

That produced fourteen "hangs" in one night which looked model-specific
(Qwen only) and problem-specific (`lcbhard_7`/`10`/`11`), because those are
simply the models and problems whose turns run long enough to cross 600 s.
Ministral's ~1k-token turns never came close. See docs/infrastructure.md.

The hook is why the margin is thin: it costs 11% at one request in flight but
55% at eight, so concurrency is what pushes a long turn over the line.
"""

from __future__ import annotations

import glob

import pytest
import yaml

from healthy_rl.rollouts import (
    DEFAULT_REQUEST_TIMEOUT_S,
    MEASURED_HOOKED_TOKENS_PER_S,
    SDK_DEFAULT_TIMEOUT_S,
    request_timeout_s,
)

# Full-length turns are what time out; a turn that stops early was never at risk.
def seconds_for(max_tokens: int) -> float:
    return max_tokens / MEASURED_HOOKED_TOKENS_PER_S


def test_default_is_not_the_sdk_default():
    """600 s is the value that caused the hang; it must not be the fallback."""
    assert request_timeout_s({}) != int(SDK_DEFAULT_TIMEOUT_S)
    assert request_timeout_s({}) == int(DEFAULT_REQUEST_TIMEOUT_S)


def test_default_outlasts_a_full_length_turn_at_the_largest_cap():
    """24576 tokens is the largest cap any shipped config uses."""
    needed = seconds_for(24576)
    assert request_timeout_s({}) > needed, (
        f"default {request_timeout_s({})}s does not cover a 24576-token turn "
        f"({needed:.0f}s at {MEASURED_HOOKED_TOKENS_PER_S} tok/s)"
    )


def test_config_value_is_respected():
    assert request_timeout_s({"request_timeout_s": 1234}) == 1234


def test_eval_generate_config_sets_the_timeout():
    """Leaving `timeout` unset is the failure mode: the SDK silently uses 600."""
    pytest.importorskip("inspect_ai", reason="inspect_ai not installed")
    from healthy_rl.rollouts import eval_generate_config

    config = eval_generate_config({"max_tokens": 24576}, {})
    assert config.timeout == int(DEFAULT_REQUEST_TIMEOUT_S)


@pytest.mark.parametrize(
    "path", sorted(glob.glob("configs/shards/rollouts-*.yaml")) or ["<none found>"]
)
def test_every_shard_config_clears_its_own_max_tokens(path):
    """Each config's timeout must cover a full-length turn AT ITS OWN cap.

    Checked per file rather than globally: the caps differ (16384, 24576), so a
    single blanket number would pass a config it does not actually protect.
    """
    if path == "<none found>":
        pytest.skip("run from the repo root")
    cfg = yaml.safe_load(open(path))
    timeout = request_timeout_s(cfg)
    needed = seconds_for(int(cfg.get("max_tokens", 2048)))
    assert timeout > needed, (
        f"{path}: request_timeout_s={timeout}s cannot cover its own "
        f"max_tokens={cfg.get('max_tokens')} ({needed:.0f}s at "
        f"{MEASURED_HOOKED_TOKENS_PER_S} tok/s) -- this is the hang"
    )
