"""``request_timeout_s`` must reach the eval's model, not only the preflight.

Until 2026-08-16 the config key was read by the preflight ``LensClient`` alone,
so every rollout generation ran on the OpenAI SDK's default 600 s per-request
timeout. A turn longer than that was abandoned and retried against a server
that was still generating it -- the "Qwen-only hang" of docs/infrastructure.md.
These tests pin the wiring; they need no server, only the config object.
"""

from __future__ import annotations

from healthy_rl.rollouts import eval_generate_config, request_timeout_s


def test_request_timeout_s_reaches_the_generate_config():
    assert eval_generate_config({"request_timeout_s": 3600}, {}).timeout == 3600


def test_the_default_timeout_is_unchanged():
    # The control shards running tonight must keep identical behaviour.
    assert eval_generate_config({}, {}).timeout == 600
    assert request_timeout_s({}) == 600


def test_a_float_timeout_is_truncated_to_the_int_inspect_wants():
    # GenerateConfig.timeout is `int | None`; the YAML may carry a float.
    timeout = eval_generate_config({"request_timeout_s": 1800.5}, {}).timeout
    assert timeout == 1800
    assert isinstance(timeout, int)


def test_the_other_sampling_settings_still_flow_through():
    cfg = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 24576,
        "max_connections": 8,
        "request_timeout_s": 3600,
    }
    config = eval_generate_config(cfg, {"apply_hooks": ["hook"]})
    assert config.temperature == 0.7
    assert config.top_p == 0.95
    assert config.max_tokens == 24576
    assert config.max_connections == 8
    assert config.extra_body == {"extra_args": {"apply_hooks": ["hook"]}}


def test_the_sampling_defaults_are_unchanged():
    config = eval_generate_config({}, {})
    assert config.temperature == 1.0
    assert config.top_p == 1.0
    assert config.max_tokens == 2048
    assert config.max_connections == 12
