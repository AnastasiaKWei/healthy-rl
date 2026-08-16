"""The vllm-lens pre-hook must not log a traceback per forward pass.

Guards `patches/vllm_lens_pre_hook_log_spam.py`. `_make_pre_hook` reads hidden
states from `args[1]`; when a model's layers are called with fewer positional
arguments that raises IndexError, vllm-lens catches it, skips the pre-hook, and
logs the full traceback at WARNING with exc_info. Unpatched, that fires on every
forward pass of every layer of every request: one 3-hour rollout job logged it
12.1 million times and produced a 5.6 GB server log, and the serve directory
reached 99 GB of a 108 GB artifact tree.

This is a logging fault, not a measurement one -- captures come from post-hooks,
and every record written during the spam carried hook_data and residuals. The
patch keeps the first warning per layer and silences the repeats.

Fails on an unpatched venv, which is the point: `uv sync` silently reverts it.
"""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("vllm_lens", reason="vllm_lens not installed")


def test_patch_is_applied():
    from vllm_lens import _worker_ext

    assert hasattr(_worker_ext, "_HEALTHY_RL_PRE_HOOK_WARNED"), (
        "unpatched vllm-lens pre-hook logger; run "
        "`.venv/bin/python patches/vllm_lens_pre_hook_log_spam.py`"
    )


def test_failing_pre_hook_warns_once_per_layer(caplog):
    """A hook that fails every call must warn on the first call only."""
    from vllm_lens import _worker_ext

    _worker_ext._HEALTHY_RL_PRE_HOOK_WARNED.discard(7)
    _worker_ext._HEALTHY_RL_PRE_HOOK_WARNED.discard(8)
    hook7 = _worker_ext._make_pre_hook(object(), 7)
    hook8 = _worker_ext._make_pre_hook(object(), 8)

    with caplog.at_level(logging.WARNING):
        for _ in range(50):
            # An empty args tuple is exactly the real failure: no args[1].
            assert hook7(None, ()) is None
        assert hook8(None, ()) is None

    spam = [r for r in caplog.records if "pre-hook error on layer" in r.getMessage()]
    layers = sorted(int(r.getMessage().split("layer ")[1].split()[0].rstrip(",")) for r in spam)
    assert layers == [7, 8], (
        f"expected one warning per layer, got {len(spam)} for layers {layers}"
    )
