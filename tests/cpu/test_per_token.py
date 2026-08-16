"""Per-token projections must reach the rollout's .npz, at every capture layer.

The hook already ships (T x 14) projections per layer to the client; until this
change ``summarise_hook_results`` reduced them to a turn mean and dropped them.
``docs/runs.md`` said they were kept. Now they are.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from healthy_rl.rollouts import ResidualStash, Vectors, summarise_hook_results


def _vectors() -> Vectors:
    rng = np.random.default_rng(1)
    d = rng.normal(size=(14, 2, 8)).astype(np.float32)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return Vectors(
        directions=d, emotions=[f"e{i}" for i in range(14)], capture_layers=[3, 5],
        probe_layer=5, mean_residual_norm={3: 10.0, 5: 12.0}, path=Path("/nonexistent"),
    )


def _hook_results(n_decode: int = 4):
    """One prefill row followed by n_decode decode rows, at layers 3 and 5, as the hook saves them."""
    saved = {}
    for layer in (3, 5):
        P = 1 + n_decode
        saved[f"proj_L{layer}"] = torch.arange(P * 14, dtype=torch.float32).reshape(P, 14) / 100 + layer
        saved[f"norm_L{layer}"] = torch.full((P,), 9.0 + layer)
        saved[f"kind_L{layer}"] = torch.tensor([1.0] + [0.0] * n_decode)
    saved["res_start_L5"] = torch.ones(8, dtype=torch.float16)
    saved["res_end_L5"] = torch.ones(8, dtype=torch.float16) * 2
    return {"hook0": saved}


def test_per_token_arrays_are_stashed_for_every_capture_layer():
    stash = ResidualStash()
    stats = summarise_hook_results(_hook_results(), _vectors(), stash)
    assert stats.error is None
    assert stats.n_generated == 4
    arrays = stash.pop(stats.residual_key)
    for layer in (3, 5):
        proj = arrays[f"proj_L{layer}"]
        norm = arrays[f"norm_L{layer}"]
        kind = arrays[f"kind_L{layer}"]
        assert proj.shape == (5, 14) and proj.dtype == np.float16
        assert norm.shape == (5,) and norm.dtype == np.float32
        assert kind.shape == (5,) and kind.dtype == np.int8
        assert kind.tolist() == [1, 0, 0, 0, 0]
        # the whole row, not one element: a transposed or mis-sliced proj would
        # still match column 0 by itself.
        np.testing.assert_allclose(
            proj[1].astype(np.float32), layer + np.arange(14, 28) / 100, rtol=1e-3
        )
    # the boundary residuals are still there, only at the residual layer
    assert "res_start_L5" in arrays and "res_end_L5" in arrays
    assert "res_start_L3" not in arrays


def test_turn_stat_is_unchanged_by_keeping_the_arrays():
    stash = ResidualStash()
    with_stash = summarise_hook_results(_hook_results(), _vectors(), stash)
    without = summarise_hook_results(_hook_results(), _vectors(), None)
    assert with_stash.stats == without.stats
    assert with_stash.observed_norm == without.observed_norm
    assert without.residual_key is None


def test_missing_layer_is_reported_and_the_others_are_still_kept():
    results = _hook_results()
    for k in ("proj_L3", "norm_L3", "kind_L3"):
        del results["hook0"][k]
    stash = ResidualStash()
    stats = summarise_hook_results(results, _vectors(), stash)
    assert "layer 3 missing" in (stats.error or "")
    arrays = stash.pop(stats.residual_key)
    assert "proj_L5" in arrays and "proj_L3" not in arrays


# ---------------------------------------------------------------------------
# Boundary residuals must survive a coordinate above the float16 max.
#
# gemma-3-12b-it's prefill row carries one coordinate near/above 65,504. Stored
# as float16 that coordinate became inf and the whole turn-start residual read
# as non-finite -- 629/629 of the non-finite cases across seven gemma
# conditions were exactly one overflowed coordinate out of 3,840
# (docs/measurement.md, "Non-finite residuals"). The per-token proj/norm arrays
# were never affected: they are reduced in float32 before any cast.
# ---------------------------------------------------------------------------


class _FakeCtx:
    """The three attributes ``project_tokens`` touches on a vllm-lens context."""

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.saved: dict = {}
        self._prefetched: dict = {}


def _run_hook(hidden_states: torch.Tensor, layer: int = 5) -> dict:
    """Drive the real hook body over one pass and return what it saved."""
    from healthy_rl.rollouts import make_projection_hook

    hook = make_projection_hook(_vectors().directions, [3, 5], [5])
    ctx = _FakeCtx(layer)
    hook.fn(ctx, hidden_states)
    return ctx.saved


def test_boundary_residual_survives_a_coordinate_above_the_float16_max():
    # bfloat16 is what vLLM hands the hook, and it holds 70000 fine; float16
    # cannot, which is the whole bug.
    hidden = torch.zeros(6, 8, dtype=torch.bfloat16)
    hidden[-1, 3] = 70000.0
    hidden[-1, 0] = 1.5

    saved = _run_hook(hidden)

    event = saved["res_start_L5"]
    assert event.dtype == torch.float32, f"boundary residual stored as {event.dtype}"
    assert torch.isfinite(event).all(), "the outsized coordinate overflowed"
    assert float(event.max()) > 65504.0, "the test lost the outsized coordinate"
    # a decode pass writes the other key, and it is float32 too
    decode = _run_hook(torch.ones(1, 8, dtype=torch.bfloat16))
    assert decode["res_end_L5"].dtype == torch.float32


def test_outsized_boundary_residual_stays_finite_through_the_summary():
    """End to end: the stashed .npz array is float32 and finite."""
    results = _hook_results()
    start = torch.zeros(8, dtype=torch.float32)
    start[3] = 70000.0
    results["hook0"]["res_start_L5"] = start

    stash = ResidualStash()
    stats = summarise_hook_results(results, _vectors(), stash)
    arrays = stash.pop(stats.residual_key)

    stashed = arrays["res_start_L5"]
    assert stashed.dtype == np.float32
    assert np.isfinite(stashed).all()
    assert stashed.max() > 65504.0
