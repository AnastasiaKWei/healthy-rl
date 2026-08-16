"""Readout conventions from docs/measurement.md, as executable checks."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from healthy_rl.dashboard import stats

E, L = 3, 2  # emotions, capture layers


def _turn(T=5, seed=0):
    rng = np.random.default_rng(seed)
    proj = rng.normal(size=(T, L, E)).astype(np.float32)
    norm = np.full((T, L), 10.0, dtype=np.float32)
    proj_prefill = rng.normal(size=(L, E)).astype(np.float32)
    norm_prefill = np.full((L,), 20.0, dtype=np.float32)
    kind = ["think", "think", "answer", "answer", "answer"]
    return proj, norm, proj_prefill, norm_prefill, kind


def test_token_cosine_divides_by_the_tokens_own_norm():
    proj, norm, *_ = _turn()
    cos = stats.token_cosine(proj, norm, layer_index=1)
    assert cos.shape == (5, E)
    np.testing.assert_allclose(cos, proj[:, 1, :] / 10.0)


def test_token_cosine_marks_nonfinite_or_zero_norm_rows_nan():
    proj, norm, *_ = _turn()
    norm = norm.copy(); norm[1, 0] = 0.0; norm[3, 0] = np.inf
    cos = stats.token_cosine(proj, norm, layer_index=0)
    assert np.isnan(cos[1]).all() and np.isnan(cos[3]).all()
    assert np.isfinite(cos[[0, 2, 4]]).all()


def test_segment_mask():
    kind = ["think", "think", "answer"]
    assert stats.segment_mask(kind, "all").tolist() == [True, True, True]
    assert stats.segment_mask(kind, "think").tolist() == [True, True, False]
    assert stats.segment_mask(kind, "answer").tolist() == [False, False, True]
    with pytest.raises(ValueError):
        stats.segment_mask(kind, "bogus")


def test_turn_readout_start_uses_prefill_row():
    proj, norm, pp, pn, kind = _turn()
    r = stats.turn_readout(proj=proj, norm=norm, proj_prefill=pp, norm_prefill=pn,
                           token_kind=kind, layer_index=0, readout="start")
    np.testing.assert_allclose(r, pp[0] / 20.0)


def test_turn_readout_think_end_answer_start_end():
    proj, norm, pp, pn, kind = _turn()
    kw = dict(proj=proj, norm=norm, proj_prefill=pp, norm_prefill=pn, token_kind=kind, layer_index=1)
    np.testing.assert_allclose(stats.turn_readout(readout="think_end", **kw), proj[1, 1] / 10.0)
    np.testing.assert_allclose(stats.turn_readout(readout="answer_start", **kw), proj[2, 1] / 10.0)
    np.testing.assert_allclose(stats.turn_readout(readout="end", **kw), proj[4, 1] / 10.0)


def test_turn_readout_end_is_none_for_a_capped_turn_whose_last_row_is_padding():
    # assemble_generation pads one all-NaN row when a generation stops at
    # max_tokens: that token has no residual, so its "end" readout is the cap,
    # not the model. The earlier readouts still land on real rows.
    proj, norm, pp, pn, kind = _turn()
    proj = proj.copy(); norm = norm.copy()
    proj[-1] = np.nan; norm[-1] = np.nan
    kw = dict(proj=proj, norm=norm, proj_prefill=pp, norm_prefill=pn, token_kind=kind, layer_index=1)
    assert stats.turn_readout(readout="end", **kw) is None
    np.testing.assert_allclose(stats.turn_readout(readout="think_end", **kw), proj[1, 1] / 10.0)
    np.testing.assert_allclose(stats.turn_readout(readout="answer_start", **kw), proj[2, 1] / 10.0)
    # the padding row is skipped by the finite mask, not counted as a zero
    m = stats.turn_mean(proj=proj, norm=norm, token_kind=kind, layer_index=1, segment="answer")
    np.testing.assert_allclose(m, proj[2:4, 1, :].mean(axis=0) / 10.0)


def test_turn_readout_none_when_no_reasoning_or_nonfinite():
    proj, norm, pp, pn, _ = _turn()
    kind = ["answer"] * 5
    kw = dict(proj=proj, norm=norm, proj_prefill=pp, norm_prefill=pn, token_kind=kind, layer_index=0)
    assert stats.turn_readout(readout="think_end", **kw) is None
    assert stats.turn_readout(readout="answer_start", **kw) is not None  # first token is an answer token
    pn2 = pn.copy(); pn2[0] = np.nan
    assert stats.turn_readout(proj=proj, norm=norm, proj_prefill=pp, norm_prefill=pn2,
                              token_kind=kind, layer_index=0, readout="start") is None
    assert stats.turn_readout(proj=np.zeros((0, L, E)), norm=np.zeros((0, L)), proj_prefill=pp,
                              norm_prefill=pn, token_kind=[], layer_index=0, readout="end") is None


def test_turn_mean_respects_segment_and_skips_nonfinite():
    proj, norm, _, _, kind = _turn()
    norm = norm.copy(); norm[0, 0] = np.nan
    m = stats.turn_mean(proj=proj, norm=norm, token_kind=kind, layer_index=0, segment="think")
    np.testing.assert_allclose(m, proj[1, 0] / 10.0)  # token 0 skipped, only token 1 remains
    assert stats.turn_mean(proj=proj, norm=norm, token_kind=["answer"] * 5, layer_index=0, segment="think") is None


def test_non_empty_index_skips_zero_token_turns():
    assert stats.non_empty_index([0, 0, 900, 0, 850, 700]) == [None, None, 0, None, 1, 2]


def test_moving_mean_keeps_length_and_edges():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(stats.moving_mean(x, 1), x)
    np.testing.assert_allclose(stats.moving_mean(x, 3), [1.5, 2.0, 3.0, 3.5])


def test_by_turn_index_counts_skips_and_ragged_lengths():
    a = [np.array([1.0, 0.0]), np.array([3.0, 0.0]), None]
    b = [np.array([3.0, 2.0]), None]
    out = stats.by_turn_index([a, b])
    np.testing.assert_allclose(out["mean"][0], [2.0, 1.0])
    np.testing.assert_allclose(out["mean"][1], [3.0, 0.0])
    assert out["n"].tolist() == [2, 1, 0]
    assert out["skipped"].tolist() == [0, 1, 1]
    assert np.isnan(out["mean"][2]).all()
    assert out["sem"].shape == (3, 2)


def test_paired_delta_last_minus_first_within_conversation():
    seqs = [[np.array([0.0]), np.array([0.5]), np.array([1.0])],
            [np.array([1.0]), np.array([3.0])],
            [np.array([2.0])],            # single turn: excluded
            [None, np.array([1.0]), np.array([4.0])]]  # leading skipped turn: first usable is index 1
    out = stats.paired_delta(seqs)
    assert out["n"] == 3
    np.testing.assert_allclose(out["mean"], [(1.0 + 2.0 + 3.0) / 3])
    assert out["sem"].shape == (1,)
    assert np.isnan(out["p"]).all()  # n < 6


def test_paired_delta_reports_wilcoxon_when_n_at_least_six():
    pytest.importorskip("scipy")
    seqs = [[np.array([0.0]), np.array([1.0 + 0.1 * i])] for i in range(8)]
    out = stats.paired_delta(seqs)
    assert out["n"] == 8 and 0.0 <= float(out["p"][0]) < 0.05


def test_turn_readout_rejects_token_kind_of_the_wrong_length():
    proj, norm, pp, pn, kind = _turn()
    with pytest.raises(ValueError, match="token_kind"):
        stats.turn_readout(proj=proj, norm=norm, proj_prefill=pp, norm_prefill=pn,
                           token_kind=kind[:3], layer_index=0, readout="end")


def test_turn_mean_rejects_token_kind_of_the_wrong_length():
    proj, norm, _, _, kind = _turn()
    with pytest.raises(ValueError, match="token_kind"):
        stats.turn_mean(proj=proj, norm=norm, token_kind=kind[:3], layer_index=0, segment="all")


def test_moving_mean_is_quiet_and_nan_on_all_nan_windows():
    x = np.array([1.0, np.nan, np.nan, np.nan, 2.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning fails the test
        out = stats.moving_mean(x, 3)
    assert np.isnan(out[2])  # window is entirely NaN
    np.testing.assert_allclose(out[[0, 4]], [1.0, 2.0])  # NaN neighbours ignored


def test_by_turn_index_keeps_its_columns_when_every_turn_is_skipped():
    """measurement.md: gemma-3-12b-it aff6 is non-finite on 144/144 turns at start."""
    out = stats.by_turn_index([[None, None], [None]], n_emotions=E)
    assert out["mean"].shape == (2, E) and out["sem"].shape == (2, E)
    assert np.isnan(out["mean"]).all()
    assert out["n"].tolist() == [0, 0]
    assert out["skipped"].tolist() == [2, 1]


def test_paired_delta_keeps_its_columns_when_no_conversation_is_usable():
    out = stats.paired_delta([[None, None], [np.array([1.0, 2.0, 3.0])]], n_emotions=E)
    assert out["n"] == 0
    for key in ("mean", "sem", "p"):
        assert out[key].shape == (E,) and np.isnan(out[key]).all()


def test_end_readout_from_boundary_row_when_no_decode_rows():
    import numpy as np
    from healthy_rl.dashboard import stats
    L, E = 2, 3
    empty = np.zeros((0, L, E)); empty_norm = np.zeros((0, L))
    pre = np.array([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]]); pre_n = np.array([1.0, 2.0])
    end = np.array([[np.nan, 0.0, 0.0], [0.5, -0.5, 0.25]]); end_n = np.array([np.nan, 5.0])
    v = stats.turn_readout(proj=empty, norm=empty_norm, proj_prefill=pre, norm_prefill=pre_n,
                           token_kind=[], layer_index=1, readout="end", proj_end=end, norm_end=end_n)
    assert np.allclose(v, [0.1, -0.1, 0.05])
    # non-finite norm at the other layer -> None, not inf
    assert stats.turn_readout(proj=empty, norm=empty_norm, proj_prefill=pre, norm_prefill=pre_n,
                              token_kind=[], layer_index=0, readout="end", proj_end=end, norm_end=end_n) is None
    # without the pair the old answer stands: no decode rows -> None
    assert stats.turn_readout(proj=empty, norm=empty_norm, proj_prefill=pre, norm_prefill=pre_n,
                              token_kind=[], layer_index=1, readout="end") is None
    # think_end / answer_start stay None with no decode rows even if the pair is passed
    assert stats.turn_readout(proj=empty, norm=empty_norm, proj_prefill=pre, norm_prefill=pre_n,
                              token_kind=[], layer_index=1, readout="think_end", proj_end=end, norm_end=end_n) is None
    # start is unaffected by the pair
    assert np.allclose(stats.turn_readout(proj=empty, norm=empty_norm, proj_prefill=pre, norm_prefill=pre_n,
                                          token_kind=[], layer_index=1, readout="start", proj_end=end, norm_end=end_n), [0.5, 1.0, 1.5])
