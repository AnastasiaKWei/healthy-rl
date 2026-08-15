"""Readout conventions from docs/measurement.md, as executable checks."""
from __future__ import annotations

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
