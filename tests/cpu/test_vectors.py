"""CPU-only tests for direction building, online covariance, and projection."""

from __future__ import annotations

import numpy as np
import pytest

from healthy_rl.vectors import (
    OnlineCovariance,
    build_directions,
    project,
    turn_statistic,
)


def _unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _cos(a, b):
    return float(np.dot(_unit(a), _unit(b)))


def _planted_setup(seed=0, d=128, n_emotions=6, nuisance_scale=20.0):
    """Emotion means = shared offset + emotion-specific signal + a shared nuisance axis."""
    rng = np.random.default_rng(seed)

    # Emotion-specific signal, made zero-mean across emotions so it survives centering intact.
    signal = rng.normal(size=(n_emotions, d))
    signal -= signal.mean(axis=0, keepdims=True)
    signal = _unit(signal)

    nuisance = _unit(rng.normal(size=d))
    offset = rng.normal(size=d) * 10.0  # grand mean, removed by centering
    loadings = rng.normal(size=(n_emotions, 1)) * nuisance_scale

    emotion_means = offset + 3.0 * signal + loadings * nuisance

    # Neutral covariance dominated by the nuisance axis.
    neutral_cov = 100.0 * np.outer(nuisance, nuisance) + 0.01 * np.eye(d)
    return signal, nuisance, emotion_means, neutral_cov


def test_build_directions_recovers_planted_signal():
    signal, nuisance, emotion_means, neutral_cov = _planted_setup()
    directions, n_removed = build_directions(emotion_means, neutral_cov, var_frac=0.5)

    assert directions.shape == emotion_means.shape
    assert n_removed == 1  # nuisance axis alone carries >50% of the neutral variance
    for i in range(directions.shape[0]):
        assert _cos(directions[i], signal[i]) > 0.95
        assert abs(_cos(directions[i], nuisance)) < 0.05
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)


def test_build_directions_without_removal_keeps_nuisance():
    """Guard: the nuisance really is present before removal, so the test above bites."""
    _, nuisance, emotion_means, neutral_cov = _planted_setup()
    directions, n_removed = build_directions(emotion_means, neutral_cov, var_frac=0.0)

    assert n_removed == 0
    assert max(abs(_cos(row, nuisance)) for row in directions) > 0.5


def test_build_directions_removes_more_components_for_higher_var_frac():
    rng = np.random.default_rng(3)
    d = 32
    basis = np.linalg.qr(rng.normal(size=(d, d)))[0]
    eigenvalues = np.array([10.0, 8.0, 6.0] + [0.01] * (d - 3))
    neutral_cov = (basis * eigenvalues) @ basis.T
    emotion_means = rng.normal(size=(4, d))

    _, few = build_directions(emotion_means, neutral_cov, var_frac=0.4)
    _, more = build_directions(emotion_means, neutral_cov, var_frac=0.9)
    assert few == 1
    assert more > few


def test_build_directions_validates_shapes():
    with pytest.raises(ValueError):
        build_directions(np.zeros((3, 8)), np.zeros((7, 7)))
    with pytest.raises(ValueError):
        build_directions(np.zeros(8), np.zeros((8, 8)))
    with pytest.raises(ValueError):
        build_directions(np.zeros((3, 8)), np.zeros((8, 8)), var_frac=1.5)


def test_online_covariance_matches_numpy_cov():
    rng = np.random.default_rng(7)
    data = rng.normal(size=(500, 16)) * np.arange(1, 17) + 5.0

    cov = OnlineCovariance(16)
    for batch in (data[:37], data[37:200], data[200:201], data[201:]):
        cov.update(batch)

    assert cov.count == data.shape[0]
    assert np.max(np.abs(cov.mean() - data.mean(axis=0))) < 1e-6
    assert np.max(np.abs(cov.covariance() - np.cov(data, rowvar=False))) < 1e-6
    assert np.max(np.abs(cov.covariance(ddof=0) - np.cov(data, rowvar=False, bias=True))) < 1e-6


def test_online_covariance_infers_dim_and_rejects_mismatch():
    rng = np.random.default_rng(11)
    cov = OnlineCovariance()
    cov.update(rng.normal(size=(10, 4)))
    assert cov.d == 4
    with pytest.raises(ValueError):
        cov.update(rng.normal(size=(10, 5)))
    with pytest.raises(ValueError):
        OnlineCovariance(4).update(np.zeros(4))


def test_online_covariance_requires_two_samples():
    cov = OnlineCovariance(3)
    cov.update(np.ones((1, 3)))
    with pytest.raises(ValueError):
        cov.covariance()


def test_project_shape_and_values():
    directions = np.array([[5.0, 0.0, 0.0], [0.0, 2.0, 0.0]])  # deliberately not unit norm
    activations = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 0.0]])

    out = project(activations, directions)

    assert out.shape == (2, 2)
    np.testing.assert_allclose(out, np.array([[1.0, 2.0], [-4.0, 0.5]]))


def test_project_accepts_single_position():
    directions = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = project(np.array([3.0, 4.0]), directions)
    assert out.shape == (1, 2)
    np.testing.assert_allclose(out, [[3.0, 4.0]])


def test_project_validates_dim():
    with pytest.raises(ValueError):
        project(np.zeros((4, 8)), np.zeros((3, 7)))


def test_turn_statistic_hand_computed():
    projections = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    # column means are [3, 4]; dividing by norm=2 gives [1.5, 2.0]
    np.testing.assert_allclose(turn_statistic(projections, 2.0), [1.5, 2.0])


def test_turn_statistic_scalar_for_one_emotion():
    out = turn_statistic(np.array([2.0, 4.0, 6.0, 8.0]), 5.0)
    assert isinstance(out, float)
    assert out == pytest.approx(1.0)


def test_turn_statistic_rejects_bad_norm():
    projections = np.ones((3, 2))
    with pytest.raises(ValueError):
        turn_statistic(projections, 0.0)
    with pytest.raises(ValueError):
        turn_statistic(np.zeros((0, 2)), 1.0)


def _massive_activation_data(n=4000, seed=13):
    """Residual-stream-like data: a few dimensions whose mean dwarfs their spread."""
    rng = np.random.default_rng(seed)
    means = np.array([1500.0, -900.0, 1200.0, 0.5, -0.25, 2.0])
    return rng.normal(size=(n, means.size)) * 2.0 + means


def test_online_covariance_survives_massive_activation_dimensions():
    """float64 accumulation must stay exact where mean >> std.

    The sum-of-squares form cancels catastrophically here: at float32 the variance
    of the mean-1500 dimension comes back ~22% low -- so a 1e-8 relative tolerance
    here discriminates by seven orders of magnitude, even though the raw sums of
    squares are ~1e10 (measured float64 error: 2e-10).
    """
    data = _massive_activation_data()

    cov = OnlineCovariance(data.shape[1])
    for start in range(0, data.shape[0], 137):
        cov.update(data[start : start + 137])

    expected = np.cov(data, rowvar=False)
    got = cov.covariance()
    scale = np.abs(expected).max()
    assert np.max(np.abs(got - expected)) / scale < 1e-8
    # The variances that would collapse at float32 are recovered to ~4 (std 2).
    np.testing.assert_allclose(np.diag(got)[:3], np.diag(expected)[:3], rtol=1e-8)


def test_online_covariance_rejects_non_float64_dtype():
    for dtype in (np.float32, np.float16, np.int64):
        with pytest.raises(ValueError, match="float64"):
            OnlineCovariance(8, dtype=dtype)


def test_online_covariance_rejects_non_float64_accumulator_state():
    """The resume path restores sum/sum_outer by assignment and must be guarded."""
    data = _massive_activation_data(n=200)
    cov = OnlineCovariance(data.shape[1])
    cov.update(data)

    cov.sum = cov.sum.astype(np.float32)
    cov.sum_outer = cov.sum_outer.astype(np.float32)
    with pytest.raises(ValueError, match="float64"):
        cov.covariance()
    with pytest.raises(ValueError, match="float64"):
        cov.mean()


def test_online_covariance_upcasts_float32_input():
    """A float32 *batch* is fine -- it is accumulated exactly in float64."""
    data = _massive_activation_data(n=500).astype(np.float32)
    cov = OnlineCovariance(data.shape[1])
    cov.update(data[:250])
    cov.update(data[250:])

    assert cov.sum.dtype == np.float64
    expected = np.cov(data.astype(np.float64), rowvar=False)
    assert np.max(np.abs(cov.covariance() - expected)) / np.abs(expected).max() < 1e-8
