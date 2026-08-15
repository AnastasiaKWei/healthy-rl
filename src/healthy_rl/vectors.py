"""Emotion direction building and projection.

Pure numpy (torch CPU tensors are accepted anywhere an array is, via
``np.asarray``). No vLLM import, nothing GPU-bound: importable on a login node.

Conventions:
- ``directions`` is ``(n_emotions, d)`` with unit-norm rows, for ONE layer.
- ``project`` returns the scalar projection onto those unit directions; the
  cosine-style normalisation by that layer's mean residual norm happens in
  ``turn_statistic``, so per-token magnitudes stay inspectable.
"""

from __future__ import annotations

import numpy as np

__all__ = ["build_directions", "OnlineCovariance", "project", "turn_statistic"]


def _as_2d(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {arr.shape}")
    return arr


def _unit_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    # Leave all-zero rows at zero rather than dividing by zero.
    norms = np.where(norms > 0, norms, 1.0)
    return x / norms


def build_directions(
    emotion_means: np.ndarray, neutral_cov: np.ndarray, var_frac: float = 0.5
) -> tuple[np.ndarray, int]:
    """Emotion directions for one layer, with dominant neutral variance removed.

    Args:
        emotion_means: ``(n_emotions, d)`` mean activation per emotion.
        neutral_cov: ``(d, d)`` covariance of activations on neutral text.
        var_frac: remove the fewest top principal components of ``neutral_cov``
            whose eigenvalues together cover at least this fraction of its total
            variance. ``0.0`` removes nothing.

    Returns:
        ``(directions, n_components_removed)`` where ``directions`` is
        ``(n_emotions, d)`` with unit-norm rows.
    """
    means = np.asarray(emotion_means, dtype=np.float64)
    cov = np.asarray(neutral_cov, dtype=np.float64)
    if means.ndim != 2:
        raise ValueError(f"emotion_means must be (n_emotions, d), got shape {means.shape}")
    d = means.shape[1]
    if cov.shape != (d, d):
        raise ValueError(f"neutral_cov must be ({d}, {d}) to match emotion_means, got {cov.shape}")
    if not 0.0 <= var_frac <= 1.0:
        raise ValueError(f"var_frac must be in [0, 1], got {var_frac}")

    # What is specific to each emotion, not shared by all of them.
    centered = means - means.mean(axis=0, keepdims=True)

    n_removed = 0
    if var_frac > 0.0:
        # Symmetric: eigh gives ascending real eigenvalues; flip to descending.
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        positive = np.clip(eigenvalues, 0.0, None)
        total = positive.sum()
        if total > 0:
            cumulative = np.cumsum(positive) / total
            n_removed = int(np.searchsorted(cumulative, var_frac, side="left") + 1)
            n_removed = min(n_removed, int(np.count_nonzero(positive)))
            top = eigenvectors[:, :n_removed]
            centered = centered - (centered @ top) @ top.T

    return _unit_rows(centered), n_removed


class OnlineCovariance:
    """Streaming covariance for one layer: keeps ``sum``, ``sum_outer``, ``count``.

    Token-level activations are never stored, only the ``(d, d)`` outer-product
    accumulator, so a full pass over rollouts costs O(d^2) memory per layer.

    **float64 only.** The sum-of-squares form cancels catastrophically when a
    dimension's mean dwarfs its standard deviation -- exactly the massive-activation
    dimensions of a residual stream. Measured on mean 1500 / std 2 data: float64
    errs by ~4e-10 while float32 returns variance 3.07 against a true 3.95, a 22%
    error. At d=5120 the accumulator is 210 MB per layer, so float32 is precisely
    what someone reaches for under memory pressure; the constructor and
    ``covariance()`` both refuse it rather than trusting the default.
    """

    def __init__(self, d: int | None = None, dtype=np.float64) -> None:
        if np.dtype(dtype) != np.float64:
            raise ValueError(
                f"OnlineCovariance requires float64, got {np.dtype(dtype)}: the "
                "sum-of-squares accumulator loses ~22% of the variance at float32 on "
                "residual-stream dimensions whose mean dwarfs their standard deviation"
            )
        self.d = d
        self.dtype = np.float64
        self.count = 0
        self.sum = np.zeros(d, dtype=np.float64) if d is not None else None
        self.sum_outer = np.zeros((d, d), dtype=np.float64) if d is not None else None

    def _check_accumulator_dtype(self) -> None:
        """Guard the resume path: callers restore ``sum``/``sum_outer`` by assignment."""
        for name in ("sum", "sum_outer"):
            arr = getattr(self, name)
            if arr is not None and np.dtype(arr.dtype) != np.float64:
                raise ValueError(
                    f"OnlineCovariance.{name} has dtype {arr.dtype}, expected float64; "
                    "a non-float64 accumulator silently loses variance on "
                    "large-mean dimensions"
                )

    def update(self, batch) -> "OnlineCovariance":
        """Accumulate a ``(n_positions, d)`` batch of activations."""
        arr = np.asarray(batch, dtype=self.dtype)
        if arr.ndim != 2:
            raise ValueError(f"batch must be (n_positions, d), got shape {arr.shape}")
        if self.d is None:
            self.d = arr.shape[1]
            self.sum = np.zeros(self.d, dtype=self.dtype)
            self.sum_outer = np.zeros((self.d, self.d), dtype=self.dtype)
        elif arr.shape[1] != self.d:
            raise ValueError(f"batch has d={arr.shape[1]}, expected d={self.d}")

        self.sum += arr.sum(axis=0)
        self.sum_outer += arr.T @ arr
        self.count += arr.shape[0]
        return self

    def mean(self) -> np.ndarray:
        if self.count == 0:
            raise ValueError("no samples accumulated")
        self._check_accumulator_dtype()
        return self.sum / self.count

    def covariance(self, ddof: int = 1) -> np.ndarray:
        """The ``(d, d)`` covariance matrix (``ddof=1`` matches ``numpy.cov``)."""
        if self.count - ddof <= 0:
            raise ValueError(
                f"need more than ddof={ddof} samples to form a covariance, have {self.count}"
            )
        self._check_accumulator_dtype()
        mean = self.mean()
        centered_outer = self.sum_outer - self.count * np.outer(mean, mean)
        cov = centered_outer / (self.count - ddof)
        return (cov + cov.T) / 2.0  # kill asymmetry from floating-point drift


def project(activations, directions) -> np.ndarray:
    """Project activations onto unit-normalised emotion directions.

    Args:
        activations: ``(n_positions, d)`` (a single ``(d,)`` position is accepted).
        directions: ``(n_emotions, d)``; re-normalised here so callers cannot
            silently pass unnormalised rows.

    This is the raw dot product with unit directions, not a cosine: the activation
    side is deliberately left unnormalised so per-token magnitude survives, and the
    "cosine-style" normalisation of brief 2 happens once per turn in
    ``turn_statistic`` (divide by the layer's mean residual norm).

    Returns:
        ``(n_positions, n_emotions)``.
    """
    acts = _as_2d(activations, "activations")
    dirs = _as_2d(directions, "directions")
    if acts.shape[1] != dirs.shape[1]:
        raise ValueError(
            f"activations have d={acts.shape[1]} but directions have d={dirs.shape[1]}"
        )
    return acts @ _unit_rows(dirs).T


def turn_statistic(projections, norm: float):
    """Mean projection over an assistant turn's generated positions, scaled by ``norm``.

    ``norm`` is that layer's mean residual-stream norm, which makes the result
    comparable across layers and models. Returns a ``(n_emotions,)`` array for
    ``(n_positions, n_emotions)`` input.

    A 1-D input is read as ``(n_positions,)`` for a single emotion and returns a
    float. Note the trap: ``project`` always returns 2-D, so a 1-D array here is
    almost always one *position's* ``(n_emotions,)`` row, which this would average
    across emotions -- a meaningless number. Pass the 2-D array.
    """
    arr = np.asarray(projections, dtype=np.float64)
    if arr.ndim not in (1, 2):
        raise ValueError(f"projections must be 1-D or 2-D, got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("projections has no token positions to average over")
    norm = float(norm)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"norm must be a positive finite float, got {norm}")

    out = arr.mean(axis=0) / norm
    return float(out) if out.ndim == 0 else out
