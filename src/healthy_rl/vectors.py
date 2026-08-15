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
    """

    def __init__(self, d: int | None = None, dtype=np.float64) -> None:
        self.d = d
        self.dtype = dtype
        self.count = 0
        self.sum = np.zeros(d, dtype=dtype) if d is not None else None
        self.sum_outer = np.zeros((d, d), dtype=dtype) if d is not None else None

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
        return self.sum / self.count

    def covariance(self, ddof: int = 1) -> np.ndarray:
        """The ``(d, d)`` covariance matrix (``ddof=1`` matches ``numpy.cov``)."""
        if self.count - ddof <= 0:
            raise ValueError(
                f"need more than ddof={ddof} samples to form a covariance, have {self.count}"
            )
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
    comparable across layers and models. Returns a float for a single emotion
    (1-D input) and a ``(n_emotions,)`` array for ``(n_positions, n_emotions)``
    input.
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
