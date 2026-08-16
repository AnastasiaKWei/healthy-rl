"""Readout conventions from docs/measurement.md, in one place.

Everything the dashboard shows as a number goes through here, so the readout
conventions live in one place: single-token cosine at one layer, ``start`` read
at the prefill row, indexing by position among non-empty turns, non-finite rows
skipped *and counted*.

Two deliberate divergences from ``scripts/live_trajectory.py``:

- **Turn indexing under skips.** ``token_sequences`` there drops a non-finite
  turn from the list, so every later turn shifts down one index and turn 3 of a
  conversation can be averaged into turn index 2. Here a skipped turn is a
  ``None`` placeholder, so positions hold and ``by_turn_index`` reports the
  skips instead of absorbing them. This convention is the intended one.
- **Turn means.** ``--stat mean`` there is measurement.md's turn-mean: mean
  projection over the turn divided by the layer's *mean* residual norm. Here
  ``turn_mean`` averages per-token cosines, each divided by its own token's
  norm -- a mean of ratios, not a ratio of means. The two are close but not
  equal, and neither is comparable to a single-token number (measurement.md:
  they differ by 2-3x and disagree about which effects exist).

Shapes: ``proj`` is ``(T, L, E)`` (decode tokens x capture layers x emotions),
``norm`` is ``(T, L)``, ``proj_prefill`` is ``(L, E)``, ``norm_prefill`` is
``(L,)``. Login-node importable: numpy only, scipy optional.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

READOUTS = ("start", "think_end", "answer_start", "end")
SEGMENTS = ("all", "think", "answer")
MIN_N_FOR_WILCOXON = 6


def token_cosine(proj: np.ndarray, norm: np.ndarray, layer_index: int) -> np.ndarray:
    """Cosine of every decode token with every direction at one layer, ``(T, E)``.

    Directions are unit-norm, so projection / token norm is the cosine. Rows
    whose norm is non-finite or zero become NaN rather than inf: callers skip
    and count NaN rows.
    """
    p = np.asarray(proj, dtype=np.float64)[:, layer_index, :]
    n = np.asarray(norm, dtype=np.float64)[:, layer_index]
    bad = ~np.isfinite(n) | (n == 0)
    safe = np.where(bad, 1.0, n)
    out = p / safe[:, None]
    out[bad] = np.nan
    out[~np.isfinite(out).all(axis=1)] = np.nan
    return out


def segment_mask(token_kind: Sequence[str], segment: str) -> np.ndarray:
    if segment not in SEGMENTS:
        raise ValueError(f"segment must be one of {SEGMENTS}, got {segment!r}")
    kinds = np.asarray(list(token_kind), dtype=object)
    if segment == "all":
        return np.ones(len(kinds), dtype=bool)
    return kinds == segment


def _check_kind_length(token_kind: Sequence[str], T: int) -> None:
    """A ``token_kind`` shorter than the decode run would silently mis-locate readouts."""
    if T > 0 and len(token_kind) != T:
        raise ValueError(f"token_kind has {len(token_kind)} entries for {T} decode tokens")


def _finite_or_none(row: np.ndarray) -> np.ndarray | None:
    return row if row is not None and np.isfinite(row).all() else None


def turn_readout(
    *,
    proj: np.ndarray,
    norm: np.ndarray,
    proj_prefill: np.ndarray,
    norm_prefill: np.ndarray,
    token_kind: Sequence[str],
    layer_index: int,
    readout: str,
) -> np.ndarray | None:
    """One turn's ``(E,)`` cosine at a named position, or None if unavailable.

    ``start`` reads the prefill row (the residual that produced the first
    generated token: the paper's Assistant-colon analogue). ``think_end`` is
    the last ``think`` token, ``answer_start`` the first ``answer`` token,
    ``end`` the last decode token. None when the position does not exist
    (no reasoning, empty turn) or the value is non-finite.

    Raises ValueError if ``token_kind`` does not have one entry per decode token
    (``start`` excepted: it reads the prefill row and ignores ``token_kind``).
    """
    if readout not in READOUTS:
        raise ValueError(f"readout must be one of {READOUTS}, got {readout!r}")
    if readout == "start":
        n = float(np.asarray(norm_prefill, dtype=np.float64)[layer_index])
        if not np.isfinite(n) or n == 0:
            return None
        return _finite_or_none(np.asarray(proj_prefill, dtype=np.float64)[layer_index] / n)
    T = int(np.asarray(proj).shape[0])
    if T == 0:
        return None
    _check_kind_length(token_kind, T)
    kinds = list(token_kind)
    if readout == "end":
        idx = T - 1
    elif readout == "think_end":
        think = [i for i, k in enumerate(kinds) if k == "think"]
        if not think:
            return None
        idx = think[-1]
    else:  # answer_start
        answer = [i for i, k in enumerate(kinds) if k == "answer"]
        if not answer:
            return None
        idx = answer[0]
    cos = token_cosine(proj, norm, layer_index)
    return _finite_or_none(cos[idx])


def turn_mean(
    *,
    proj: np.ndarray,
    norm: np.ndarray,
    token_kind: Sequence[str],
    layer_index: int,
    segment: str,
) -> np.ndarray | None:
    """Mean per-token cosine over a segment's finite tokens, or None if there are none.

    A mean of per-token ratios, not measurement.md's turn-mean statistic -- see
    the module docstring. Raises ValueError if ``token_kind`` does not have one
    entry per decode token.
    """
    T = int(np.asarray(proj).shape[0])
    if T == 0:
        return None
    _check_kind_length(token_kind, T)
    cos = token_cosine(proj, norm, layer_index)
    keep = segment_mask(token_kind, segment) & np.isfinite(cos).all(axis=1)
    if not keep.any():
        return None
    return cos[keep].mean(axis=0)


def non_empty_index(n_generated: Sequence[int]) -> list[int | None]:
    """Position among turns that generated tokens; None for empty turns."""
    out: list[int | None] = []
    k = 0
    for n in n_generated:
        if int(n) > 0:
            out.append(k)
            k += 1
        else:
            out.append(None)
    return out


def moving_mean(x: np.ndarray, k: int) -> np.ndarray:
    """Centred moving mean with the window shrunk at the edges; same length as ``x``.

    NaN entries are ignored; a window that is entirely NaN yields NaN quietly,
    without numpy's "Mean of empty slice" warning. All-NaN stretches are normal
    input here -- ``by_turn_index`` emits an all-NaN row for any turn index
    whose conversations all skipped.
    """
    x = np.asarray(x, dtype=np.float64)
    if k <= 1 or x.size == 0:
        return x.copy()
    half = k // 2
    out = np.empty_like(x)
    for i in range(x.size):
        lo, hi = max(0, i - half), min(x.size, i + half + 1)
        win = x[lo:hi]
        out[i] = np.nan if np.isnan(win).all() else np.nanmean(win)
    return out


def by_turn_index(
    sequences: list[list[np.ndarray | None]], *, n_emotions: int | None = None
) -> dict:
    """Mean/SEM/n/skipped by non-empty turn index over conversations.

    Each inner list is one conversation's per-turn ``(E,)`` values in
    non-empty-turn order; None marks a turn skipped as non-finite.

    Pass ``n_emotions`` to fix the column count. Without it E is inferred from
    the first value present, which is nothing at all when every turn was
    skipped -- a real condition, not a corner case: measurement.md has
    ``gemma-3-12b-it`` ``aff6`` non-finite at turn start on 144/144 turns. The
    honest answer there is an all-NaN ``(K, E)`` table with ``skipped == n``,
    not a zero-column one that makes ``mean[:, e]`` raise IndexError.
    """
    K = max((len(s) for s in sequences), default=0)
    if n_emotions is not None:
        E = int(n_emotions)
    else:
        E = next((v.shape[0] for s in sequences for v in s if v is not None), 0)
    mean = np.full((K, E), np.nan)
    sem = np.full((K, E), np.nan)
    n = np.zeros(K, dtype=int)
    skipped = np.zeros(K, dtype=int)
    for k in range(K):
        vals = []
        for s in sequences:
            if len(s) <= k:
                continue
            if s[k] is None:
                skipped[k] += 1
            else:
                vals.append(np.asarray(s[k], dtype=np.float64))
        n[k] = len(vals)
        if vals:
            arr = np.stack(vals)
            mean[k] = arr.mean(axis=0)
            if len(vals) > 1:
                sem[k] = arr.std(axis=0, ddof=1) / np.sqrt(len(vals))
    return {"mean": mean, "sem": sem, "n": n, "skipped": skipped}


def paired_delta(
    sequences: list[list[np.ndarray | None]], *, n_emotions: int | None = None
) -> dict:
    """Last-minus-first usable turn, paired within conversation.

    Conversations with fewer than two usable (non-None) turns are excluded.
    ``p`` is the Wilcoxon signed-rank p-value per emotion when ``n >= 6`` and
    scipy imports, else NaN.

    ``n_emotions`` fixes the width of the all-NaN result when no conversation
    has two usable turns -- see ``by_turn_index``.
    """
    deltas = []
    for s in sequences:
        usable = [v for v in s if v is not None]
        if len(usable) >= 2:
            deltas.append(np.asarray(usable[-1], dtype=np.float64) - np.asarray(usable[0], dtype=np.float64))
    n = len(deltas)
    if n == 0:
        width = int(n_emotions) if n_emotions is not None else 0
        empty = np.full(width, np.nan)
        return {"mean": empty, "sem": empty.copy(), "p": empty.copy(), "n": 0}
    d = np.stack(deltas)
    E = d.shape[1]
    mean = d.mean(axis=0)
    sem = d.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(E, np.nan)
    p = np.full(E, np.nan)
    if n >= MIN_N_FOR_WILCOXON:
        try:
            from scipy import stats as sps
        except ImportError:
            sps = None
        if sps is not None:
            for e in range(E):
                col = d[:, e]
                if np.any(col != 0):
                    p[e] = float(sps.wilcoxon(col).pvalue)
    return {"mean": mean, "sem": sem, "p": p, "n": n}
