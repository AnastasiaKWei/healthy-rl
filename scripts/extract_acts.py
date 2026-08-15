#!/usr/bin/env python
"""Stage 3: extract per-emotion mean activations and a neutral covariance.

Runs on a compute node against an already-running vllm-lens server. For every story
we prefill only (``max_tokens=1``), read the residual stream at ``capture_layers``, and
mean-pool token positions from ``skip_positions`` onward -- the recipe from the
reference method: "averaging across all token positions within each story, beginning
with the 50th token". Stories shorter than ``min_tokens`` are skipped and counted.

Emotion stories accumulate into a per-emotion running mean. Neutral stories additionally
feed an ``OnlineCovariance`` per layer (over *token* positions, matching "the top
principal components of the activations on this dataset") and a running mean residual
norm per layer.

Outputs, in ``$ARTIFACT_DIR/activations/<model>/<version>/``:
  emotion_means.safetensors  key ``emotion_means``  (n_emotions, n_capture_layers, d)
  neutral_cov.safetensors    key ``neutral_cov``    (n_capture_layers, d, d)
  norms.json                 mean residual norm per layer + story counts
  manifest.json

Two transports, selected by ``transport``:

``pooled_hook`` (default)
    A ``Hook`` mean-pools inside the server and returns only ``(n_layers, d)`` per
    story: 0.10 MB of JSON against 9.5 MB for the raw stream, measured -- ~95x less.

    It also sidesteps a real vllm-lens bug. vllm-lens keeps PROCESS-GLOBAL
    ``ZstdCompressor``/``ZstdDecompressor`` objects in THREE modules and calls them
    from concurrent handlers; a shared instance reuses one ``ZSTD_CCtx``/``ZSTD_DCtx``,
    so concurrent calls interleave and corrupt frames. Locally reproducible on the
    client-side pair in ``_helpers/_serialize.py`` at 9.5 MB payloads: clean at 1
    thread, and at 8+ threads both ``ZstdError: Data corruption detected`` AND -- worse
    -- payloads that decompress without raising and return the WRONG BYTES.

    ``_serialize_value`` routes ONLY ``torch.Tensor`` through that code; JSON-safe
    values pass through untouched. This hook therefore saves plain Python floats and
    calls ``serialize_tensor`` exactly zero times, which makes it immune to both the
    raising and the silent variant rather than merely less exposed.

``capture_layers``
    The original path: the server ships ``(n_layers, total_pos, d)`` and we pool here.
    Kept for fallback and for cross-checking the hook against it.

**Neutral stories always use ``capture_layers``**, whichever transport is set. Their
covariance is over token positions, not story means, so the pooled hook would throw
away exactly the data the covariance needs, and a per-story ``(d, d)`` outer product is
far too large to ship. Neutral is only 1200 of 18,000 requests, but it is the one phase
still moving tensors through the zstd path, and **throttling does not fix that** -- the
corruption was observed at ``batch_size: 4`` as well as 32. ``neutral_batch_size``
reduces exposure; only patching the shared zstd objects removes it. Run the neutral
phase against a patched venv.

This job runs unattended for hours, so accumulator state is checkpointed every
``checkpoint_every`` stories and a killed job resumes from the last checkpoint.
Work is processed in fixed order in slices of ``batch_size`` concurrent requests, so
the resume cursor is an exact high-water mark rather than a guess. Stories whose
request failed are remembered by index and retried in a sweep after the main pass, so
a transient server fault costs a retry rather than a hole in the mean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from healthy_rl.artifacts import artifact_dir, check_upstream, verify_upstreams, write_manifest
from healthy_rl.config import load_config, load_env, repo_root
from healthy_rl.models import ModelSpec
from healthy_rl.vectors import OnlineCovariance

DEFAULT_CONFIG = repo_root() / "configs" / "extract_acts.yaml"

NEUTRAL_GROUP = "__neutral__"
STATE_NAME = "_state.npz"
COV_STATE_NAME = "_neutral_cov.npz"
NORMS_NAME = "norms.json"


# ---------------------------------------------------------------------------
# CLI / server address
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=None, help="override the config's model name")
    parser.add_argument("--base-url", default=None, help="vllm-lens server base URL")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--limit", type=int, default=None, help="process at most N stories (debugging)"
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="discard any checkpoint and start from the first story",
    )
    parser.add_argument(
        "--verify-transport",
        type=int,
        default=0,
        metavar="N",
        help=(
            "run N stories through BOTH transports and report the disagreement, "
            "then exit without writing an activations artifact"
        ),
    )
    return parser.parse_args(argv)


def resolve_base_url(cli_value: str | None) -> str:
    """``--base-url``, else ``$HEALTHY_RL_SERVER_URL``, else the file it names.

    ``slurm/serve.slurm`` exports the endpoint path under both spellings; either works.
    """
    if cli_value and cli_value.strip():
        return cli_value.strip()
    url = os.environ.get("HEALTHY_RL_SERVER_URL")
    if url and url.strip():
        return url.strip()
    url_file = os.environ.get("HEALTHY_RL_ENDPOINT_FILE") or os.environ.get(
        "HEALTHY_RL_SERVER_URL_FILE"
    )
    if url_file:
        path = Path(url_file)
        if not path.is_file():
            raise RuntimeError(
                f"endpoint file {url_file} does not exist; "
                "has the server job written its URL yet?"
            )
        text = path.read_text().strip()
        if not text:
            raise RuntimeError(f"endpoint file {url_file} is empty")
        return text
    raise RuntimeError(
        "no server URL: pass --base-url, or set HEALTHY_RL_SERVER_URL, "
        "or set HEALTHY_RL_ENDPOINT_FILE to a file containing it"
    )


# ---------------------------------------------------------------------------
# Work list
# ---------------------------------------------------------------------------


def build_work_list(
    stories: pd.DataFrame,
    neutral: pd.DataFrame,
    emotions: list[str],
    emotion_column: str,
    text_column: str,
    stories_per_emotion: int | None,
    n_neutral: int | None,
) -> list[tuple[str, int, str]]:
    """``(group, index_within_group, text)`` in a fixed, reproducible order.

    Emotions come first in config order, neutral last: the ``(d, d)`` covariance
    accumulator is only touched in the neutral phase, so the expensive part of the
    checkpoint is written only during the tail of the run.
    """
    missing = [e for e in emotions if e not in set(stories[emotion_column].unique())]
    if missing:
        raise ValueError(f"emotions missing from the stories parquet: {missing}")

    work: list[tuple[str, int, str]] = []
    for emotion in emotions:
        texts = stories.loc[stories[emotion_column] == emotion, text_column].tolist()
        if stories_per_emotion is not None:
            if len(texts) < stories_per_emotion:
                raise ValueError(
                    f"emotion {emotion!r} has {len(texts)} stories, "
                    f"config asks for {stories_per_emotion}"
                )
            texts = texts[:stories_per_emotion]
        work.extend((emotion, i, str(text)) for i, text in enumerate(texts))

    neutral_texts = neutral[text_column].tolist()
    if n_neutral is not None:
        if len(neutral_texts) < n_neutral:
            raise ValueError(
                f"neutral parquet has {len(neutral_texts)} rows, config asks for {n_neutral}"
            )
        neutral_texts = neutral_texts[:n_neutral]
    work.extend((NEUTRAL_GROUP, i, str(text)) for i, text in enumerate(neutral_texts))
    return work


def fingerprint(payload: dict[str, Any]) -> str:
    """Identity of the run a checkpoint belongs to; a mismatch forbids resuming."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# Accumulators
# ---------------------------------------------------------------------------


class Accumulators:
    """Everything the run carries forward, plus its checkpoint round trip."""

    def __init__(self, n_emotions: int, n_layers: int, d: int, fingerprint: str) -> None:
        self.n_emotions = n_emotions
        self.n_layers = n_layers
        self.d = d
        self.fingerprint = fingerprint

        self.emotion_sum = np.zeros((n_emotions, n_layers, d), dtype=np.float64)
        self.emotion_count = np.zeros(n_emotions, dtype=np.int64)
        self.covs = [OnlineCovariance(d) for _ in range(n_layers)]
        self.norm_sum = np.zeros(n_layers, dtype=np.float64)
        self.norm_count = np.zeros(n_layers, dtype=np.int64)

        self.cursor = 0
        self.n_used = 0
        self.n_skipped = 0
        # Work-list indices whose request failed. A set, not a counter, so the retry
        # sweep knows WHICH stories to re-request and the count can go back down.
        self.failed_indices: set[int] = set()
        self.cov_dirty = False

    @property
    def n_failed(self) -> int:
        return len(self.failed_indices)

    # -- accumulate ---------------------------------------------------------

    def add_emotion(self, emotion_index: int, pooled: np.ndarray) -> None:
        self.emotion_sum[emotion_index] += pooled
        self.emotion_count[emotion_index] += 1

    def add_neutral(self, positions: np.ndarray) -> None:
        """Feed one neutral story's ``(n_layers, n_positions, d)`` kept positions.

        The covariance is over token positions, not over story means: the reference
        method takes "the top principal components of the activations on this dataset".
        """
        for layer in range(self.n_layers):
            block = positions[layer]
            self.covs[layer].update(block)
            self.norm_sum[layer] += float(np.linalg.norm(block, axis=1).sum())
            self.norm_count[layer] += block.shape[0]
        self.cov_dirty = True

    # -- checkpoint ---------------------------------------------------------

    def save(self, out_dir: Path) -> None:
        _atomic_npz(
            out_dir / STATE_NAME,
            fingerprint=np.array(self.fingerprint),
            emotion_sum=self.emotion_sum,
            emotion_count=self.emotion_count,
            norm_sum=self.norm_sum,
            norm_count=self.norm_count,
            counters=np.array([self.cursor, self.n_used, self.n_skipped]),
            failed_indices=np.array(sorted(self.failed_indices), dtype=np.int64),
        )
        if self.cov_dirty:
            _atomic_npz(
                out_dir / COV_STATE_NAME,
                fingerprint=np.array(self.fingerprint),
                cov_sum=np.stack([c.sum for c in self.covs]),
                cov_sum_outer=np.stack([c.sum_outer for c in self.covs]),
                cov_count=np.array([c.count for c in self.covs], dtype=np.int64),
            )
            self.cov_dirty = False

    def load(self, out_dir: Path) -> bool:
        """Restore from a checkpoint. Returns False when there is nothing to resume."""
        state_path = out_dir / STATE_NAME
        if not state_path.is_file():
            return False
        state = np.load(state_path, allow_pickle=False)
        found = str(state["fingerprint"])
        if found != self.fingerprint:
            raise RuntimeError(
                f"checkpoint {state_path} belongs to a different run "
                f"(fingerprint {found[:12]} != {self.fingerprint[:12]}): the config, the "
                "model or the stories artifact changed. Re-run with --restart to discard it."
            )
        self.emotion_sum = state["emotion_sum"]
        self.emotion_count = state["emotion_count"]
        self.norm_sum = state["norm_sum"]
        self.norm_count = state["norm_count"]
        self.cursor, self.n_used, self.n_skipped = (int(x) for x in state["counters"])
        self.failed_indices = {int(x) for x in state["failed_indices"]}

        cov_path = out_dir / COV_STATE_NAME
        if cov_path.is_file():
            cov_state = np.load(cov_path, allow_pickle=False)
            if str(cov_state["fingerprint"]) != self.fingerprint:
                raise RuntimeError(f"{cov_path} belongs to a different run; use --restart")
            sums = cov_state["cov_sum"]
            outers = cov_state["cov_sum_outer"]
            counts = cov_state["cov_count"]
            for layer, cov in enumerate(self.covs):
                cov.sum = sums[layer]
                cov.sum_outer = outers[layer]
                cov.count = int(counts[layer])
        return True

    def clear_checkpoint(self, out_dir: Path) -> None:
        for name in (STATE_NAME, COV_STATE_NAME):
            (out_dir / name).unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write via a temp file in the same directory, then rename.

    A job killed mid-write must not leave a half-written checkpoint that the next
    attempt happily loads.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# One story
# ---------------------------------------------------------------------------



class StoryTooShort(RuntimeError):
    """Fewer than ``min_tokens`` prompt tokens: no usable pooled window."""


# ---------------------------------------------------------------------------
# Transport A: server-side pooling hook
# ---------------------------------------------------------------------------


def make_pool_hook(skip_positions: int):
    """Build the hook function that mean-pools inside the server.

    Returned as a CLOSURE on purpose. ``Hook`` cloudpickles ``fn`` to a process that
    has no ``healthy_rl`` on its path; cloudpickle serialises nested functions by value
    but may serialise a module-level function by reference, which would fail to import
    on the server. A closure also bakes ``skip_positions`` in without a global.

    What it saves, per layer, into ``ctx.saved["L<idx>"]``::

        {"sum": [float, ...] | None, "n_pooled": int, "n_seen": int}

    Three properties matter and each is load-bearing:

    * **JSON-safe values only.** ``_serialize_value`` sends ``torch.Tensor`` through the
      process-global ``ZstdCompressor`` that corrupts under concurrency, but passes
      plain floats/ints/dicts through untouched. Saving a list of floats keeps this
      payload off the racing code path entirely.
    * **A dict, not a list.** ``_merge_hook_results`` concatenates list values across
      ranks but overwrites everything else. The residual stream is replicated across TP
      ranks, so every rank computes an identical dict and the overwrite is idempotent --
      whereas a list would be duplicated once per TP rank.
    * **``n_seen`` carries the absolute position offset**, so chunked prefill (the hook
      firing several times per layer for one request) pools the right window rather
      than restarting the count at each chunk.

    Sums are accumulated in float64 and divided client-side, which makes the result
    match the ``capture_layers`` path's float64 pooling to round-off.
    """

    def pool_hook(ctx, hidden_states):
        key = f"L{ctx.layer_idx}"
        record = ctx.saved.get(key)
        if record is None:
            record = {"sum": None, "n_pooled": 0, "n_seen": 0}
            ctx.saved[key] = record

        seen = record["n_seen"]
        n_rows = int(hidden_states.shape[0])
        # Absolute index of the first row this call is allowed to pool.
        start = skip_positions - seen
        if start < 0:
            start = 0
        if start < n_rows:
            total = hidden_states[start:].double().sum(dim=0).cpu().tolist()
            if record["sum"] is None:
                record["sum"] = total
            else:
                record["sum"] = [a + b for a, b in zip(record["sum"], total)]
            record["n_pooled"] += n_rows - start
        record["n_seen"] = seen + n_rows
        return None  # hidden states are read, never modified

    return pool_hook


def pooled_from_hook(
    hook_results: dict[str, Any],
    raw: dict[str, Any],
    capture_layers: list[int],
    d_model: int,
    min_tokens: int,
) -> tuple[np.ndarray, int]:
    """``(pooled, n_seen)`` from one story's hook results: ``(n_layers, d)`` float64.

    Verifies the window the server actually pooled rather than trusting it: every layer
    must report the same ``n_seen``, and that count must match the prompt length the
    server reports, so a silently truncated or re-chunked prefill is caught here.
    """
    if not hook_results:
        raise RuntimeError("server returned no hook_results; was the hook registered?")
    # {hook_index: ctx.saved}; we register exactly one hook.
    saved: dict[str, Any] = {}
    for per_hook in hook_results.values():
        saved.update(per_hook)

    missing = [layer for layer in capture_layers if f"L{layer}" not in saved]
    if missing:
        raise RuntimeError(f"hook never fired on layers {missing} (saw {sorted(saved)})")

    seen_counts = {int(saved[f"L{layer}"]["n_seen"]) for layer in capture_layers}
    if len(seen_counts) != 1:
        raise RuntimeError(
            f"layers disagree on how many positions they saw: {sorted(seen_counts)}"
        )
    n_seen = seen_counts.pop()
    if n_seen < min_tokens:
        raise StoryTooShort(f"{n_seen} prompt tokens < min_tokens={min_tokens}")

    usage = raw.get("usage") if isinstance(raw, dict) else None
    if isinstance(usage, dict) and isinstance(usage.get("prompt_tokens"), int):
        n_prompt = int(usage["prompt_tokens"])
        if n_seen != n_prompt:
            raise RuntimeError(
                f"hook pooled over {n_seen} positions but the server reports "
                f"{n_prompt} prompt tokens; the pooling window is not what we asked for"
            )

    pooled = np.zeros((len(capture_layers), d_model), dtype=np.float64)
    for i, layer in enumerate(capture_layers):
        record = saved[f"L{layer}"]
        n_pooled = int(record["n_pooled"])
        if not n_pooled or record["sum"] is None:
            raise StoryTooShort(
                f"layer {layer} pooled no positions out of {n_seen} seen"
            )
        total = np.asarray(record["sum"], dtype=np.float64)
        if total.shape != (d_model,):
            raise RuntimeError(
                f"layer {layer} returned {total.shape}, expected ({d_model},)"
            )
        pooled[i] = total / n_pooled
    return pooled, n_seen


# ---------------------------------------------------------------------------
# Transport B: client-side pooling of the full residual stream
# ---------------------------------------------------------------------------


def pooled_from_output(
    activations: dict[str, Any],
    raw: dict[str, Any],
    n_layers: int,
    d_model: int,
    skip_positions: int,
    min_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``(positions, pooled)`` as float64: ``(n_layers, n_kept, d)`` and ``(n_layers, d)``.

    ``residual_stream`` is ``(n_layers, total_pos, d)`` with layers in ascending
    capture-layer order. ``total_pos`` can include the sampled position, so it is
    trimmed back to the prompt length reported by the server when that is available.
    """
    if not activations or "residual_stream" not in activations:
        raise RuntimeError(
            f"server returned no residual_stream activations (keys: {list(activations or {})})"
        )
    acts = activations["residual_stream"]
    if acts.ndim != 3:
        raise RuntimeError(f"residual_stream is {acts.ndim}-D {tuple(acts.shape)}, expected 3-D")
    if acts.shape[0] != n_layers:
        raise RuntimeError(
            f"residual_stream has {acts.shape[0]} layers, expected {n_layers} capture layers"
        )
    if acts.shape[2] != d_model:
        raise RuntimeError(
            f"residual_stream has d={acts.shape[2]}, expected d_model={d_model}"
        )

    n_prompt = None
    usage = raw.get("usage") if isinstance(raw, dict) else None
    if isinstance(usage, dict) and isinstance(usage.get("prompt_tokens"), int):
        n_prompt = int(usage["prompt_tokens"])

    total_pos = int(acts.shape[1])
    n_tokens = total_pos if n_prompt is None else min(n_prompt, total_pos)
    if n_tokens < min_tokens:
        raise StoryTooShort(f"{n_tokens} prompt tokens < min_tokens={min_tokens}")

    positions = acts[:, skip_positions:n_tokens, :].float().numpy().astype(np.float64)
    if positions.shape[1] == 0:
        raise StoryTooShort(
            f"{n_tokens} prompt tokens leave no positions after skipping {skip_positions}"
        )
    return positions, positions.mean(axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_env()
    cfg = load_config(args.config)

    model_name = args.model or cfg["model"]
    version = str(cfg.get("version", "v1"))
    emotions = list(cfg["emotions"])
    emotion_index = {emotion: i for i, emotion in enumerate(emotions)}
    skip_positions = int(cfg["skip_positions"])
    min_tokens = int(cfg["min_tokens"])
    batch_size = max(1, int(cfg["batch_size"]))
    transport = str(cfg.get("transport", "pooled_hook"))
    if transport not in ("pooled_hook", "capture_layers"):
        raise ValueError(
            f"transport must be 'pooled_hook' or 'capture_layers', got {transport!r}"
        )
    use_hook_for_emotions = transport == "pooled_hook"
    # Neutral stories always ship full residual streams, so they get their own, lower
    # concurrency; that is the knob that controls the vllm-lens zstd corruption.
    neutral_batch_size = max(1, int(cfg.get("neutral_batch_size", 4)))
    checkpoint_every = max(1, int(cfg["checkpoint_every"]))
    max_failure_rate = float(cfg.get("max_failure_rate", 0.02))
    min_failure_sample = int(cfg.get("min_failure_sample", 50))

    spec = ModelSpec.from_checkpoint(Path(cfg["model_dir"]) / model_name, name=model_name)
    capture_layers = list(spec.capture_layers)
    n_layers = len(capture_layers)
    d_model = spec.d_model

    stories_dir = Path(cfg["stories_dir"])
    # An upstream that was rewritten under us invalidates everything downstream.
    check_upstream(stories_dir)  # raises a named error when stage 1 never ran
    verify_upstreams(stories_dir)

    out_dir = Path(args.out_dir) if args.out_dir else artifact_dir("activations", model_name, version)
    if (out_dir / "manifest.json").is_file():
        verify_upstreams(out_dir)

    stories = pd.read_parquet(stories_dir / cfg["stories_file"])
    neutral = pd.read_parquet(stories_dir / cfg["neutral_file"])
    work = build_work_list(
        stories,
        neutral,
        emotions,
        cfg["emotion_column"],
        cfg["text_column"],
        cfg.get("stories_per_emotion"),
        cfg.get("n_neutral"),
    )
    if args.limit is not None:
        work = work[: args.limit]

    from healthy_rl.artifacts import manifest_sha256

    run_id = fingerprint(
        {
            "model": model_name,
            "capture_layers": capture_layers,
            "d_model": d_model,
            "emotions": emotions,
            "skip_positions": skip_positions,
            "min_tokens": min_tokens,
            "n_work": len(work),
            "stories_manifest": manifest_sha256(stories_dir),
        }
    )

    acc = Accumulators(len(emotions), n_layers, d_model, run_id)
    if args.restart:
        acc.clear_checkpoint(out_dir)
        print("--restart: discarded any existing checkpoint", flush=True)
    elif acc.load(out_dir):
        print(
            f"resumed from checkpoint at story {acc.cursor}/{len(work)} "
            f"(used={acc.n_used} skipped={acc.n_skipped} failed={acc.n_failed})",
            flush=True,
        )

    base_url = resolve_base_url(args.base_url)
    timeout_s = float(cfg.get("request_timeout_s", 600.0))
    print(
        f"stage 3 extract_acts: model={model_name} arch={spec.architecture} "
        f"d_model={d_model} capture_layers={capture_layers} probe_layer={spec.probe_layer}\n"
        f"  server={base_url} stories={len(work)} transport={transport} "
        f"batch_size={batch_size} neutral_batch_size={neutral_batch_size}\n"
        f"  skip_positions={skip_positions} min_tokens={min_tokens}\n"
        f"  out={out_dir}",
        flush=True,
    )

    # One client per worker thread: requests.Session is not documented thread-safe,
    # and a shared session serialises on its connection pool anyway.
    local = threading.local()

    def client() -> Any:
        existing = getattr(local, "client", None)
        if existing is None:
            from healthy_rl.server import LensClient

            existing = LensClient(
                base_url,
                timeout=timeout_s,
                max_attempts=int(cfg.get("max_attempts", 5)),
            )
            local.client = existing
        return existing

    def request(text: str, use_hook: bool) -> tuple[Any, Any, Any]:
        """One prefill. Returns ``(activations, hook_results, raw)``."""
        kwargs: dict[str, Any] = {
            "max_tokens": int(cfg.get("max_tokens", 1)),
            "temperature": 0.0,
        }
        if use_hook:
            from vllm_lens import Hook

            kwargs["hooks"] = [
                Hook(fn=make_pool_hook(skip_positions), layer_indices=capture_layers)
            ]
        else:
            kwargs["capture_layers"] = capture_layers
        out = client().generate(text, **kwargs)
        return out.activations, out.hook_results, out.raw

    def fetch(job: tuple[int, tuple[str, int, str], bool]):
        """Never raises: a failed request is one lost story, not a lost overnight job."""
        index, item, use_hook = job
        try:
            activations, hook_results, raw = request(item[2], use_hook)
        except Exception as exc:  # noqa: BLE001 - reported and counted by the caller
            return index, item, use_hook, exc, None, None
        return index, item, use_hook, activations, hook_results, raw

    def accumulate(jobs: list[tuple[int, tuple[str, int, str], bool]], pool) -> tuple[int, int]:
        """Run one slice of work and fold it in. Returns ``(n_ok, n_failed)`` this slice."""
        n_ok = n_bad = 0
        for index, item, use_hook, activations, hook_results, raw in pool.map(fetch, jobs):
            group = item[0]
            if isinstance(activations, BaseException):
                acc.failed_indices.add(index)
                n_bad += 1
                print(
                    f"  story {group}[{index}] request failed: "
                    f"{type(activations).__name__}: {activations}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            try:
                if use_hook:
                    positions = None
                    pooled, _ = pooled_from_hook(
                        hook_results, raw, capture_layers, d_model, min_tokens
                    )
                else:
                    positions, pooled = pooled_from_output(
                        activations, raw, n_layers, d_model, skip_positions, min_tokens
                    )
            except StoryTooShort:
                acc.failed_indices.discard(index)
                acc.n_skipped += 1
                continue
            except Exception as exc:  # noqa: BLE001 - one bad story must not end the run
                acc.failed_indices.add(index)
                n_bad += 1
                print(
                    f"  story {group}[{index}] unusable: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            if group == NEUTRAL_GROUP:
                # The covariance is over token positions, so neutral never uses the hook.
                acc.add_neutral(positions)
            else:
                acc.add_emotion(emotion_index[group], pooled)
            # A retry that succeeded stops being a failure.
            acc.failed_indices.discard(index)
            acc.n_used += 1
            n_ok += 1
        return n_ok, n_bad

    def plan(index: int, item: tuple[str, int, str]) -> tuple[int, tuple[str, int, str], bool]:
        """Neutral stories always take the capture path; emotion stories take `transport`."""
        return index, item, use_hook_for_emotions and item[0] != NEUTRAL_GROUP

    def run_transport_check(n_stories: int) -> int:
        """Send the same stories both ways and report the disagreement.

        This is what lets the pooled hook be trusted without re-deriving the science:
        if it agrees with the path stage 0 validated, it is the same measurement.
        """
        tolerance = float(cfg.get("transport_tolerance", 1e-4))
        sample = [item for item in work if item[0] != NEUTRAL_GROUP][:n_stories]
        rows = []
        worst = 0.0
        for order, item in enumerate(sample):
            _, hook_results, hook_raw = request(item[2], True)
            activations, _, cap_raw = request(item[2], False)
            hook_pooled, n_seen = pooled_from_hook(
                hook_results, hook_raw, capture_layers, d_model, min_tokens
            )
            _, cap_pooled = pooled_from_output(
                activations, cap_raw, n_layers, d_model, skip_positions, min_tokens
            )
            scale = np.abs(cap_pooled).mean()
            abs_diff = float(np.abs(hook_pooled - cap_pooled).max())
            rel_diff = abs_diff / float(scale) if scale > 0 else float("inf")
            worst = max(worst, rel_diff)
            rows.append(
                {
                    "emotion": item[0],
                    "n_seen": int(n_seen),
                    "max_abs_diff": abs_diff,
                    "max_rel_diff": rel_diff,
                    "mean_abs_value": float(scale),
                }
            )
            print(
                f"  [{order + 1}/{len(sample)}] {item[0]:<14} n_seen={n_seen:>4} "
                f"max|diff|={abs_diff:.3e} rel={rel_diff:.3e}",
                flush=True,
            )

        passed = bool(rows) and worst <= tolerance
        report = {
            "stage": "transport_check",
            "model": model_name,
            "capture_layers": capture_layers,
            "skip_positions": skip_positions,
            "n_stories": len(rows),
            "tolerance": tolerance,
            "worst_rel_diff": worst,
            "passed": passed,
            "note": (
                "pooled_hook sums in float64 on the server; capture_layers ships bf16 "
                "and is pooled in float64 here. Both average the identical bf16 values, "
                "so only accumulation order differs and agreement should be ~1e-6."
            ),
            "per_story": rows,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "transport_check.json"
        target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(
            f"transport check {'PASSED' if passed else 'FAILED'}: worst relative "
            f"disagreement {worst:.3e} over {len(rows)} stories "
            f"(tolerance {tolerance:.1e}); wrote {target}",
            flush=True,
        )
        return 0 if passed else 1

    if args.verify_transport:
        return run_transport_check(args.verify_transport)

    started = time.monotonic()
    start_cursor = acc.cursor
    since_checkpoint = 0
    pass_ok = pass_bad = 0
    with ThreadPoolExecutor(max_workers=max(batch_size, neutral_batch_size)) as pool:
        while acc.cursor < len(work):
            # Neutral runs at its own (lower) concurrency: it is the only phase still
            # shipping full residual streams, and concurrency is what corrupts them.
            width = neutral_batch_size if work[acc.cursor][0] == NEUTRAL_GROUP else batch_size
            chunk = [
                plan(acc.cursor + offset, item)
                for offset, item in enumerate(work[acc.cursor : acc.cursor + width])
            ]
            n_ok, n_bad = accumulate(chunk, pool)
            pass_ok += n_ok
            pass_bad += n_bad
            acc.cursor += len(chunk)
            since_checkpoint += len(chunk)

            # Judged on THIS pass, not on history: a resumed run must not inherit an
            # abort from failures it is about to retry. Needs a sample worth judging --
            # two rejected requests out of the first four is noise, not a sick server.
            seen = pass_ok + pass_bad + acc.n_skipped
            if seen >= min_failure_sample and pass_bad / seen > max_failure_rate:
                acc.save(out_dir)
                raise RuntimeError(
                    f"{pass_bad}/{seen} requests failed in this pass, above "
                    f"max_failure_rate={max_failure_rate}; the server is unhealthy. "
                    f"Checkpoint saved at story {acc.cursor}, re-run to resume "
                    f"(failed stories are retried automatically)."
                )

            if since_checkpoint >= checkpoint_every or acc.cursor >= len(work):
                acc.save(out_dir)
                since_checkpoint = 0
                rate = (acc.cursor - start_cursor) / max(time.monotonic() - started, 1e-9)
                remaining = (len(work) - acc.cursor) / rate if rate > 0 else float("nan")
                print(
                    f"  {acc.cursor}/{len(work)} stories "
                    f"(used {acc.n_used}, skipped {acc.n_skipped}, failed {acc.n_failed}) "
                    f"{rate:.2f}/s eta {remaining / 60:.1f} min",
                    flush=True,
                )

        # Retry sweep. The corruption that motivated the pooled hook is transient, so a
        # failed story is usually recoverable; without this it would be a permanent hole
        # in that emotion's mean.
        for round_index in range(int(cfg.get("retry_rounds", 2))):
            if not acc.failed_indices:
                break
            pending = sorted(acc.failed_indices)
            print(
                f"  retry round {round_index + 1}: {len(pending)} failed stories",
                flush=True,
            )
            for offset in range(0, len(pending), neutral_batch_size):
                slice_ = pending[offset : offset + neutral_batch_size]
                accumulate([plan(index, work[index]) for index in slice_], pool)
            acc.save(out_dir)
            print(f"  retry round {round_index + 1}: {acc.n_failed} still failing", flush=True)

    write_outputs(out_dir, acc, cfg, spec, emotions, capture_layers, stories_dir, model_name, version)
    acc.clear_checkpoint(out_dir)
    return 0


def write_outputs(
    out_dir: Path,
    acc: Accumulators,
    cfg: dict,
    spec: ModelSpec,
    emotions: list[str],
    capture_layers: list[int],
    stories_dir: Path,
    model_name: str,
    version: str,
) -> None:
    """Materialise the artifact. Any emotion with no usable story is a hard error."""
    from safetensors.numpy import save_file

    empty = [emotions[i] for i, n in enumerate(acc.emotion_count) if n == 0]
    if empty:
        raise RuntimeError(f"no usable stories for emotions {empty}; refusing to write means")
    if any(cov.count < 2 for cov in acc.covs):
        raise RuntimeError(
            f"neutral covariance has {[c.count for c in acc.covs]} positions per layer; "
            "need at least 2 to form a covariance"
        )

    means = (acc.emotion_sum / acc.emotion_count[:, None, None]).astype(np.float32)
    cov = np.stack([c.covariance() for c in acc.covs]).astype(np.float32)
    save_file({"emotion_means": np.ascontiguousarray(means)}, str(out_dir / "emotion_means.safetensors"))
    save_file({"neutral_cov": np.ascontiguousarray(cov)}, str(out_dir / "neutral_cov.safetensors"))

    mean_norm = {
        str(layer): float(acc.norm_sum[i] / acc.norm_count[i]) if acc.norm_count[i] else float("nan")
        for i, layer in enumerate(capture_layers)
    }
    norms = {
        "mean_residual_norm": mean_norm,
        "n_stories_used": int(acc.n_used),
        "n_stories_skipped": int(acc.n_skipped),
        # Beyond the contract, but a request the server rejected is not a short story.
        "n_stories_failed": int(acc.n_failed),
        "n_neutral_positions": [int(c.count) for c in acc.covs],
        "emotion_story_counts": {e: int(n) for e, n in zip(emotions, acc.emotion_count)},
    }
    (out_dir / NORMS_NAME).write_text(json.dumps(norms, indent=2, sort_keys=True) + "\n")

    extra = {
        "emotions": emotions,
        "capture_layers": capture_layers,
        "probe_layer": spec.probe_layer,
        "n_layers": spec.n_layers,
        "d_model": spec.d_model,
        "architecture": spec.architecture,
        "checkpoint_path": str(spec.path),
        **norms,
    }
    write_manifest(
        out_dir,
        stage=str(cfg.get("stage", "extract_acts")),
        config={**cfg, "model": model_name, "version": version, "extra": extra},
        upstreams={"stories": stories_dir},
    )

    print(
        f"wrote {out_dir}\n"
        f"  emotion_means {tuple(means.shape)}  neutral_cov {tuple(cov.shape)}\n"
        f"  used {acc.n_used}, skipped {acc.n_skipped} (short), failed {acc.n_failed}",
        flush=True,
    )
    for layer in capture_layers:
        print(f"  layer {layer:>3}  mean residual norm {mean_norm[str(layer)]:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
