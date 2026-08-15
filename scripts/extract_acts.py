#!/usr/bin/env python
"""Stage 3: extract per-emotion mean activations and a neutral covariance.

Runs on a compute node against an already-running vllm-lens server. For every story
we prefill only (``max_tokens=1``), capture the residual stream at ``capture_layers``,
and mean-pool token positions from ``skip_positions`` onward -- the recipe from the
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

This job runs unattended for hours, so accumulator state is checkpointed every
``checkpoint_every`` stories and a killed job resumes from the last checkpoint.
Work is processed in fixed order in slices of ``batch_size`` concurrent requests, so
the resume cursor is an exact high-water mark rather than a guess.
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
    return parser.parse_args(argv)


def resolve_base_url(cli_value: str | None) -> str:
    """``--base-url``, else ``$HEALTHY_RL_SERVER_URL``, else the file it names."""
    if cli_value and cli_value.strip():
        return cli_value.strip()
    url = os.environ.get("HEALTHY_RL_SERVER_URL")
    if url and url.strip():
        return url.strip()
    url_file = os.environ.get("HEALTHY_RL_ENDPOINT_FILE")
    if url_file:
        path = Path(url_file)
        if not path.is_file():
            raise RuntimeError(
                f"HEALTHY_RL_ENDPOINT_FILE={url_file} does not exist; "
                "has the server job written its URL yet?"
            )
        text = path.read_text().strip()
        if not text:
            raise RuntimeError(f"HEALTHY_RL_ENDPOINT_FILE={url_file} is empty")
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
        self.n_failed = 0
        self.cov_dirty = False

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
            counters=np.array([self.cursor, self.n_used, self.n_skipped, self.n_failed]),
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
        self.cursor, self.n_used, self.n_skipped, self.n_failed = (
            int(x) for x in state["counters"]
        )

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
        f"  server={base_url} stories={len(work)} batch_size={batch_size} "
        f"skip_positions={skip_positions} min_tokens={min_tokens}\n"
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

    def fetch(item: tuple[str, int, str]) -> tuple[tuple[str, int, str], Any, Any]:
        """Never raises: a failed request is one lost story, not a lost overnight job."""
        _, _, text = item
        try:
            out = client().generate(
                text,
                max_tokens=int(cfg.get("max_tokens", 1)),
                temperature=0.0,
                capture_layers=capture_layers,
            )
        except Exception as exc:  # noqa: BLE001 - reported and counted by the caller
            return item, exc, None
        return item, out.activations, out.raw

    started = time.monotonic()
    start_cursor = acc.cursor
    since_checkpoint = 0
    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        while acc.cursor < len(work):
            chunk = work[acc.cursor : acc.cursor + batch_size]
            for item, activations, raw in pool.map(fetch, chunk):
                group, _, _ = item
                if isinstance(activations, BaseException):
                    acc.n_failed += 1
                    print(
                        f"  story {group} request failed: "
                        f"{type(activations).__name__}: {activations}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                try:
                    positions, pooled = pooled_from_output(
                        activations, raw, n_layers, d_model, skip_positions, min_tokens
                    )
                except StoryTooShort:
                    acc.n_skipped += 1
                    continue
                except Exception as exc:  # noqa: BLE001 - one bad story must not end the run
                    acc.n_failed += 1
                    print(
                        f"  story {group} unusable: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                if group == NEUTRAL_GROUP:
                    acc.add_neutral(positions)
                else:
                    acc.add_emotion(emotion_index[group], pooled)
                acc.n_used += 1

            acc.cursor += len(chunk)
            since_checkpoint += len(chunk)

            # Only judge the failure rate once there is a sample worth judging: two
            # rejected requests out of the first four is noise, not an unhealthy server.
            seen = acc.n_used + acc.n_skipped + acc.n_failed
            if seen >= min_failure_sample and acc.n_failed / seen > max_failure_rate:
                acc.save(out_dir)
                raise RuntimeError(
                    f"{acc.n_failed}/{seen} requests failed, above max_failure_rate="
                    f"{max_failure_rate}; the server is unhealthy. Checkpoint saved at "
                    f"story {acc.cursor}, re-run to resume."
                )

            if since_checkpoint >= checkpoint_every or acc.cursor >= len(work):
                acc.save(out_dir)
                since_checkpoint = 0
                # Rate over work done in THIS process: a resumed job's cursor starts high.
                rate = (acc.cursor - start_cursor) / max(time.monotonic() - started, 1e-9)
                remaining = (len(work) - acc.cursor) / rate if rate > 0 else float("nan")
                print(
                    f"  {acc.cursor}/{len(work)} stories "
                    f"(used {acc.n_used}, skipped {acc.n_skipped}, failed {acc.n_failed}) "
                    f"{rate:.2f}/s eta {remaining / 60:.1f} min",
                    flush=True,
                )

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
