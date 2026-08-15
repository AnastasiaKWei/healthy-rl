#!/usr/bin/env python
"""Stage 4: turn stage 3's activation statistics into one direction per emotion per layer.

Login-node CPU stage: numpy only, no torch, no vLLM.

For each capture layer, ``healthy_rl.vectors.build_directions`` subtracts the
across-emotion grand mean and projects out the top principal components of the neutral
covariance (the fewest covering ``var_frac`` of its variance), then unit-normalises.

The number of components removed is the silent-wrongness risk of this stage. If the
neutral spectrum is near-flat, ``var_frac=0.5`` strips hundreds of directions and can
take real emotion signal with it, and nothing downstream would look obviously wrong.
So ``n_removed`` is printed per layer, written into ``vectors.json``, and reported
alongside two things that make it interpretable:

  ``retained_norm_frac``  per emotion per layer, how much of the emotion-specific mean
                          survived the projection. Near 1.0 means the removal was
                          harmless; near 0.0 means the direction is mostly noise.
  ``neutral_spectrum``    top-1 variance share and the component counts for 50/90/99%.

Outputs, in ``$ARTIFACT_DIR/vectors/<model>/<version>/``:
  vectors.safetensors  key ``directions``  (n_emotions, n_capture_layers, d)
  vectors.json
  manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file

from healthy_rl.artifacts import artifact_dir, check_upstream, verify_upstreams, write_manifest
from healthy_rl.config import load_config, load_env, repo_root
from healthy_rl.vectors import build_directions

DEFAULT_CONFIG = repo_root() / "configs" / "build_vectors.yaml"
VECTORS_NAME = "vectors.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=None, help="override the config's model name")
    parser.add_argument("--acts-dir", type=Path, default=None, help="stage 3 artifact dir")
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args(argv)


def spectrum_summary(cov: np.ndarray, fractions: tuple[float, ...] = (0.5, 0.9, 0.99)) -> dict:
    """How concentrated the neutral covariance is, for reading ``n_removed`` against."""
    eigenvalues = np.clip(np.linalg.eigvalsh(np.asarray(cov, dtype=np.float64)), 0.0, None)[::-1]
    total = float(eigenvalues.sum())
    if total <= 0:
        return {"total_variance": 0.0, "top1_frac": float("nan"), "n_components": {}}
    cumulative = np.cumsum(eigenvalues) / total
    return {
        "total_variance": total,
        "top1_frac": float(eigenvalues[0] / total),
        "n_components": {
            str(f): int(np.searchsorted(cumulative, f, side="left") + 1) for f in fractions
        },
        "rank": int(np.count_nonzero(eigenvalues > eigenvalues[0] * 1e-12)),
    }


def retained_norm_frac(centered: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Fraction of each emotion's centered-mean norm that survived the PC removal.

    ``directions`` are the unit-normalised residuals, so the residual norm is exactly
    ``centered . direction`` -- no second eigendecomposition needed.
    """
    norms = np.linalg.norm(centered, axis=1)
    retained = np.einsum("ij,ij->i", centered, directions)
    return np.where(norms > 0, retained / np.where(norms > 0, norms, 1.0), 0.0)


def load_activations(acts_dir: Path) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """``(emotion_means, neutral_cov, norms, extra)`` from a stage 3 artifact."""
    manifest = check_upstream(acts_dir)
    verify_upstreams(acts_dir)

    means = load_file(str(acts_dir / "emotion_means.safetensors"))["emotion_means"]
    cov = load_file(str(acts_dir / "neutral_cov.safetensors"))["neutral_cov"]
    norms = json.loads((acts_dir / "norms.json").read_text())
    extra = (manifest.get("config") or {}).get("extra") or {}
    if not extra.get("emotions") or not extra.get("capture_layers"):
        raise ValueError(
            f"{acts_dir}/manifest.json has no config.extra.emotions / capture_layers; "
            "it was not written by scripts/extract_acts.py"
        )
    return means, cov, norms, extra


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_env()
    cfg = load_config(args.config)

    model_name = args.model or cfg["model"]
    version = str(cfg.get("version", "v1"))
    var_frac = float(cfg["var_frac"])

    acts_dir = (
        Path(args.acts_dir)
        if args.acts_dir
        else artifact_dir("activations", model_name, str(cfg.get("activations_version", "v1")))
    )
    means, cov, norms, extra = load_activations(acts_dir)

    emotions: list[str] = list(extra["emotions"])
    capture_layers: list[int] = [int(x) for x in extra["capture_layers"]]
    probe_layer = int(extra["probe_layer"])

    # The emotion axis must be identical in every artifact; a config that disagrees with
    # what stage 3 actually accumulated is a silent relabelling of every direction.
    if cfg.get("emotions") and list(cfg["emotions"]) != emotions:
        raise ValueError(
            f"configs emotion order {list(cfg['emotions'])} != the order stage 3 used "
            f"{emotions}; the directions would be mislabelled"
        )

    n_emotions, n_layers, d = means.shape
    if (n_emotions, n_layers) != (len(emotions), len(capture_layers)):
        raise ValueError(
            f"emotion_means is {means.shape} but the manifest names {len(emotions)} emotions "
            f"and {len(capture_layers)} capture layers"
        )
    if cov.shape != (n_layers, d, d):
        raise ValueError(f"neutral_cov is {cov.shape}, expected {(n_layers, d, d)}")

    print(
        f"stage 4 build_vectors: model={model_name} emotions={n_emotions} "
        f"layers={capture_layers} probe_layer={probe_layer} d={d} var_frac={var_frac}\n"
        f"  activations={acts_dir}",
        flush=True,
    )

    directions = np.zeros((n_emotions, n_layers, d), dtype=np.float32)
    n_removed: dict[str, int] = {}
    retained: dict[str, dict[str, float]] = {}
    spectra: dict[str, Any] = {}

    want_spectrum = bool(cfg.get("spectrum_diagnostics", True))
    for i, layer in enumerate(capture_layers):
        layer_means = np.asarray(means[:, i, :], dtype=np.float64)
        layer_cov = np.asarray(cov[i], dtype=np.float64)
        layer_dirs, removed = build_directions(layer_means, layer_cov, var_frac=var_frac)
        directions[:, i, :] = layer_dirs.astype(np.float32)
        n_removed[str(layer)] = int(removed)

        centered = layer_means - layer_means.mean(axis=0, keepdims=True)
        fracs = retained_norm_frac(centered, layer_dirs)
        retained[str(layer)] = {e: float(f) for e, f in zip(emotions, fracs)}
        if want_spectrum:
            spectra[str(layer)] = spectrum_summary(layer_cov)

        detail = ""
        if want_spectrum:
            summary = spectra[str(layer)]
            detail = f"  top1 {summary['top1_frac']:.3f}  rank {summary['rank']}"
        print(
            f"  layer {layer:>3}: n_removed {removed:>5} / {d}   "
            f"retained norm frac min {fracs.min():.3f} median {np.median(fracs):.3f}"
            f"{detail}",
            flush=True,
        )

    out_dir = Path(args.out_dir) if args.out_dir else artifact_dir("vectors", model_name, version)
    save_file(
        {"directions": np.ascontiguousarray(directions)}, str(out_dir / "vectors.safetensors")
    )

    mean_norm = {str(k): float(v) for k, v in (norms.get("mean_residual_norm") or {}).items()}
    payload = {
        "emotions": emotions,
        "capture_layers": capture_layers,
        "probe_layer": probe_layer,
        "n_removed": n_removed,
        "mean_residual_norm": mean_norm,
        # Beyond the contract: what makes n_removed readable by a human.
        "var_frac": var_frac,
        "d_model": int(d),
        "retained_norm_frac": retained,
        "neutral_spectrum": spectra,
        "n_stories_used": norms.get("n_stories_used"),
        "n_stories_skipped": norms.get("n_stories_skipped"),
        "model": model_name,
    }
    (out_dir / VECTORS_NAME).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    write_manifest(
        out_dir,
        stage=str(cfg.get("stage", "build_vectors")),
        config={**cfg, "model": model_name, "version": version, "extra": {
            "emotions": emotions,
            "capture_layers": capture_layers,
            "probe_layer": probe_layer,
            "d_model": int(d),
            "n_removed": n_removed,
        }},
        upstreams={"activations": acts_dir},
    )

    print(f"wrote {out_dir}/vectors.safetensors {tuple(directions.shape)}", flush=True)
    print("  n_removed per layer: " + ", ".join(f"L{k}={v}" for k, v in n_removed.items()))
    worst = min(
        ((layer, e, f) for layer, per in retained.items() for e, f in per.items()),
        key=lambda t: t[2],
    )
    print(
        f"  weakest surviving direction: {worst[1]} at layer {worst[0]} kept "
        f"{worst[2]:.3f} of its centered-mean norm",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
