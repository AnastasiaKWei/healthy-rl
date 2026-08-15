#!/usr/bin/env python
"""Stage 5: logit-lens gate -- do the emotion directions point at the emotion's own words?

CPU stage. The unembedding is read straight off the checkpoint with ``safetensors``,
not through the vLLM server (ruling R2): the weight is static, the server is a scarce
GPU resource, and a slice read costs a few hundred MB instead of a model load. Gemma
ties its embeddings and has no ``lm_head.weight``, so ``embed_tokens.weight`` is the
fallback -- and on the multimodal wrappers both live under a ``language_model.`` prefix,
so keys are matched by suffix rather than assumed.

For each of the emotion directions at ``probe_layer`` we apply the final RMSNorm weight,
multiply by the unembedding, and record the top ``top_k`` tokens. Two scores:

  ``self_token_rate``     the emotion's own word appears verbatim among its top tokens.
  ``latin_initial_rate``  a looser control: some top token is Latin-script and shares
                          the emotion word's first ``latin_initial_prefix`` characters
                          ("desp" catching "despair", "desperation", "desper").

Like stage 0 this stage records rather than raises: a failed gate is the result it
exists to produce, so ``gate.json`` is always written and the exit code is always 0.
``n_removed`` from stage 4 is echoed into ``gate.json``, because a gate that fails
because half the space was projected away is a different problem from a gate that fails
because the model has no such representation.
"""

from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file

from healthy_rl.artifacts import artifact_dir, check_upstream, verify_upstreams, write_manifest
from healthy_rl.config import load_config, load_env, repo_root

DEFAULT_CONFIG = repo_root() / "configs" / "gate.yaml"
GATE_NAME = "gate.json"

# Sentencepiece and byte-level-BPE word-boundary markers.
_MARKERS = ("▁", "Ġ", "Ċ")
_LATIN_RE = re.compile(r"^[a-z]+$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=None)
    parser.add_argument("--vectors-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Reading one tensor out of a checkpoint without loading the checkpoint
# ---------------------------------------------------------------------------


def weight_map(ckpt: Path) -> dict[str, str]:
    """``tensor name -> shard filename``, for sharded and single-file checkpoints."""
    index = ckpt / "model.safetensors.index.json"
    if index.is_file():
        return dict(json.loads(index.read_text())["weight_map"])
    single = ckpt / "model.safetensors"
    if single.is_file():
        from safetensors import safe_open

        with safe_open(str(single), framework="pt") as handle:
            return {name: single.name for name in handle.keys()}
    raise FileNotFoundError(
        f"no model.safetensors.index.json and no model.safetensors in {ckpt}"
    )


def pick_key(names: list[str], exact: list[str], suffix: str, exclude: tuple[str, ...]) -> str | None:
    """First matching name: preferred exact spellings, then a suffix match.

    ``exclude`` drops the vision-tower twins (``embed_vision``, ``visual.*``) that would
    otherwise win a bare suffix match on a multimodal checkpoint.
    """
    available = set(names)
    for candidate in exact:
        if candidate in available:
            return candidate
    matches = [
        name
        for name in sorted(names)
        if name.endswith(suffix) and not any(bad in name for bad in exclude)
    ]
    return matches[0] if len(matches) >= 1 else None


_VISION = ("vision", "visual", "audio", "layers.")


def find_unembedding_key(names: list[str]) -> tuple[str, bool]:
    """``(key, tied)``. ``lm_head.weight`` when present, else the tied embedding."""
    key = pick_key(
        names,
        ["lm_head.weight", "language_model.lm_head.weight", "model.lm_head.weight"],
        "lm_head.weight",
        _VISION,
    )
    if key is not None:
        return key, False
    key = pick_key(
        names,
        [
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
            "language_model.model.embed_tokens.weight",
        ],
        "embed_tokens.weight",
        _VISION,
    )
    if key is None:
        raise KeyError(
            "checkpoint has neither an lm_head.weight nor an embed_tokens.weight "
            f"(saw {len(names)} tensors)"
        )
    return key, True


def find_final_norm_key(names: list[str]) -> str | None:
    return pick_key(
        names,
        [
            "model.language_model.norm.weight",
            "model.norm.weight",
            "language_model.model.norm.weight",
        ],
        ".norm.weight",
        _VISION,
    )


def read_tensor(ckpt: Path, wmap: dict[str, str], key: str) -> np.ndarray:
    """One tensor as float32 numpy. Goes through torch: numpy has no bfloat16."""
    from safetensors import safe_open

    with safe_open(str(ckpt / wmap[key]), framework="pt") as handle:
        return handle.get_tensor(key).float().numpy()


def scores_against_unembedding(
    ckpt: Path, wmap: dict[str, str], key: str, vectors: np.ndarray, chunk_rows: int
) -> np.ndarray:
    """``(n_vectors, vocab)`` logits, reading the unembedding in vocab-row chunks.

    The full ``(vocab, d)`` matrix is 5-6 GB in float32 for these models; the score
    matrix it produces is 15 MB. Streaming rows keeps this runnable on a login node.
    """
    from safetensors import safe_open

    with safe_open(str(ckpt / wmap[key]), framework="pt") as handle:
        slice_ = handle.get_slice(key)
        vocab, d = (int(x) for x in slice_.get_shape())
        if d != vectors.shape[1]:
            raise ValueError(
                f"{key} is ({vocab}, {d}) but the directions have d={vectors.shape[1]}"
            )
        out = np.zeros((vectors.shape[0], vocab), dtype=np.float32)
        for start in range(0, vocab, chunk_rows):
            end = min(start + chunk_rows, vocab)
            block = slice_[start:end].float().numpy()
            out[:, start:end] = (vectors @ block.T).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def normalise_token(token: str) -> str:
    """Strip word-boundary markers and case so ``'▁Desperate'`` reads as ``'desperate'``."""
    text = token
    for marker in _MARKERS:
        text = text.replace(marker, "")
    return text.strip().lower()


def is_latin(token: str) -> bool:
    return bool(_LATIN_RE.match(token))


def score_emotion(emotion: str, tokens: list[str], prefix_len: int) -> dict[str, Any]:
    """Per-emotion hit flags plus the tokens that produced them."""
    normalised = [normalise_token(t) for t in tokens]
    word = emotion.lower()
    self_hits = [t for t in normalised if t == word]
    # A word shorter than prefix_len ("sad") uses its whole self as the prefix, so the
    # looser control can never be stricter than self_token_rate.
    prefix = word[: min(prefix_len, len(word))]
    initial_hits = [
        t for t in normalised if is_latin(t) and len(t) >= len(prefix) and t.startswith(prefix)
    ]
    return {
        "self_token": bool(self_hits),
        "self_token_matches": self_hits,
        "latin_initial": bool(initial_hits),
        "latin_initial_matches": sorted(set(initial_hits)),
        "latin_prefix": prefix,
    }


def load_vectors(vectors_dir: Path) -> tuple[np.ndarray, dict]:
    check_upstream(vectors_dir)
    verify_upstreams(vectors_dir)
    directions = load_file(str(vectors_dir / "vectors.safetensors"))["directions"]
    meta = json.loads((vectors_dir / "vectors.json").read_text())
    return directions, meta


def run_gate(cfg: dict, args: argparse.Namespace, result: dict[str, Any]) -> None:
    """Fill ``result`` in place. Raising here is caught by ``main`` and recorded."""
    model_name = args.model or cfg["model"]
    result["model"] = model_name
    top_k = int(cfg["top_k"])
    prefix_len = int(cfg["latin_initial_prefix"])

    vectors_dir = (
        Path(args.vectors_dir)
        if args.vectors_dir
        else artifact_dir("vectors", model_name, str(cfg.get("vectors_version", "v1")))
    )
    result["vectors_dir"] = str(vectors_dir)
    directions, meta = load_vectors(vectors_dir)

    emotions: list[str] = list(meta["emotions"])
    capture_layers: list[int] = [int(x) for x in meta["capture_layers"]]
    probe_layer = int(cfg.get("probe_layer") or meta["probe_layer"])
    if probe_layer not in capture_layers:
        raise ValueError(f"probe_layer {probe_layer} is not among capture_layers {capture_layers}")
    layer_index = capture_layers.index(probe_layer)
    result["probe_layer"] = probe_layer
    result["emotions"] = emotions
    # Echoed so a failed gate can be read against how much space stage 4 removed.
    result["n_removed"] = meta.get("n_removed", {})
    result["n_removed_at_probe_layer"] = (meta.get("n_removed") or {}).get(str(probe_layer))
    result["retained_norm_frac_at_probe_layer"] = (meta.get("retained_norm_frac") or {}).get(
        str(probe_layer)
    )
    result["var_frac"] = meta.get("var_frac")

    probe = np.asarray(directions[:, layer_index, :], dtype=np.float32)
    if probe.shape[0] != len(emotions):
        raise ValueError(f"directions have {probe.shape[0]} rows but {len(emotions)} emotions")

    ckpt = Path(cfg["model_dir"]) / model_name
    wmap = weight_map(ckpt)
    names = list(wmap)
    unembed_key, tied = find_unembedding_key(names)
    norm_key = find_final_norm_key(names)
    result["unembedding_key"] = unembed_key
    result["tied_embeddings"] = tied
    result["final_norm_key"] = norm_key

    # The final RMSNorm's per-channel weight reweights the residual basis before the
    # unembedding sees it; the RMS division itself is a positive scalar and cannot
    # change a ranking, so it is skipped. `zero_centered` picks between the plain
    # `x * w` convention and Gemma's `x * (1 + w)`.
    zero_centered = bool(cfg.get("rms_norm_zero_centered", False))
    variants: dict[str, np.ndarray] = {"no_norm": probe}
    if norm_key is not None and bool(cfg.get("apply_final_norm", True)):
        norm_w = read_tensor(ckpt, wmap, norm_key)
        result["final_norm_stats"] = {
            "mean": float(norm_w.mean()),
            "min": float(norm_w.min()),
            "max": float(norm_w.max()),
            "zero_centered": zero_centered,
        }
        effective = (norm_w + 1.0) if zero_centered else norm_w
        variants["norm_weight"] = (probe * effective[None, :]).astype(np.float32)
    primary = "norm_weight" if "norm_weight" in variants else "no_norm"
    result["primary_variant"] = primary

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))

    chunk_rows = int(cfg.get("vocab_chunk_rows", 16384))
    per_variant: dict[str, Any] = {}
    for variant, vecs in variants.items():
        logits = scores_against_unembedding(ckpt, wmap, unembed_key, vecs, chunk_rows)
        order = np.argsort(-logits, axis=1)[:, :top_k]
        per_emotion = []
        for i, emotion in enumerate(emotions):
            ids = [int(x) for x in order[i]]
            tokens = tokenizer.convert_ids_to_tokens(ids)
            tokens = [t if isinstance(t, str) else f"<unk id {ids[j]}>" for j, t in enumerate(tokens)]
            entry = {
                "emotion": emotion,
                "top_tokens": tokens,
                "top_token_ids": ids,
                "top_decoded": [tokenizer.decode([i_]) for i_ in ids],
                "top_scores": [float(logits[i, j]) for j in ids],
            }
            entry.update(score_emotion(emotion, tokens, prefix_len))
            per_emotion.append(entry)
        per_variant[variant] = {
            "per_emotion": per_emotion,
            "self_token_rate": float(np.mean([e["self_token"] for e in per_emotion])),
            "latin_initial_rate": float(np.mean([e["latin_initial"] for e in per_emotion])),
        }

    result["variants"] = per_variant
    result["self_token_rate"] = per_variant[primary]["self_token_rate"]
    result["latin_initial_rate"] = per_variant[primary]["latin_initial_rate"]
    result["per_emotion"] = per_variant[primary]["per_emotion"]

    self_threshold = float(cfg["self_token_rate_threshold"])
    latin_threshold = float(cfg["latin_initial_rate_threshold"])
    result["thresholds"] = {
        "self_token_rate": self_threshold,
        "latin_initial_rate": latin_threshold,
    }
    result["passed"] = bool(
        result["self_token_rate"] >= self_threshold
        and result["latin_initial_rate"] >= latin_threshold
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result: dict[str, Any] = {
        "stage": "gate",
        "model": None,
        "passed": False,
        "error": None,
        "definitions": {
            "self_token_rate": (
                "fraction of emotions whose top_k tokens include the emotion word "
                "itself, after stripping word-boundary markers and case"
            ),
            "latin_initial_rate": (
                "looser control: fraction of emotions with at least one top_k token "
                "that is Latin-script and starts with the emotion word's first "
                "latin_initial_prefix characters (or the whole word, when it is "
                "shorter than that), so it always subsumes self_token_rate"
            ),
            "variants": (
                "'norm_weight' applies the final RMSNorm weight before the "
                "unembedding, 'no_norm' does not; both are reported so the gate "
                "result can be checked for sensitivity to that treatment"
            ),
        },
    }

    out_dir: Path | None = None
    cfg: dict[str, Any] = {}
    try:
        load_env()
        cfg = load_config(args.config)
        out_dir = (
            Path(args.out_dir)
            if args.out_dir
            else artifact_dir("gate", args.model or cfg["model"], str(cfg.get("version", "v1")))
        )
        run_gate(cfg, args, result)
    except BaseException as exc:  # noqa: BLE001 - a failed gate is the result, not a crash
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()

    if out_dir is not None:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / GATE_NAME).write_text(
                json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
            )
            upstreams = {}
            if result.get("vectors_dir") and Path(result["vectors_dir"], "manifest.json").is_file():
                upstreams["vectors"] = Path(result["vectors_dir"])
            write_manifest(
                out_dir,
                stage=str(cfg.get("stage", "gate")),
                config={**cfg, "model": result["model"] or cfg.get("model")},
                upstreams=upstreams,
            )
        except BaseException as exc:  # noqa: BLE001
            result["error"] = (result["error"] or "") + f" | writing {GATE_NAME}: {exc}"

    print(f"stage 5 gate: model={result['model']} probe_layer={result.get('probe_layer')}")
    if result["error"]:
        print(f"ERROR: {result['error']}")
    for entry in result.get("per_emotion", []):
        flags = ("S" if entry["self_token"] else "-") + ("L" if entry["latin_initial"] else "-")
        print(f"  [{flags}] {entry['emotion']:<14} {' '.join(entry['top_tokens'][:12])}")
    if result.get("n_removed"):
        print("  n_removed per layer: " + ", ".join(f"L{k}={v}" for k, v in result["n_removed"].items()))
    print(
        f"  self_token_rate={result.get('self_token_rate')} "
        f"latin_initial_rate={result.get('latin_initial_rate')} "
        f"thresholds={result.get('thresholds')}"
    )
    print(f"gate {'PASSED' if result['passed'] else 'FAILED'} for {result['model']}; wrote {out_dir}")
    # Exit 0 unconditionally: a failed gate is a recorded result, not a job failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
