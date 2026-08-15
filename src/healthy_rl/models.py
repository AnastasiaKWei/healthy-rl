"""Model introspection: read shapes off a checkpoint's ``config.json``.

``d_model`` and ``n_layers`` are never hardcoded; they come from the checkpoint.
Multimodal wrappers (e.g. ``Gemma4ForConditionalGeneration``,
``MuseGlimmerForConditionalGeneration``) nest the language-model config under
``text_config``, so we descend into it when it is present.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ModelSpec"]

# Accepted spellings, most standard first.
_LAYER_KEYS = ("num_hidden_layers", "n_layers", "num_layers", "n_layer")
_WIDTH_KEYS = ("hidden_size", "d_model", "n_embd", "model_dim")

CAPTURE_RADIUS = 2


def _first_key(cfg: dict, keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = cfg.get(key)
        if isinstance(value, int):
            return value
    return None


def probe_layer_for(n_layers: int) -> int:
    """Two thirds of the way up the stack.

    ``round`` is safe here: ``2 * n / 3`` has fractional part 0, 1/3 or 2/3, so a
    ``.5`` tie -- where Python would round half to even -- can never occur. This
    reproduces the pilot table exactly: 60 -> 40, 64 -> 43, 52 -> 35.
    """
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    return round(2 * n_layers / 3)


def capture_layers_for(probe_layer: int, n_layers: int) -> list[int]:
    """The ``2 * CAPTURE_RADIUS + 1`` layers around ``probe_layer``, clipped to the stack."""
    window = range(probe_layer - CAPTURE_RADIUS, probe_layer + CAPTURE_RADIUS + 1)
    clipped = {min(max(layer, 0), n_layers - 1) for layer in window}
    return sorted(clipped)


@dataclass(frozen=True)
class ModelSpec:
    """Everything downstream stages need to know about a checkpoint's shape."""

    name: str
    path: Path
    n_layers: int
    d_model: int
    architecture: str
    probe_layer: int
    capture_layers: list[int] = field(default_factory=list)

    @classmethod
    def from_checkpoint(
        cls, path: str | os.PathLike[str], name: str | None = None
    ) -> "ModelSpec":
        ckpt = Path(path)
        cfg_path = ckpt / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"no config.json in checkpoint directory: {ckpt}")
        cfg = json.loads(cfg_path.read_text())

        text_cfg = cfg.get("text_config")
        inner = text_cfg if isinstance(text_cfg, dict) else {}

        n_layers = _first_key(inner, _LAYER_KEYS)
        if n_layers is None:
            n_layers = _first_key(cfg, _LAYER_KEYS)
        d_model = _first_key(inner, _WIDTH_KEYS)
        if d_model is None:
            d_model = _first_key(cfg, _WIDTH_KEYS)
        if n_layers is None or d_model is None:
            raise KeyError(
                f"could not read layer count / hidden size from {cfg_path}; "
                f"looked for {_LAYER_KEYS} and {_WIDTH_KEYS} at the top level and in text_config"
            )

        architectures = cfg.get("architectures") or inner.get("architectures") or []
        architecture = architectures[0] if architectures else cfg.get("model_type", "unknown")

        probe_layer = probe_layer_for(n_layers)
        return cls(
            name=name if name is not None else ckpt.name,
            path=ckpt,
            n_layers=n_layers,
            d_model=d_model,
            architecture=architecture,
            probe_layer=probe_layer,
            capture_layers=capture_layers_for(probe_layer, n_layers),
        )
