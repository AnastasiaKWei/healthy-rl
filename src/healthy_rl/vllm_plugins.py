"""vLLM general plugin: make transformers >= 5.15 Gemma 4 configs readable by vLLM 0.27.

Registered under the ``vllm.general_plugins`` entry-point group in pyproject.toml,
which vLLM loads in every process (API server, engine core, workers) before it
parses the checkpoint config. It only touches the Gemma 4 text-config classes.

Why. transformers 5.15 moved Gemma 4's per-layer attention geometry into a
"heterogeneous config" (``config.per_layer_config[i]``). Two consequences for
vLLM 0.27.1, which predates that change:

1. ``head_dim`` and ``num_key_value_heads`` become *per-layer* attributes, and
   reading them on the text config raises ``AmbiguousGlobalPerLayerAttributeError``
   (a RuntimeError, so ``getattr(cfg, "head_dim", 0)`` does not swallow it).
   vLLM's ``Gemma4ModelArchConfigConvertor.get_head_size`` does exactly that read,
   so ``vllm serve`` dies while building ``ModelConfig``. That is the
   ``'head_dim' is a per-layer attribute`` traceback -- not a vLLM limitation on
   heterogeneous head sizes, which vLLM's ``gemma4.py`` handles itself.
2. ``global_head_dim`` and ``num_global_key_value_heads`` are *popped* out of the
   kwargs and stored only in ``per_layer_config``. vLLM's decoder layer reads
   ``getattr(config, "global_head_dim", config.head_dim)``, so once (1) is
   silenced it would build the full-attention layers with head_dim 256 instead
   of 512 (and the wrong KV head count under ``attention_k_eq_v``), failing at
   weight load -- or worse.

What the patch does. After the original ``__post_init__`` has built
``per_layer_config``, it (a) sets ``allow_global_per_layer_attribute_access``
so global reads warn once instead of raising, and (b) restores
``global_head_dim`` / ``num_global_key_value_heads`` as plain attributes,
derived from the first full-attention layer's per-layer config. Deriving from
``per_layer_config`` rather than from kwargs makes it survive a
``to_dict``/``from_dict`` round trip, where the original kwargs are gone.

Idempotent; a no-op if the Gemma 4 config modules are absent or already
expose the attributes (older transformers).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_MARK = "_healthy_rl_gemma4_patched"


def _restore_global_attention_attrs(cfg: Any) -> None:
    """Populate ``global_head_dim`` / ``num_global_key_value_heads`` on ``cfg``.

    Read from the heterogeneity spec's per-layer overrides (``per_layer_config``
    itself is a property that builds layer configs on demand). Every
    full-attention layer carries the same override, so the first one suffices.
    """
    d = cfg.__dict__
    # Older transformers keeps these as ordinary attributes; leave them alone.
    if "global_head_dim" in d and "num_global_key_value_heads" in d:
        return
    spec = d.get("_heterogeneity_spec")
    layer_types = d.get("layer_types") or []
    if spec is None:
        return
    try:
        full_idx = layer_types.index("full_attention")
    except ValueError:
        return
    overrides = spec.per_layer_overrides.get(full_idx) or {}
    if "global_head_dim" not in d and "head_dim" in overrides:
        d["global_head_dim"] = overrides["head_dim"]
    if "num_global_key_value_heads" not in d and "num_key_value_heads" in overrides:
        d["num_global_key_value_heads"] = overrides["num_key_value_heads"]


def _patch_text_config_class(cls: type) -> bool:
    """Wrap ``cls.__post_init__``. Returns True if this call installed the wrapper."""
    original = cls.__dict__.get("__post_init__")
    if original is None or getattr(original, _MARK, False):
        return False

    def __post_init__(self, **kwargs):
        original(self, **kwargs)
        # Only meaningful for heterogeneous configs; harmless otherwise.
        if "_heterogeneity_spec" in self.__dict__:
            self.allow_global_per_layer_attribute_access = True
            _restore_global_attention_attrs(self)

    setattr(__post_init__, _MARK, True)
    __post_init__.__wrapped__ = original  # type: ignore[attr-defined]
    cls.__post_init__ = __post_init__
    return True


def register() -> None:
    """Entry point for ``vllm.general_plugins``."""
    patched: list[str] = []
    for module_name, class_name in (
        ("transformers.models.gemma4.configuration_gemma4", "Gemma4TextConfig"),
        (
            "transformers.models.gemma4_unified.configuration_gemma4_unified",
            "Gemma4UnifiedTextConfig",
        ),
    ):
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
        # Only patch classes that actually pop the global attrs, i.e. the
        # heterogeneous-config transformers. Detect by the mixin's flag property.
        if not hasattr(cls, "allow_global_per_layer_attribute_access"):
            continue
        if _patch_text_config_class(cls):
            patched.append(class_name)
    if patched:
        logger.info("healthy_rl: patched %s for vLLM compatibility", ", ".join(patched))
