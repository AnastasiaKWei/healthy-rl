"""Gemma 4 configs must expose what vLLM 0.27's gemma4.py reads.

Guards `healthy_rl.vllm_plugins`, the vllm.general_plugins entry point that
reconciles transformers >= 5.15's heterogeneous Gemma 4 text config with vLLM
0.27.1. Without it, `vllm serve` on any Gemma 4 checkpoint dies in
`Gemma4ModelArchConfigConvertor.get_head_size` with
`AmbiguousGlobalPerLayerAttributeError` -- and if only that guard were
silenced, the full-attention layers would be built with head_dim 256 instead
of 512 because transformers pops `global_head_dim` out of the config.

Constructed from a small in-memory config, so no checkpoint is needed.
"""

from __future__ import annotations

import pickle

import pytest

transformers = pytest.importorskip("transformers")
pytest.importorskip("vllm", reason="vllm not installed")

from healthy_rl import vllm_plugins  # noqa: E402

TEXT_KW = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=6,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    global_head_dim=32,
    num_global_key_value_heads=1,
    attention_k_eq_v=True,
    vocab_size=256,
    layer_types=["sliding_attention"] * 5 + ["full_attention"],
)


def _classes():
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    out = [Gemma4TextConfig]
    try:
        from transformers.models.gemma4_unified.configuration_gemma4_unified import (
            Gemma4UnifiedTextConfig,
        )
    except ImportError:
        pass
    else:
        out.append(Gemma4UnifiedTextConfig)
    return out


@pytest.fixture(scope="module", autouse=True)
def _registered():
    vllm_plugins.register()
    vllm_plugins.register()  # idempotent


def _check(cfg):
    # What vllm/model_executor/models/gemma4.py and the arch convertor read.
    assert cfg.head_dim == 16
    assert getattr(cfg, "global_head_dim", cfg.head_dim) == 32
    assert cfg.num_key_value_heads == 2
    assert getattr(cfg, "num_global_key_value_heads", cfg.num_key_value_heads) == 1
    assert cfg.layer_types[-1] == "full_attention"


@pytest.mark.parametrize("cls", _classes(), ids=lambda c: c.__name__)
def test_global_attention_attrs_readable(cls):
    _check(cls(**TEXT_KW))


@pytest.mark.parametrize("cls", _classes(), ids=lambda c: c.__name__)
def test_survives_pickle_and_dict_roundtrip(cls):
    cfg = cls(**TEXT_KW)
    _check(pickle.loads(pickle.dumps(cfg)))
    _check(cls.from_dict(cfg.to_dict()))


def test_registered_as_vllm_general_plugin():
    from importlib.metadata import entry_points

    names = {ep.name: ep.value for ep in entry_points(group="vllm.general_plugins")}
    assert names.get("healthy_rl_gemma4") == "healthy_rl.vllm_plugins:register"
