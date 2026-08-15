from __future__ import annotations

import numpy as np
import pytest

from healthy_rl.dashboard.generation import (
    Generation, assemble_generation, merge_hook_results, split_reasoning, token_kinds,
)

E, LAYERS, PROBE = 2, [10, 20], 20


def _saved(n_decode=4, n_prefill_rows=1, drop_layer=None, probe_res=True):
    saved = {}
    P = n_prefill_rows + n_decode
    for l in LAYERS:
        if l == drop_layer:
            continue
        saved[f"proj_L{l}"] = np.arange(P * E, dtype=np.float32).reshape(P, E) + l
        saved[f"norm_L{l}"] = np.full(P, 2.0, np.float32)
        saved[f"kind_L{l}"] = np.array([1.0] * n_prefill_rows + [0.0] * n_decode, np.float32)
    if probe_res:
        saved[f"res_start_L{PROBE}"] = np.ones(8, np.float16)
        saved[f"res_end_L{PROBE}"] = np.zeros(8, np.float16)
    return saved


def test_split_reasoning_handles_each_tag_style_and_none():
    assert split_reasoning("<think>a b</think>\nanswer") == ("a b", "answer", len("<think>a b</think>"))
    r, a, k = split_reasoning("[THINK]x[/THINK]y")
    assert (r, a) == ("x", "y") and k == len("[THINK]x[/THINK]")
    r, a, k = split_reasoning("<SCRATCHPAD_REASONING>s</SCRATCHPAD_REASONING> final")
    assert (r, a) == ("s", "final")
    assert split_reasoning("plain") == (None, "plain", 0)


def test_split_reasoning_unclosed_tag_is_all_reasoning():
    r, a, k = split_reasoning("<think>ran out of tok")
    assert r == "ran out of tok" and a == "" and k == len("<think>ran out of tok")


def test_split_reasoning_close_tag_without_open():
    r, a, k = split_reasoning("thoughts</think>ans")
    assert (r, a) == ("thoughts", "ans") and k == len("thoughts</think>")


def test_token_kinds_by_char_offset():
    toks = ["<think>", "a", " b", "</think>", "\n", "ans"]
    kinds = token_kinds(toks, len("<think>a b</think>"))
    assert kinds == ["think", "think", "think", "think", "answer", "answer"]
    assert token_kinds(toks, 0) == ["answer"] * 6


def test_assemble_generation_aligned():
    toks = ["<think>", "x", "</think>", "y"]
    g = assemble_generation(text="<think>x</think>y", reasoning_content=None, tokens=toks,
                            finish_reason="stop", hook_saved=_saved(n_decode=4), capture_layers=LAYERS,
                            probe_layer=PROBE, n_emotions=E, max_tokens=8, seconds=1.5)
    assert isinstance(g, Generation) and not g.misaligned and g.error is None
    assert g.proj.shape == (4, 2, E) and g.norm.shape == (4, 2)
    assert g.proj_prefill.shape == (2, E) and g.norm_prefill.shape == (2,)
    np.testing.assert_allclose(g.proj[:, 1, :], (np.arange(5 * E).reshape(5, E) + 20)[1:])
    np.testing.assert_allclose(g.proj_prefill[1], np.arange(E) + 20)
    assert g.token_kind == ["think", "think", "think", "answer"] and g.n_think == 3
    assert g.n_generated == 4 and g.at_cap is False and g.finish_reason == "stop"
    assert g.res_start.shape == (8,) and g.res_end.shape == (8,)
    assert g.reasoning == "x" and g.answer == "y" and g.seconds == 1.5


def test_assemble_generation_uses_last_prefill_row_under_chunked_prefill():
    g = assemble_generation(text="abcd", reasoning_content=None, tokens=list("abcd"), finish_reason="length",
                            hook_saved=_saved(n_decode=4, n_prefill_rows=3), capture_layers=LAYERS,
                            probe_layer=PROBE, n_emotions=E, max_tokens=4, seconds=0.1)
    np.testing.assert_allclose(g.proj_prefill[0], np.arange(E) + 10 + 2 * E)  # third prefill row
    assert g.at_cap is True and g.n_generated == 4


def test_assemble_generation_flags_misalignment_and_keeps_data():
    g = assemble_generation(text="abc", reasoning_content=None, tokens=list("abc"), finish_reason="stop",
                            hook_saved=_saved(n_decode=4), capture_layers=LAYERS, probe_layer=PROBE,
                            n_emotions=E, max_tokens=8, seconds=0.1)
    assert g.misaligned and "3" in g.error and "4" in g.error
    assert g.proj.shape[0] == 4 and g.n_generated == 4 and len(g.token_kind) == 3


def test_assemble_generation_missing_layer_is_an_error_not_a_crash():
    g = assemble_generation(text="ab", reasoning_content=None, tokens=list("ab"), finish_reason="stop",
                            hook_saved=_saved(n_decode=2, drop_layer=10), capture_layers=LAYERS,
                            probe_layer=PROBE, n_emotions=E, max_tokens=8, seconds=0.1)
    assert g.error and "layer 10" in g.error
    assert np.isnan(g.proj[:, 0, :]).all() and np.isfinite(g.proj[:, 1, :]).all()


def test_assemble_generation_with_reasoning_content_from_parser():
    toks = ["th", "ink", "ans"]
    g = assemble_generation(text="ans", reasoning_content="think", tokens=toks, finish_reason="stop",
                            hook_saved=_saved(n_decode=3), capture_layers=LAYERS, probe_layer=PROBE,
                            n_emotions=E, max_tokens=8, seconds=0.1)
    assert g.token_kind == ["think", "think", "answer"] and g.reasoning == "think" and g.answer == "ans"


def test_merge_hook_results_flattens_and_converts():
    torch = pytest.importorskip("torch")
    merged = merge_hook_results({"0": {"proj_L1": torch.ones(2, 3), "kind_L1": torch.zeros(2)}})
    assert isinstance(merged["proj_L1"], np.ndarray) and merged["proj_L1"].shape == (2, 3)
    assert merge_hook_results(None) == {}
