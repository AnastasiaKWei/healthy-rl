"""Synthetic rollout cells for the RolloutStore tests. Nothing here touches MODEL_DIR/ARTIFACT_DIR."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

EMOTIONS = ("desperate", "frustrated", "joyful")


class WhitespaceTokenizer:
    """Looks like a fast HF tokenizer for the one call the store makes."""
    is_fast = True

    def tokenize(self, text: str) -> list[str]:
        return [m.group(0) for m in re.finditer(r"\s*\S+", text)]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        spans = [(m.start(), m.end()) for m in re.finditer(r"\s*\S+", text)]
        out = {"input_ids": list(range(len(spans)))}
        if return_offsets_mapping:
            out["offset_mapping"] = spans
        return out


class FakeEvalSamples:
    """eval_loader stand-in: path -> samples. Missing path -> FileNotFoundError like the real one."""
    def __init__(self, mapping: dict[str, list[dict]]):
        self.mapping = {str(k): v for k, v in mapping.items()}
        self.calls = 0

    def __call__(self, path):
        self.calls += 1
        if str(path) not in self.mapping:
            raise FileNotFoundError(path)
        return self.mapping[str(path)]


def _npz_for(rows_ngen: list[int], *, token_arrays: bool, layers, probe, E, d, rng) -> dict:
    arrs = {}
    for t, n in enumerate(rows_ngen):
        if n == 0:
            continue                      # a zero-token turn wrote no hook rows
        arrs[f"t{t}_res_start_L{probe}"] = rng.normal(size=d).astype(np.float32)
        arrs[f"t{t}_res_end_L{probe}"] = rng.normal(size=d).astype(np.float32)
        if token_arrays:
            for L in layers:
                P = n + 1                 # 1 prefill row + n decode rows
                arrs[f"t{t}_proj_L{L}"] = (rng.normal(size=(P, E)) * 0.05).astype(np.float16)
                arrs[f"t{t}_norm_L{L}"] = np.full(P, 10.0, np.float32)
                kind = np.zeros(P, np.int8); kind[0] = 1
                arrs[f"t{t}_kind_L{L}"] = kind
    return arrs


def make_cell(root: Path, model: str, version: str, *, rows: list[dict], token_arrays: bool = True,
              max_tokens: int | None = 4, emotions=EMOTIONS, capture_layers=(10, 20), probe_layer=20,
              d_model=8, shard: str = "0/2", seed: int = 0) -> Path:
    cell = Path(root) / model / version
    (cell / "residuals").mkdir(parents=True, exist_ok=True)
    a, b = shard.split("/")
    (cell / "inspect-logs" / f"shard{a}of{b}").mkdir(parents=True, exist_ok=True)
    (cell / "inspect-logs" / f"shard{a}of{b}" / "x.eval").write_bytes(b"")
    if max_tokens is not None:
        (cell / "manifest.json").write_text(json.dumps({"config": {"max_tokens": max_tokens, "model": model}}))
    tok = WhitespaceTokenizer(); rng = np.random.default_rng(seed)
    lines = []
    for r in rows:
        comps = list(r["completions"])
        ngen = r.get("n_generated") or [len(tok.tokenize(c)) + 1 if c else 0 for c in comps]
        rel = f"residuals/{r['task_id']}_s{r.get('sample', 0)}.npz"
        np.savez(cell / rel, **_npz_for(ngen, token_arrays=token_arrays, layers=capture_layers,
                                        probe=probe_layer, E=len(emotions), d=d_model, rng=rng))
        lines.append({
            "model": model, "task_id": r["task_id"], "sample": r.get("sample", 0), "epoch": 1,
            "shard": shard, "run_id": "run", "condition_name": "readout", "tier": 1,
            "bench_split": r.get("bench_split", "conflicting"), "passed": r.get("passed", False), "score": "I",
            "n_turns": len(comps), "emotions": list(emotions), "probe_layer": probe_layer,
            "capture_layers": list(capture_layers), "turn_n_generated": ngen,
            "turn_after_test_failure": [i > 0 for i in range(len(comps))],
            "residuals": rel, "hook_data": True, "turn_errors": [], "sample_error": None,
            "scratchpad_reasoning": r.get("scratchpad", False), "affect_prompt": r.get("affect", False),
            "mindset": r.get("mindset", []), "mindset_version": 2, "turn_completion": comps,
        })
    (cell / f"rollouts.shard{a}of{b}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    return cell
