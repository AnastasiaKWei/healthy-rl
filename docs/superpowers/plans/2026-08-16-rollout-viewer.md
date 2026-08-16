# Rollout Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Open pilot rollout cells (`$ARTIFACT_DIR/rollouts/<model>/<version>/`) in the Affect Scope dashboard, read-only on the login node: per-token emotion strips where the arrays exist, per-turn readouts where only boundary residuals exist, aggregates grouped by (model, cell).

**Architecture:** A `RolloutStore` presents cell directories through the interface `SessionStore` already has (one record per rollout turn, built lazily: re-tokenised `turn_completion`, EOS-tolerant alignment against the npz decode rows, surrounding messages from the cell's `.eval` logs, arrays assembled from the rollout npz or from boundary residuals × the model's directions). `app.py` stops assuming one `Vectors`: direction metadata is looked up per record, and `/api/aggregate` returns groups. The page grows a rollouts mode (rail by model ▸ cell, grouped aggregate picker) and otherwise reuses its transcript, token-strip, Tokens and Trajectory code.

**Tech Stack:** Python 3.12, numpy, FastAPI + TestClient, `transformers.AutoTokenizer` (fast tokenizers, offsets), `inspect_ai.log.read_eval_log_async`, vanilla JS + inline SVG (no build step). Tests: pytest under `tests/cpu/`, login-node runnable, no GPU, no `MODEL_DIR`.

**Spec:** `docs/superpowers/specs/2026-08-16-rollout-viewer-design.md` — read it first; this plan argues from it. Where this plan is more specific than the spec, the plan wins and the spec's "Deviations" section is updated in Task 11.

## Global Constraints

- Run everything from the repo root with `.venv/bin/python` / `.venv/bin/pytest`; never on a GPU node (this is login-node work).
- Tests must not touch `MODEL_DIR` or `ARTIFACT_DIR`: tokenizer, vectors and `.eval` loaders are injected; the default loaders are exercised only by the manual gate.
- Live (`--fake`) and replay (`--replay`) behaviour must not change: every existing test in `tests/cpu/test_dashboard_*.py` keeps passing unchanged (except the deliberate `groups[0]` reshaping in Task 7, where the assertions are updated in the same commit).
- Never pool `conflicting` and `original` (400), never relabel emotion columns (readouts `None` + `emotion_order_mismatch`), skip-and-count non-finite values. These rules are in `app.py` today; keep them.
- Deterministic ids: `record_id = <model>/<version>/<task_id>/s<sample>/t<turn>`, `conversation_id = <model>/<version>/<task_id>/s<sample>` (`sample` is the row's global sample index; the spec wrote `ep<epoch>`, but a cell has several samples per task all at Inspect epoch 1, so `sample` is the identity — see Task 2).
- One page file, no CDN, both themes; `tests/cpu/test_dashboard_page.py::test_javascript_parses` must pass after every page edit (needs `node` on PATH; skip is acceptable only if node is absent).
- Commit after every task with a message in the repo's style (imperative, one line, what and why); end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

## File structure

| file | responsibility |
|---|---|
| `src/healthy_rl/dashboard/stats.py` (modify) | `turn_readout` gains an `end` path for records without decode rows (`proj_end`/`norm_end`) |
| `src/healthy_rl/dashboard/rollout_store.py` (create) | `discover_cells`, `Cell`, `RolloutStore`; pure helpers `records_from_row`, `tokenise`, `align_tokens`, `arrays_from_npz`, `sample_messages`; default loaders `load_hf_tokenizer`, `load_model_vectors`, `load_eval_samples` |
| `src/healthy_rl/dashboard/app.py` (modify) | `AppState.mode`, `VectorsMeta`, per-record `_meta`/`_layer`; `/api/session`, `/api/conversations` filters, tokens route, grouped `/api/aggregate` |
| `src/healthy_rl/dashboard/__main__.py` (modify) | `--rollouts` entry, startup table |
| `src/healthy_rl/dashboard/static/index.html` (modify) | rollouts mode: rail, conversation header/notes, aggregate groups picker, settings tables, per-model layers |
| `tests/cpu/rollout_cell.py` (create) | `make_cell(...)` synthetic-cell builder + `WhitespaceTokenizer` + `FakeEvalSamples`, shared by the tests below |
| `tests/cpu/test_rollout_store.py` (create) | store tests |
| `tests/cpu/test_dashboard_stats.py`, `test_dashboard_app.py`, `test_dashboard_main.py`, `test_dashboard_page.py` (modify) | additions |
| `docs/runs.md`, `docs/measurement.md`, `docs/infrastructure.md`, the spec (modify) | Task 11 |

---

### Task 1: `turn_readout` end-readout without decode rows

**Files:**
- Modify: `src/healthy_rl/dashboard/stats.py:73-120`
- Test: `tests/cpu/test_dashboard_stats.py`

**Interfaces:**
- Produces: `stats.turn_readout(*, proj, norm, proj_prefill, norm_prefill, token_kind, layer_index, readout, proj_end=None, norm_end=None) -> np.ndarray | None`. When `readout == "end"` and `proj` has zero rows and `proj_end` is given, returns `proj_end[layer_index] / norm_end[layer_index]` under the same finite/zero rules `start` uses on the prefill row. Everything else unchanged.

- [ ] **Step 1: Write the failing test** (append to `tests/cpu/test_dashboard_stats.py`)

```python
def test_end_readout_from_boundary_row_when_no_decode_rows():
    import numpy as np
    from healthy_rl.dashboard import stats
    L, E = 2, 3
    empty = np.zeros((0, L, E)); empty_norm = np.zeros((0, L))
    pre = np.array([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]]); pre_n = np.array([1.0, 2.0])
    end = np.array([[np.nan, 0.0, 0.0], [0.5, -0.5, 0.25]]); end_n = np.array([np.nan, 5.0])
    v = stats.turn_readout(proj=empty, norm=empty_norm, proj_prefill=pre, norm_prefill=pre_n,
                           token_kind=[], layer_index=1, readout="end", proj_end=end, norm_end=end_n)
    assert np.allclose(v, [0.1, -0.1, 0.05])
    # non-finite norm at the other layer -> None, not inf
    assert stats.turn_readout(proj=empty, norm=empty_norm, proj_prefill=pre, norm_prefill=pre_n,
                              token_kind=[], layer_index=0, readout="end", proj_end=end, norm_end=end_n) is None
    # without the pair the old answer stands: no decode rows -> None
    assert stats.turn_readout(proj=empty, norm=empty_norm, proj_prefill=pre, norm_prefill=pre_n,
                              token_kind=[], layer_index=1, readout="end") is None
    # think_end / answer_start stay None with no decode rows even if the pair is passed
    assert stats.turn_readout(proj=empty, norm=empty_norm, proj_prefill=pre, norm_prefill=pre_n,
                              token_kind=[], layer_index=1, readout="think_end", proj_end=end, norm_end=end_n) is None
    # start is unaffected by the pair
    assert np.allclose(stats.turn_readout(proj=empty, norm=empty_norm, proj_prefill=pre, norm_prefill=pre_n,
                                          token_kind=[], layer_index=1, readout="start", proj_end=end, norm_end=end_n), [0.5, 1.0, 1.5])
```

- [ ] **Step 2: Run it** — `.venv/bin/pytest tests/cpu/test_dashboard_stats.py::test_end_readout_from_boundary_row_when_no_decode_rows -v` → FAIL (`unexpected keyword argument 'proj_end'`).

- [ ] **Step 3: Implement.** In `turn_readout` add `proj_end: np.ndarray | None = None, norm_end: np.ndarray | None = None` to the signature and, right after the `start` branch (before `T = ...`):

```python
    T = int(np.asarray(proj).shape[0])
    if T == 0:
        if readout == "end" and proj_end is not None and norm_end is not None:
            # A record that kept only boundary residuals (rollouts before 2026-08-16):
            # the end row is stored on its own, exactly like the prefill row.
            n = float(np.asarray(norm_end, dtype=np.float64)[layer_index])
            if not np.isfinite(n) or n == 0:
                return None
            return _finite_or_none(np.asarray(proj_end, dtype=np.float64)[layer_index] / n)
        return None
```
(remove the old `T = ...; if T == 0: return None` pair). Update the docstring: "`end` may also be read from an explicit `proj_end`/`norm_end` row when there are no decode rows." Update the module docstring's Shapes line: "`proj_end` (`L, E`) / `norm_end` (`L,`) optional".

- [ ] **Step 4: Run** `.venv/bin/pytest tests/cpu/test_dashboard_stats.py -v` → all PASS.

- [ ] **Step 5: Commit** — `git add src/healthy_rl/dashboard/stats.py tests/cpu/test_dashboard_stats.py && git commit -m "Let turn_readout read 'end' from a stored boundary row when a record has no decode rows"`.

---

### Task 2: Synthetic cell fixture, cell discovery, light records and conversations

**Files:**
- Create: `tests/cpu/rollout_cell.py`
- Create: `src/healthy_rl/dashboard/rollout_store.py`
- Create: `tests/cpu/test_rollout_store.py`

**Interfaces:**
- Produces (test helper): `make_cell(root: Path, model: str, version: str, *, rows: list[dict], token_arrays: bool = True, max_tokens: int | None = 4, emotions=("desperate","frustrated","joyful"), capture_layers=(10, 20), probe_layer=20, d_model=8, shard="0/2") -> Path` — writes `root/model/version/rollouts.shard0of2.jsonl`, `residuals/<task>_s<sample>.npz`, `manifest.json`, `inspect-logs/shard0of2/x.eval` (empty placeholder file). Each entry of `rows` is `{"task_id", "sample", "completions": [str,...], "n_generated": [int,...] | None, "passed": bool, "bench_split": "conflicting", "mindset": [], "scratchpad": False, "affect": False}`; `n_generated` defaults to `len(WhitespaceTokenizer().tokenize(c)) + 1` per completion (the EOS row). Also `WhitespaceTokenizer` (callable like a fast HF tokenizer: `tok(text, add_special_tokens=False, return_offsets_mapping=True)` → `{"input_ids": [...], "offset_mapping": [(s, e), ...]}`, `.is_fast = True`) and `FakeEvalSamples(mapping: dict[str, list[dict]])` — a callable `path -> list[{"id","epoch","messages":[{"role","content"}]}]`.
- Produces (module): `discover_cells(paths) -> tuple[list[Cell], list[Path]]`; `Cell(model, version, path, max_tokens, manifest)`; `records_from_row(row, *, model, version, max_tokens, created_at) -> list[dict]` (light records: everything in spec §3.1 except `tokens`, `token_kind`, `misaligned`, `error`, `messages_in`, `feedback`, `has_token_arrays`; those are filled by `record()` in Tasks 3–5; light records carry `tokenised: False`); `RolloutStore.open(paths, *, tokenizer_loader=None, vectors_loader=None, eval_loader=None) -> RolloutStore`; `.session`, `.root`, `.records()`, `.record(rid)`, `.arrays(rid)`, `.conversations()`, `.refresh()`, `.cells`. `record()`/`arrays()` raise `NotImplementedError` until Tasks 3–4 fill them in (tests for them come with those tasks).

- [ ] **Step 1: Write the fixture helper** `tests/cpu/rollout_cell.py`:

```python
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
```

- [ ] **Step 2: Write the failing tests** `tests/cpu/test_rollout_store.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rollout_cell import EMOTIONS, FakeEvalSamples, WhitespaceTokenizer, make_cell
from healthy_rl.dashboard.rollout_store import RolloutStore, discover_cells, records_from_row

ROWS = [
    {"task_id": "lcbhard_0", "sample": 0, "completions": ["a b c", "[THINK]x y[/THINK] z"], "passed": False},
    {"task_id": "lcbhard_0", "sample": 1, "completions": ["p q", "r s t u"], "passed": True},
    {"task_id": "lcbhard_1", "sample": 0, "completions": ["", "k l m"], "n_generated": [0, 4], "passed": False},
]


def _store(tmp_path, **kw):
    make_cell(tmp_path / "rollouts", "fake-model", "appr6", rows=ROWS)
    make_cell(tmp_path / "rollouts", "fake-model", "d6", rows=ROWS[:1], token_arrays=False)
    (tmp_path / "rollouts" / "fake-model" / "scratchpad-sanity").mkdir()
    (tmp_path / "rollouts" / "fake-model" / "scratchpad-sanity" / "transcripts.jsonl").write_text("{}\n")
    return RolloutStore.open([tmp_path / "rollouts"], tokenizer_loader=lambda m: WhitespaceTokenizer(),
                             vectors_loader=lambda m: None, eval_loader=FakeEvalSamples({}), **kw)


def test_discover_cells_from_root_model_and_cell(tmp_path):
    make_cell(tmp_path / "r", "m1", "d6", rows=ROWS[:1]); make_cell(tmp_path / "r", "m2", "aff6", rows=ROWS[:1])
    (tmp_path / "r" / "m1" / "residuals-only").mkdir()
    cells, ignored = discover_cells([tmp_path / "r"])
    assert [(c.model, c.version) for c in cells] == [("m1", "d6"), ("m2", "aff6")]
    assert [p.name for p in ignored] == ["residuals-only"]
    assert [(c.model, c.version) for c in discover_cells([tmp_path / "r" / "m1"])[0]] == [("m1", "d6")]
    assert [(c.model, c.version) for c in discover_cells([tmp_path / "r" / "m2" / "aff6"])[0]] == [("m2", "aff6")]
    assert cells[0].max_tokens == 4 and cells[0].path == tmp_path / "r" / "m1" / "d6"


def test_discover_cells_dedupes_and_reads_missing_manifest(tmp_path):
    c = make_cell(tmp_path / "r", "m1", "d6", rows=ROWS[:1], max_tokens=None)
    cells, _ = discover_cells([tmp_path / "r", c])
    assert len(cells) == 1 and cells[0].max_tokens is None


def test_records_from_row_shape(tmp_path):
    cell = make_cell(tmp_path / "r", "m1", "appr6", rows=ROWS)
    rows = [json.loads(l) for l in (cell / "rollouts.shard0of2.jsonl").read_text().splitlines()]
    recs = records_from_row(rows[2], model="m1", version="appr6", max_tokens=4, created_at="2026-08-16T00:00:00+00:00")
    assert [r["record_id"] for r in recs] == ["m1/appr6/lcbhard_1/s0/t0", "m1/appr6/lcbhard_1/s0/t1"]
    assert recs[0]["conversation_id"] == "m1/appr6/lcbhard_1/s0" and recs[0]["source"] == "rollout"
    assert recs[0]["n_generated"] == 0 and recs[0]["non_empty_turn_index"] is None
    assert recs[1]["non_empty_turn_index"] == 0 and recs[1]["at_cap"] is True   # 4 >= max_tokens 4
    assert recs[1]["turn_index"] == 1 and recs[1]["after_test_failure"] is True
    assert recs[1]["text"] == "k l m" and recs[1]["reasoning"] is None and recs[1]["answer"] == "k l m"
    assert recs[1]["emotions"] == list(EMOTIONS) and recs[1]["probe_layer"] == 20 and recs[1]["capture_layers"] == [10, 20]
    assert recs[1]["passed"] is False and recs[1]["bench_split"] == "conflicting" and recs[1]["mindset"] == []
    assert recs[1]["tokenised"] is False and recs[1]["arrays"] == "virtual"
    r1 = records_from_row(rows[0], model="m1", version="appr6", max_tokens=None, created_at="x")[1]
    assert r1["reasoning"] == "x y" and r1["answer"] == "z" and r1["at_cap"] is None and "cap unknown" in " ".join(r1["warnings"])


def test_records_from_row_defaults_for_old_rows(tmp_path):
    cell = make_cell(tmp_path / "r", "m1", "d6", rows=ROWS[:1])
    row = json.loads((cell / "rollouts.shard0of2.jsonl").read_text().splitlines()[0])
    for k in ("bench_split", "mindset", "mindset_version"):
        row.pop(k)
    r = records_from_row(row, model="m1", version="d6", max_tokens=4, created_at="x")[0]
    assert r["bench_split"] == "conflicting" and r["mindset"] == [] and r["mindset_version"] == 0


def test_store_records_conversations_session(tmp_path):
    st = _store(tmp_path)
    recs = st.records()
    assert len(recs) == 3 * 2 + 1 * 2                      # appr6: 3 rollouts x 2 turns; d6: 1 x 2
    convs = st.conversations()
    assert len(convs) == 4
    c = next(c for c in convs if c["conversation_id"] == "fake-model/appr6/lcbhard_0/s1")
    assert c["source"] == "rollout" and c["model"] == "fake-model" and c["version"] == "appr6"
    assert c["task_id"] == "lcbhard_0" and c["sample"] == 1 and c["passed"] is True and c["n_turns"] == 2
    assert c["bench_split"] == "conflicting" and c["mindset"] == [] and c["has_token_arrays"] is True
    d6 = next(c for c in convs if c["version"] == "d6")
    assert d6["has_token_arrays"] is False
    s = st.session
    assert s["mode"] == "rollouts" and s["models"]["fake-model"]["probe_layer"] == 20
    assert s["models"]["fake-model"]["emotions"] == list(EMOTIONS)
    assert s["models"]["fake-model"]["tokenizer"] == "ok" and s["models"]["fake-model"]["vectors"] == "missing"
    cells = {(c["model"], c["version"]): c for c in s["cells"]}
    assert cells[("fake-model", "appr6")]["n_rollouts"] == 3 and cells[("fake-model", "appr6")]["n_with_token_arrays"] == 3
    assert cells[("fake-model", "d6")]["n_with_token_arrays"] == 0 and cells[("fake-model", "d6")]["max_tokens"] == 4
    assert cells[("fake-model", "appr6")]["n_tokenised"] == 0
    assert st.root == tmp_path / "rollouts" and s["ignored"] == [str(tmp_path / "rollouts" / "fake-model" / "scratchpad-sanity")]


def test_refresh_sees_appended_row_and_new_shard(tmp_path):
    st = _store(tmp_path)
    assert len(st.records()) == 8
    cell = tmp_path / "rollouts" / "fake-model" / "appr6"
    f = cell / "rollouts.shard0of2.jsonl"
    row = json.loads(f.read_text().splitlines()[0]); row["task_id"] = "lcbhard_9"
    with f.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    import os, time
    os.utime(f, (time.time() + 5, time.time() + 5))       # mtime moves forward even on coarse filesystems
    assert len(st.records()) == 10
    (cell / "rollouts.shard1of2.jsonl").write_text(json.dumps({**row, "task_id": "lcbhard_8"}) + "\n")
    assert len(st.records()) == 12
    assert st.session["cells"][0]["n_rollouts"] in (5, 1)  # session recomputed after refresh (order by (model, version))


def test_light_records_do_not_load_tokenizer(tmp_path):
    calls = []
    make_cell(tmp_path / "rollouts", "fake-model", "appr6", rows=ROWS)
    st = RolloutStore.open([tmp_path / "rollouts"], tokenizer_loader=lambda m: calls.append(m) or WhitespaceTokenizer(),
                           vectors_loader=lambda m: None, eval_loader=FakeEvalSamples({}))
    st.records(); st.conversations(); st.session
    assert calls == []
```
(`refresh` ordering: `session["cells"]` is sorted by `(model, version)`, so `cells[0]` is `appr6` with 5 rollouts.)

- [ ] **Step 3: Run** `.venv/bin/pytest tests/cpu/test_rollout_store.py -v` → FAIL (`No module named healthy_rl.dashboard.rollout_store`).

- [ ] **Step 4: Implement** `src/healthy_rl/dashboard/rollout_store.py` (this task's part; later tasks add to it):

```python
"""Pilot rollout cells, presented as dashboard records.

A rollout record (one JSONL row = one whole rollout, per-turn arrays in one npz)
becomes one dashboard record per turn, so the page's transcript, token strip,
Tokens chart and aggregates work unchanged. Everything expensive is lazy and
cached: ``records()`` is a cheap JSON parse; ``record(rid)`` re-tokenises that
turn's ``turn_completion`` and checks it against the npz decode rows;
``arrays(rid)`` reads the npz. Cells that were still running when the store
opened grow on the next ``records()`` call.

Loaders are injected so the login-node tests never touch MODEL_DIR/ARTIFACT_DIR.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from healthy_rl.dashboard import stats
from healthy_rl.dashboard.generation import split_reasoning
from healthy_rl.rollouts import read_jsonl

CELL_GLOB = "rollouts*.jsonl"
SOURCE = "rollout"


@dataclass
class Cell:
    model: str
    version: str
    path: Path
    max_tokens: int | None
    manifest: dict

    @property
    def key(self) -> tuple[str, str]:
        return (self.model, self.version)


def _is_cell(p: Path) -> bool:
    return p.is_dir() and any(p.glob(CELL_GLOB))


def _cell_at(p: Path) -> Cell:
    manifest: dict = {}
    mp = p / "manifest.json"
    if mp.is_file():
        try:
            manifest = json.loads(mp.read_text())
        except (OSError, ValueError):
            manifest = {}
    mt = (manifest.get("config") or {}).get("max_tokens")
    return Cell(model=p.parent.name, version=p.name, path=p, max_tokens=int(mt) if mt is not None else None, manifest=manifest)


def discover_cells(paths: Sequence[str | os.PathLike[str]]) -> tuple[list[Cell], list[Path]]:
    """Cells under each path (a cell, a model, or a root), and the directories skipped.

    Deduped by path, ordered by (model, version). A directory is a cell when it
    holds ``rollouts*.jsonl``; a model is walked one level, a root two.
    """
    found: dict[Path, Cell] = {}
    ignored: list[Path] = []

    def visit(p: Path, depth: int) -> None:
        if _is_cell(p):
            found.setdefault(p.resolve(), _cell_at(p))
            return
        if depth == 0 or not p.is_dir():
            ignored.append(p)
            return
        kids = sorted(k for k in p.iterdir() if k.is_dir())
        if not kids:
            ignored.append(p)
            return
        for k in kids:
            visit(k, depth - 1)

    for raw in paths:
        visit(Path(raw), 2)
    cells = sorted(found.values(), key=lambda c: c.key)
    return cells, ignored


def _now_iso(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).isoformat(timespec="seconds")


def records_from_row(row: dict, *, model: str, version: str, max_tokens: int | None, created_at: str) -> list[dict]:
    """One light record per turn of a rollout row (spec §3.1, minus the lazy fields)."""
    n_turns = int(row.get("n_turns") or len(row.get("turn_completion") or []))
    ngen = list(row.get("turn_n_generated") or [0] * n_turns)
    comps = list(row.get("turn_completion") or [""] * n_turns)
    after = list(row.get("turn_after_test_failure") or [False] * n_turns)
    nei = stats.non_empty_index(ngen)
    sample = int(row.get("sample", 0))
    task_id = str(row.get("task_id"))
    cid = f"{model}/{version}/{task_id}/s{sample}"
    base = {
        "conversation_id": cid, "source": SOURCE, "model": model, "version": version,
        "mindset": list(row.get("mindset") or []), "mindset_version": int(row.get("mindset_version") or 0),
        "scratchpad_reasoning": bool(row.get("scratchpad_reasoning")), "affect_prompt": bool(row.get("affect_prompt")),
        "bench_split": row.get("bench_split") or "conflicting", "task_id": task_id, "sample": sample,
        "epoch": int(row.get("epoch") or 1), "passed": row.get("passed"), "shard": row.get("shard"),
        "run_id": row.get("run_id"), "condition_name": row.get("condition_name"),
        "emotions": list(row.get("emotions") or []), "capture_layers": [int(l) for l in row.get("capture_layers") or []],
        "probe_layer": row.get("probe_layer"), "residuals": row.get("residuals"), "created_at": created_at,
        "arrays": "virtual", "tokenised": False, "n_turns_total": n_turns,
    }
    out = []
    for t in range(n_turns):
        text = comps[t] if t < len(comps) and comps[t] is not None else ""
        reasoning, answer, _ = split_reasoning(text)
        n = int(ngen[t]) if t < len(ngen) else 0
        warnings: list[str] = []
        if max_tokens is None:
            at_cap = None
            warnings.append("cap unknown: the cell's manifest.json has no config.max_tokens")
        else:
            at_cap = n >= max_tokens
        out.append({**base, "record_id": f"{cid}/t{t}", "turn_index": t, "non_empty_turn_index": nei[t],
                    "n_generated": n, "at_cap": at_cap, "after_test_failure": bool(after[t]) if t < len(after) else False,
                    "text": text, "reasoning": reasoning, "answer": answer, "warnings": warnings})
    return out


class RolloutStore:
    """Read-only, ``SessionStore``-shaped view over rollout cells."""

    def __init__(self, cells: list[Cell], ignored: list[Path], roots: list[Path], *,
                 tokenizer_loader: Callable[[str], Any] | None = None,
                 vectors_loader: Callable[[str], Any] | None = None,
                 eval_loader: Callable[[Path], list[dict]] | None = None) -> None:
        self.cells = cells
        self.ignored = ignored
        self.roots = roots
        self.root = roots[0] if roots else Path(".")
        self._tokenizer_loader = tokenizer_loader or load_hf_tokenizer
        self._vectors_loader = vectors_loader or load_model_vectors
        self._eval_loader = eval_loader or load_eval_samples
        self._files: dict[Path, tuple[float, int]] = {}          # jsonl -> (mtime, size) seen
        self._rows_by_file: dict[Path, list[dict]] = {}
        self._light: dict[str, dict] = {}                        # record_id -> light record
        self._order: list[str] = []
        self._full: dict[str, dict] = {}                         # record_id -> tokenised record (Task 3)
        self._tokenizers: dict[str, Any] = {}                    # model -> tokenizer | None
        self._vectors: dict[str, Any] = {}                       # model -> Vectors | None
        self._session: dict | None = None
        self.refresh()

    @classmethod
    def open(cls, paths: Sequence[str | os.PathLike[str]], **kw) -> "RolloutStore":
        cells, ignored = discover_cells(paths)
        if not cells:
            raise FileNotFoundError(f"no rollout cells under {[str(p) for p in paths]}")
        return cls(cells, ignored, [Path(p) for p in paths], **kw)

    # ---- growth ---------------------------------------------------------
    def refresh(self) -> bool:
        """Re-read changed/new shard files. Returns True if anything changed."""
        changed = False
        for cell in self.cells:
            for f in sorted(cell.path.glob(CELL_GLOB)):
                try:
                    stt = f.stat()
                except FileNotFoundError:
                    continue
                sig = (stt.st_mtime, stt.st_size)
                if self._files.get(f) == sig:
                    continue
                self._files[f] = sig
                self._rows_by_file[f] = [(cell, r) for r in read_jsonl(f)]
                changed = True
        if changed:
            self._light.clear(); self._order.clear()
            for f in sorted(self._rows_by_file):
                created = _now_iso(self._files[f][0])
                for cell, row in self._rows_by_file[f]:
                    for rec in records_from_row(row, model=cell.model, version=cell.version,
                                                max_tokens=cell.max_tokens, created_at=created):
                        self._light[rec["record_id"]] = rec
                        self._order.append(rec["record_id"])
            self._session = None
        return changed

    # ---- SessionStore interface -----------------------------------------
    def records(self) -> list[dict]:
        self.refresh()
        return [self._full.get(rid) or self._light[rid] for rid in self._order]

    def record(self, record_id: str) -> dict:
        raise NotImplementedError  # Task 3

    def arrays(self, record_id: str) -> dict[str, np.ndarray]:
        raise NotImplementedError  # Task 4

    def conversations(self) -> list[dict]:
        out: dict[str, dict] = {}
        for r in self.records():
            cid = r["conversation_id"]
            c = out.get(cid)
            if c is None:
                c = out[cid] = {"conversation_id": cid, "source": SOURCE, "model": r["model"], "version": r["version"],
                                "bench_split": r["bench_split"], "task_id": r["task_id"], "sample": r["sample"],
                                "epoch": r["epoch"], "mindset": r["mindset"], "passed": r["passed"], "title": None,
                                "n_turns": 0, "has_token_arrays": self._has_token_arrays(r), "n_misaligned": 0,
                                "last_created_at": r["created_at"]}
            c["n_turns"] += 1
            if r.get("misaligned"):
                c["n_misaligned"] += 1
        return list(out.values())

    def append(self, *a, **k):
        raise PermissionError("rollout store is read-only")

    def close(self) -> None:
        return None

    # ---- session ---------------------------------------------------------
    def _has_token_arrays(self, rec: dict) -> bool:
        """Does this rollout's npz carry per-token projections? Cached per npz path."""
        cache = getattr(self, "_hta", None)
        if cache is None:
            cache = self._hta = {}
        rel = rec.get("residuals")
        if not rel:
            return False
        path = self._cell_of(rec).path / rel
        if path not in cache:
            try:
                with np.load(path) as z:
                    cache[path] = any(k.startswith("t") and "_proj_L" in k for k in z.files)
            except (OSError, ValueError):
                cache[path] = False
        return cache[path]

    def _cell_of(self, rec: dict) -> Cell:
        return next(c for c in self.cells if c.key == (rec["model"], rec["version"]))

    def _model_meta(self, model: str) -> dict:
        first = next((r for r in self._light.values() if r["model"] == model), None)
        vec = self._vectors_for(model)
        emotions = list(vec.emotions) if vec is not None else list(first["emotions"] if first else [])
        return {"emotions": emotions,
                "capture_layers": list(first["capture_layers"]) if first else [],
                "probe_layer": first["probe_layer"] if first else None,
                "tokenizer": "ok" if self._tokenizer_for(model, probe=True) else "missing",
                "vectors": "ok" if vec is not None else "missing"}

    def _tokenizer_for(self, model: str, *, probe: bool = False):
        """The model's tokenizer, or None. ``probe=True`` only reports availability
        without loading (used by session) -- see Task 3 for the loading path."""
        if model in self._tokenizers:
            return self._tokenizers[model]
        if probe:
            return self._tokenizer_available(model)
        try:
            tok = self._tokenizer_loader(model)
        except Exception:      # any failure to load reads as "missing", never as a 500
            tok = None
        self._tokenizers[model] = tok
        return tok

    def _tokenizer_available(self, model: str) -> bool:
        """Availability without loading: injected loaders are always 'available';
        the default loader checks the directory exists."""
        if self._tokenizer_loader is not load_hf_tokenizer:
            return True
        return _hf_tokenizer_dir(model) is not None

    def _vectors_for(self, model: str):
        if model not in self._vectors:
            try:
                self._vectors[model] = self._vectors_loader(model)
            except Exception:
                self._vectors[model] = None
        return self._vectors[model]

    @property
    def session(self) -> dict:
        self.refresh()
        if self._session is None:
            models = sorted({c.model for c in self.cells})
            cells = []
            for c in self.cells:
                recs = [r for r in self._light.values() if (r["model"], r["version"]) == c.key]
                convs = {r["conversation_id"]: r for r in recs}
                first = next(iter(convs.values()), None)
                tokd = [self._full[r["record_id"]] for r in recs if r["record_id"] in self._full]
                cells.append({"model": c.model, "version": c.version, "path": str(c.path),
                              "bench_split": first["bench_split"] if first else None,
                              "mindset": first["mindset"] if first else [],
                              "scratchpad_reasoning": first["scratchpad_reasoning"] if first else False,
                              "affect_prompt": first["affect_prompt"] if first else False,
                              "n_rollouts": len(convs),
                              "n_with_token_arrays": sum(1 for r in convs.values() if self._has_token_arrays(r)),
                              "n_tokenised": len(tokd), "n_misaligned": sum(1 for r in tokd if r.get("misaligned")),
                              "max_tokens": c.max_tokens})
            self._session = {"mode": "rollouts", "model": ", ".join(models), "roots": [str(p) for p in self.roots],
                             "ignored": [str(p) for p in self.ignored],
                             "models": {m: self._model_meta(m) for m in models}, "cells": cells}
        return self._session


# ---- default loaders (not exercised by tests) ------------------------------
def _hf_tokenizer_dir(model: str) -> Path | None:
    root = os.environ.get("MODEL_DIR")
    if not root:
        return None
    p = Path(root) / model
    return p if (p / "tokenizer_config.json").is_file() or (p / "tokenizer.json").is_file() else None


def load_hf_tokenizer(model: str):
    """``AutoTokenizer`` from ``$MODEL_DIR/<model>``; None when the directory has no tokenizer."""
    p = _hf_tokenizer_dir(model)
    if p is None:
        return None
    from transformers import AutoTokenizer  # heavy import, only on the login node path
    return AutoTokenizer.from_pretrained(str(p))


def load_model_vectors(model: str):
    """``load_vectors($ARTIFACT_DIR/vectors/<model>/v1)`` or None when absent."""
    root = os.environ.get("ARTIFACT_DIR")
    if not root:
        return None
    p = Path(root) / "vectors" / model / "v1"
    if not (p / "vectors.json").is_file():
        return None
    from healthy_rl.rollouts import load_vectors
    return load_vectors(p)


def load_eval_samples(path: Path) -> list[dict]:
    """Samples of one ``.eval`` as plain dicts: ``{"id", "epoch", "messages": [{"role", "content"}]}``.

    Uses the async reader under its own loop: the sync ``read_eval_log`` dies with
    "no running event loop" as soon as a log has samples (scripts/_read_transcript.py).
    """
    import asyncio
    from inspect_ai.log import read_eval_log_async
    log = asyncio.run(read_eval_log_async(str(path)))
    out = []
    for s in log.samples or []:
        msgs = []
        for m in s.messages:
            c = m.content
            if not isinstance(c, str):
                c = "".join((getattr(x, "text", None) or getattr(x, "reasoning", None) or "") for x in c)
            msgs.append({"role": m.role, "content": c})
        out.append({"id": str(s.id), "epoch": int(getattr(s, "epoch", 1)), "messages": msgs})
    return out
```

- [ ] **Step 5: Run** `.venv/bin/pytest tests/cpu/test_rollout_store.py -v` → the six tests of this task PASS. (If `refresh` mtime handling makes `test_refresh_sees_appended_row_and_new_shard` flaky on the shared filesystem, the size half of the signature catches the append; keep both.)

- [ ] **Step 6: Commit** — `git add tests/cpu/rollout_cell.py tests/cpu/test_rollout_store.py src/healthy_rl/dashboard/rollout_store.py && git commit -m "Add RolloutStore: discover pilot cells and present each rollout turn as a dashboard record"`.

---

### Task 3: Tokenisation, the EOS rule, and `record(rid)`

**Files:**
- Modify: `src/healthy_rl/dashboard/rollout_store.py`
- Test: `tests/cpu/test_rollout_store.py`

**Interfaces:**
- Produces: `tokenise(text: str, tokenizer) -> tuple[list[str], list[int]]` — tokens tile `text` exactly (`"".join(tokens) == text`), `starts[i]` is the true start offset of token i (a leading gap is folded into the token's text but not into its start). Fast tokenizers via `return_offsets_mapping`; slow ones via per-id `decode` with cumulative offsets. `align_tokens(tokens, starts, think_end_char, n_decode: int | None) -> tuple[list[str], list[str], bool, str | None]` → `(tokens, kinds, misaligned, error)` under the EOS rule (§3.3). `RolloutStore.record(rid)` returns the full record: light fields + `tokens`, `token_kind`, `n_think`, `misaligned`, `error`, `has_token_arrays`, `n_decode`, `tokenised: True`, `warnings` extended; cached in `self._full`. `RolloutStore._decode_rows(rec) -> int | None` reads `t{t}_kind_L{probe}` and counts zeros.

- [ ] **Step 1: Write the failing tests** (append):

```python
from healthy_rl.dashboard.rollout_store import align_tokens, tokenise


def test_tokenise_tiles_text_and_keeps_true_starts():
    toks, starts = tokenise("ab  cd e", WhitespaceTokenizer())
    assert toks == ["ab", "  cd", " e"] and "".join(toks) == "ab  cd e"
    assert starts == [0, 2, 6]
    assert tokenise("", WhitespaceTokenizer()) == ([], [])


def test_align_tokens_eos_rule():
    toks, starts = ["a", " b", " c"], [0, 1, 3]
    assert align_tokens(toks, starts, 0, 3) == (toks, ["answer"] * 3, False, None)
    t, k, mis, err = align_tokens(toks, starts, 0, 4)
    assert t == toks + ["<eos>"] and k == ["answer"] * 4 and mis is False and err is None
    t, k, mis, err = align_tokens(toks, starts, 0, 5)
    assert mis is True and "3 tokens" in err and "5 decode rows" in err and t == toks
    t, k, mis, err = align_tokens(toks, starts, 0, 2)
    assert mis is True
    # no arrays at all: nothing to check against
    assert align_tokens(toks, starts, 0, None) == (toks, ["answer"] * 3, False, None)
    # think/answer split by start offset; eos inherits the last kind
    t, k, _, _ = align_tokens(["[THINK]x", " y[/THINK]", " z"], [0, 8, 18], 18, 4)
    assert k == ["think", "think", "answer", "answer"]
    t, k, _, _ = align_tokens(["[THINK]x", " y[/THINK]"], [0, 8], 18, 3)
    assert k == ["think", "think", "think"]


def test_record_is_tokenised_and_cached(tmp_path):
    st = _store(tmp_path)
    r = st.record("fake-model/appr6/lcbhard_0/s0/t1")
    assert r["tokenised"] is True and r["tokens"] == ["[THINK]x", " y[/THINK]", " z", "<eos>"]
    assert r["token_kind"] == ["think", "think", "answer", "answer"] and r["n_think"] == 2
    assert r["misaligned"] is False and r["has_token_arrays"] is True and r["n_decode"] == 4
    assert st.record("fake-model/appr6/lcbhard_0/s0/t1") is r
    # records() now hands back the full record for that id
    assert next(x for x in st.records() if x["record_id"] == r["record_id"])["tokenised"] is True
    # zero-token turn: no rows, no tokens, not misaligned
    z = st.record("fake-model/appr6/lcbhard_1/s0/t0")
    assert z["tokens"] == [] and z["misaligned"] is False and z["has_token_arrays"] is False
    # old cell: tokens exist (text is there) but there are no arrays to align against
    o = st.record("fake-model/d6/lcbhard_0/s0/t0")
    assert o["tokens"] == ["a", " b", " c"] and o["has_token_arrays"] is False and o["misaligned"] is False and o["n_decode"] is None


def test_record_misaligned_when_counts_disagree(tmp_path):
    rows = [{"task_id": "lcbhard_0", "sample": 0, "completions": ["a b c"], "n_generated": [7]}]
    make_cell(tmp_path / "rollouts", "fake-model", "appr6", rows=rows)
    st = RolloutStore.open([tmp_path / "rollouts"], tokenizer_loader=lambda m: WhitespaceTokenizer(),
                           vectors_loader=lambda m: None, eval_loader=FakeEvalSamples({}))
    r = st.record("fake-model/appr6/lcbhard_0/s0/t0")
    assert r["misaligned"] is True and "3 tokens" in r["error"] and "7 decode rows" in r["error"]
    assert st.session["cells"][0]["n_tokenised"] == 1 and st.session["cells"][0]["n_misaligned"] == 1
    assert st.conversations()[0]["n_misaligned"] == 1


def test_record_without_tokenizer(tmp_path):
    make_cell(tmp_path / "rollouts", "fake-model", "appr6", rows=ROWS[:1])
    st = RolloutStore.open([tmp_path / "rollouts"], tokenizer_loader=lambda m: None,
                           vectors_loader=lambda m: None, eval_loader=FakeEvalSamples({}))
    r = st.record("fake-model/appr6/lcbhard_0/s0/t0")
    assert r["tokens"] == [] and r["misaligned"] is True and r["error"] == "no tokenizer for fake-model"
    assert st.session["models"]["fake-model"]["tokenizer"] == "missing"


def test_record_unknown_id(tmp_path):
    with pytest.raises(KeyError):
        _store(tmp_path).record("nope")
```

- [ ] **Step 2: Run** → FAIL (`cannot import name 'tokenise'`).

- [ ] **Step 3: Implement.** Add to `rollout_store.py`:

```python
EOS_TOKEN = "<eos>"


def tokenise(text: str, tokenizer) -> tuple[list[str], list[int]]:
    """Tokens that tile ``text`` exactly, plus each token's true start offset.

    Fast tokenizers give offsets; a leading gap (SentencePiece drops the space
    from some spans) is folded into the following token's text so the strip
    reproduces the text character for character, but the start offset stays the
    span's own start so a think/answer boundary is not moved by a space.
    """
    if not text:
        return [], []
    if getattr(tokenizer, "is_fast", False):
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        spans = list(enc["offset_mapping"])
        tokens, starts, prev = [], [], 0
        for i, (s, e) in enumerate(spans):
            e = max(int(e), prev)
            if i == len(spans) - 1:
                e = max(e, len(text))
            tokens.append(text[prev:e]); starts.append(int(s))
            prev = e
        return tokens, starts
    ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = [tokenizer.decode([i]) for i in ids]
    starts, pos = [], 0
    for t in tokens:
        starts.append(pos); pos += len(t)
    return tokens, starts


def align_tokens(tokens: list[str], starts: list[int], think_end_char: int, n_decode: int | None
                 ) -> tuple[list[str], list[str], bool, str | None]:
    """Apply the EOS rule (spec §3.3): N == D aligned; N + 1 == D aligned with an appended
    ``<eos>``; anything else misaligned. ``n_decode=None`` means there are no arrays."""
    kinds = ["think" if s < think_end_char else "answer" for s in starts]
    N = len(tokens)
    if n_decode is None or N == n_decode:
        return list(tokens), kinds, False, None
    if N + 1 == n_decode:
        return list(tokens) + [EOS_TOKEN], kinds + [kinds[-1] if kinds else "answer"], False, None
    return list(tokens), kinds, True, f"re-tokenised {N} tokens, {n_decode} decode rows"
```

and in `RolloutStore`:

```python
    def _npz_path(self, rec: dict) -> Path | None:
        rel = rec.get("residuals")
        return (self._cell_of(rec).path / rel) if rel else None

    def _decode_rows(self, rec: dict) -> int | None:
        """Decode-row count at the probe layer, or None when the turn has no per-token arrays."""
        path = self._npz_path(rec)
        if path is None:
            return None
        key = f"t{rec['turn_index']}_kind_L{rec['probe_layer']}"
        try:
            with np.load(path) as z:
                if key not in z.files:
                    return None
                return int((np.asarray(z[key]).reshape(-1) == 0).sum())
        except (OSError, ValueError):
            return None

    def record(self, record_id: str) -> dict:
        full = self._full.get(record_id)
        if full is not None:
            return full
        self.refresh()
        light = self._light.get(record_id)
        if light is None:
            raise KeyError(record_id)
        rec = dict(light)
        warnings = list(rec.get("warnings") or [])
        n_decode = self._decode_rows(rec)
        rec["n_decode"] = n_decode
        rec["has_token_arrays"] = n_decode is not None
        tok = self._tokenizer_for(rec["model"])
        if tok is None:
            rec.update(tokens=[], token_kind=[], n_think=0, misaligned=True, error=f"no tokenizer for {rec['model']}")
            warnings.append("tokenizer missing: text and turn readouts only")
        else:
            _, _, think_end = split_reasoning(rec["text"])
            toks, starts = tokenise(rec["text"], tok)
            toks, kinds, mis, err = align_tokens(toks, starts, think_end, n_decode)
            rec.update(tokens=toks, token_kind=kinds, n_think=sum(1 for k in kinds if k == "think"),
                       misaligned=mis, error=err)
        rec["warnings"] = warnings
        rec["tokenised"] = True
        self._full[record_id] = rec
        self._session = None          # cell table counts changed
        return rec
```
Task 5 adds `messages_in`/`feedback` inside `record()`; leave a comment `# messages_in / feedback: Task 5` where they will go.

- [ ] **Step 4: Run** `.venv/bin/pytest tests/cpu/test_rollout_store.py -v` → PASS.

- [ ] **Step 5: Commit** — `git commit -am "RolloutStore: re-tokenise turn completions and align them to the decode rows under the EOS rule"`.

---

### Task 4: `arrays(rid)` — per-token arrays, and boundary residuals for old cells

**Files:**
- Modify: `src/healthy_rl/dashboard/rollout_store.py`
- Test: `tests/cpu/test_rollout_store.py`

**Interfaces:**
- Produces: `arrays_from_npz(z, *, turn: int, capture_layers: list[int], probe_layer: int, vectors=None) -> tuple[dict[str, np.ndarray], list[str]]` — returns `(arrays, problems)`; `RolloutStore.arrays(rid)` — the arrays dict for a record: `proj` (T×L×E), `norm` (T×L), `proj_prefill` (L×E), `norm_prefill` (L), and, when only residuals exist and `vectors` is loaded, `proj_end` (L×E) / `norm_end` (L) with NaN at non-probe layers; `res_start_L{p}`/`res_end_L{p}` passed through. `problems` non-empty ⇒ the record is marked `misaligned` with the message (layer mismatch between npz and row).

- [ ] **Step 1: Write the failing tests** (append):

```python
def test_arrays_for_token_cell(tmp_path):
    st = _store(tmp_path)
    r = st.record("fake-model/appr6/lcbhard_0/s0/t1")
    a = st.arrays(r["record_id"])
    assert a["proj"].shape == (4, 2, 3) and a["proj"].dtype == np.float32 and a["norm"].shape == (4, 2)
    assert a["proj_prefill"].shape == (2, 3) and a["norm_prefill"].shape == (2,)
    assert a["res_start_L20"].shape == (8,) and "proj_end" not in a
    z = np.load(tmp_path / "rollouts" / "fake-model" / "appr6" / "residuals" / "lcbhard_0_s0.npz")
    assert np.allclose(a["proj"][:, 1, :], z["t1_proj_L20"][1:].astype(np.float32))
    assert np.allclose(a["proj_prefill"][1], z["t1_proj_L20"][0].astype(np.float32))
    # readouts flow through stats unchanged
    from healthy_rl.dashboard import stats
    v = stats.turn_readout(proj=a["proj"], norm=a["norm"], proj_prefill=a["proj_prefill"], norm_prefill=a["norm_prefill"],
                           token_kind=r["token_kind"], layer_index=1, readout="think_end")
    assert v is not None and v.shape == (3,)


def test_arrays_for_old_cell_project_residuals(tmp_path):
    from healthy_rl.rollouts import Vectors
    E, L, d = 3, 2, 8
    dirs = np.zeros((E, L, d), np.float32); dirs[:, 1, :3] = np.eye(3)     # probe layer 20 = index 1
    vec = Vectors(directions=dirs, emotions=list(EMOTIONS), capture_layers=[10, 20], probe_layer=20,
                  mean_residual_norm={10: 1.0, 20: 1.0}, path=Path("fake"))
    make_cell(tmp_path / "rollouts", "fake-model", "d6", rows=ROWS[:1], token_arrays=False)
    st = RolloutStore.open([tmp_path / "rollouts"], tokenizer_loader=lambda m: WhitespaceTokenizer(),
                           vectors_loader=lambda m: vec, eval_loader=FakeEvalSamples({}))
    r = st.record("fake-model/d6/lcbhard_0/s0/t0")
    a = st.arrays(r["record_id"])
    assert a["proj"].shape == (0, 2, 3) and a["norm"].shape == (0, 2)
    z = np.load(tmp_path / "rollouts" / "fake-model" / "d6" / "residuals" / "lcbhard_0_s0.npz")
    h = z["t0_res_start_L20"].astype(np.float64)
    assert np.allclose(a["proj_prefill"][1], h[:3]) and np.isclose(a["norm_prefill"][1], np.linalg.norm(h))
    assert np.isnan(a["proj_prefill"][0]).all() and np.isnan(a["norm_prefill"][0])
    he = z["t0_res_end_L20"].astype(np.float64)
    assert np.allclose(a["proj_end"][1], he[:3]) and np.isclose(a["norm_end"][1], np.linalg.norm(he))
    from healthy_rl.dashboard import stats
    s = stats.turn_readout(proj=a["proj"], norm=a["norm"], proj_prefill=a["proj_prefill"], norm_prefill=a["norm_prefill"],
                           token_kind=[], layer_index=1, readout="start")
    e = stats.turn_readout(proj=a["proj"], norm=a["norm"], proj_prefill=a["proj_prefill"], norm_prefill=a["norm_prefill"],
                           token_kind=[], layer_index=1, readout="end", proj_end=a["proj_end"], norm_end=a["norm_end"])
    assert np.allclose(s, h[:3] / np.linalg.norm(h)) and np.allclose(e, he[:3] / np.linalg.norm(he))
    assert st.session["models"]["fake-model"]["vectors"] == "ok"


def test_arrays_for_old_cell_without_vectors(tmp_path):
    st = _store(tmp_path)          # vectors_loader -> None
    r = st.record("fake-model/d6/lcbhard_0/s0/t0")
    a = st.arrays(r["record_id"])
    assert a["proj"].shape == (0, 2, 3) and np.isnan(a["proj_prefill"]).all() and np.isnan(a["norm_prefill"]).all()
    assert "proj_end" not in a
    assert any("vectors" in w for w in st.record(r["record_id"])["warnings"])


def test_arrays_zero_token_turn_and_missing_npz(tmp_path):
    st = _store(tmp_path)
    a = st.arrays("fake-model/appr6/lcbhard_1/s0/t0")
    assert a["proj"].shape == (0, 2, 3) and np.isnan(a["norm_prefill"]).all()
    import os
    os.remove(tmp_path / "rollouts" / "fake-model" / "appr6" / "residuals" / "lcbhard_1_s0.npz")
    st2 = _store(tmp_path)
    r = st2.record("fake-model/appr6/lcbhard_1/s0/t1")
    a = st2.arrays(r["record_id"])
    assert a["proj"].shape[0] == 0 and st2.record(r["record_id"])["misaligned"] is True and "npz" in st2.record(r["record_id"])["error"]


def test_arrays_layer_mismatch_marks_misaligned(tmp_path):
    cell = make_cell(tmp_path / "rollouts", "fake-model", "appr6", rows=ROWS[:1], capture_layers=(10, 20, 30))
    f = cell / "rollouts.shard0of2.jsonl"
    row = json.loads(f.read_text()); row["capture_layers"] = [10, 20]; f.write_text(json.dumps(row) + "\n")   # row lies about its layers
    st = RolloutStore.open([tmp_path / "rollouts"], tokenizer_loader=lambda m: WhitespaceTokenizer(),
                           vectors_loader=lambda m: None, eval_loader=FakeEvalSamples({}))
    a = st.arrays("fake-model/appr6/lcbhard_0/s0/t0")
    assert a["proj"].shape == (0, 2, 3)      # nothing usable is served under the wrong layer list
    r = st.record("fake-model/appr6/lcbhard_0/s0/t0")
    assert r["misaligned"] is True and "L30" in r["error"]
```

- [ ] **Step 2: Run** → FAIL (`arrays` NotImplementedError).

- [ ] **Step 3: Implement.** In `rollout_store.py`:

```python
def _project_residual(h: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, float]:
    """``(proj (E,), norm)`` of one residual on the probe-layer directions; NaN when non-finite."""
    h = np.asarray(h, dtype=np.float64)
    if not np.isfinite(h).all():
        return np.full(directions.shape[0], np.nan), np.nan
    n = float(np.linalg.norm(h))
    return directions @ h, n


def arrays_from_npz(z, *, turn: int, capture_layers: list[int], probe_layer: int, n_emotions: int, vectors=None
                    ) -> tuple[dict[str, np.ndarray], list[str]]:
    """Dashboard-shaped arrays for one turn of a rollout npz (spec §3.2, §3.4)."""
    L, files = len(capture_layers), set(z.files)
    problems: list[str] = []
    E = int(n_emotions)
    per_layer: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = []
    for layer in capture_layers:
        k = f"t{turn}_proj_L{layer}"
        if k in files:
            proj = np.asarray(z[k], dtype=np.float32); norm = np.asarray(z[f"t{turn}_norm_L{layer}"], dtype=np.float32)
            kind = np.asarray(z[f"t{turn}_kind_L{layer}"]).reshape(-1)
            if proj.shape[1] != E:
                problems.append(f"{k}: proj has {proj.shape[1]} emotions, record lists {E}")
            per_layer.append((proj, norm, kind))
        else:
            per_layer.append(None)
    extra = sorted(k for k in files if k.startswith(f"t{turn}_proj_L") and int(k.rsplit("L", 1)[1]) not in capture_layers)
    if extra:
        problems.append("npz has layers the record does not list: " + ", ".join("L" + k.rsplit("L", 1)[1] for k in extra))
    have = [p for p in per_layer if p is not None]
    out: dict[str, np.ndarray] = {}
    if have and not problems and len(have) == L:
        T = int((have[0][2] == 0).sum())
        if any(int((p[2] == 0).sum()) != T for p in have):
            problems.append("decode-row count differs across layers")
    if have and not problems and len(have) == L:
        out["proj"] = np.stack([p[0][p[2] == 0] for p in have], axis=1)                # T x L x E
        out["norm"] = np.stack([p[1][p[2] == 0] for p in have], axis=1)                # T x L
        pre = [np.where(p[2] == 1)[0] for p in have]
        out["proj_prefill"] = np.stack([p[0][i[-1]] if len(i) else np.full(E, np.nan) for p, i in zip(have, pre)])
        out["norm_prefill"] = np.array([p[1][i[-1]] if len(i) else np.nan for p, i in zip(have, pre)], np.float32)
    else:
        if have and len(have) != L and not problems:
            missing = [f"L{l}" for l, p in zip(capture_layers, per_layer) if p is None]
            problems.append("npz lacks per-token arrays at " + ", ".join(missing))
        out["proj"] = np.zeros((0, L, E), np.float32); out["norm"] = np.zeros((0, L), np.float32)
        out["proj_prefill"] = np.full((L, E), np.nan, np.float32); out["norm_prefill"] = np.full(L, np.nan, np.float32)
        rs, re_ = f"t{turn}_res_start_L{probe_layer}", f"t{turn}_res_end_L{probe_layer}"
        if vectors is not None and probe_layer in capture_layers and (rs in files or re_ in files):
            li = capture_layers.index(probe_layer)
            D = np.asarray(vectors.probe_directions(), dtype=np.float64)
            if rs in files:
                p, n = _project_residual(z[rs], D); out["proj_prefill"][li] = p; out["norm_prefill"][li] = n
            if re_ in files:
                out["proj_end"] = np.full((L, E), np.nan, np.float32); out["norm_end"] = np.full(L, np.nan, np.float32)
                p, n = _project_residual(z[re_], D); out["proj_end"][li] = p; out["norm_end"][li] = n
    for k in (f"t{turn}_res_start_L{probe_layer}", f"t{turn}_res_end_L{probe_layer}"):
        if k in files:
            out[k.split("_", 1)[1]] = np.asarray(z[k])          # "res_start_L20"
    return out, problems
```

In `RolloutStore`:

```python
    def arrays(self, record_id: str) -> dict[str, np.ndarray]:
        rec = self.record(record_id)
        L, E = len(rec["capture_layers"]), len(rec["emotions"])
        empty = {"proj": np.zeros((0, L, E), np.float32), "norm": np.zeros((0, L), np.float32),
                 "proj_prefill": np.full((L, E), np.nan, np.float32), "norm_prefill": np.full(L, np.nan, np.float32)}
        path = self._npz_path(rec)
        if path is None or not path.is_file():
            self._mark(rec, f"npz missing: {path}")
            return empty
        vec = self._vectors_for(rec["model"])
        try:
            with np.load(path) as z:
                out, problems = arrays_from_npz(z, turn=rec["turn_index"], capture_layers=rec["capture_layers"],
                                                probe_layer=rec["probe_layer"], n_emotions=E, vectors=vec)
        except (OSError, ValueError) as exc:
            self._mark(rec, f"npz unreadable: {exc}")
            return empty
        if problems:
            self._mark(rec, "; ".join(problems))
            return empty
        if out["proj"].shape[0] == 0 and rec["n_generated"] > 0 and vec is None and not rec["has_token_arrays"]:
            if "vectors missing: start/end readouts unavailable for this cell" not in rec["warnings"]:
                rec["warnings"].append("vectors missing: start/end readouts unavailable for this cell")
        return out

    def _mark(self, rec: dict, error: str) -> None:
        """Flag a full record misaligned; the page hides the strip and readouts go None."""
        rec["misaligned"] = True
        rec["error"] = error if not rec.get("error") else rec["error"] + "; " + error
        self._session = None
```

- [ ] **Step 4: Run** `.venv/bin/pytest tests/cpu/test_rollout_store.py -v` → PASS.

- [ ] **Step 5: Commit** — `git commit -am "RolloutStore: serve per-token arrays from the rollout npz, and boundary-residual readouts for older cells"`.

---

### Task 5: `.eval` messages — `messages_in`, `feedback`, per-turn `passed`

**Files:**
- Modify: `src/healthy_rl/dashboard/rollout_store.py`
- Test: `tests/cpu/test_rollout_store.py`

**Interfaces:**
- Produces: `sample_messages(samples: list[dict], task_id: str, completions: list[str]) -> list[dict] | None` — the messages of the sample whose assistant messages match `completions` (first non-empty completion equal, then all present ones); `None` if none matches. `RolloutStore._eval_samples_for(rec) -> list[dict]` — all samples from every `.eval` under `inspect-logs/shard{a}of{b}` (from `rec["shard"]` = `"a/b"`; if unparsable, every `.eval` under `inspect-logs`), newest file first, cached per file. `record()` fills `messages_in` (messages before assistant turn t), `feedback` (the user message right after it, else `None`), and per-turn `passed`: `False` when `feedback` is present, the rollout's `passed` on the last turn, else `None`. `conversations()` keeps the rollout-level `passed`.

- [ ] **Step 1: Write the failing tests** (append):

```python
from healthy_rl.dashboard.rollout_store import sample_messages

SAMPLES = [
    {"id": "lcbhard_0", "epoch": 1, "messages": [
        {"role": "user", "content": "PROBLEM"}, {"role": "assistant", "content": "a b c"},
        {"role": "user", "content": "Your previous attempt failed the tests. FAIL1"},
        {"role": "assistant", "content": "[THINK]x y[/THINK] z"},
        {"role": "user", "content": "Your previous attempt failed the tests. FAIL2"}]},
    {"id": "lcbhard_0", "epoch": 1, "messages": [
        {"role": "user", "content": "PROBLEM"}, {"role": "assistant", "content": "p q"},
        {"role": "user", "content": "FAILP"}, {"role": "assistant", "content": "r s t u"}]},
]


def test_sample_messages_matches_by_completion():
    m = sample_messages(SAMPLES, "lcbhard_0", ["p q", "r s t u"])
    assert m[1]["content"] == "p q"
    assert sample_messages(SAMPLES, "lcbhard_0", ["a b c", "[THINK]x y[/THINK] z"])[3]["content"].endswith(" z")
    assert sample_messages(SAMPLES, "lcbhard_0", ["nope"]) is None
    assert sample_messages(SAMPLES, "lcbhard_7", ["a b c"]) is None
    # a rollout whose first turn generated nothing matches on its first non-empty completion
    assert sample_messages(SAMPLES, "lcbhard_0", ["", "p q"]) is not None


def test_record_messages_in_and_feedback(tmp_path):
    make_cell(tmp_path / "rollouts", "fake-model", "appr6", rows=ROWS)
    evals = FakeEvalSamples({str(tmp_path / "rollouts" / "fake-model" / "appr6" / "inspect-logs" / "shard0of2" / "x.eval"): SAMPLES})
    st = RolloutStore.open([tmp_path / "rollouts"], tokenizer_loader=lambda m: WhitespaceTokenizer(),
                           vectors_loader=lambda m: None, eval_loader=evals)
    r0 = st.record("fake-model/appr6/lcbhard_0/s0/t0"); r1 = st.record("fake-model/appr6/lcbhard_0/s0/t1")
    assert r0["messages_in"] == [{"role": "user", "content": "PROBLEM"}]
    assert r0["feedback"].endswith("FAIL1") and r0["passed"] is False
    assert [m["role"] for m in r1["messages_in"]] == ["user", "assistant", "user"]
    assert r1["feedback"].endswith("FAIL2") and r1["passed"] is False           # last turn, rollout failed
    s1 = st.record("fake-model/appr6/lcbhard_0/s1/t1")
    assert s1["feedback"] is None and s1["passed"] is True                       # last turn, rollout passed
    assert st.record("fake-model/appr6/lcbhard_0/s1/t0")["passed"] is False
    assert evals.calls == 1                                                      # one parse per file
    # a rollout the eval does not know: empty messages, a warning, not misaligned
    r = st.record("fake-model/appr6/lcbhard_1/s0/t1")
    assert r["messages_in"] == [] and r["feedback"] is None and any(".eval" in w for w in r["warnings"]) and r["misaligned"] is False


def test_record_without_eval_file(tmp_path):
    st = _store(tmp_path)              # FakeEvalSamples({}) raises FileNotFoundError
    r = st.record("fake-model/appr6/lcbhard_0/s0/t0")
    assert r["messages_in"] == [] and any(".eval" in w for w in r["warnings"])
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement.**

```python
def sample_messages(samples: list[dict], task_id: str, completions: list[str]) -> list[dict] | None:
    """The messages of the sample that produced these completions.

    Several samples share a task id at Inspect epoch 1 (resumed shards restart the
    numbering), so the id alone is ambiguous; the completion text is not -- the
    ``.eval``'s assistant messages equal ``turn_completion`` verbatim.
    """
    want = [c for c in completions if c]
    for s in samples:
        if str(s.get("id")) != str(task_id):
            continue
        got = [m["content"] for m in s.get("messages", []) if m.get("role") == "assistant"]
        got_ne = [g for g in got if g]
        if want and got_ne[:len(want)] == want:
            return list(s["messages"])
    return None
```
In `RolloutStore`:

```python
    def _eval_files(self, rec: dict) -> list[Path]:
        cell = self._cell_of(rec)
        shard = str(rec.get("shard") or "")
        sub = None
        if "/" in shard:
            a, b = shard.split("/", 1)
            if a.isdigit() and b.isdigit():
                sub = cell.path / "inspect-logs" / f"shard{a}of{b}"
        base = sub if sub is not None and sub.is_dir() else cell.path / "inspect-logs"
        return sorted(base.rglob("*.eval"), reverse=True) if base.is_dir() else []

    def _eval_samples(self, path: Path) -> list[dict]:
        cache = getattr(self, "_evals", None)
        if cache is None:
            cache = self._evals = {}
        if path not in cache:
            try:
                cache[path] = self._eval_loader(path)
            except Exception as exc:          # unreadable log: no messages, and say so
                cache[path] = exc
        v = cache[path]
        if isinstance(v, Exception):
            raise v
        return v

    def _messages_for(self, rec: dict) -> tuple[list[dict] | None, str | None]:
        """(messages, warning) -- the sample's whole message list, or None + why."""
        files = self._eval_files(rec)
        if not files:
            return None, "no .eval log under inspect-logs for this shard"
        comps = self._completions_of(rec)
        errors = 0
        for f in files:
            try:
                samples = self._eval_samples(f)
            except Exception:
                errors += 1
                continue
            m = sample_messages(samples, rec["task_id"], comps)
            if m is not None:
                return m, None
        if errors == len(files):
            return None, "the .eval log(s) could not be read"
        return None, "no .eval sample matches this rollout's completions"

    def _completions_of(self, rec: dict) -> list[str]:
        cid = rec["conversation_id"]
        return [self._light[rid]["text"] for rid in self._order if self._light[rid]["conversation_id"] == cid]
```
and in `record()`, before `rec["warnings"] = warnings`:

```python
        msgs, why = self._messages_for(rec)
        t = rec["turn_index"]
        rec["messages_in"], rec["feedback"] = [], None
        if msgs is None:
            warnings.append(f"transcript context unavailable: {why}")
        else:
            idx = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
            if len(idx) != rec["n_turns_total"]:
                warnings.append(f".eval has {len(idx)} assistant messages, record has {rec['n_turns_total']} turns")
            if t < len(idx):
                i = idx[t]
                rec["messages_in"] = [dict(m) for m in msgs[:i]]
                nxt = msgs[i + 1] if i + 1 < len(msgs) else None
                rec["feedback"] = nxt["content"] if nxt is not None and nxt.get("role") == "user" else None
        last = t == rec["n_turns_total"] - 1
        rec["passed"] = False if rec["feedback"] else (rec["passed"] if last else None)
```
`conversations()` reads `passed` from the *light* record (`self._light[cid-first]`), so change its `passed` line to `"passed": self._light[r["record_id"]]["passed"]`.

- [ ] **Step 4: Run** `.venv/bin/pytest tests/cpu/test_rollout_store.py -v` → PASS.

- [ ] **Step 5: Commit** — `git commit -am "RolloutStore: reconstruct each turn's context and feedback from the cell's .eval logs"`.

---

### Task 6: `app.py` — rollouts mode, per-record direction metadata, routes

**Files:**
- Modify: `src/healthy_rl/dashboard/app.py`
- Test: `tests/cpu/test_dashboard_app.py`

**Interfaces:**
- Produces: `AppState.mode: str = "live"` (set `"replay"`/`"rollouts"` by `__main__`), `AppState.vectors: Vectors | None`; `VectorsMeta(emotions, capture_layers, probe_layer)` with `.layer_index(layer)`; inside `create_app`: `_meta(rec) -> Vectors | VectorsMeta`, `_layer(layer, rec) -> int`; `_readouts_for(rec, arrays, meta)` also passes `proj_end`/`norm_end` from `arrays` when present. Routes per spec §4 (aggregate is Task 7).

- [ ] **Step 1: Write the failing tests** (append to `tests/cpu/test_dashboard_app.py`):

```python
from pathlib import Path
from rollout_cell import EMOTIONS, FakeEvalSamples, WhitespaceTokenizer, make_cell
from healthy_rl.dashboard.rollout_store import RolloutStore

RROWS = [
    {"task_id": "lcbhard_0", "sample": 0, "completions": ["a b c", "[THINK]x y[/THINK] z"], "passed": False},
    {"task_id": "lcbhard_0", "sample": 1, "completions": ["p q", "r s t u"], "passed": True, "bench_split": "original"},
]


def _rollout_client(tmp_path):
    make_cell(tmp_path / "r", "m-a", "appr6", rows=RROWS[:1])
    make_cell(tmp_path / "r", "m-a", "d6", rows=RROWS[:1], token_arrays=False)
    make_cell(tmp_path / "r", "m-b", "appr6", rows=RROWS, capture_layers=(5, 15), probe_layer=15)
    store = RolloutStore.open([tmp_path / "r"], tokenizer_loader=lambda m: WhitespaceTokenizer(),
                              vectors_loader=lambda m: None, eval_loader=FakeEvalSamples({}))
    st = AppState(engine=None, sandbox=None, store=store, vectors=None, cfg={}, read_only=True, mode="rollouts")
    return TestClient(create_app(st))


def test_rollouts_session_and_conversations(tmp_path):
    c = _rollout_client(tmp_path)
    s = c.get("/api/session").json()
    assert s["mode"] == "rollouts" and s["read_only"] is True
    assert set(s["session"]["models"]) == {"m-a", "m-b"} and s["session"]["models"]["m-b"]["probe_layer"] == 15
    assert len(s["session"]["cells"]) == 3 and s["emotions"] == list(EMOTIONS)
    convs = c.get("/api/conversations").json()["conversations"]
    assert len(convs) == 4 and all(x["source"] == "rollout" for x in convs)
    assert len(c.get("/api/conversations", params={"model": "m-b"}).json()["conversations"]) == 2
    assert len(c.get("/api/conversations", params={"model": "m-a", "version": "d6"}).json()["conversations"]) == 1


def test_rollouts_conversation_readouts_at_own_probe_layer(tmp_path):
    c = _rollout_client(tmp_path)
    conv = c.get("/api/conversations/m-b/appr6/lcbhard_0/s0").json()
    t = conv["turns"][1]
    assert t["probe_layer"] == 15 and t["has_token_arrays"] is True and t["misaligned"] is False
    assert isinstance(t["readouts"]["desperate"]["start"], float) and isinstance(t["readouts"]["desperate"]["think_end"], float)
    assert t["tokens"][-1] == "<eos>" and t["emotion_order_mismatch"] is False
    old = c.get("/api/conversations/m-a/d6/lcbhard_0/s0").json()["turns"][0]
    assert old["has_token_arrays"] is False and old["readouts"]["desperate"]["start"] is None   # no vectors loaded
    assert any("vectors" in w for w in old["warnings"])
    assert c.get("/api/conversations/nope").status_code == 404


def test_rollouts_tokens_route_validates_record_layers(tmp_path):
    c = _rollout_client(tmp_path)
    rid = "m-b/appr6/lcbhard_0/s0/t1"
    p = c.get(f"/api/records/{rid}/tokens").json()
    assert p["layer"] == 15 and len(p["tokens"]) == 4 and len(p["cosine"]) == 4 and p["markers"]["think_end"] == 1
    assert c.get(f"/api/records/{rid}/tokens", params={"layer": 5}).status_code == 200
    assert c.get(f"/api/records/{rid}/tokens", params={"layer": 20}).status_code == 400
    assert c.get("/api/records/nope/tokens").status_code == 404


def test_rollouts_mode_refuses_generation(tmp_path):
    c = _rollout_client(tmp_path)
    assert c.post("/api/chat/new/send", json={"text": "x"}).status_code == 409
    assert c.post("/api/task/start", json={"split": "original", "task_id": "lcbhard_0"}).status_code == 409
    assert c.get("/api/problems").status_code == 409
```
Also add one regression line to `test_index_and_session`: `assert s["mode"] == "live"`.

- [ ] **Step 2: Run** `.venv/bin/pytest tests/cpu/test_dashboard_app.py -v -k rollouts` → FAIL (`unexpected keyword 'mode'`).

- [ ] **Step 3: Implement.** In `app.py`:

```python
@dataclass(frozen=True)
class VectorsMeta:
    """Enough of a ``Vectors`` to read stored projections: order, layers, probe layer."""
    emotions: tuple[str, ...]
    capture_layers: tuple[int, ...]
    probe_layer: int | None

    def layer_index(self, layer: int) -> int:
        return self.capture_layers.index(layer)
```
`AppState`: `vectors: Vectors | None`, add `mode: str = "live"`.

`_readouts_for(rec, arrays, vectors)` → keep the name of the third parameter as `meta`; pass `proj_end=arrays.get("proj_end"), norm_end=arrays.get("norm_end")` to `stats.turn_readout`; `li = meta.layer_index(meta.probe_layer)` guarded: if `meta.probe_layer is None or meta.probe_layer not in meta.capture_layers` return all-None.

In `create_app`:

```python
    V = st.vectors            # None in rollouts mode
    ROLL = st.mode == "rollouts"

    def _meta(rec: dict | None = None):
        if not ROLL:
            return V
        models = st.store.session.get("models", {})
        m = models.get(rec["model"], {}) if rec else {}
        emotions = m.get("emotions") or (rec or {}).get("emotions") or []
        return VectorsMeta(tuple(emotions), tuple(int(l) for l in (rec or {}).get("capture_layers", m.get("capture_layers", []))),
                           (rec or {}).get("probe_layer", m.get("probe_layer")))

    def _layer(layer: int | None, rec: dict | None = None) -> int:
        meta = _meta(rec)
        layer = meta.probe_layer if layer is None else layer
        if layer not in meta.capture_layers:
            raise HTTPException(400, f"layer must be one of {list(meta.capture_layers)}" + (f" for {rec['model']}" if rec and ROLL else ""))
        return layer

    def _first_meta():
        """Session-level emotions/layers for the page's boot: the first model's in rollouts mode."""
        if not ROLL:
            return V
        models = st.store.session.get("models", {})
        first = next(iter(models.values()), {})
        return VectorsMeta(tuple(first.get("emotions", [])), tuple(first.get("capture_layers", [])), first.get("probe_layer"))
```
Replace every `V.emotions` / `V.capture_layers` / `V.probe_layer` / `V.layer_index` in the routes: `session` uses `_first_meta()`; `conversation` uses `_meta(r)` per turn (`emotion` validation against `_first_meta().emotions` ∪ each turn's — simplest: validate `emotion` against the union of `models[*].emotions` in rollouts mode, else `V.emotions`); `tokens` uses `_meta(rec)`; `_readouts_for(r, arrays, _meta(r))`; `_emotion_order_mismatch(r, _meta(r))`. Add `"mode": st.mode` to `/api/session`. In `conversation`, records for a conversation come from `st.store.record(r["record_id"])` when `ROLL` (full records) — write:

```python
        recs = [r for r in st.store.records() if r["conversation_id"] == cid]
        if ROLL:
            recs = [st.store.record(r["record_id"]) for r in recs]
```
`/api/conversations`: add `model: list[str] = Query(default=[])`, `version: list[str] = Query(default=[])`; filter `convs` by membership when non-empty. `_writable()` also raises when `ROLL`; `_problems` already 409s on `sandbox is None`. `_rehydrate_chat` is only reached after `_writable()`; confirm `chat_send` and `task_start`/`task_continue` call `_writable()` before anything else (they do in the current file; keep it).

- [ ] **Step 4: Run** `.venv/bin/pytest tests/cpu/test_dashboard_app.py tests/cpu/test_dashboard_main.py tests/cpu/test_dashboard_stage.py -v` → all PASS (old tests unchanged).

- [ ] **Step 5: Commit** — `git commit -am "Dashboard app: rollouts mode with per-record direction metadata and read-only routes"`.

---

### Task 7: Grouped `/api/aggregate` (+ page consumers)

**Files:**
- Modify: `src/healthy_rl/dashboard/app.py:437-512`
- Modify: `src/healthy_rl/dashboard/static/index.html` (`renderTrajectory`, `renderAggregate`, `aggDraw`: read `groups[0]`)
- Test: `tests/cpu/test_dashboard_app.py`

**Interfaces:**
- Produces: `GET /api/aggregate?source=task|chat|rollout&split=&position=&stat=&segment=&include_cap=&layer=probe|<int>&model=(repeat)&version=(repeat)` → `{"groups": [...], "emotions": [...], "params": {...}}` per spec §4; each group: `model, version, bench_split, mindset, layer, n_conversations, n_records, excluded_cap, skipped, by_turn, delta`. `skipped` = total of `by_turn.skipped`. Helper `_aggregate_group(recs, *, meta, layer, position, stat, segment, drop_cap) -> dict`.

- [ ] **Step 1: Write the failing tests** (append):

```python
def test_aggregate_live_is_a_single_group(client):
    with client.stream("POST", "/api/chat/new/send", json={"text": "hello"}) as r:
        r.read()
    a = client.get("/api/aggregate", params={"source": "chat"}).json()
    assert len(a["groups"]) == 1 and a["groups"][0]["n_conversations"] == 1
    g = a["groups"][0]
    assert g["model"] == "fake" and g["version"] is None and g["layer"] == 20
    assert "mean" in g["by_turn"] and "mean" in g["delta"] and a["emotions"] == ["desperate", "frustrated", "joyful"]


def test_aggregate_rollout_groups(tmp_path):
    c = _rollout_client(tmp_path)
    a = c.get("/api/aggregate", params={"source": "rollout", "split": "conflicting"}).json()
    keys = {(g["model"], g["version"]) for g in a["groups"]}
    assert keys == {("m-a", "appr6"), ("m-a", "d6"), ("m-b", "appr6")}
    gb = next(g for g in a["groups"] if g["model"] == "m-b")
    assert gb["layer"] == 15 and gb["n_conversations"] == 1 and gb["bench_split"] == "conflicting"
    ga = next(g for g in a["groups"] if (g["model"], g["version"]) == ("m-a", "appr6"))
    assert ga["layer"] == 20 and len(ga["by_turn"]["mean"]) == 2 and ga["skipped"] == 0
    old = next(g for g in a["groups"] if g["version"] == "d6")
    assert old["skipped"] == 2                                     # no vectors: both turns None, counted
    # filters
    a = c.get("/api/aggregate", params={"source": "rollout", "split": "conflicting", "model": ["m-b"]}).json()
    assert [(g["model"], g["version"]) for g in a["groups"]] == [("m-b", "appr6")]
    a = c.get("/api/aggregate", params={"source": "rollout", "split": "conflicting", "model": ["m-a"], "version": ["d6"]}).json()
    assert [(g["model"], g["version"]) for g in a["groups"]] == [("m-a", "d6")]
    # layer must exist for every selected model
    assert c.get("/api/aggregate", params={"source": "rollout", "split": "conflicting", "layer": 20}).status_code == 400
    assert "m-b" in c.get("/api/aggregate", params={"source": "rollout", "split": "conflicting", "layer": 20}).json()["detail"]
    assert c.get("/api/aggregate", params={"source": "rollout", "split": "conflicting", "model": ["m-a"], "layer": 10}).status_code == 200
    # splits are never pooled
    assert c.get("/api/aggregate", params={"source": "rollout"}).status_code == 400
    assert c.get("/api/aggregate", params={"source": "rollout", "split": "original"}).json()["groups"][0]["model"] == "m-b"
```
Existing tests that read `a["by_turn"]` / `a["delta"]` / `a["n_conversations"]` at the top level (`grep -n "aggregate" tests/cpu/test_dashboard_app.py`) are updated to read them from `["groups"][0]` in this same commit.

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement.** Replace the aggregate route body:

```python
    def _aggregate_group(recs: list[dict], *, meta, layer: int, position: str, stat: str, segment: str, drop_cap: bool) -> dict:
        li = meta.layer_index(layer)
        by_conv: dict[str, list] = {}
        for r in recs:
            by_conv.setdefault(r["conversation_id"], []).append(r)
        seqs, excluded = [], 0
        for rows in by_conv.values():
            rows.sort(key=lambda r: r["turn_index"])
            seq = []
            for r in rows:
                if r.get("n_generated", 0) == 0:
                    continue
                if drop_cap and r.get("at_cap"):
                    excluded += 1; seq.append(None); continue
                if _emotion_order_mismatch(r, meta):
                    seq.append(None); continue
                a = st.store.arrays(r["record_id"])
                if stat == "token":
                    v = _readout_or_none(stats.turn_readout, proj=a["proj"], norm=a["norm"], proj_prefill=a["proj_prefill"],
                                         norm_prefill=a["norm_prefill"], token_kind=r.get("token_kind", []), layer_index=li,
                                         readout=position, proj_end=a.get("proj_end"), norm_end=a.get("norm_end"))
                else:
                    v = _readout_or_none(stats.turn_mean, proj=a["proj"], norm=a["norm"], token_kind=r.get("token_kind", []),
                                         layer_index=li, segment=segment)
                seq.append(v)
            if seq:
                seqs.append(seq)
        E = len(meta.emotions)
        bt = stats.by_turn_index(seqs, n_emotions=E)
        return {"emotions": list(meta.emotions), "layer": layer, "n_conversations": len(seqs), "n_records": len(recs),
                "excluded_cap": excluded, "skipped": int(np.asarray(bt["skipped"]).sum()),
                "by_turn": bt, "delta": stats.paired_delta(seqs, n_emotions=E)}

    def _widen(group: dict, emotions: list[str]) -> dict:
        """Re-index a group's per-emotion columns onto the union order; NaN where the model lacks one."""
        idx = [group["emotions"].index(e) if e in group["emotions"] else None for e in emotions]
        def cols(a):
            a = np.asarray(a, dtype=np.float64)
            out = np.full(a.shape[:-1] + (len(emotions),), np.nan)
            for j, i in enumerate(idx):
                if i is not None:
                    out[..., j] = a[..., i]
            return out
        bt, de = group["by_turn"], group["delta"]
        group["by_turn"] = {**bt, "mean": cols(bt["mean"]), "sem": cols(bt["sem"])}
        group["delta"] = {**de, "mean": cols(de["mean"]), "sem": cols(de["sem"]), "p": cols(de["p"])}
        return group

    @app.get("/api/aggregate")
    def aggregate(source: str = "task", split: str | None = None, position: str = "start", stat: str = "token",
                  segment: str = "all", include_cap: bool = False, layer: str | None = None,
                  model: list[str] = Query(default=[]), version: list[str] = Query(default=[])):
        """(keep the existing docstring, then:) In rollouts mode groups are (model, version)
        cells; ``layer`` is ``probe`` (each group at its model's probe layer) or an integer
        that every selected model must capture."""
        if source not in ("task", "chat", "rollout"):
            raise HTTPException(400, "source must be 'task', 'chat' or 'rollout'")
        if position not in stats.READOUTS or stat not in ("token", "mean") or segment not in stats.SEGMENTS:
            raise HTTPException(400, "bad position/stat/segment")
        if split is not None and source != "rollout":
            _split(split)
        want_layer: int | None = None
        if layer not in (None, "", "probe"):
            try:
                want_layer = int(layer)
            except ValueError:
                raise HTTPException(400, "layer must be 'probe' or an integer")
        recs = [r for r in st.store.records() if r.get("source") == source]
        if source == "rollout":
            if model:
                recs = [r for r in recs if r["model"] in model]
            if version:
                recs = [r for r in recs if r["version"] in version]
            recs = [st.store.record(r["record_id"]) for r in recs]
        if source in ("task", "rollout"):
            splits = {r.get("bench_split") for r in recs}
            if split is None and len(splits) > 1:
                raise HTTPException(400, "choose a split; conflicting and original cannot be pooled")
            if split is not None:
                recs = [r for r in recs if r.get("bench_split") == split]
        drop_cap = stat == "token" and position == "end" and not include_cap
        groups_in: dict[tuple, list[dict]] = {}
        for r in recs:
            key = (r["model"], r["version"]) if source == "rollout" else (st.store.session.get("model"), None)
            groups_in.setdefault(key, []).append(r)
        if not groups_in and source != "rollout":
            groups_in[(st.store.session.get("model"), None)] = []
        groups = []
        for (m, v), rs in sorted(groups_in.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
            meta = _meta(rs[0]) if rs else _first_meta()
            lyr = meta.probe_layer if want_layer is None else want_layer
            if lyr not in meta.capture_layers:
                raise HTTPException(400, f"layer L{lyr} is not a capture layer of {m} (has {list(meta.capture_layers)})")
            g = _aggregate_group(rs, meta=meta, layer=lyr, position=position, stat=stat, segment=segment, drop_cap=drop_cap)
            g.update(model=m, version=v, bench_split=(rs[0].get("bench_split") if rs else split),
                     mindset=(rs[0].get("mindset") if rs else None))
            groups.append(g)
        emotions: list[str] = []
        for g in groups:
            for e in g["emotions"]:
                if e not in emotions:
                    emotions.append(e)
        groups = [_widen(g, emotions) for g in groups]
        return _clean({"groups": groups, "emotions": emotions,
                       "params": {"source": source, "split": split, "position": position, "stat": stat, "segment": segment,
                                  "include_cap": include_cap, "layer": "probe" if want_layer is None else want_layer,
                                  "model": model, "version": version}})
```
`_clean` must serialise numpy — it already does for the old shape (check `_json_default`/`_clean` handle nested dicts and arrays; they do).

Page (`index.html`): in `renderTrajectory` replace `S.trajSess = data;` with `S.trajSess = (data.groups && data.groups[0]) || null;` and where `S.trajSess` is consumed (`trajDraw`, `renderTrajWarnings`) it already expects the group shape (`by_turn`, `emotions`, `n_conversations`). In `renderAggregate` after `data = await aggregateData(params)`: `const g = (data.groups && data.groups[0]) || {by_turn: {}, delta: {}, emotions: [], n_conversations: 0, n_records: 0, excluded_cap: 0};` and use `g.emotions`, `g.by_turn`, `g.delta`, `g.n_conversations`, `g.n_records`, `g.excluded_cap` in the table/subtitle, and `aggDraw(g)`. (Task 10 replaces this single-group rendering in rollouts mode; live/replay keep it.) Also `aggParams` sends `layer: S.layer` — the API now accepts an int string; fine.

- [ ] **Step 4: Run** `.venv/bin/pytest tests/cpu -q` → all PASS (including `test_dashboard_page.py::test_javascript_parses`).

- [ ] **Step 5: Commit** — `git commit -am "Aggregate by (model, cell) group; live sessions are a group of one"`.

---

### Task 8: `--rollouts` entry point and startup table

**Files:**
- Modify: `src/healthy_rl/dashboard/__main__.py`
- Test: `tests/cpu/test_dashboard_main.py`

**Interfaces:**
- Produces: `build_state(*, fake, replay, session_dir, vectors_dir, cfg, rollouts: list[str] | None = None) -> AppState`; `python -m healthy_rl.dashboard --rollouts P [P ...]`; `startup_report(store) -> str` (the cell/model table printed at start).

- [ ] **Step 1: Write the failing test** (append to `tests/cpu/test_dashboard_main.py`):

```python
def test_rollouts_state_opens_cells_read_only(tmp_path, capsys):
    from rollout_cell import make_cell
    from healthy_rl.dashboard.__main__ import startup_report
    make_cell(tmp_path / "r", "m-a", "appr6", rows=[{"task_id": "lcbhard_0", "sample": 0, "completions": ["a b"], "passed": False}])
    st = build_state(fake=False, replay=None, session_dir=None, vectors_dir=None, cfg={}, rollouts=[str(tmp_path / "r")])
    assert st.mode == "rollouts" and st.read_only and st.vectors is None
    rep = startup_report(st.store)
    assert "m-a" in rep and "appr6" in rep and "tokenizer" in rep
    c = TestClient(create_app(st))
    assert c.get("/api/session").json()["mode"] == "rollouts"
    assert c.post("/api/chat/new/send", json={"text": "x"}).status_code == 409


def test_rollouts_state_with_no_cells_exits(tmp_path):
    import pytest
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit):
        build_state(fake=False, replay=None, session_dir=None, vectors_dir=None, cfg={}, rollouts=[str(tmp_path / "empty")])
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement.** In `__main__.py`: docstring gains `python -m healthy_rl.dashboard --rollouts $ARTIFACT_DIR/rollouts[/<model>[/<version>]] ... --port 8765`. Add:

```python
from healthy_rl.dashboard.rollout_store import RolloutStore


def startup_report(store: RolloutStore) -> str:
    s = store.session
    lines = ["cells:"]
    for c in s["cells"]:
        arms = ("+".join(c["mindset"]) or "-")
        lines.append(f"  {c['model']:32s} {c['version']:12s} {c['bench_split'] or '?':12s} mindset={arms:20s} "
                     f"rollouts={c['n_rollouts']:3d} with_token_arrays={c['n_with_token_arrays']:3d} max_tokens={c['max_tokens']}")
    lines.append("models:")
    for m, meta in s["models"].items():
        lines.append(f"  {m:32s} tokenizer={meta['tokenizer']:8s} vectors={meta['vectors']:8s} probe=L{meta['probe_layer']} "
                     f"layers={meta['capture_layers']}")
    if s["ignored"]:
        lines.append("ignored (no rollouts*.jsonl): " + ", ".join(s["ignored"]))
    return "\n".join(lines)
```
In `build_state` add parameter `rollouts: list[str] | None = None` and, first:

```python
    if rollouts:
        try:
            store = RolloutStore.open(rollouts)
        except FileNotFoundError as exc:
            raise SystemExit(f"{exc}\nnothing to open: give a cell, a model directory, or the rollouts root") from None
        return AppState(engine=None, sandbox=None, store=store, vectors=None, cfg=cfg, read_only=True, mode="rollouts")
```
Set `mode="replay"` on the replay branch. `main()`: `g.add_argument("--rollouts", nargs="+", metavar="PATH")`; pass `rollouts=args.rollouts`; the final print becomes `... ({'rollouts' if args.rollouts else 'replay' if args.replay else 'fake engine'}; records: {state.store.root})` and, when `args.rollouts`, `print(startup_report(state.store), flush=True)` before uvicorn.

Also handle the exit-code contract for `main()` when `--rollouts` finds nothing: `SystemExit` with a message is exit code 1 — the spec says 2. Wrap: in `main()`, `except SystemExit` is clumsy; instead raise `SystemExit(2)` after printing the message to stderr in `build_state`: `print(msg, file=sys.stderr); raise SystemExit(2)`. Test asserts `SystemExit` only.

- [ ] **Step 4: Run** `.venv/bin/pytest tests/cpu/test_dashboard_main.py -v` → PASS.

- [ ] **Step 5: Smoke on real data (login node, quick):** `set -a; . ./.env; set +a; timeout 120 .venv/bin/python -m healthy_rl.dashboard --rollouts $ARTIFACT_DIR/rollouts/Ministral-3-14B-Reasoning-2512/appr6 --port 8799 &` then `curl -s localhost:8799/api/session | head -c 600`, `curl -s "localhost:8799/api/conversations" | head -c 400`, and one conversation: `curl -s "localhost:8799/api/conversations/$(curl -s localhost:8799/api/conversations | .venv/bin/python -c 'import json,sys;print(json.load(sys.stdin)["conversations"][0]["conversation_id"])')" | .venv/bin/python -c 'import json,sys; d=json.load(sys.stdin); t=d["turns"][0]; print(t["misaligned"], t["error"], len(t["tokens"]), t["n_decode"], t["warnings"][:2], t["readouts"]["desperate"])'`; then `kill %1`. Expected: `misaligned False`, tokens == n_decode, finite readouts. If Ministral's tokenizer offsets do not tile the text, `tokenise` is where to fix it — record what you find in the task report.

- [ ] **Step 6: Commit** — `git commit -am "python -m healthy_rl.dashboard --rollouts: open pilot cells read-only with a startup cell table"`.

---

### Task 9: Page — rollouts mode (rail, conversation, settings, per-model layers)

**Files:**
- Modify: `src/healthy_rl/dashboard/static/index.html`
- Test: `tests/cpu/test_dashboard_page.py`

**Interfaces:**
- Consumes: `/api/session` (`mode`, `session.models`, `session.cells`, `session.roots`), `/api/conversations` rollout fields, `/api/conversations/{cid}` turn fields (`has_token_arrays`, `misaligned`, `feedback`, `passed`, `probe_layer`, `capture_layers`, `model`, `version`, `mindset`, `sample`, `bench_split`).
- Produces: `S.rollouts` (bool), `S.models` (session.models), `applyModelLayers(model)`; rail grouping; `renderConversation` rollout branch; Settings model/cell tables; the composer/health/task controls hidden in rollouts mode.

- [ ] **Step 1: Extend the structural test** (append to `tests/cpu/test_dashboard_page.py`):

```python
def test_rollouts_mode_strings_present():
    for s in ["S.rollouts", "applyModelLayers", "no per-token arrays", "session.models", "railFilter"]:
        assert s in PAGE, s
```
Run → FAIL.

- [ ] **Step 2: Implement — boot and state.** In `S` add `rollouts: false, models: {}, railFilter: ""`. In `boot()` after `S.session = ...`:

```javascript
  S.rollouts = S.session.mode === "rollouts";
  S.models = (S.session.session && S.session.session.models) || {};
  if (S.rollouts) {
    // union of every model's emotion order, first-seen; a model lacking one shows "—"
    const seen = [];
    Object.values(S.models).forEach(m => (m.emotions || []).forEach(e => { if (seen.indexOf(e) < 0) seen.push(e); }));
    S.session.emotions = seen.length ? seen : S.session.emotions;
    document.body.classList.add("rollouts");
  }
```
CSS: `body.rollouts #composer, body.rollouts #btnNewChat, body.rollouts #btnNewTask, body.rollouts #healthChip { display:none }` — use the page's real ids (`grep -n 'id="' index.html` for the composer container, the two rail buttons and the health chip; name them exactly in the rule). `startHealthPoll()` is skipped when `S.rollouts` (there is no server to poll: `if (!S.rollouts) startHealthPoll();`).

`applyModelLayers(model)`:

```javascript
function applyModelLayers(model) {
  const m = (S.models || {})[model];
  if (!m) return;
  S.layers = m.capture_layers || []; S.probeLayer = m.probe_layer === undefined ? null : m.probe_layer;
  if (S.layers.indexOf(S.layer) < 0) S.layer = S.probeLayer !== null ? S.probeLayer : (S.layers[0] || null);
  renderChips();          // the layer chips are rebuilt from S.layers
}
```
Call it at the top of `openConversation` once `data` is in: `if (S.rollouts && data.conversation.model) applyModelLayers(data.conversation.model);`. Check `renderChips()` reads `S.layers` (it does: `grep -n "S.layers" index.html`); if it only runs at boot, make sure calling it again does not double-wire click handlers (it rebuilds innerHTML, so wiring is per build — verify by reading `renderChips`).

- [ ] **Step 3: Implement — rail.** In `refreshRail`, before the `tasks/chats` grouping:

```javascript
  if (S.rollouts) { renderRolloutRail(list); renderRailFoot(); fillAggFilter(); return; }
```
and add:

```javascript
function renderRolloutRail(list) {
  const q = (S.railFilter || "").toLowerCase();
  const byModel = {};
  S.conversations.forEach(c => { ((byModel[c.model] ||= {})[c.version] ||= []).push(c); });
  const models = Object.keys(byModel).sort();
  const oneCell = models.length === 1 && Object.keys(byModel[models[0]]).length === 1;
  const cellsMeta = {}; ((S.session.session || {}).cells || []).forEach(c => { cellsMeta[c.model + "/" + c.version] = c; });
  models.forEach(model => {
    const g = el("div", "group eyebrow", model); g.style.marginTop = "8px"; list.appendChild(g);
    Object.keys(byModel[model]).sort().forEach(version => {
      const items = byModel[model][version].filter(c => !q || (c.task_id + " " + version + " " + model).toLowerCase().indexOf(q) >= 0);
      if (!items.length) return;
      const meta = cellsMeta[model + "/" + version] || {};
      const d = document.createElement("details"); d.className = "cell"; d.open = oneCell || !!q;
      const s = document.createElement("summary");
      const mind = (meta.mindset || []).length ? " · " + meta.mindset.join("+") : "";
      s.innerHTML = "<b>" + esc(version) + "</b> <span class=\"meta\">" + esc(meta.bench_split || "?") + esc(mind) + " · " + items.length + " rollouts" +
        (meta.n_with_token_arrays ? " · " + meta.n_with_token_arrays + " per-token" : "") +
        (meta.n_misaligned ? ' · <span class="cap">' + meta.n_misaligned + " misaligned</span>" : "") + "</span>";
      d.appendChild(s);
      const multi = new Set(items.map(c => c.task_id)).size !== items.length;
      items.sort((a, b) => a.task_id.localeCompare(b.task_id, undefined, {numeric: true}) || a.sample - b.sample).forEach(c => {
        const b = el("button", "item" + (S.current && S.current.conversation.conversation_id === c.conversation_id ? " sel" : ""));
        b.dataset.cid = c.conversation_id;
        const st = c.passed === true ? "pass" : c.passed === false ? "fail" : "idle";
        b.innerHTML = '<span class="st ' + st + '"></span><span class="id">' + esc(c.task_id) + (multi ? " · s" + c.sample : "") + "</span>" +
          '<span class="meta">' + c.n_turns + "t" + (c.has_token_arrays ? " · ●" : "") + (c.passed === true ? " · ✓" : c.passed === false ? " · ✕" : "") + "</span>";
        b.title = c.has_token_arrays ? "per-token arrays available" : "turn readouts only (no per-token arrays)";
        b.addEventListener("click", () => openConversation(c.conversation_id));
        d.appendChild(b);
      });
      list.appendChild(d);
    });
  });
  if (!S.conversations.length) list.innerHTML = '<div class="group" style="color:var(--muted);font-size:11.5px">no rollouts found</div>';
  const n = S.conversations.reduce((a, c) => a + (c.n_turns || 0), 0);
  $("#recChip").textContent = n + " turns · " + S.conversations.length + " rollouts";
}
```
Add a filter input above `#railList` in the HTML: `<input id="railFilter" class="railfilter" placeholder="filter task / cell" hidden>`; show it in rollouts mode (`body.rollouts #railFilter{display:block}`), wire in `wireControls`: `$("#railFilter").addEventListener("input", e => { S.railFilter = e.target.value; refreshRail(); });`. CSS: `details.cell summary{cursor:pointer;padding:4px 8px;font-size:12px}`, `.railfilter{width:calc(100% - 16px);margin:6px 8px;padding:4px 6px;font-size:12px;background:var(--bg2);color:var(--fg);border:1px solid var(--line);border-radius:4px}` — use the page's actual token names (`grep -n "^  --" index.html` for the palette; do not invent tokens).

`renderRailFoot`: when `S.rollouts` print `Rollouts → <code>roots joined</code><br>Read-only: pilot records are never modified.`

- [ ] **Step 4: Implement — conversation.** In `renderConversation`, compute `const isRoll = conv.source === "rollout";` and treat it as `isTask` for message layout (`const isTask = conv.source === "task" || isRoll;`). Header: when `isRoll`, `$("#convTitle").textContent = conv.task_id + (conv.sample !== undefined ? " · s" + conv.sample : "");` and `bits` = `[conv.model, conv.version, (conv.bench_split||"?") + " split", conv.mindset && conv.mindset.length ? "mindset:" + conv.mindset.join("+") : null, turns.length + " turns"].filter(Boolean)`; state chip: `✓ passed`/`✕ failed` from `conv.passed`. Problem statement: the existing `msgs0.find(m => m.role === "user")` works from `messages_in`; when `msgs0` is empty and `isRoll`, append `plainMessage("user", "problem statement", "(unavailable: " + ((first.warnings||[]).find(w => w.indexOf("transcript context") >= 0) || "no .eval log") + ")")`. Feedback: `testMessage(t, i)` already keys off `t.passed` (`False` when feedback exists, rollout result on the last turn); nothing to change. In `assistantMessage` facts, when `t.n_decode !== undefined && t.n_decode !== null` add `t.n_decode + " rows"`. In `renderTurnBody`, before the misaligned check:

```javascript
  if (S.rollouts && turn.has_token_arrays === false && !turn.misaligned) {
    const w = el("div", "warnbox"); w.innerHTML = "<b>No per-token arrays</b> for this record (written before the mindset merge, 2026-08-16): text and start/end readouts only.";
    host.appendChild(w);
  }
```
and make the text branch condition `if (S.view === "text" || turn.misaligned || (S.rollouts && turn.has_token_arrays === false))`. In `assistantMessage`, if `S.rollouts && turn.tokenised === false` nothing special (full records always come through `/api/conversations/{cid}`).

- [ ] **Step 5: Implement — settings.** In `renderSettings`, when `S.rollouts`, replace the Server list with `[["mode","rollouts (read-only)"],["roots",(meta.roots||[]).join(", ")],["ignored",(meta.ignored||[]).join(", ")||"—"]]` and add two tables after `#setSession` in a new card `#setCells` (`<div class="card"><div class="ch"><h3>Models and cells</h3></div><table class="tbl" id="setCells"></table></div>` in the HTML): rows for models (`model, tokenizer, vectors, probe, layers, emotions`) then cells (`model, version, split, mindset, rollouts, per-token, tokenised, misaligned, max_tokens`). Use the page's existing table class (`grep -n 'class="tbl\|<table' index.html`).

- [ ] **Step 6: Verify.** `.venv/bin/pytest tests/cpu/test_dashboard_page.py -v` → PASS. Then run the app on the synthetic cell used by the tests (write a tiny script under the scratchpad that calls `make_cell` into a temp dir with 3 rollouts incl. one `d6`, then `python -m healthy_rl.dashboard --rollouts <dir> --port 8799`) and `curl` `/`; open it in a browser if one is available to you (the user's tunnel: `ssh -L 8799:localhost:8799 <login-host>`), otherwise say so plainly in the task report. Then run it on the real `Ministral-3-14B-Reasoning-2512` model dir (`--rollouts $ARTIFACT_DIR/rollouts/Ministral-3-14B-Reasoning-2512`) and check the rail lists every cell, a `d6` rollout opens in text view with start/end readouts, an `appr6` rollout opens in tokens view with a strip. Fix what you find.

- [ ] **Step 7: Commit** — `git commit -am "Page: rollouts mode — rail by model and cell, rollout transcript header, per-model layers, cell tables"`.

---

### Task 10: Page — grouped aggregate picker and chart

**Files:**
- Modify: `src/healthy_rl/dashboard/static/index.html`
- Test: `tests/cpu/test_dashboard_page.py`

**Interfaces:**
- Consumes: `/api/aggregate` groups (Task 7); `S.session.session.cells`.
- Produces: `S.aggGroups = {models: Set, versions: Set, split: string|null, layer: "probe"|int}`; `renderAggPicker()`, `renderAggregateGroups(data)`, `aggDrawGroups(data)`; `api.get` sends array params repeated; group colours `--g1..--g8` in all three theme blocks.

- [ ] **Step 1: Test strings** (append to `test_rollouts_mode_strings_present`): `"renderAggPicker", "aggDrawGroups", "--g1", "with base"`. Run → FAIL.

- [ ] **Step 2: `api.get` arrays.** Change the params loop to:
```javascript
    if (params) Object.entries(params).forEach(([k, v]) => {
      if (v === undefined || v === null) return;
      if (Array.isArray(v)) v.forEach(x => u.searchParams.append(k, x)); else u.searchParams.set(k, v);
    });
```

- [ ] **Step 3: Colours.** Add to each of the three palette blocks: light `--g1:#2a78d6; --g2:#eb6834; --g3:#1baf7a; --g4:#eda100; --g5:#8b5cf6; --g6:#0e9aa7; --g7:#c0392b; --g8:#6b7280;` dark `--g1:#3987e5; --g2:#d95926; --g3:#199e70; --g4:#c98500; --g5:#a78bfa; --g6:#22b8c8; --g7:#e06055; --g8:#9ca3af;`. Helper `const gvar = i => "var(--g" + ((i % 8) + 1) + ")";`.

- [ ] **Step 4: Picker.** HTML: next to `#aggFilter` add `<div id="aggPicker" hidden></div>`; in rollouts mode `#aggFilter` is hidden and `#aggPicker` shown (`body.rollouts #aggFilter{display:none} body.rollouts #aggPicker{display:block}`). `fillAggFilter` begins with `if (S.rollouts) { renderAggPicker(); return; }`.

```javascript
function renderAggPicker() {
  const host = $("#aggPicker"); host.innerHTML = "";
  const cells = ((S.session.session || {}).cells || []).slice().sort((a, b) => (a.model + a.version).localeCompare(b.model + b.version));
  const models = Array.from(new Set(cells.map(c => c.model)));
  const splits = Array.from(new Set(cells.map(c => c.bench_split).filter(Boolean)));
  if (!S.aggGroups) {
    const cur = S.current && S.current.conversation;
    S.aggGroups = {models: new Set(cur ? [cur.model] : models.slice(0, 1)),
                   versions: new Set(cur ? [cur.version] : cells.filter(c => c.model === models[0]).slice(0, 1).map(c => c.version)),
                   split: cur ? cur.bench_split : (splits[0] || null), layer: "probe"};
  }
  const G = S.aggGroups;
  const row = (label, node) => { const r = el("div", "pickrow"); r.appendChild(el("span", "eyebrow", label)); r.appendChild(node); host.appendChild(r); };
  const sp = el("div", "seg");
  splits.forEach(s => { const b = el("button", null, s); b.setAttribute("aria-pressed", String(G.split === s)); b.onclick = () => { G.split = s; renderAggPicker(); renderAggregate(); }; sp.appendChild(b); });
  row("split", sp);
  const mm = el("div", "checks");
  models.forEach(m => { const l = el("label"); const c = document.createElement("input"); c.type = "checkbox"; c.checked = G.models.has(m);
    c.onchange = () => { c.checked ? G.models.add(m) : G.models.delete(m); renderAggPicker(); renderAggregate(); }; l.appendChild(c); l.appendChild(document.createTextNode(" " + m)); mm.appendChild(l); });
  row("models", mm);
  const vv = el("div", "checks");
  cells.filter(c => G.models.has(c.model) && (!G.split || c.bench_split === G.split)).forEach(c => {
    const l = el("label"); const x = document.createElement("input"); x.type = "checkbox"; x.checked = G.versions.has(c.version);
    x.onchange = () => { x.checked ? G.versions.add(c.version) : G.versions.delete(c.version); renderAggregate(); };
    l.appendChild(x); l.appendChild(document.createTextNode(" " + c.model + "/" + c.version + ((c.mindset || []).length ? " (" + c.mindset.join("+") + ")" : "")));
    vv.appendChild(l);
  });
  row("cells", vv);
  const base = el("button", "tog", "with base");
  base.title = "add each selected mindset arm's base cell (d6 / aff6 / sp6 by its affect and scratchpad flags)";
  base.onclick = () => {
    cells.filter(c => G.models.has(c.model) && G.versions.has(c.version) && (c.mindset || []).length).forEach(c => {
      const want = c.affect_prompt ? "aff6" : c.scratchpad_reasoning ? "sp6" : "d6";
      if (cells.some(b => b.model === c.model && b.version === want)) G.versions.add(want);
    });
    renderAggPicker(); renderAggregate();
  };
  const ly = el("div", "seg");
  const layers = ["probe"].concat(Array.from(new Set([].concat(...Array.from(G.models).map(m => (S.models[m] || {}).capture_layers || [])))).sort((a, b) => a - b));
  layers.forEach(L => { const b = el("button", null, L === "probe" ? "probe" : "L" + L); b.setAttribute("aria-pressed", String(String(G.layer) === String(L)));
    b.onclick = () => { G.layer = L; renderAggregate(); renderAggPicker(); }; ly.appendChild(b); });
  const r2 = el("div", "pickrow"); r2.appendChild(el("span", "eyebrow", "layer")); r2.appendChild(ly); r2.appendChild(base); host.appendChild(r2);
}
```
CSS: `.pickrow{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:4px 0} .checks{display:flex;gap:10px;flex-wrap:wrap;font-size:12px}` (reuse the page's `.seg` button group and `.tog` styles — check their real class names with grep and match).

- [ ] **Step 5: Render.** In `renderAggregate`, when `S.rollouts` build params as
```javascript
  const G = S.aggGroups || {models: new Set(), versions: new Set(), split: null, layer: "probe"};
  params = {source: "rollout", split: G.split, model: Array.from(G.models), version: Array.from(G.versions),
            position: S.readout, stat: S.stat, segment: S.segment, include_cap: S.includeCap, layer: G.layer};
  $("#aggParams").textContent = "readout " + READOUT_LABEL[S.readout] + " · stat " + S.stat + " · segment " + S.segment + " · layer " + (G.layer === "probe" ? "probe (per model)" : "L" + G.layer) + " · capped turns " + (S.includeCap ? "counted as skipped" : "excluded");
```
and after `data` arrives: `if (S.rollouts) { renderAggregateGroups(data); return; }`.

```javascript
function renderAggregateGroups(data) {
  const table = $("#aggTable"), groups = data.groups || [], emos = data.emotions || [];
  if (!groups.length) { table.innerHTML = ""; $("#aggChart").innerHTML = '<p class="hint">Pick at least one cell.</p>'; $("#aggChartSub").textContent = ""; return; }
  let html = "<tr><th>cell</th><th>direction</th><th style=\"text-align:right\">t0</th><th style=\"text-align:right\">tlast</th><th style=\"text-align:right\">Δ</th><th style=\"text-align:right\">sem</th><th style=\"text-align:right\">p</th><th style=\"text-align:right\">n</th><th style=\"text-align:right\">skip</th><th>L</th></tr>";
  groups.forEach((g, gi) => {
    const bt = g.by_turn || {}, del = g.delta || {}, rows = bt.mean || [], K = rows.length;
    const show = emos.filter(e => isHead(e) || S.shown.has(e));
    show.forEach((e, k0) => {
      const k = emos.indexOf(e);
      const t0 = K && isNum(rows[0][k]) ? rows[0][k] : null, tl = K && isNum(rows[K - 1][k]) ? rows[K - 1][k] : null;
      html += '<tr><td>' + (k0 === 0 ? '<span class="sw" style="background:' + gvar(gi) + '"></span>' + esc(g.model + "/" + g.version) + ((g.mindset || []).length ? " (" + esc(g.mindset.join("+")) + ")" : "") : "") + "</td>" +
        "<td>" + esc(e) + '</td><td class="n">' + fmt(t0) + '</td><td class="n">' + fmt(tl) + '</td><td class="n"><b>' + fmt(isNum((del.mean || [])[k]) ? del.mean[k] : null) + "</b></td>" +
        '<td class="n">' + (isNum((del.sem || [])[k]) ? del.sem[k].toFixed(4) : "—") + '</td><td class="n">' + (isNum((del.p || [])[k]) ? del.p[k].toFixed(3) : "—") + "</td>" +
        '<td class="n">' + (del.n || 0) + '</td><td class="n">' + (g.skipped || 0) + '</td><td>L' + g.layer + "</td></tr>";
    });
  });
  table.innerHTML = html;
  $("#aggChartSub").textContent = S.colourBy + " · " + groups.map(g => g.n_conversations).reduce((a, b) => a + b, 0) + " rollouts across " + groups.length + " cell" + (groups.length === 1 ? "" : "s") + " · band = ±1 SEM · one line per cell";
  aggDrawGroups(data);
}

function aggDrawGroups(data) {
  const host = $("#aggChart"); host.innerHTML = "";
  const groups = data.groups || [], emos = data.emotions || [], ei = emos.indexOf(S.colourBy);
  if (ei < 0 || !groups.length) { host.innerHTML = '<p class="hint">' + esc(S.colourBy) + ' is not measured for the selected cells.</p>'; return; }
  const K = Math.max(...groups.map(g => ((g.by_turn || {}).mean || []).length));
  if (!K) { host.innerHTML = '<p class="hint">No by-turn series for these cells.</p>'; return; }
  const val = (g, k) => { const r = (g.by_turn.mean || [])[k]; return r && isNum(r[ei]) ? r[ei] : null; };
  const semv = (g, k) => { const r = (g.by_turn.sem || [])[k]; return r && isNum(r[ei]) ? r[ei] : 0; };
  const all = [];
  groups.forEach(g => { for (let k = 0; k < K; k++) { const v = val(g, k); if (isNum(v)) { all.push(v + semv(g, k)); all.push(v - semv(g, k)); } } });
  const sc = scaleY(all, 0.06);
  const W = 400, H = 180, m = {l: 44, r: 110, t: 10, b: 26};
  const s = svgEl(W, H);
  const y = v => m.t + (H - m.t - m.b) * (1 - (v - sc.lo) / (sc.hi - sc.lo));
  const x = t => m.l + (W - m.l - m.r) * (K > 1 ? t / (K - 1) : 0.5);
  const gr = N("g", {class: "grid"});
  sc.ticks.forEach(v => { gr.appendChild(N("line", {x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: Math.abs(v) < 1e-12 ? "zero" : ""}));
    const t = N("text", {x: m.l - 6, y: y(v) + 3.5, "text-anchor": "end"}); t.textContent = tickLabel(v, sc.step); gr.appendChild(t); });
  s.appendChild(gr);
  for (let t = 0; t < K; t++) { if (K > 12 && t % Math.ceil(K / 8) !== 0 && t !== K - 1) continue; const tx = N("text", {x: x(t), y: H - 8, "text-anchor": "middle"}); tx.textContent = "t" + t; s.appendChild(tx); }
  const labels = [];
  groups.forEach((g, gi) => {
    const vals = []; for (let k = 0; k < K; k++) vals.push(val(g, k));
    let up = "", dn = "", pen = false;
    for (let k = 0; k < K; k++) { const v = vals[k]; if (!isNum(v)) { pen = false; continue; } up += (pen ? "L" : "M") + x(k) + "," + y(v + semv(g, k)); pen = true; }
    for (let k = K - 1; k >= 0; k--) { const v = vals[k]; if (!isNum(v)) continue; dn += "L" + x(k) + "," + y(v - semv(g, k)); }
    if (up && dn) s.appendChild(N("path", {d: up + dn + "Z", fill: gvar(gi), opacity: 0.12}));
    const d = pathOf(vals, x, y);
    if (d) s.appendChild(N("path", {d: d, fill: "none", stroke: gvar(gi), "stroke-width": 2}));
    const lastIdx = vals.reduce((acc, v, i) => isNum(v) ? i : acc, -1);
    if (lastIdx >= 0) labels.push({y: y(vals[lastIdx]) + 3.5, x: x(lastIdx) + 8, e: g.version + (groups.some(o => o !== g && o.version === g.version) ? " · " + g.model : ""), gi: gi});
  });
  layoutLabels(labels, 11).forEach(L => { const t = N("text", {x: L.x, y: L.y, class: "lbl"}); t.textContent = L.e; t.style.fill = gvar(L.gi); s.appendChild(t); });
  const cross = N("line", {class: "cross", y1: m.t, y2: H - m.b, style: "display:none"}); s.appendChild(cross);
  const hit = N("rect", {x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent"}); s.appendChild(hit);
  hit.addEventListener("mousemove", ev => {
    const t = hoverIndex(ev, s, W, m, K);
    cross.setAttribute("x1", x(t)); cross.setAttribute("x2", x(t)); cross.style.display = "";
    let html = "<b>turn t" + t + "</b> · " + esc(S.colourBy) + "<br>";
    groups.forEach((g, gi) => { html += '<div class="row"><span><i style="background:' + gvar(gi) + '"></i>' + esc(g.model + "/" + g.version) + "</span><span>" + fmt(val(g, t)) + " ± " + (semv(g, t) ? semv(g, t).toFixed(4) : "—") + " · n=" + ((g.by_turn.n || [])[t] || 0) + "</span></div>"; });
    showTip(ev, html);
  });
  hit.addEventListener("mouseleave", () => { cross.style.display = "none"; hideTip(); });
  host.appendChild(s);
}
```
`renderTrajectory` in rollouts mode: the session-mean overlay call becomes `aggregateData({source: "rollout", split: conv.bench_split || null, model: [conv.model], version: [conv.version], position: S.readout, stat: "token", segment: "all", include_cap: S.includeCap, layer: "probe"})` and `S.trajSess = data.groups[0]`. Also, when `S.colourBy` changes (`renderChips` / colour-by wiring), call `renderAggregate()` if the aggregate tab is open — check `redrawEverything()` covers it.

- [ ] **Step 6: Verify.** `.venv/bin/pytest tests/cpu -q` → PASS. Run on the real Ministral model dir, open the Aggregate tab: pick `appr6`, press **with base** → `d6` joins; two lines, two colours, labels at line ends; the table lists both cells; switch layer to `L27` (Ministral's probe) and to `probe`. Then `--rollouts $ARTIFACT_DIR/rollouts` (everything): the picker lists every model; selecting two models with different probe layers and `layer=probe` draws both; picking an explicit layer one model lacks shows the 400 message in `#aggMsg`. If no browser is available to you, do the same through `curl` against `/api/aggregate` and say in the report which of the visual checks are unverified.

- [ ] **Step 7: Commit** — `git commit -am "Page: grouped aggregate picker and per-cell lines for rollouts mode"`.

---

### Task 11: Docs, spec deviations, final gate

**Files:**
- Modify: `docs/runs.md` ("Reading a run"), `docs/measurement.md` (after "The dashboard's readouts"), `docs/infrastructure.md` (only if a trap was met), `docs/superpowers/specs/2026-08-16-rollout-viewer-design.md` (append "## Deviations")

- [ ] **Step 1: `docs/runs.md`** — in "Reading a run" add:

````bash
# transcripts + per-token emotion strips + cross-cell aggregates, in the browser
.venv/bin/python -m healthy_rl.dashboard --rollouts $ARTIFACT_DIR/rollouts/gemma-3-12b-it --port 8765
# then from your machine:  ssh -L 8765:localhost:8765 <login-host>   and open http://localhost:8765
````
plus two sentences: one path may be a cell, a model or the root; per-token strips exist only for records with `t0_proj_L*` (post-2026-08-16); older cells show start/end readouts from the boundary residuals and need `$ARTIFACT_DIR/vectors/<model>/v1`.

- [ ] **Step 2: `docs/measurement.md`** — after "The dashboard's readouts" add "### Rollout token strips": the strip is `turn_completion` re-tokenised with the model's HF tokenizer (fast-tokenizer offsets, tokens tile the text); alignment is checked against the npz decode rows with the EOS rule (`N == D` or `N + 1 == D`, the extra row is EOS and is drawn as `<eos>`); any other count withholds the strip and marks the turn misaligned rather than shifting it; the identity of a rollout is `(task_id, sample)`, and the `.eval` context is matched by completion text because several samples share an id at Inspect epoch 1; the aggregate draws cells side by side at each model's own probe layer and does not claim cosines are commensurable across models.

- [ ] **Step 3: Spec deviations** — append to the spec a `## Deviations` section listing, at minimum: (1) ids use `s<sample>` not `ep<epoch>` and why; (2) `records()` returns light records and `record()` tokenises lazily, so the cell table's misaligned count is "among tokenised" (`n_tokenised`, `n_misaligned`); (3) token strings come from fast-tokenizer offsets tiling the text, not per-id decode; (4) `.eval` samples are matched by completion text; (5) `arrays_from_npz` takes `n_emotions` from the record; (6) anything else found in Tasks 8–10 (e.g. a tokenizer whose offsets do not tile — record the model and the fix). Also add "Reading a run" cross-reference to the spec status line: `**Status:** implemented 2026-08-16`.

- [ ] **Step 4: `docs/infrastructure.md`** — add a trap entry only if one was hit (e.g. `read_eval_log` sync failing in a thread, `AutoTokenizer` on a Gemma dir needing `sentencepiece`, apptainer-only inspect versions). Otherwise skip and say so.

- [ ] **Step 5: Final gate.** `.venv/bin/pytest tests/cpu -q` all green; the manual gate from spec §7 run on the login node (`--rollouts $ARTIFACT_DIR/rollouts`): one Ministral `appr6` rollout in tokens view, one `d6` rollout in text view with start/end readouts, an aggregate of `appr6` with base `d6`; screenshots at 1280 and 1920 px in both themes if a browser is available, else state which checks were done by curl only.

- [ ] **Step 6: Commit** — `git add docs && git commit -m "Document the rollout viewer: command, token-strip alignment rule, spec deviations"`.

---

## Self-review against the spec

- §2 usage/paths/startup table → Tasks 2 (discover), 8 (entry, report). §3.1 record fields → Task 2 (light) + 3 (tokens/misaligned) + 5 (messages/feedback/passed) + 4 (`arrays` marking). §3.2 arrays → Task 4. §3.3 tokenisation + EOS → Task 3 (offsets instead of per-id decode — deviation logged in Task 11). §3.4 old cells → Tasks 1 + 4. §3.5 refresh → Task 2. §3.6 session → Task 2 (+ `n_tokenised`, deviation logged). §4 app → Tasks 6, 7. §5 page → Tasks 9, 10 (Trajectory/Tokens reuse; Settings tables in 9). §6 error table → 2 (no cells → exit 2 in Task 8), 3 (tokenizer missing), 4 (vectors missing, npz missing/layer mismatch), 5 (`.eval` missing), 6/7 (layer 400, split 400). §7 tests → each task; manual gate → Tasks 8–11. §8 docs → Task 11. §9 out of scope respected.
- Names used consistently: `RolloutStore.open/records/record/arrays/conversations/session/refresh/cells/root`; `records_from_row`, `tokenise`, `align_tokens`, `arrays_from_npz(z, *, turn, capture_layers, probe_layer, n_emotions, vectors)`, `sample_messages`; `VectorsMeta`, `_meta`, `_layer(layer, rec)`, `_first_meta`, `_aggregate_group`, `_widen`; `AppState.mode`; `startup_report`; page `S.rollouts`, `S.models`, `S.aggGroups`, `applyModelLayers`, `renderRolloutRail`, `renderAggPicker`, `renderAggregateGroups`, `aggDrawGroups`, `gvar`.
