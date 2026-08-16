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


def test_duplicate_row_is_collapsed_and_counted(tmp_path):
    st = _store(tmp_path)
    assert len(st.records()) == 8
    cell = tmp_path / "rollouts" / "fake-model" / "appr6"
    f = cell / "rollouts.shard0of2.jsonl"
    row = json.loads(f.read_text().splitlines()[0])       # lcbhard_0/s0 again: a re-run after a crash
    row["passed"] = True                                  # the later row is the one that should win
    with f.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    import os, time
    os.utime(f, (time.time() + 5, time.time() + 5))
    recs = st.records()
    assert len(recs) == 8 and len({r["record_id"] for r in recs}) == 8
    assert next(r for r in recs if r["record_id"] == "fake-model/appr6/lcbhard_0/s0/t0")["passed"] is True
    convs = {c["conversation_id"]: c for c in st.conversations()}
    assert len(convs) == 4 and convs["fake-model/appr6/lcbhard_0/s0"]["n_turns"] == 2
    cells = {(c["model"], c["version"]): c for c in st.session["cells"]}
    assert cells[("fake-model", "appr6")]["n_duplicate_rows"] == 1
    assert cells[("fake-model", "appr6")]["n_rollouts"] == 3
    assert cells[("fake-model", "d6")]["n_duplicate_rows"] == 0
