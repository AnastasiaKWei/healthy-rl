from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rollout_cell import EMOTIONS, FakeEvalSamples, GappedTokenizer, WhitespaceTokenizer, make_cell
from healthy_rl.dashboard.generation import split_reasoning
from healthy_rl.dashboard.rollout_store import (RolloutStore, align_tokens, discover_cells, records_from_row,
                                                sample_messages, tokenise)
from healthy_rl.rollouts import Vectors

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
    cells, ignored = discover_cells([tmp_path / "r", c])
    assert len(cells) == 1 and cells[0].max_tokens is None
    assert ignored == []                       # the same directory twice is not a second cell


def test_discover_cells_keeps_one_path_per_model_version(tmp_path):
    """Two roots holding the same cell: the key is what every record resolves its npz through,
    so keeping both would serve root A's arrays under root B's label."""
    a = make_cell(tmp_path / "A", "m", "v1", rows=ROWS[:1])
    b = make_cell(tmp_path / "B", "m", "v1", rows=ROWS[:1])
    cells, ignored = discover_cells([tmp_path / "A", tmp_path / "B"])
    assert len(cells) == 1 and cells[0].path == a.resolve()
    assert ignored == [b.resolve()]


def test_discover_cells_names_a_relative_cell_path(tmp_path, monkeypatch):
    """``--rollouts .`` from inside a cell: model/version come off the path, so resolve first."""
    c = make_cell(tmp_path / "r", "m1", "d6", rows=ROWS[:1])
    monkeypatch.chdir(c)
    cells, _ = discover_cells(["."])
    assert [(x.model, x.version) for x in cells] == [("m1", "d6")]
    assert cells[0].path == c.resolve()


def test_a_truncated_npz_does_not_break_the_store(tmp_path):
    """A half-written npz raises zipfile.BadZipFile, whose only base is Exception: every np.load
    site has to survive it, or one partial file under the root takes the whole dashboard down."""
    st = _store(tmp_path)
    npz = tmp_path / "rollouts" / "fake-model" / "appr6" / "residuals" / "lcbhard_0_s0.npz"
    npz.write_bytes(npz.read_bytes()[:100])
    st = RolloutStore.open([tmp_path / "rollouts"], tokenizer_loader=lambda m: WhitespaceTokenizer(),
                           vectors_loader=lambda m: None, eval_loader=FakeEvalSamples({}))
    assert st.session["cells"] and len(st.records()) == 8      # session reads every rollout's npz
    assert len(st.conversations()) == 4
    r = st.record("fake-model/appr6/lcbhard_0/s0/t0")
    assert r["misaligned"] is True and "npz unreadable" in r["error"]
    assert st.arrays("fake-model/appr6/lcbhard_0/s0/t0")["proj"].shape[0] == 0
    ok = st.record("fake-model/appr6/lcbhard_0/s1/t0")         # the neighbouring rollout is untouched
    assert ok["misaligned"] is False and ok["error"] is None


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
    # Rows predating the hash guard get "", the same value a base-arm row carries.
    assert r["mindset_hash"] == ""


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
    # the last turn is the one whose `passed` is the rollout's verdict, so it shows which row was read
    stale = st.record("fake-model/appr6/lcbhard_0/s0/t1")  # tokenised from the row about to be replaced
    untouched = st.record("fake-model/appr6/lcbhard_0/s1/t0")  # same shard file, a rollout nobody rewrote
    assert stale["passed"] is False
    with f.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    import os, time
    os.utime(f, (time.time() + 5, time.time() + 5))
    fresh = st.record("fake-model/appr6/lcbhard_0/s0/t1")
    assert fresh is not stale and fresh["passed"] is True and fresh["tokenised"] is True
    assert st.record("fake-model/appr6/lcbhard_0/s1/t0") is untouched   # the file's new mtime alone is not a change
    recs = st.records()
    assert len(recs) == 8 and len({r["record_id"] for r in recs}) == 8
    assert next(r for r in recs if r["record_id"] == "fake-model/appr6/lcbhard_0/s0/t1")["passed"] is True
    convs = {c["conversation_id"]: c for c in st.conversations()}
    assert len(convs) == 4 and convs["fake-model/appr6/lcbhard_0/s0"]["n_turns"] == 2
    cells = {(c["model"], c["version"]): c for c in st.session["cells"]}
    assert cells[("fake-model", "appr6")]["n_duplicate_rows"] == 1
    assert cells[("fake-model", "appr6")]["n_rollouts"] == 3
    assert cells[("fake-model", "d6")]["n_duplicate_rows"] == 0


def test_tokenise_tiles_text_and_keeps_true_starts():
    toks, starts = tokenise("ab  cd e", WhitespaceTokenizer())
    assert toks == ["ab", "  cd", " e"] and "".join(toks) == "ab  cd e"
    assert starts == [0, 2, 6]
    assert tokenise("", WhitespaceTokenizer()) == ([], [])


class _SpecialSpanTokenizer:
    """Fast tokenizer that emits a zero-width span, the way HF does for a special token."""
    is_fast = True

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        spans = [(0, 2), (0, 0), (2, 6)]
        out = {"input_ids": list(range(len(spans)))}
        if return_offsets_mapping:
            out["offset_mapping"] = spans
        return out


def test_tokenise_keeps_span_starts_when_offsets_have_gaps():
    # the gap goes into the token's text, but the start stays the span's own start
    assert tokenise("ab  cd", GappedTokenizer()) == (["ab", "  cd"], [0, 4])
    toks, starts = tokenise("ab ", GappedTokenizer())          # trailing gap: no span covers it
    assert "".join(toks) == "ab " and starts == [0]
    # a whitespace-only completion has no spans at all
    assert tokenise("  ", GappedTokenizer()) == (["  "], [0])
    # a zero-width span keeps its slot in the strip instead of jumping back to 0
    toks, starts = tokenise("ab  cd", _SpecialSpanTokenizer())
    assert toks == ["ab", "", "  cd"] and starts == [0, 2, 2] and "".join(toks) == "ab  cd"


def test_tokenise_start_offsets_decide_the_think_boundary():
    text = "[THINK]x[/THINK]  z"
    _, _, think_end = split_reasoning(text)
    toks, starts = tokenise(text, GappedTokenizer())
    assert toks == ["[THINK]x[/THINK]", "  z"] and starts == [0, 18]   # cumulative offsets would say 16
    assert think_end == 16
    _, kinds, _, _ = align_tokens(toks, starts, think_end, None)
    assert kinds == ["think", "answer"]


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
    assert z["error"] is None and o["error"] is None       # neither is a problem, just no arrays


def test_record_misaligned_when_counts_disagree(tmp_path):
    rows = [{"task_id": "lcbhard_0", "sample": 0, "completions": ["a b c"], "n_generated": [7]}]
    make_cell(tmp_path / "rollouts", "fake-model", "appr6", rows=rows)
    st = RolloutStore.open([tmp_path / "rollouts"], tokenizer_loader=lambda m: WhitespaceTokenizer(),
                           vectors_loader=lambda m: None, eval_loader=FakeEvalSamples({}))
    r = st.record("fake-model/appr6/lcbhard_0/s0/t0")
    assert r["misaligned"] is True and "3 tokens" in r["error"] and "7 decode rows" in r["error"]
    assert st.session["cells"][0]["n_tokenised"] == 1 and st.session["cells"][0]["n_misaligned"] == 1
    assert st.conversations()[0]["n_misaligned"] == 1


def test_record_reports_npz_problems(tmp_path):
    st = _store(tmp_path)
    res = tmp_path / "rollouts" / "fake-model" / "appr6" / "residuals"
    (res / "lcbhard_0_s0.npz").unlink()
    r = st.record("fake-model/appr6/lcbhard_0/s0/t1")
    assert r["misaligned"] is True and "npz missing" in r["error"] and str(res / "lcbhard_0_s0.npz") in r["error"]
    assert r["has_token_arrays"] is False and r["n_decode"] is None
    assert r["tokens"] == ["[THINK]x", " y[/THINK]", " z"]        # the text is still fine
    # half a pair is a problem too -- it is not the honest array-less turn
    np.savez(res / "lcbhard_0_s1.npz", **{"t1_kind_L20": np.zeros(5, np.int8)})
    h = st.record("fake-model/appr6/lcbhard_0/s1/t1")
    assert h["misaligned"] is True and "without" in h["error"] and "t1_proj_L20" in h["error"]
    assert h["has_token_arrays"] is False and h["n_decode"] is None


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
    # a fresh store over the same cells: re-running _store would re-create the npz just removed
    st2 = RolloutStore.open([tmp_path / "rollouts"], tokenizer_loader=lambda m: WhitespaceTokenizer(),
                            vectors_loader=lambda m: None, eval_loader=FakeEvalSamples({}))
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


def _fake_vectors(*, emotions=EMOTIONS, probe_layer=20, capture_layers=(10, 20), d=8):
    """A vectors artifact whose probe-layer directions read the first ``E`` residual components."""
    E, L = len(emotions), len(capture_layers)
    dirs = np.zeros((E, L, d), np.float32)
    dirs[:, list(capture_layers).index(probe_layer), :E] = np.eye(E)
    return Vectors(directions=dirs, emotions=list(emotions), capture_layers=list(capture_layers),
                   probe_layer=probe_layer, mean_residual_norm={l: 1.0 for l in capture_layers}, path=Path("fake"))


def _old_cell_store(tmp_path, vec):
    """One old cell (boundary residuals only), with ``vec`` as the model's vectors artifact."""
    make_cell(tmp_path / "rollouts", "fake-model", "d6", rows=ROWS[:1], token_arrays=False)
    return RolloutStore.open([tmp_path / "rollouts"], tokenizer_loader=lambda m: WhitespaceTokenizer(),
                             vectors_loader=lambda m: vec, eval_loader=FakeEvalSamples({}))


def test_arrays_for_row_without_a_residuals_file(tmp_path):
    cell = make_cell(tmp_path / "rollouts", "fake-model", "d6", rows=ROWS[:1], token_arrays=False)
    f = cell / "rollouts.shard0of2.jsonl"
    row = json.loads(f.read_text()); row["residuals"] = None; f.write_text(json.dumps(row) + "\n")
    st = RolloutStore.open([tmp_path / "rollouts"], tokenizer_loader=lambda m: WhitespaceTokenizer(),
                           vectors_loader=lambda m: None, eval_loader=FakeEvalSamples({}))
    a = st.arrays("fake-model/d6/lcbhard_0/s0/t0")
    assert a["proj"].shape == (0, 2, 3) and np.isnan(a["norm_prefill"]).all() and np.isnan(a["proj_prefill"]).all()
    r = st.record("fake-model/d6/lcbhard_0/s0/t0")
    assert r["misaligned"] is False and r["error"] is None    # no npz to be missing: an honest array-less row


def test_arrays_rejects_vectors_from_a_different_probe_layer(tmp_path):
    st = _old_cell_store(tmp_path, _fake_vectors(capture_layers=(10, 30), probe_layer=30))
    a = st.arrays("fake-model/d6/lcbhard_0/s0/t0")
    assert a["proj"].shape == (0, 2, 3) and np.isnan(a["proj_prefill"]).all()
    assert "proj_end" not in a and "res_start_L20" not in a   # nothing is served off the wrong artifact
    r = st.record("fake-model/d6/lcbhard_0/s0/t0")
    assert r["misaligned"] is True and "vectors probe layer L30 differs from the record's L20" in r["error"]


def test_arrays_rejects_vectors_with_a_different_emotion_count(tmp_path):
    st = _old_cell_store(tmp_path, _fake_vectors(emotions=(*EMOTIONS, "calm")))
    a = st.arrays("fake-model/d6/lcbhard_0/s0/t0")
    assert a["proj"].shape == (0, 2, 3) and np.isnan(a["proj_prefill"]).all() and "proj_end" not in a
    r = st.record("fake-model/d6/lcbhard_0/s0/t0")
    assert r["misaligned"] is True and "vectors list 4 emotions, record lists 3" in r["error"]


def test_arrays_rejects_vectors_whose_emotion_order_differs(tmp_path):
    st = _old_cell_store(tmp_path, _fake_vectors(emotions=(EMOTIONS[1], EMOTIONS[0], EMOTIONS[2])))
    a = st.arrays("fake-model/d6/lcbhard_0/s0/t0")
    assert a["proj"].shape == (0, 2, 3) and np.isnan(a["proj_prefill"]).all() and "proj_end" not in a
    r = st.record("fake-model/d6/lcbhard_0/s0/t0")
    assert r["misaligned"] is True and "vectors emotion order differs from the record's" in r["error"]


def test_records_from_row_separates_steering_conditions(tmp_path):
    """A steering sweep re-runs one (task, sample) once per condition; the rows are different
    rollouts and must not collapse onto each other."""
    cell = make_cell(tmp_path / "r", "m1", "v1", rows=ROWS[:1])
    row = json.loads((cell / "rollouts.shard0of2.jsonl").read_text().splitlines()[0])
    kw = dict(model="m1", version="v1", max_tokens=4, created_at="2026-08-16T00:00:00+00:00")
    base = records_from_row(dict(row, condition_name="readout"), **kw)
    steer = records_from_row(dict(row, condition_name="calm+0.1"), **kw)
    assert base[0]["conversation_id"] == "m1/v1/lcbhard_0/s0"
    assert steer[0]["conversation_id"] == "m1/v1/lcbhard_0/s0/ccalm+0.1"
    assert steer[0]["record_id"] == "m1/v1/lcbhard_0/s0/ccalm+0.1/t0"
    assert records_from_row(dict(row, condition_name=None), **kw)[0]["conversation_id"] == base[0]["conversation_id"]


def test_store_keeps_a_steering_sweeps_conditions_apart(tmp_path):
    cell = make_cell(tmp_path / "r", "m1", "v1", rows=ROWS[:1])
    f = cell / "rollouts.shard0of2.jsonl"
    row = json.loads(f.read_text().splitlines()[0])
    f.write_text("".join(json.dumps(dict(row, condition_name=c)) + "\n" for c in ("readout", "calm+0.1")))
    st = RolloutStore.open([tmp_path / "r"], tokenizer_loader=lambda m: WhitespaceTokenizer(),
                           vectors_loader=lambda m: None, eval_loader=FakeEvalSamples({}))
    convs = st.conversations()
    assert len(convs) == 2 and len({c["conversation_id"] for c in convs}) == 2
    assert sorted(c["condition_name"] for c in convs) == ["calm+0.1", "readout"]
    assert st.session["cells"][0]["n_duplicate_rows"] == 0


SAMPLES = [
    {"id": "lcbhard_0", "epoch": 1, "messages": [
        {"role": "user", "content": "PROBLEM"}, {"role": "assistant", "content": "a b c"},
        {"role": "user", "content": "Your previous attempt failed the tests. FAIL1"},
        {"role": "assistant", "content": "[THINK]x y[/THINK] z"},
        {"role": "user", "content": "Your previous attempt failed the tests. FAIL2"}]},
    {"id": "lcbhard_0", "epoch": 1, "messages": [
        {"role": "user", "content": "PROBLEM"}, {"role": "assistant", "content": "p q"},
        {"role": "user", "content": "FAILP"}, {"role": "assistant", "content": "r s t u"}]},
    # ROWS[2]: turn 0 generated nothing, so the .eval holds one assistant message for two turns
    {"id": "lcbhard_1", "epoch": 1, "messages": [
        {"role": "user", "content": "PROBLEM"}, {"role": "assistant", "content": "k l m"},
        {"role": "user", "content": "Your previous attempt failed the tests. FAILK"}]},
]

EMPTY_ROW = {"task_id": "lcbhard_2", "sample": 0, "completions": ["", ""], "n_generated": [0, 0], "passed": False}
EMPTY_SAMPLE = {"id": "lcbhard_2", "epoch": 1, "messages": [{"role": "user", "content": "PROBLEM"}]}


def _eval_store(tmp_path, samples, rows=ROWS, **kw):
    """A one-cell store whose shard holds exactly one .eval, carrying ``samples``."""
    make_cell(tmp_path / "rollouts", "fake-model", "appr6", rows=rows, **kw)
    log = tmp_path / "rollouts" / "fake-model" / "appr6" / "inspect-logs" / "shard0of2" / "x.eval"
    evals = FakeEvalSamples({str(log): samples})
    st = RolloutStore.open([tmp_path / "rollouts"], tokenizer_loader=lambda m: WhitespaceTokenizer(),
                           vectors_loader=lambda m: None, eval_loader=evals)
    return st, evals


def test_sample_messages_matches_by_completion():
    m, rule = sample_messages(SAMPLES, "lcbhard_0", ["p q", "r s t u"])
    assert m[1]["content"] == "p q" and rule == "completion"
    assert sample_messages(SAMPLES, "lcbhard_0", ["a b c", "[THINK]x y[/THINK] z"])[0][3]["content"].endswith(" z")
    assert sample_messages(SAMPLES, "lcbhard_0", ["nope"]) == (None, None)
    assert sample_messages(SAMPLES, "lcbhard_7", ["a b c"]) == (None, None)
    # a rollout whose first turn generated nothing matches on its first non-empty completion
    assert sample_messages(SAMPLES, "lcbhard_0", ["", "p q"])[0] is not None
    # nothing generated at all: no completion to match on, so the id alone -- if it is unambiguous
    assert sample_messages([EMPTY_SAMPLE], "lcbhard_2", ["", ""])[0][0]["content"] == "PROBLEM"
    assert sample_messages([EMPTY_SAMPLE, EMPTY_SAMPLE], "lcbhard_2", ["", ""]) == (None, None)


OLD_SAMPLES = [
    {"id": "lcbhard_5", "epoch": 1, "messages": [
        {"role": "user", "content": "PROBLEM ep1"}, {"role": "assistant", "content": "a b c"},
        {"role": "user", "content": "FAIL ep1"}, {"role": "assistant", "content": "d e f g"}]},
    {"id": "lcbhard_5", "epoch": 2, "messages": [
        {"role": "user", "content": "PROBLEM ep2"}, {"role": "assistant", "content": "[THINK]x y[/THINK] z"},
        {"role": "user", "content": "FAIL ep2"}, {"role": "assistant", "content": "q r s t"}]},
]

# an old cell (before the mindset merge, 2026-08-16): token counts but no turn_completion, no arrays
OLD_ROW = {"task_id": "lcbhard_5", "sample": 1, "completions": ["", ""], "n_generated": [3, 4], "passed": False}


def test_sample_messages_matches_an_old_cell_by_epoch():
    """No completion text anywhere, and two samples share the id: the epoch tells them apart."""
    m, rule = sample_messages(OLD_SAMPLES, "lcbhard_5", ["", ""], epoch=2)
    assert rule == "epoch" and m[0]["content"] == "PROBLEM ep2"
    assert sample_messages(OLD_SAMPLES, "lcbhard_5", ["", ""], epoch=1)[0][0]["content"] == "PROBLEM ep1"
    assert sample_messages(OLD_SAMPLES, "lcbhard_5", ["", ""], epoch=3) == (None, None)   # ambiguous id
    assert sample_messages(OLD_SAMPLES, "lcbhard_5", ["", ""]) == (None, None)
    # one candidate and no epoch of its own: the id rule still answers
    assert sample_messages([OLD_SAMPLES[0]], "lcbhard_5", ["", ""], epoch=9)[1] == "id"


def test_record_recovers_an_old_cells_text_from_the_eval(tmp_path):
    """d6/aff6/sp6/v1 rows store turn_n_generated but not turn_completion. The text is in the
    .eval log; it is put in the bubble and flagged, never aligned against arrays (there are none)."""
    st, _ = _eval_store(tmp_path, OLD_SAMPLES, rows=[OLD_ROW], token_arrays=False)
    t0 = st.record("fake-model/appr6/lcbhard_5/s1/t0")
    t1 = st.record("fake-model/appr6/lcbhard_5/s1/t1")
    assert t0["text"] == "[THINK]x y[/THINK] z" and t0["text_source"] == "eval"     # epoch 2, not epoch 1
    assert t0["reasoning"] == "x y" and t0["answer"] == "z"
    assert t0["has_token_arrays"] is False and t0["misaligned"] is False and t0["error"] is None
    assert t0["messages_in"] == [{"role": "user", "content": "PROBLEM ep2"}]
    assert t0["feedback"] == "FAIL ep2"
    assert any("matched by epoch" in w for w in t0["warnings"])
    assert any("taken from the .eval log" in w for w in t0["warnings"])
    assert t1["text"] == "q r s t" and t1["answer"] == "q r s t" and t1["text_source"] == "eval"
    assert [m["role"] for m in t1["messages_in"]] == ["user", "assistant", "user"]


def test_record_of_a_new_cell_keeps_its_own_text(tmp_path):
    st, _ = _eval_store(tmp_path, SAMPLES)
    r = st.record("fake-model/appr6/lcbhard_0/s0/t0")
    assert r["text"] == "a b c" and r["text_source"] == "record"


def _two_log_store(tmp_path, x_samples, y_samples, rows, **kw):
    """A cell whose shard holds two .eval logs -- what a steering sweep writes, one run per arm."""
    make_cell(tmp_path / "rollouts", "fake-model", "appr6", rows=rows, **kw)
    d = tmp_path / "rollouts" / "fake-model" / "appr6" / "inspect-logs" / "shard0of2"
    (d / "y.eval").write_bytes(b"")
    evals = FakeEvalSamples({str(d / "x.eval"): x_samples, str(d / "y.eval"): y_samples})
    return RolloutStore.open([tmp_path / "rollouts"], tokenizer_loader=lambda m: WhitespaceTokenizer(),
                             vectors_loader=lambda m: None, eval_loader=evals)


def test_an_old_cell_refuses_a_rollout_two_eval_logs_could_both_be(tmp_path):
    """The id and the epoch are unique inside one log, not across a shard's logs. Olmo-3.1-32B-Think/v1
    is a 9-condition sweep with 9-13 logs per shard, all carrying the same (id, epoch): taking the
    first put another arm's completion in this rollout's bubble."""
    other = [{"id": "lcbhard_5", "epoch": 2, "messages": [
        {"role": "user", "content": "PROBLEM other arm"}, {"role": "assistant", "content": "not this rollout"}]}]
    st = _two_log_store(tmp_path, OLD_SAMPLES, other, [OLD_ROW], token_arrays=False)
    r = st.record("fake-model/appr6/lcbhard_5/s1/t0")
    assert r["text"] == "" and r["text_source"] == "record" and r["messages_in"] == []
    assert any("tell them apart" in w for w in r["warnings"])


def test_a_new_cells_completion_text_still_picks_its_log_out(tmp_path):
    """Text equality is its own proof, so the strong rule is unaffected by a second log."""
    st = _two_log_store(tmp_path, [], SAMPLES, ROWS)
    r = st.record("fake-model/appr6/lcbhard_0/s0/t0")
    assert r["text"] == "a b c" and r["text_source"] == "record"
    assert r["messages_in"] == [{"role": "user", "content": "PROBLEM"}] and r["feedback"].endswith("FAIL1")


def test_record_of_an_old_cell_the_eval_cannot_identify(tmp_path):
    """Neither the epoch nor the id picks a sample out: no context, no text, and it says so."""
    st, _ = _eval_store(tmp_path, OLD_SAMPLES, rows=[dict(OLD_ROW, sample=7)], token_arrays=False)
    r = st.record("fake-model/appr6/lcbhard_5/s7/t0")
    assert r["messages_in"] == [] and r["text"] == "" and r["text_source"] == "record"
    assert any("no .eval sample matches" in w for w in r["warnings"])


def test_record_messages_in_and_feedback(tmp_path):
    st, evals = _eval_store(tmp_path, SAMPLES)
    r0 = st.record("fake-model/appr6/lcbhard_0/s0/t0"); r1 = st.record("fake-model/appr6/lcbhard_0/s0/t1")
    assert r0["messages_in"] == [{"role": "user", "content": "PROBLEM"}]
    assert r0["feedback"].endswith("FAIL1") and r0["passed"] is False
    assert [m["role"] for m in r1["messages_in"]] == ["user", "assistant", "user"]
    assert r1["feedback"].endswith("FAIL2") and r1["passed"] is False           # last turn, rollout failed
    s1 = st.record("fake-model/appr6/lcbhard_0/s1/t1")
    assert s1["feedback"] is None and s1["passed"] is True                       # last turn, rollout passed
    assert st.record("fake-model/appr6/lcbhard_0/s1/t0")["passed"] is False
    assert evals.calls == 1                                                      # one parse per file


def test_record_maps_turns_past_an_empty_turn(tmp_path):
    """A turn that generated nothing wrote no assistant message, so later turns must not shift."""
    st, _ = _eval_store(tmp_path, SAMPLES)
    t0 = st.record("fake-model/appr6/lcbhard_1/s0/t0")       # generated 0 tokens
    t1 = st.record("fake-model/appr6/lcbhard_1/s0/t1")
    assert t0["messages_in"] == [{"role": "user", "content": "PROBLEM"}]   # what the empty turn was given
    assert t0["feedback"] is None and t0["passed"] is None
    assert t1["messages_in"] == [{"role": "user", "content": "PROBLEM"}]   # not the next turn's context
    assert t1["feedback"].endswith("FAILK") and t1["passed"] is False
    assert [w for w in t0["warnings"] + t1["warnings"] if "assistant messages" in w] == []


def test_record_of_a_rollout_that_generated_nothing(tmp_path):
    st, _ = _eval_store(tmp_path, [EMPTY_SAMPLE], rows=[EMPTY_ROW])
    for rid in ("fake-model/appr6/lcbhard_2/s0/t0", "fake-model/appr6/lcbhard_2/s0/t1"):
        r = st.record(rid)
        assert r["messages_in"] == [{"role": "user", "content": "PROBLEM"}]   # the prompt is still recovered
        assert r["feedback"] is None and r["misaligned"] is False
        # sample 0 <-> epoch 1: the epoch rule answers before the id rule does, and says so
        assert [w for w in r["warnings"] if "matched by epoch" not in w] == []
    assert st.record("fake-model/appr6/lcbhard_2/s0/t1")["passed"] is False    # last turn: the rollout's verdict


def test_record_when_the_eval_does_not_identify_the_rollout(tmp_path):
    """Two samples share the id and nothing was generated: nothing tells them apart."""
    st, _ = _eval_store(tmp_path, [EMPTY_SAMPLE, EMPTY_SAMPLE], rows=[EMPTY_ROW])
    r = st.record("fake-model/appr6/lcbhard_2/s0/t0")
    assert r["messages_in"] == [] and r["feedback"] is None and r["misaligned"] is False
    assert any("no .eval sample matches" in w for w in r["warnings"])


def test_record_without_eval_file(tmp_path):
    st = _store(tmp_path)              # FakeEvalSamples({}) raises FileNotFoundError
    r = st.record("fake-model/appr6/lcbhard_0/s0/t0")
    assert r["messages_in"] == [] and any(".eval" in w for w in r["warnings"])


def test_concurrent_reads_and_refresh_do_not_race(tmp_path):
    """The dashboard's sync routes run in a threadpool: several requests hit one store at once.

    Reads and refreshes are interleaved deliberately -- a shard file grows while other
    threads are inside ``records()``/``record()``/``arrays()``/``session``/``conversations()``
    -- because ``refresh()`` rebuilds ``_light``/``_order`` and prunes ``_full`` in place.
    Unsynchronised this raises (``dictionary changed size during iteration`` in ``refresh``,
    ``KeyError`` from ``records()`` reading ``_light`` mid-rebuild). The short switch
    interval is what makes that certain rather than occasional: the tasks are small
    enough that the default 5 ms rarely preempts a thread inside ``refresh()``.
    """
    import os
    import sys
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    st = _store(tmp_path)
    rids = [r["record_id"] for r in st.records()]
    assert len(rids) == 8
    f = tmp_path / "rollouts" / "fake-model" / "appr6" / "rollouts.shard0of2.jsonl"
    template = json.loads(f.read_text().splitlines()[0])
    write_lock = threading.Lock()          # the writer is the pilot job, not a second dashboard
    n_appended = 20

    def append(i: int) -> None:
        row = dict(template, task_id=f"lcbhard_new{i}")     # a new rollout: 2 more turns
        with write_lock:
            with f.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            os.utime(f, (time.time() + 5 + i, time.time() + 5 + i))   # forward, for coarse mtimes

    def task(i: int) -> None:
        if i % 10 == 9:
            append(i // 10)
        k, rid = i % 5, rids[i % len(rids)]
        if k == 0:
            st.records()
        elif k == 1:
            st.record(rid)
        elif k == 2:
            st.arrays(rid)
        elif k == 3:
            st.session
        else:
            st.conversations()

    switch = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        with ThreadPoolExecutor(8) as ex:
            futures = [ex.submit(task, i) for i in range(200)]
            failures = [repr(fut.exception()) for fut in futures if fut.exception() is not None]
    finally:
        sys.setswitchinterval(switch)
    assert failures == [], f"{len(failures)} worker(s) raised: {failures[:3]}"

    recs = st.records()
    assert len(recs) == 8 + 2 * n_appended
    assert len({r["record_id"] for r in recs}) == len(recs)
    cells = {(c["model"], c["version"]): c for c in st.session["cells"]}
    assert cells[("fake-model", "appr6")]["n_rollouts"] == 3 + n_appended
    assert len(st.conversations()) == 4 + n_appended
