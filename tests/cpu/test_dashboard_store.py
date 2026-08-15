from __future__ import annotations

import json

import numpy as np
import pytest

from healthy_rl.dashboard.store import SessionStore


def _rec(conv="c1", source="chat", turn=0, **kw):
    r = {"conversation_id": conv, "source": source, "turn_index": turn, "text": "hi",
         "n_generated": 3, "emotions": ["a", "b"], "created_at": f"2026-08-15T00:00:0{turn}"}
    r.update(kw)
    return r


def _arrays():
    return {"proj": np.zeros((3, 2, 2), np.float32), "norm": np.ones((3, 2), np.float32),
            "proj_prefill": np.zeros((2, 2), np.float32), "norm_prefill": np.ones((2,), np.float32)}


def test_create_writes_session_json_and_open_reads_it(tmp_path):
    st = SessionStore.create(tmp_path / "s", {"model": "m", "probe_layer": 27})
    assert (tmp_path / "s" / "session.json").is_file()
    again = SessionStore.open(tmp_path / "s")
    assert again.session["model"] == "m" and "created_at" in again.session
    with pytest.raises(FileNotFoundError):
        SessionStore.open(tmp_path / "nope")


def test_append_writes_row_and_npz_and_assigns_id(tmp_path):
    st = SessionStore.create(tmp_path / "s", {"model": "m"})
    rid = st.append(_rec(), _arrays())
    rows = [json.loads(l) for l in (tmp_path / "s" / "records.jsonl").read_text().splitlines()]
    assert rows[0]["record_id"] == rid and rows[0]["arrays"] == f"proj/{rid}.npz"
    arr = st.arrays(rid)
    assert arr["proj"].shape == (3, 2, 2) and arr["norm_prefill"].shape == (2,)
    assert st.record(rid)["text"] == "hi"
    with pytest.raises(KeyError):
        st.record("missing")


def test_conversations_groups_in_first_seen_order(tmp_path):
    st = SessionStore.create(tmp_path / "s", {"model": "m"})
    st.append(_rec("t1", "task", 0, bench_split="original", task_id="lcbhard_3", passed=False), _arrays())
    st.append(_rec("c1", "chat", 0, title="Deadline"), _arrays())
    st.append(_rec("t1", "task", 1, bench_split="original", task_id="lcbhard_3", passed=True), _arrays())
    convs = st.conversations()
    assert [c["conversation_id"] for c in convs] == ["t1", "c1"]
    assert convs[0]["n_turns"] == 2 and convs[0]["passed"] is True and convs[0]["task_id"] == "lcbhard_3"
    assert convs[1]["title"] == "Deadline" and convs[1]["passed"] is None


def test_records_survive_reopen(tmp_path):
    st = SessionStore.create(tmp_path / "s", {"model": "m"})
    st.append(_rec(), _arrays()); st.close()
    assert len(SessionStore.open(tmp_path / "s").records()) == 1
