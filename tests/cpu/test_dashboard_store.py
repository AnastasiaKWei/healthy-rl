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


def test_concurrent_appends_share_one_writer_and_lose_no_rows(tmp_path, monkeypatch):
    """A task run and a chat send append at the same time, through one handle.

    The writer is created lazily on the first append, so without a lock several
    threads each open their own handle to ``records.jsonl``. The extra handles
    are never closed and the rows they wrote are not counted by the survivor.
    The sleep in the patched constructor holds that window open, so a missing
    lock fails this test every time instead of one run in five.
    """
    import threading
    import time

    from healthy_rl.dashboard import store as store_mod

    built: list[object] = []

    class SlowWriter(store_mod.JsonlWriter):
        def __init__(self, path):
            built.append(path)
            time.sleep(0.05)
            super().__init__(path)

    monkeypatch.setattr(store_mod, "JsonlWriter", SlowWriter)

    st = SessionStore.create(tmp_path / "s", {"model": "m"})
    errors: list[BaseException] = []
    together = threading.Barrier(8)

    def work(t: int) -> None:
        try:
            together.wait()
            for i in range(25):
                st.append(_rec(f"c{t}", "chat", i), _arrays())
        except BaseException as exc:  # a torn write surfaces here, not as a silent short file
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(built) == 1, f"{len(built)} handles opened on one records.jsonl"
    assert st._writer.n_written == 200
    st.close()

    rows = [json.loads(l) for l in (tmp_path / "s" / "records.jsonl").read_text().splitlines()]
    assert len(rows) == 200
    assert len({r["record_id"] for r in rows}) == 200
    assert all((tmp_path / "s" / r["arrays"]).is_file() for r in rows)
