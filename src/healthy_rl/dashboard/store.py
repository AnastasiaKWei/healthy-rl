"""Session records: ``session.json``, append-only ``records.jsonl``, ``proj/<id>.npz``.

Layout under ``$ARTIFACT_DIR/dashboard/<model>/<jobid>/`` (see the spec §3.3).
Field names follow the pilot's rollout records where they overlap
(``emotions``, ``bench_split``, ``passed``, ``n_generated``); the npz keys for
boundary residuals are the pilot's ``res_start_L{probe}`` / ``res_end_L{probe}``.
Login-node importable: numpy + stdlib + healthy_rl.rollouts' pure half.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from healthy_rl.rollouts import JsonlWriter, read_jsonl

SESSION_FILE = "session.json"
RECORDS_FILE = "records.jsonl"
ARRAYS_DIR = "proj"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


class SessionStore:
    def __init__(self, root: str | os.PathLike[str], session: dict[str, Any]) -> None:
        self.root = Path(root)
        self.session = session
        self.records_path = self.root / RECORDS_FILE
        self.arrays_dir = self.root / ARRAYS_DIR
        self.arrays_dir.mkdir(parents=True, exist_ok=True)
        self._writer: JsonlWriter | None = None
        # One dashboard session appends from several threads at once -- a task
        # run and a chat send can be in flight together -- and they share one
        # file handle. Without this, two records interleave on that handle and
        # the JSONL grows a torn line that no reader can parse.
        self._lock = threading.Lock()

    @classmethod
    def create(cls, root: str | os.PathLike[str], session: dict[str, Any]) -> "SessionStore":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        meta = dict(session)
        meta.setdefault("created_at", _now())
        (root / SESSION_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True, default=str))
        return cls(root, meta)

    @classmethod
    def open(cls, root: str | os.PathLike[str]) -> "SessionStore":
        root = Path(root)
        path = root / SESSION_FILE
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist; not a dashboard session directory")
        return cls(root, json.loads(path.read_text()))

    def append(self, record: dict[str, Any], arrays: dict[str, np.ndarray]) -> str:
        """Write the arrays and the row. Thread-safe; the row lands last.

        The lock covers the npz write too, so a row is never visible before the
        arrays it points at, and ``self._writer`` is created exactly once.
        """
        with self._lock:
            rid = record.get("record_id") or uuid.uuid4().hex
            record["record_id"] = rid
            record.setdefault("created_at", _now())
            np.savez(self.arrays_dir / f"{rid}.npz", **arrays)
            record["arrays"] = f"{ARRAYS_DIR}/{rid}.npz"
            if self._writer is None:
                self._writer = JsonlWriter(self.records_path)
            self._writer.write(record)
        return rid

    def records(self) -> list[dict[str, Any]]:
        if not self.records_path.is_file():
            return []
        return read_jsonl(self.records_path)

    def record(self, record_id: str) -> dict[str, Any]:
        for r in self.records():
            if r.get("record_id") == record_id:
                return r
        raise KeyError(record_id)

    def arrays(self, record_id: str) -> dict[str, np.ndarray]:
        with np.load(self.arrays_dir / f"{record_id}.npz") as z:
            return {k: z[k] for k in z.files}

    def conversations(self) -> list[dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for r in self.records():
            cid = r["conversation_id"]
            c = out.get(cid)
            if c is None:
                c = out[cid] = {
                    "conversation_id": cid, "source": r.get("source"),
                    "bench_split": r.get("bench_split"), "task_id": r.get("task_id"),
                    "title": r.get("title"), "n_turns": 0, "passed": None,
                    "last_created_at": r.get("created_at"),
                }
            c["n_turns"] += 1
            if r.get("passed") is not None:
                c["passed"] = r["passed"]
            if r.get("title") and not c["title"]:
                c["title"] = r["title"]
            c["last_created_at"] = r.get("created_at", c["last_created_at"])
        return list(out.values())

    def close(self) -> None:
        with self._lock:
            if self._writer is not None:
                self._writer.close()
                self._writer = None
