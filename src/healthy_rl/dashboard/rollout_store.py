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
from dataclasses import dataclass
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
        self._duplicate_rows: dict[tuple[str, str], int] = {}     # cell key -> rows re-using a rollout id
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
            self._light.clear(); self._order.clear(); self._duplicate_rows.clear()
            seen_rollouts: set[str] = set()
            for f in sorted(self._rows_by_file):
                created = _now_iso(self._files[f][0])
                for cell, row in self._rows_by_file[f]:
                    recs = records_from_row(row, model=cell.model, version=cell.version,
                                            max_tokens=cell.max_tokens, created_at=created)
                    if recs:
                        cid = recs[0]["conversation_id"]
                        if cid in seen_rollouts:   # a re-run row, or a second epoch of one sample
                            self._duplicate_rows[cell.key] = self._duplicate_rows.get(cell.key, 0) + 1
                        seen_rollouts.add(cid)
                    for rec in recs:
                        rid = rec["record_id"]
                        if rid not in self._light:      # order holds each id once; the row read last wins
                            self._order.append(rid)
                        self._light[rid] = rec
            self._session = None
        return changed

    # ---- SessionStore interface -----------------------------------------
    def records(self) -> list[dict]:
        self.refresh()
        return [self._full.get(rid) or self._light[rid] for rid in self._order]

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
        # messages_in / feedback: Task 5
        rec["warnings"] = warnings
        rec["tokenised"] = True
        self._full[record_id] = rec
        self._session = None          # cell table counts changed
        return rec

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
                              "n_duplicate_rows": self._duplicate_rows.get(c.key, 0),
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
