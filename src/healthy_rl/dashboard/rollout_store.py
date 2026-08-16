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
import functools
import json
import os
import threading
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
    p = Path(p).resolve()          # model/version are read off the path, so "." must be named first
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

    Deduped by ``(model, version)``, ordered by it. A directory is a cell when it
    holds ``rollouts*.jsonl``; a model is walked one level, a root two. Two
    directories that resolve to the same ``(model, version)`` -- the same cell
    staged under two roots -- cannot both be kept: the key is what every record
    resolves its npz and its ``.eval`` through, so the second would read the
    first's arrays under its own label. The first wins; the later path is
    ignored, and the startup line names it.
    """
    found: dict[tuple[str, str], Cell] = {}
    seen: set[Path] = set()
    ignored: list[Path] = []

    def visit(p: Path, depth: int) -> None:
        if _is_cell(p):
            rp = p.resolve()
            if rp in seen:              # the same directory reached twice (a root and the cell itself)
                return
            seen.add(rp)
            cell = _cell_at(p)
            if cell.key in found:
                ignored.append(rp)
            else:
                found[cell.key] = cell
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


def _same_light(a: dict | None, b: dict | None) -> bool:
    """Did a re-read leave this light record alone? ``created_at`` does not count: it is the
    shard file's mtime, which moves for every row of the file each time the file grows."""
    if a is None or b is None:
        return a is b
    return {k: v for k, v in a.items() if k != "created_at"} == {k: v for k, v in b.items() if k != "created_at"}


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
    # A steering sweep re-runs one (task, sample) once per condition; without the condition in
    # the id the rows collapse onto each other last-writer-wins (Olmo-3.1-32B-Think/v1: 172 rows
    # over 36 pairs). "readout" is the unsteered arm every ordinary cell writes, so it stays bare.
    condition = row.get("condition_name")
    if condition not in (None, "", "readout"):
        cid += f"/c{condition}"
    base = {
        "conversation_id": cid, "source": SOURCE, "model": model, "version": version,
        "mindset": list(row.get("mindset") or []), "mindset_version": int(row.get("mindset_version") or 0),
        "mindset_hash": str(row.get("mindset_hash") or ""),
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
            s, e = int(s), int(e)
            if e <= prev:            # a special token's (0, 0) span keeps its place in the strip
                s = max(s, prev)
                e = prev
            if i == len(spans) - 1:
                e = max(e, len(text))
            tokens.append(text[prev:e]); starts.append(s)
            prev = e
        if prev < len(text):         # no spans at all (whitespace-only text): keep the tiling
            tokens.append(text[prev:]); starts.append(prev)
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


def _project_residual(h: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, float]:
    """``(proj (E,), norm)`` of one residual on the probe-layer directions; NaN when non-finite."""
    h = np.asarray(h, dtype=np.float64)
    if not np.isfinite(h).all():
        return np.full(directions.shape[0], np.nan), np.nan
    n = float(np.linalg.norm(h))
    return directions @ h, n


def arrays_from_npz(z, *, turn: int, capture_layers: list[int], probe_layer: int, n_emotions: int,
                    emotions: list[str] | None = None, vectors=None) -> tuple[dict[str, np.ndarray], list[str]]:
    """Dashboard-shaped arrays for one turn of a rollout npz (spec §3.2, §3.4)."""
    L, files = len(capture_layers), set(z.files)
    problems: list[str] = []
    E = int(n_emotions)
    if vectors is not None:
        # The artifact names the columns the page labels, so a disagreement with the record is a
        # problem for the record, not something to paper over by reordering or relabelling columns.
        mismatch: list[str] = []
        if vectors.probe_layer != probe_layer:
            mismatch.append(f"vectors probe layer L{vectors.probe_layer} differs from the record's L{probe_layer}")
        vem = list(vectors.emotions)
        if len(vem) != E:
            mismatch.append(f"vectors list {len(vem)} emotions, record lists {E}")
        elif emotions is not None and list(emotions) != vem:
            mismatch.append("vectors emotion order differs from the record's")
        if mismatch:
            problems += mismatch
            vectors = None                     # never project with an artifact that disagrees
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


def sample_messages(samples: list[dict], task_id: str, completions: list[str], epoch: int | None = None
                    ) -> tuple[list[dict] | None, str | None]:
    """``(messages, rule)`` for the sample that produced these completions.

    Several samples share a task id at Inspect epoch 1 (resumed shards restart the
    numbering), so the id alone is ambiguous; the completion text is not -- the
    ``.eval``'s assistant messages equal ``turn_completion`` verbatim. Turns that
    generated nothing wrote no assistant message, so both sides drop the empties.

    Cells written before the mindset merge (2026-08-16) store no ``turn_completion``
    at all, so there is no text to match on for any of their rollouts. There the
    ``.eval``'s ``epoch`` is tried instead (``epoch == sample + 1``), and only when
    exactly one candidate carries it. ``rule`` names which of the three matched
    (``"completion"``, ``"epoch"``, ``"id"``) so the caller can say so on the page.
    """
    want = [c for c in completions if c]
    cands = [s for s in samples if str(s.get("id")) == str(task_id)]
    if not want:
        # No text to be matched on, but the prompt (and, for an old cell, the completions
        # themselves) are still worth recovering: the epoch first, then an unambiguous id.
        if epoch is not None:
            by_epoch = [s for s in cands if int(s.get("epoch") or 1) == int(epoch)]
            if len(by_epoch) == 1:
                return list(by_epoch[0]["messages"]), "epoch"
        return (list(cands[0]["messages"]), "id") if len(cands) == 1 else (None, None)
    for s in cands:
        got = [m["content"] for m in s.get("messages", []) if m.get("role") == "assistant" and m["content"]]
        if got[:len(want)] == want:
            return list(s["messages"]), "completion"
    return None, None


def _locked(fn):
    """Run the method holding ``self._lock``. See ``RolloutStore``'s docstring for why."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kw):
        with self._lock:
            return fn(self, *args, **kw)
    return wrapper


class RolloutStore:
    """Read-only, ``SessionStore``-shaped view over rollout cells.

    Every entry point takes ``self._lock``. The dashboard's routes are sync, so
    FastAPI runs them in a threadpool and several requests share one store, whose
    caches (``_light``/``_order``/``_full``, the tokenizers, vectors and ``.eval``
    samples) are mutable state that ``refresh()`` rebuilds in place: concurrent
    ``/api/aggregate`` requests over four models produced a 500 with
    ``RuntimeError: dictionary changed size during iteration`` in ``refresh()``.
    The lock is an ``RLock`` because these methods call each other (``record()``
    calls ``refresh()``, ``arrays()`` calls ``record()``). It is coarse: tokenising
    one record (~ms per turn) blocks the other requests, and the first tokenizer
    load for a model (~20 s) blocks them once -- acceptable for a single-user tool,
    and far cheaper than a torn read.
    """

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
        self._evals: dict[Path, list[dict] | Exception] = {}     # .eval path -> samples, or why not
        self._hta: dict[Path, bool] = {}                         # npz path -> any per-token arrays
        self._session: dict | None = None
        self._lock = threading.RLock()           # before refresh(): refresh() takes it
        self.refresh()

    @classmethod
    def open(cls, paths: Sequence[str | os.PathLike[str]], **kw) -> "RolloutStore":
        cells, ignored = discover_cells(paths)
        if not cells:
            raise FileNotFoundError(f"no rollout cells under {[str(p) for p in paths]}")
        return cls(cells, ignored, [Path(p) for p in paths], **kw)

    # ---- growth ---------------------------------------------------------
    @_locked
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
            prev_light = dict(self._light)                # to spot rows a re-read rewrote
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
            # a rewritten (or vanished) row invalidates its tokenised record; the rest stay cached
            for rid in [r for r in list(self._full) if not _same_light(self._light.get(r), prev_light.get(r))]:
                del self._full[rid]
            self._hta.clear()         # a growing npz can gain per-token arrays between reads
            self._session = None
        return changed

    # ---- SessionStore interface -----------------------------------------
    @_locked
    def records(self) -> list[dict]:
        self.refresh()
        return [self._full.get(rid) or self._light[rid] for rid in self._order]

    @_locked
    def record(self, record_id: str) -> dict:
        self.refresh()                    # first: a re-read row drops its stale tokenised record
        full = self._full.get(record_id)
        if full is not None:
            return full
        light = self._light.get(record_id)
        if light is None:
            raise KeyError(record_id)
        rec = dict(light)
        warnings = list(rec.get("warnings") or [])
        n_decode, problem = self._decode_rows(rec)
        rec["n_decode"] = n_decode
        rec["has_token_arrays"] = n_decode is not None
        # The .eval is read before the text is tokenised: an old cell stores no turn_completion,
        # so the text being tokenised may be the one recovered from the log just below.
        msgs, why, rule = self._messages_for(rec)
        t = rec["turn_index"]
        rec["messages_in"], rec["feedback"], rec["text_source"] = [], None, "record"
        if msgs is None:
            warnings.append(f"transcript context unavailable: {why}")
        else:
            if rule == "epoch":
                warnings.append("context matched by epoch: the record stores no completion text")
            # A turn that generated nothing wrote no assistant message, so the .eval's assistant
            # messages line up with the *non-empty* turns, not with turn_index.
            turns = self._conversation_records(rec)
            idx = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
            n_non_empty = sum(1 for r in turns if r["n_generated"] > 0)
            if len(idx) != n_non_empty:
                warnings.append(f".eval has {len(idx)} assistant messages, "
                                f"record has {n_non_empty} non-empty turns")
            k = rec["non_empty_turn_index"]
            if k is None:
                # an empty turn: its context is everything up to the next assistant message there is
                k = sum(1 for r in turns[:t] if r["n_generated"] > 0)
                rec["messages_in"] = [dict(m) for m in msgs[:idx[k] if k < len(idx) else len(msgs)]]
            elif k < len(idx):
                i = idx[k]
                rec["messages_in"] = [dict(m) for m in msgs[:i]]
                nxt = msgs[i + 1] if i + 1 < len(msgs) else None
                rec["feedback"] = nxt["content"] if nxt is not None and nxt.get("role") == "user" else None
                text = msgs[i].get("content") or ""
                if not rec["text"] and rec["n_generated"] > 0 and text:
                    # An old cell wrote the token count but not the text. The .eval has it, so the
                    # bubble is filled from there and labelled: this is the log's copy, not the row's.
                    rec["text"] = text
                    rec["reasoning"], rec["answer"], _ = split_reasoning(text)
                    rec["text_source"] = "eval"
                    warnings.append("completion text taken from the .eval log (not stored in the record)")
        tok = self._tokenizer_for(rec["model"])
        if tok is None:
            rec.update(tokens=[], token_kind=[], n_think=0, misaligned=True, error=f"no tokenizer for {rec['model']}")
            warnings.append("tokenizer missing: text and turn readouts only")
        else:
            # Recovered text is never aligned against arrays: an old cell has none, so n_decode is
            # None and align_tokens leaves the record aligned whatever the re-tokenised count is.
            _, _, think_end = split_reasoning(rec["text"])
            toks, starts = tokenise(rec["text"], tok)
            toks, kinds, mis, err = align_tokens(toks, starts, think_end, n_decode)
            rec.update(tokens=toks, token_kind=kinds, n_think=sum(1 for k in kinds if k == "think"),
                       misaligned=mis, error=err)
        if problem is not None:
            # the text and the turn readouts are still good; only the token strip is not.
            # A model with no tokenizer already put its own error here: keep both.
            prev = rec.get("error")
            rec.update(misaligned=True, has_token_arrays=False,
                       error=problem if not prev else f"{prev}; {problem}")
        last = t == rec["n_turns_total"] - 1
        # a turn that drew feedback failed the tests; only the last turn's verdict is the rollout's
        rec["passed"] = False if rec["feedback"] is not None else (rec["passed"] if last else None)
        rec["warnings"] = warnings
        rec["tokenised"] = True
        self._full[record_id] = rec
        self._session = None          # cell table counts changed
        return rec

    @_locked
    def arrays(self, record_id: str) -> dict[str, np.ndarray]:
        rec = self.record(record_id)
        L, E = len(rec["capture_layers"]), len(rec["emotions"])
        empty = {"proj": np.zeros((0, L, E), np.float32), "norm": np.zeros((0, L), np.float32),
                 "proj_prefill": np.full((L, E), np.nan, np.float32), "norm_prefill": np.full(L, np.nan, np.float32)}
        path = self._npz_path(rec)
        if path is None:
            return empty       # the row names no residuals file: array-less, like _decode_rows reads it
        if not path.is_file():
            self._mark(rec, f"npz missing: {path}")
            return empty
        vec = self._vectors_for(rec["model"])
        try:
            with np.load(path) as z:
                out, problems = arrays_from_npz(z, turn=rec["turn_index"], capture_layers=rec["capture_layers"],
                                                probe_layer=rec["probe_layer"], n_emotions=E,
                                                emotions=rec["emotions"], vectors=vec)
        except Exception as exc:          # BadZipFile from a half-written npz has no narrower base
            self._mark(rec, f"npz unreadable: {exc}")
            return empty
        if problems:
            self._mark(rec, "; ".join(problems))
            return empty
        if out["proj"].shape[0] == 0 and rec["n_generated"] > 0 and vec is None and not rec["has_token_arrays"]:
            if "vectors missing: start/end readouts unavailable for this cell" not in rec["warnings"]:
                rec["warnings"].append("vectors missing: start/end readouts unavailable for this cell")
        return out

    @_locked
    def _mark(self, rec: dict, error: str) -> None:
        """Flag a full record misaligned; the page hides the strip and readouts go None.

        ``record()`` may already have reported the same problem (a missing npz is seen
        by ``_decode_rows`` first), so an error string that is already there is not
        appended a second time.
        """
        rec["misaligned"] = True
        prev = rec.get("error")
        if not prev:
            rec["error"] = error
        elif error not in prev:
            rec["error"] = prev + "; " + error
        self._session = None

    @_locked
    def conversations(self) -> list[dict]:
        out: dict[str, dict] = {}
        for r in self.records():
            cid = r["conversation_id"]
            c = out.get(cid)
            if c is None:
                c = out[cid] = {"conversation_id": cid, "source": SOURCE, "model": r["model"], "version": r["version"],
                                "bench_split": r["bench_split"], "task_id": r["task_id"], "sample": r["sample"],
                                "epoch": r["epoch"], "mindset": r["mindset"], "title": None,
                                "condition_name": r["condition_name"],
                                "passed": self._light[r["record_id"]]["passed"],  # per-turn on a full record
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
    @_locked
    def _has_token_arrays(self, rec: dict) -> bool:
        """Does *any* turn of this rollout carry per-token projections (``t*_proj_L*``)?

        Rollout-level, and cached per npz path: it answers the conversation and cell
        tables. The record-level ``has_token_arrays`` is the stricter per-turn question
        -- see ``_decode_rows``.
        """
        cache = self._hta
        rel = rec.get("residuals")
        if not rel:
            return False
        path = self._cell_of(rec).path / rel
        if path not in cache:
            try:
                with np.load(path) as z:
                    cache[path] = any(k.startswith("t") and "_proj_L" in k for k in z.files)
            except Exception:                 # a truncated npz is not "no arrays" to the page, but
                cache[path] = False           # it must not take /api/session down either
        return cache[path]

    def _npz_path(self, rec: dict) -> Path | None:
        rel = rec.get("residuals")
        return (self._cell_of(rec).path / rel) if rel else None

    def _decode_rows(self, rec: dict) -> tuple[int | None, str | None]:
        """``(decode rows at the probe layer, problem)`` for this one turn.

        A turn has per-token arrays when *both* ``t{t}_proj_L{probe}`` and
        ``t{t}_kind_L{probe}`` are in the npz -- that pair is what the record-level
        ``has_token_arrays`` means. ``(None, None)`` is the honest array-less turn
        (a zero-token turn, or an old cell that only wrote the turn readouts): the
        record stays aligned and the token strip simply has no per-token data.
        Everything that should have worked and did not comes back as a problem
        string, so it reaches the page instead of being swallowed.
        """
        path = self._npz_path(rec)
        if path is None:
            return None, None
        proj = f"t{rec['turn_index']}_proj_L{rec['probe_layer']}"
        kind = f"t{rec['turn_index']}_kind_L{rec['probe_layer']}"
        try:
            with np.load(path) as z:
                files = set(z.files)
                if proj not in files and kind not in files:
                    return None, None
                if kind not in files:
                    return None, f"npz has {proj} without {kind}"
                if proj not in files:
                    return None, f"npz has {kind} without {proj}"
                return int((np.asarray(z[kind]).reshape(-1) == 0).sum()), None
        except FileNotFoundError:
            return None, f"npz missing: {path}"     # same wording as arrays() so the page dedupes
        except Exception as exc:              # BadZipFile, EOFError, UnpicklingError, ...
            return None, f"npz unreadable: {exc}"

    def _eval_files(self, rec: dict) -> list[Path]:
        """The cell's ``.eval`` logs for this rollout's shard, newest name first."""
        cell = self._cell_of(rec)
        shard = str(rec.get("shard") or "")
        sub = None
        if "/" in shard:
            a, b = shard.split("/", 1)
            if a.isdigit() and b.isdigit():
                sub = cell.path / "inspect-logs" / f"shard{a}of{b}"
        base = sub if sub is not None and sub.is_dir() else cell.path / "inspect-logs"
        return sorted(base.rglob("*.eval"), reverse=True) if base.is_dir() else []

    @_locked
    def _eval_samples(self, path: Path) -> list[dict]:
        """One parse per ``.eval``; an unreadable log is cached as its exception, not retried."""
        if path not in self._evals:
            try:
                self._evals[path] = self._eval_loader(path)
            except Exception as exc:          # unreadable log: no messages, and say so
                self._evals[path] = exc
        v = self._evals[path]
        if isinstance(v, Exception):
            raise v
        return v

    def _messages_for(self, rec: dict) -> tuple[list[dict] | None, str | None, str | None]:
        """(messages, warning, rule) -- the sample's whole message list, or None + why."""
        files = self._eval_files(rec)
        if not files:
            return None, "no .eval log under inspect-logs for this shard", None
        comps = self._completions_of(rec)
        errors = 0
        weak: list[tuple[list[dict], str]] = []
        for f in files:
            try:
                samples = self._eval_samples(f)
            except Exception:
                errors += 1
                continue
            m, rule = sample_messages(samples, rec["task_id"], comps, epoch=int(rec["sample"]) + 1)
            if m is None:
                continue
            if rule == "completion":
                return m, None, rule          # the text is its own proof: the first match is the match
            # The id and the epoch are only unique *within* one log. A steering sweep runs the
            # same shard once per condition and writes one log per run, so several of them hold a
            # sample with this id and epoch -- and nothing in the row says which run it came from.
            # Taking the first would put another arm's completion in this rollout's bubble.
            weak.append((m, rule))
            if len(weak) > 1:
                return None, (f"{len(files)} .eval logs in this shard hold a sample with this id"
                              " and epoch, and the record stores no completion text to tell them apart"), None
        if len(weak) == 1:
            return weak[0][0], None, weak[0][1]
        if errors == len(files):
            return None, "the .eval log(s) could not be read", None
        return None, "no .eval sample matches this rollout's completions", None

    def _conversation_records(self, rec: dict) -> list[dict]:
        """The light records of every turn of this rollout, in turn order."""
        cid = rec["conversation_id"]
        return [self._light[rid] for rid in self._order if self._light[rid]["conversation_id"] == cid]

    def _completions_of(self, rec: dict) -> list[str]:
        """This rollout's ``turn_completion``, in turn order."""
        return [r["text"] for r in self._conversation_records(rec)]

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

    @_locked
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

    @_locked
    def _vectors_for(self, model: str):
        if model not in self._vectors:
            try:
                self._vectors[model] = self._vectors_loader(model)
            except Exception:
                self._vectors[model] = None
        return self._vectors[model]

    @property
    @_locked
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
