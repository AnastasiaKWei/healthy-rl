"""FastAPI application: JSON routes plus SSE for chat sends and task attempts.

Every number the page draws is computed here through ``stats``, so the readout
conventions (single-token cosine at one capture layer, ``start`` read at the
prefill row, non-finite turns skipped and counted) hold for the transcript, the
token strip and the aggregate alike.

Two rules the routes enforce rather than trust the caller with:

- ``conflicting`` and ``original`` are never pooled into one aggregate (400).
- turns that hit the token cap are dropped from an ``end`` readout unless the
  caller asks for them, because a truncated turn's last token is where the
  generator ran out of budget, not where it finished.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from healthy_rl.dashboard import stats
from healthy_rl.dashboard.chat import ChatSession
from healthy_rl.dashboard.generation import context_text_of
from healthy_rl.dashboard.store import SessionStore
from healthy_rl.dashboard.tasks import TaskConfig, TaskRun
from healthy_rl.rollouts import (MINDSET_KEY, SCRATCHPAD_SYSTEM_PROMPT, Vectors, mindset_for,
                                 sort_task_ids)

STATIC = Path(__file__).parent / "static"
NO_HEALTH = {"ok": True, "last_ok_at": None, "last_error": None}


@dataclass(frozen=True)
class VectorsMeta:
    """Enough of a ``Vectors`` to read stored projections: order, layers, probe layer.

    A rollouts session has no single loaded ``Vectors``: each cell was captured
    with its own directions, layers and probe layer, and the numbers already
    live in the record. This carries just the fields the readouts need, per
    record, so nothing is read at one model's probe layer and labelled with
    another's emotion order.
    """
    emotions: tuple[str, ...]
    capture_layers: tuple[int, ...]
    probe_layer: int | None

    def layer_index(self, layer: int) -> int:
        return self.capture_layers.index(layer)


class HealthMonitor:
    """Background poll of the served model's ``/health``, for the top-bar chip."""

    def __init__(self, base_url: str, interval_s: float = 5.0) -> None:
        self.base_url, self.interval_s = base_url.rstrip("/"), interval_s
        self._status = {"ok": False, "last_ok_at": None, "last_error": "not polled yet"}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def poll_once(self) -> None:
        import urllib.request
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=4) as resp:
                ok = 200 <= resp.status < 300
                detail = None if ok else f"HTTP {resp.status}"
            self._status = {"ok": ok, "last_ok_at": time.strftime("%H:%M:%S") if ok else self._status["last_ok_at"],
                            "last_error": detail}
        except Exception as exc:
            self._status = {**self._status, "ok": False, "last_error": f"{type(exc).__name__}: {exc}"}

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.poll_once()
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> dict:
        return dict(self._status)


@dataclass
class AppState:
    engine: Any
    sandbox: Any
    store: SessionStore
    vectors: Vectors | None
    cfg: dict
    health: HealthMonitor | None = None
    read_only: bool = False
    mode: str = "live"
    job: dict | None = None
    chats: dict[str, ChatSession] = field(default_factory=dict)
    tasks: dict[str, TaskRun] = field(default_factory=dict)
    problems_cache: dict = field(default_factory=dict)

    def problems(self, split: str, affect: bool, mindset: Sequence[str] = ()) -> dict:
        # The mindset arm is part of the key: each arm has its own turn-1 text, and
        # one entry for all of them would serve the wrong instruction.
        key = (split, affect, tuple(mindset))
        if key not in self.problems_cache:
            self.problems_cache[key] = self.sandbox.problems(split, affect=affect, mindset=tuple(mindset))
        return self.problems_cache[key]


def _json_default(v):
    if isinstance(v, (np.floating, np.integer)):
        return None if (isinstance(v, np.floating) and not np.isfinite(v)) else v.item()
    if isinstance(v, np.ndarray):
        return _clean(v)
    return str(v)


def _clean(x):
    """NaN -> None recursively so JSON stays valid.

    ``json.dumps`` writes bare ``NaN``, which no browser will parse. Everything
    leaving this module goes through here first, and ``allow_nan=False`` below
    turns anything that slips past into a loud failure rather than a page that
    silently shows nothing.
    """
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def _sse(events: Iterator[dict]) -> StreamingResponse:
    def gen():
        for ev in events:
            payload = json.dumps(_clean(ev["data"]), default=_json_default, allow_nan=False)
            yield f"event: {ev['event']}\ndata: {payload}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _drain_until(q: queue.Queue, terminal: set[str]) -> Iterator[dict]:
    while True:
        ev = q.get()
        yield ev
        if ev["event"] in terminal:
            return


def _pump(make_events: Callable[[], Iterator[dict]]) -> Iterator[dict]:
    """Run a generator of events on a worker thread and relay it through a queue.

    ``ChatSession.send`` appends the record when its generator completes, so a
    browser that closes the stream mid-turn would abandon the generator and lose
    the turn. Draining it on a thread that nothing else can stop means the
    record lands whatever the client does.
    """
    q: queue.Queue = queue.Queue()

    def work():
        try:
            for ev in make_events():
                q.put(ev)
        except BaseException as exc:  # a raising generator must not hang the stream
            q.put({"event": "error", "data": {"message": f"{type(exc).__name__}: {exc}"}})
        finally:
            q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while True:
        ev = q.get()
        if ev is None:
            return
        yield ev


def _queued_first(data: dict, events: Iterator[dict]) -> Iterator[dict]:
    """Announce the conversation id before any generation, so the page can attach."""
    yield {"event": "queued", "data": data}
    yield from events


def _emotion_order_mismatch(rec: dict, meta: Vectors | VectorsMeta) -> bool:
    """Does this record's direction order disagree with the reference order?

    A record's ``proj`` columns are in the order of the ``emotions`` list that
    was live when it was written. Plotting them against a different order
    relabels every column silently -- joy drawn as despair -- so the dashboard
    checks the same way the rollout analysis does (docs/runs.md) and refuses.
    """
    return list(rec.get("emotions") or []) != list(meta.emotions)


def _readouts_for(rec: dict, arrays: dict, meta: Vectors | VectorsMeta) -> dict:
    """All four readouts x all emotions for one turn; None where unavailable."""
    none_for_all = {e: {r: None for r in stats.READOUTS} for e in meta.emotions}
    # A rollout cell whose probe layer was not captured (or is unknown) has no
    # column to read: better an empty readout than one taken at another layer.
    if meta.probe_layer is None or meta.probe_layer not in meta.capture_layers:
        return none_for_all
    li = meta.layer_index(meta.probe_layer)
    out: dict[str, dict[str, float | None]] = {e: {} for e in meta.emotions}
    if _emotion_order_mismatch(rec, meta):
        return none_for_all
    for readout in stats.READOUTS:
        v = _readout_or_none(stats.turn_readout, proj=arrays["proj"], norm=arrays["norm"],
                             proj_prefill=arrays["proj_prefill"], norm_prefill=arrays["norm_prefill"],
                             proj_end=arrays.get("proj_end"), norm_end=arrays.get("norm_end"),
                             token_kind=rec.get("token_kind", []), layer_index=li, readout=readout)
        for i, e in enumerate(meta.emotions):
            out[e][readout] = None if v is None else float(v[i])
    return out


def _readout_or_none(fn, **kw):
    """``stats`` refuses a ``token_kind`` that does not match the decode run.

    That is right -- a mis-sized ``token_kind`` mis-locates every position -- but
    it is a record-level defect (``Generation.misaligned``), not a request-level
    one, so the turn is unreadable rather than the request being a 500.
    """
    try:
        return fn(**kw)
    except ValueError:
        return None


def create_app(state: AppState) -> FastAPI:
    app = FastAPI(title="Affect Scope")
    st = state
    V = st.vectors            # None in rollouts mode
    ROLL = st.mode == "rollouts"

    def _health() -> dict:
        return st.health.status() if st.health else dict(NO_HEALTH)

    def _meta(rec: dict | None = None) -> Vectors | VectorsMeta:
        """The direction metadata to read ``rec`` with: the record's own in rollouts mode.

        The reference emotion order is the *model's* (the vectors artifact when
        there is one, else the first row seen), so a record whose own order
        disagrees is caught by ``_emotion_order_mismatch`` rather than silently
        relabelled. Layers and probe layer come from the record itself.
        """
        if not ROLL:
            return V
        models = st.store.session.get("models", {})
        m = models.get(rec["model"], {}) if rec else {}
        emotions = m.get("emotions") or (rec or {}).get("emotions") or []
        layers = (rec or {}).get("capture_layers", m.get("capture_layers", []))
        return VectorsMeta(tuple(emotions), tuple(int(l) for l in layers),
                           (rec or {}).get("probe_layer", m.get("probe_layer")))

    def _first_meta() -> Vectors | VectorsMeta:
        """Session-level emotions/layers for the page's boot: the first model's in rollouts mode."""
        if not ROLL:
            return V
        first = next(iter(st.store.session.get("models", {}).values()), {})
        return VectorsMeta(tuple(first.get("emotions", [])), tuple(first.get("capture_layers", [])),
                           first.get("probe_layer"))

    def _known_emotions() -> list[str]:
        """Every direction name the session can serve; the union across models in rollouts mode."""
        if not ROLL:
            return list(V.emotions)
        names = [e for m in st.store.session.get("models", {}).values() for e in m.get("emotions", [])]
        return list(dict.fromkeys(names))

    def _layer(layer: int | None, rec: dict | None = None) -> int:
        meta = _meta(rec)
        layer = meta.probe_layer if layer is None else layer
        if layer not in meta.capture_layers:
            raise HTTPException(400, f"layer must be one of {list(meta.capture_layers)}"
                                     + (f" for {rec['model']}" if rec and ROLL else ""))
        return layer

    def _writable() -> None:
        if st.read_only or ROLL:
            raise HTTPException(409, "replay session is read-only")

    def _split(split: str) -> str:
        """Reject an unknown split here rather than let the sandbox 500 on it."""
        mapping = getattr(st.sandbox, "split_parquets", None)
        known = tuple(mapping) if mapping else ("conflicting", "original")
        if split not in known:
            raise HTTPException(400, f"split must be one of {sorted(known)}")
        return split

    def _mindset(raw: Any) -> tuple[str, ...]:
        """Validated block names, in MINDSET order. Unknown ones are a 400 here.

        Left to the sandbox, a typo would come back as a nonzero exit from inside
        the container, which reads as a broken dashboard rather than a bad request.
        """
        try:
            return mindset_for({MINDSET_KEY: raw})
        except KeyError as exc:
            raise HTTPException(400, str(exc.args[0])) from None
        except TypeError as exc:
            raise HTTPException(400, f"bad mindset: {exc}") from None

    def _problems(split: str, affect: bool, mindset: Sequence[str] = ()) -> dict:
        # A replay session has no sandbox, and the problem list only exists inside
        # one. Without this the attribute error on ``None`` surfaces as a 500,
        # which reads as a broken dashboard rather than a read-only one.
        if st.sandbox is None:
            raise HTTPException(409, "replay session is read-only")
        try:
            return st.problems(_split(split), affect, _mindset(mindset))
        except ValueError as exc:  # a sandbox that knows its own splits better than we do
            raise HTTPException(400, str(exc)) from None

    def _rehydrate_chat(cid: str) -> ChatSession:
        """Rebuild a ChatSession for a conversation whose session object is gone.

        Without this a send to an unknown id starts a *fresh* session under that
        id: the conversation grows a second turn 0 with no history, and a send
        aimed at a task conversation quietly writes a chat record into it.
        History is replayed from the last record's ``messages_in`` plus its own
        reply, which is exactly what the next turn's prompt would have been.
        """
        recs = [r for r in st.store.records() if r["conversation_id"] == cid]
        if not recs:
            raise HTTPException(404, f"no conversation {cid}")
        if any(r.get("source") != "chat" for r in recs):
            raise HTTPException(409, "conversation is not a chat")
        last = recs[-1]
        cond = last.get("condition") or {}
        chat = ChatSession(st.engine, st.store, V, conversation_id=cid, title=recs[0].get("title"),
                           max_tokens=int(cond.get("max_tokens", st.cfg.get("max_tokens", 2048))),
                           temperature=float(cond.get("temperature", st.cfg.get("temperature", 0.0))))
        chat.messages = ([dict(m) for m in last.get("messages_in", [])]
                         + [{"role": "assistant", "content": context_text_of(last)}])
        chat.turn = len(recs)
        chat._non_empty = sum(1 for r in recs if r.get("n_generated", 0) > 0)
        st.chats[cid] = chat
        return chat

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/session")
    def session():
        meta = _first_meta()
        return _clean({"session": st.store.session, "health": _health(), "read_only": st.read_only,
                       "mode": st.mode, "job": st.job or {}, "emotions": meta.emotions,
                       "capture_layers": meta.capture_layers, "probe_layer": meta.probe_layer,
                       "n_records": len(st.store.records()), "defaults": st.cfg,
                       "records_dir": str(st.store.root)})

    @app.get("/api/health")
    def health():
        return _health()

    @app.get("/api/conversations")
    def conversations(model: list[str] = Query(default=[]), version: list[str] = Query(default=[])):
        live = {cid: t.state for cid, t in st.tasks.items()}
        convs = st.store.conversations()
        if model:
            convs = [c for c in convs if c.get("model") in model]
        if version:
            convs = [c for c in convs if c.get("version") in version]
        for c in convs:
            c["state"] = live.get(c["conversation_id"], "done")
        return _clean({"conversations": convs})

    # A rollout conversation id is "model/version/task/sN", so the id spans path segments.
    @app.get("/api/conversations/{cid:path}")
    def conversation(cid: str, emotion: str | None = None, readout: str = "start"):
        known = _known_emotions()
        emotion = (known[0] if known else None) if emotion is None else emotion
        if emotion not in known:
            raise HTTPException(400, f"emotion must be one of {known}")
        if readout not in stats.READOUTS:
            raise HTTPException(400, f"readout must be one of {list(stats.READOUTS)}")
        recs = [r for r in st.store.records() if r["conversation_id"] == cid]
        if not recs:
            raise HTTPException(404, f"no conversation {cid}")
        if ROLL:
            recs = [st.store.record(r["record_id"]) for r in recs]   # tokens, context, feedback
        turns = []
        for r in recs:
            # Read the arrays first: doing so can append a warning to the record and
            # mark it misaligned, and the copy below has to carry that, not precede it.
            arrays = st.store.arrays(r["record_id"])
            meta = _meta(r)
            t = {k: v for k, v in r.items() if k != "arrays"}
            t["readouts"] = _readouts_for(r, arrays, meta)
            t["emotion_order_mismatch"] = _emotion_order_mismatch(r, meta)
            turns.append(t)
        conv = next(c for c in st.store.conversations() if c["conversation_id"] == cid)
        conv["state"] = st.tasks[cid].state if cid in st.tasks else "done"
        conv["messages_in_last"] = recs[-1]["messages_in"]
        # Every emotion x readout is sent; these two say which pair the page opened on.
        return _clean({"conversation": conv, "turns": turns, "emotion": emotion, "readout": readout})

    @app.post("/api/chat/{cid}/send")
    async def chat_send(cid: str, request: Request):
        _writable()
        body = await request.json()
        text = str(body.get("text", "")).strip()
        if not text:
            raise HTTPException(400, "empty message")
        if cid == "new":
            try:
                chat = ChatSession(st.engine, st.store, V, title=body.get("title") or text[:40],
                                   system_prompt=SCRATCHPAD_SYSTEM_PROMPT if body.get("scratchpad") else None,
                                   max_tokens=int(body.get("max_tokens", st.cfg.get("max_tokens", 2048))),
                                   temperature=float(body.get("temperature", st.cfg.get("temperature", 0.0))))
            except (TypeError, ValueError) as exc:
                raise HTTPException(400, f"bad request body: {exc}") from None
            st.chats[chat.conversation_id] = chat
        else:
            chat = st.chats.get(cid) or _rehydrate_chat(cid)
        return _sse(_queued_first({"conversation_id": chat.conversation_id, "turn_index": chat.turn},
                                  _pump(lambda: chat.send(text))))

    @app.get("/api/problems")
    def problems(split: str = "conflicting", affect: bool = False,
                 mindset: list[str] = Query(default=[])):
        names = _mindset(mindset)
        probs = _problems(split, affect, names)
        items = [{"task_id": tid, "entry_point": p.get("entry_point"), "n_chars": len(p.get("input", ""))}
                 for tid, p in probs.items()]
        order = {t: i for i, t in enumerate(sort_task_ids([i["task_id"] for i in items]))}
        items.sort(key=lambda i: order[i["task_id"]])
        return {"split": split, "affect": affect, "mindset": list(names), "problems": items}

    @app.post("/api/task/start")
    async def task_start(request: Request):
        _writable()
        body = await request.json()
        try:
            cfg = TaskConfig(split=str(body["split"]), task_id=str(body["task_id"]),
                             attempts=int(body.get("attempts", st.cfg.get("max_attempts", 6))),
                             max_tokens=int(body.get("max_tokens", st.cfg.get("max_tokens", 2048))),
                             temperature=float(body.get("temperature", st.cfg.get("temperature", 0.0))),
                             scratchpad=bool(body.get("scratchpad", False)),
                             affect_prompt=bool(body.get("affect_prompt", False)),
                             mindset=_mindset(body.get("mindset", [])),
                             auto_continue=bool(body.get("auto_continue", False)))
        except KeyError as exc:
            raise HTTPException(400, f"missing {exc.args[0]!r}") from None
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, f"bad request body: {exc}") from None
        # Same affect flag and mindset arm as the run: both change the turn-1 text.
        probs = _problems(cfg.split, cfg.affect_prompt, cfg.mindset)
        if cfg.task_id not in probs:
            raise HTTPException(404, f"{cfg.task_id} not in the {cfg.split} split")
        run = TaskRun(cfg, probs[cfg.task_id], st.engine, st.sandbox, st.store, V)
        st.tasks[run.conversation_id] = run
        threading.Thread(target=run.run, daemon=True).start()
        return _sse(_queued_first({"conversation_id": run.conversation_id, "task_id": cfg.task_id, "split": cfg.split},
                                  _drain_until(run.events, {"awaiting_user", "done"})))

    @app.post("/api/task/{cid}/continue")
    async def task_continue(cid: str, request: Request):
        run = st.tasks.get(cid)
        if run is None:
            raise HTTPException(404, f"no live task {cid}")
        body = await request.json()
        # State is checked twice: once for the message, and once by resume() itself,
        # which is the only check that cannot lose a race with the run's own thread.
        if run.state != "awaiting_user" or not run.resume(body.get("intervention")):
            raise HTTPException(409, f"task is {run.state}, not awaiting_user")
        return _sse(_drain_until(run.events, {"awaiting_user", "done"}))

    @app.post("/api/task/{cid}/stop")
    def task_stop(cid: str):
        run = st.tasks.get(cid)
        if run is None:
            raise HTTPException(404, f"no live task {cid}")
        run.stop()
        return {"state": run.state}

    @app.get("/api/records/{rid:path}/tokens")
    def tokens(rid: str, layer: int | None = None, smooth: int = 1):
        try:
            rec = st.store.record(rid)
        except KeyError:
            raise HTTPException(404, f"no record {rid}")
        meta = _meta(rec)
        layer = _layer(layer, rec)
        arrays = st.store.arrays(rid)
        cos = stats.token_cosine(arrays["proj"], arrays["norm"], meta.layer_index(layer))
        if smooth > 1 and cos.size:
            cos = np.stack([stats.moving_mean(cos[:, e], smooth) for e in range(cos.shape[1])], axis=1)
        kinds = rec.get("token_kind", [])
        think = [i for i, k in enumerate(kinds) if k == "think"]
        answer = [i for i, k in enumerate(kinds) if k == "answer"]
        return _clean({"tokens": rec.get("tokens", []), "token_kind": kinds, "cosine": cos, "layer": layer,
                       "emotions": meta.emotions, "norm": arrays["norm"][:, meta.layer_index(layer)], "smooth": smooth,
                       "markers": {"think_end": think[-1] if think else None,
                                   "answer_start": answer[0] if answer else None}})

    @app.get("/api/aggregate")
    def aggregate(source: str = "task", split: str | None = None, position: str = "start", stat: str = "token",
                  segment: str = "all", include_cap: bool = False, layer: int | None = None):
        """By-turn series and paired last-minus-first delta over one source.

        ``position`` names the readout when ``stat="token"``; ``segment`` names
        the span when ``stat="mean"``. Each is inert under the other stat.

        ``include_cap`` only bites on the ``end`` *readout* (``stat="token"``,
        ``position="end"``), where a capped turn's last token is wherever the
        budget ran out rather than where the model finished. A turn mean over a
        segment is not an end readout and is never dropped for the cap (ruling,
        2026-08-15).

        It buys back less than it looks like. A capped turn's last token has no
        residual row at all -- ``assemble_generation`` pads an all-NaN one -- so
        including it moves the turn from ``excluded_cap`` into the skipped-and-
        counted column rather than producing a value. The only turn it actually
        recovers is one flagged ``at_cap`` that finished on ``stop`` exactly at
        the cap, which still has its last row.
        """
        if source not in ("task", "chat"):
            raise HTTPException(400, "source must be 'task' or 'chat'")
        if position not in stats.READOUTS or stat not in ("token", "mean") or segment not in stats.SEGMENTS:
            raise HTTPException(400, "bad position/stat/segment")
        if split is not None:
            _split(split)
        layer = _layer(layer)
        li = V.layer_index(layer)
        recs = [r for r in st.store.records() if r.get("source") == source]
        if source == "task":
            splits = {r.get("bench_split") for r in recs}
            if split is None and len(splits) > 1:
                raise HTTPException(400, "choose a split; conflicting and original cannot be pooled")
            if split is not None:
                recs = [r for r in recs if r.get("bench_split") == split]
        by_conv: dict[str, list] = {}
        for r in recs:
            by_conv.setdefault(r["conversation_id"], []).append(r)
        # The cap only contaminates the end readout: it is the position that moves to
        # wherever the budget ran out. A segment mean over the same turn is still fine.
        drop_cap = stat == "token" and position == "end" and not include_cap
        seqs, excluded = [], 0
        for rows in by_conv.values():
            rows.sort(key=lambda r: r["turn_index"])
            seq = []
            for r in rows:
                if r.get("n_generated", 0) == 0:
                    continue  # an empty turn holds no position among non-empty turns
                if drop_cap and r.get("at_cap"):
                    excluded += 1
                    seq.append(None)
                    continue
                if _emotion_order_mismatch(r, V):
                    seq.append(None)  # unreadable columns; skipped and counted, never relabelled
                    continue
                a = st.store.arrays(r["record_id"])
                if stat == "token":
                    v = _readout_or_none(stats.turn_readout, proj=a["proj"], norm=a["norm"],
                                         proj_prefill=a["proj_prefill"], norm_prefill=a["norm_prefill"],
                                         token_kind=r.get("token_kind", []), layer_index=li, readout=position)
                else:
                    v = _readout_or_none(stats.turn_mean, proj=a["proj"], norm=a["norm"],
                                         token_kind=r.get("token_kind", []), layer_index=li, segment=segment)
                seq.append(v)
            if seq:
                seqs.append(seq)
        E = len(V.emotions)
        return _clean({"emotions": V.emotions,
                       "by_turn": stats.by_turn_index(seqs, n_emotions=E),
                       "delta": stats.paired_delta(seqs, n_emotions=E),
                       "n_conversations": len(seqs), "n_records": len(recs), "excluded_cap": excluded,
                       "params": {"source": source, "split": split, "position": position, "stat": stat,
                                  "segment": segment, "include_cap": include_cap, "layer": layer}})

    return app
