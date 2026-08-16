"""Login-node entry point: run the dashboard against fakes, or read a past session or pilot
rollout cells read-only.

    python -m healthy_rl.dashboard --fake --port 8765
    python -m healthy_rl.dashboard --replay $ARTIFACT_DIR/dashboard/<model>/<jobid> --port 8765
    python -m healthy_rl.dashboard --rollouts $ARTIFACT_DIR/rollouts[/<model>[/<version>]] ... --port 8765

The GPU-backed stage is scripts/dashboard.py (run by slurm/serve.slurm).
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from healthy_rl.dashboard.app import AppState, create_app
from healthy_rl.dashboard.fake import FakeEngine, FakeSandbox
from healthy_rl.dashboard.rollout_store import RolloutStore
from healthy_rl.dashboard.store import SessionStore
from healthy_rl.rollouts import Vectors


def _vectors_from_session(session: dict) -> Vectors:
    """Enough of a ``Vectors`` to read stored projections back.

    A replay never projects anything: the records already hold ``proj``/``norm``,
    so all the app needs from ``Vectors`` is the emotion order, the capture
    layers and the probe layer. The directions themselves are dead weight (and
    the artifact they came from may not be on this filesystem), so they are
    zeros of shape ``(E, L, 1)``.
    """
    emotions = list(session["emotions"]); layers = [int(l) for l in session["capture_layers"]]
    return Vectors(directions=np.zeros((len(emotions), len(layers), 1), np.float32), emotions=emotions,
                   capture_layers=layers, probe_layer=int(session["probe_layer"]),
                   mean_residual_norm={l: 1.0 for l in layers}, path=Path(session.get("vectors_dir", "replay")))


def session_meta(vectors: Vectors, model: str, **extra) -> dict:
    """The ``session.json`` header. ``_vectors_from_session`` is its inverse."""
    return {"model": model, "emotions": list(vectors.emotions), "capture_layers": list(vectors.capture_layers),
            "probe_layer": vectors.probe_layer, "vectors_dir": str(vectors.path), **extra}


IGNORED_SHOWN = 8


def startup_report(store: RolloutStore) -> str:
    """The cell/model table printed when the dashboard opens rollout cells."""
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
        # The real rollouts root holds ~200 scratch directories; naming them all buries the table.
        shown, rest = s["ignored"][:IGNORED_SHOWN], len(s["ignored"]) - IGNORED_SHOWN
        lines.append("ignored (no rollouts*.jsonl): " + ", ".join(shown) + (f" (+{rest} more)" if rest > 0 else ""))
    return "\n".join(lines)


def build_state(*, fake: bool, replay: str | None, session_dir: str | None, vectors_dir: str | None, cfg: dict,
                rollouts: list[str] | None = None) -> AppState:
    if rollouts:
        try:
            store = RolloutStore.open(rollouts)
        except FileNotFoundError as exc:
            print(f"{exc}\nnothing to open: give a cell, a model directory, or the rollouts root", file=sys.stderr)
            raise SystemExit(2) from None
        return AppState(engine=None, sandbox=None, store=store, vectors=None, cfg=cfg, read_only=True, mode="rollouts")
    if replay:
        store = SessionStore.open(replay)
        return AppState(engine=None, sandbox=None, store=store, vectors=_vectors_from_session(store.session), cfg=cfg,
                        read_only=True, mode="replay")
    if fake:
        engine = FakeEngine()
        root = session_dir or (os.path.join(os.environ["ARTIFACT_DIR"], "dashboard", "fake", str(os.getpid()))
                               if os.environ.get("ARTIFACT_DIR") else tempfile.mkdtemp(prefix="affect-scope-fake-"))
        store = SessionStore.create(root, session_meta(engine.vectors, "fake-model", fake=True))
        return AppState(engine=engine, sandbox=FakeSandbox(pass_on_attempt=3), store=store, vectors=engine.vectors, cfg=cfg)
    raise SystemExit("one of --fake/--replay/--rollouts is required here; the GPU path is scripts/dashboard.py")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m healthy_rl.dashboard")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fake", action="store_true"); g.add_argument("--replay", metavar="DIR")
    g.add_argument("--rollouts", nargs="+", metavar="PATH")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--session-dir", default=None)
    args = ap.parse_args(argv)
    state = build_state(fake=args.fake, replay=args.replay, session_dir=args.session_dir, vectors_dir=None,
                        cfg={"max_tokens": 64, "max_attempts": 3, "temperature": 0.0}, rollouts=args.rollouts)
    import uvicorn
    kind = 'rollouts' if args.rollouts else 'replay' if args.replay else 'fake engine'
    print(f"Affect Scope on http://{args.host}:{args.port}  ({kind}; records: {state.store.root})", flush=True)
    if args.rollouts:
        print(startup_report(state.store), flush=True)
    uvicorn.run(create_app(state), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
