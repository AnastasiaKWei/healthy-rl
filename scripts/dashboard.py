#!/usr/bin/env python
"""Affect Scope stage: run the interactive dashboard next to the vllm-lens server.

Launched by slurm/serve.slurm:

    sbatch slurm/serve.slurm --model Ministral-3-14B-Reasoning-2512 \
        --config configs/dashboard.yaml --stage scripts/dashboard.py

Binds uvicorn on 0.0.0.0:<free port>, writes ``dashboard-endpoint`` beside the
vLLM ``endpoint`` file, prints the ssh tunnel command, and serves until the job
ends. ``--smoke`` runs one chat turn and one two-attempt task instead and exits.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

from healthy_rl.artifacts import artifact_dir
from healthy_rl.config import load_config, repo_root
from healthy_rl.dashboard.__main__ import session_meta
from healthy_rl.dashboard.app import AppState, HealthMonitor, create_app
from healthy_rl.dashboard.engine import Engine
from healthy_rl.dashboard.sandbox import Sandbox
from healthy_rl.dashboard.store import SessionStore
from healthy_rl.rollouts import Vectors, load_vectors, make_zstd_threadsafe
from healthy_rl.server import LensClient, base_url_from_env


def startup_checks(vectors_dir: Path) -> tuple[Vectors, dict]:
    """Load the vectors and make this process' zstd use safe under threads.

    Deviation from the spec: a reverted *file-level* zstd patch is recorded and
    printed as a WARNING rather than refused, because the in-memory shim makes
    this process safe either way. The flag lands in ``session.json`` and in the
    Settings tab, so a session recorded without the file patch says so.
    """
    vectors_dir = Path(vectors_dir)
    if not (vectors_dir / "vectors.json").is_file():
        raise SystemExit(
            f"vectors not found: {vectors_dir}/vectors.json (build them first; see docs/runs.md)"
        )
    vectors = load_vectors(vectors_dir)
    from vllm_lens._helpers import _serialize  # imports torch; GPU-node path only

    file_patch = type(getattr(_serialize, "_ZSTD_COMPRESSOR", None)).__name__ == "_PerCallZstd"
    make_zstd_threadsafe()
    if not file_patch:
        print(
            "WARNING: vllm-lens zstd file patch is NOT applied (uv sync reverts it); in-memory "
            "shim installed for this process. Re-run patches/vllm_lens_zstd_threadsafe.py.",
            file=sys.stderr, flush=True,
        )
    return vectors, {"zstd_file_patch_present": file_patch, "zstd_inmemory_shim": True}


def job_info(port: int | None = None, node: str | None = None) -> dict:
    """Job id, node, remaining walltime and the ssh tunnel line for the top bar."""
    jid = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    node = node or socket.gethostname().split(".")[0]
    time_left = None
    try:
        out = subprocess.run(["squeue", "-h", "-j", jid, "-o", "%L"],
                             capture_output=True, text=True, timeout=5)
        time_left = out.stdout.strip() or None
    except Exception:  # no slurm on this host, or squeue is wedged: not worth failing over
        pass
    login = os.environ.get("HEALTHY_RL_LOGIN_HOST", "<login-host>")
    return {"id": jid, "node": node, "time_left": time_left,
            "tunnel_cmd": f"ssh -L {port}:{node}:{port} {login}" if port else None}


def write_endpoint(model: str, job_id: str, host: str, port: int) -> Path:
    """``$ARTIFACT_DIR/serve/<model>/<job_id>/dashboard-endpoint`` <- ``host:port``."""
    d = Path(os.environ["ARTIFACT_DIR"]) / "serve" / model / job_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / "dashboard-endpoint"
    p.write_text(f"{host}:{port}\n")
    return p


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("0.0.0.0", 0))
        return s.getsockname()[1]


def smoke(state: AppState) -> int:
    """One chat turn and one two-attempt task through the real engine and sandbox."""
    from fastapi.testclient import TestClient

    c = TestClient(create_app(state))
    ok, notes = True, {}
    with c.stream("POST", "/api/chat/new/send",
                  json={"text": "What is 17 + 25? Answer with just the number.", "max_tokens": 16}) as r:
        body = r.read().decode()
    notes["chat_turn_event"] = "event: turn" in body
    ok &= notes["chat_turn_event"]
    probs = c.get("/api/problems", params={"split": "original"}).json()["problems"]
    tid = probs[0]["task_id"]
    with c.stream("POST", "/api/task/start",
                  json={"split": "original", "task_id": tid, "attempts": 2,
                        "auto_continue": True, "max_tokens": 512}) as r:
        body = r.read().decode()
    notes["task_done_event"] = "event: done" in body
    ok &= notes["task_done_event"]
    recs = state.store.records()
    notes["n_records"] = len(recs)
    notes["misaligned"] = [r["record_id"] for r in recs if r.get("misaligned")]
    notes["errors"] = [r["error"] for r in recs if r.get("error")]
    ok &= len(recs) >= 2 and not notes["misaligned"]
    # Guarded: the JSON line below is the whole point of the gate, so a smoke that
    # recorded nothing must still print why rather than die on recs[0].
    start = None
    if recs:
        conv = c.get(f"/api/conversations/{recs[0]['conversation_id']}").json()
        start = conv["turns"][0]["readouts"][state.vectors.emotions[0]]["start"]
    notes["first_start_readout"] = start
    ok &= start is not None
    print(json.dumps({"smoke_ok": bool(ok), **notes}, default=str), flush=True)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--vectors-version", default="v1")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    dash = dict(cfg.get("dashboard") or {})
    base_url = args.base_url or base_url_from_env()
    vectors_dir = Path(dash.get("vectors_dir") or artifact_dir("vectors", args.model, args.vectors_version))
    vectors, checks = startup_checks(vectors_dir)
    client = LensClient(base_url, model=args.model)
    engine = Engine(client, vectors)
    root = Path(os.environ["ARTIFACT_DIR"])
    job = job_info()
    # bench_dir is the bench ROOT; split_parquets maps a split to its parquet
    # under it, because the two splits are fetched into separate directories.
    sandbox = Sandbox(sif=Path(os.environ.get("HEALTHY_RL_EVAL_SIF") or repo_root() / "apptainer/eval.sif"),
                      project_dir=Path(os.environ.get("PROJECT_DIR") or repo_root()),
                      bench_dir=Path(dash.get("bench_dir") or root / "bench"),
                      split_parquets=dash.get("split_parquets"),
                      scratch_dir=root / "dashboard" / ".scratch" / job["id"],
                      timeout_s=int(dash.get("sandbox_timeout_s", 30)))
    store = SessionStore.create(root / "dashboard" / args.model / job["id"],
                                session_meta(vectors, args.model, job=job, base_url=base_url,
                                             config=dash, **checks))
    health = HealthMonitor(base_url)
    health.poll_once()
    state = AppState(engine=engine, sandbox=sandbox, store=store, vectors=vectors, health=health, job=job,
                     cfg={"max_tokens": int(dash.get("max_tokens", 2048)),
                          "max_attempts": int(dash.get("max_attempts", 6)),
                          "temperature": float(dash.get("temperature", 0.0)),
                          "message_limit": int(dash.get("message_limit", 40))})
    if args.smoke:
        rc = smoke(state)
        store.close()
        return rc
    port = args.port or free_port()
    # AppState.job is this same dict, so /api/session sees the port and tunnel line.
    job.update(job_info(port=port))
    ep = write_endpoint(args.model, job["id"], job["node"], port)
    print(f"[dashboard] http://{job['node']}:{port}  endpoint file {ep}\n"
          f"[dashboard] tunnel: {job['tunnel_cmd']}  then open http://localhost:{port}", flush=True)
    health.start()
    import uvicorn
    try:
        uvicorn.run(create_app(state), host=args.host, port=port, log_level="warning")
    finally:
        health.stop()
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
