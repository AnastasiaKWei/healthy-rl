#!/usr/bin/env python
"""Isolate the never-returning Qwen request: is it our hook, or vLLM?

Some Qwen3.5-9B rollout requests never return. The engine reports throughput
while GPU KV cache usage sits flat, no error appears on either side, and the
client waits forever. It is problem-specific: `lcbhard_7`, `10`, `11`, `4`
strand; the other eight readout problems do not. Ministral and Nemotron never
trip it. See docs/infrastructure.md.

Two candidate causes, and exactly one variable separates them:

    hooks OFF   plain /v1/chat/completions
    hooks ON    the same request carrying `apply_hooks`, i.e. the projection
                hook every rollout uses

If both hang, the fault is in vLLM or the model and our instrumentation is
innocent. If only the hooked one hangs, it is ours. Anything else (both fine)
means the trigger needs more than a single first turn to appear, which is itself
worth knowing because the observed hangs all froze at `Attempt 1/6`.

The prompt is rebuilt exactly as ImpossibleBench's `record_to_sample` does, from
the same parquet the rollouts read and the `instruction` string recorded verbatim
in every run's summary.json -- so this sends the real first turn, not a paraphrase.

Usage (via serve.slurm, so it runs against a real server):
    --stage scripts/diagnose_stuck_request.py::--tasks lcbhard_10,lcbhard_0
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import time
from pathlib import Path

import requests


def build_input(row, instruction: str) -> str:
    """Reproduce impossiblebench.record_to_sample's input_text exactly."""
    return (
        instruction
        + "\n\n```\n"
        + row["prompt"]
        + "\n\n"
        + row["test"]
        + f"\n\n# Use check({row['entry_point']}) to run tests.\n```"
    )


def instruction_for(artifact_root: str, model: str) -> str:
    """The exact instruction the neutral runs used, read from their summary.

    Taken from this model's `d6` run rather than hardcoded. `d6` is the arm with
    no affect prompt and no mindset block, so its `instruction` is the plain
    benchmark text. Reading it beats asserting a constant: the affect and mindset
    arms both legitimately rewrite this string, so a hardcoded value plus a
    "does anything differ?" guard fails on healthy runs -- which is exactly what
    happened twice before this was written.
    """
    import glob

    for path in sorted(glob.glob(f"{artifact_root}/rollouts/{model}/d6/summary*.json")):
        summary = json.load(open(path))
        if summary.get("affect_prompt") or summary.get("mindset"):
            continue
        recorded = summary.get("instruction")
        if recorded:
            return recorded
    raise SystemExit(
        f"no neutral d6 summary for {model} under {artifact_root}; cannot "
        "establish the instruction this diagnostic should send"
    )


def one_request(base_url: str, model: str, text: str, max_tokens: int,
                hooks, timeout_s: float) -> dict:
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
    }
    if hooks is not None:
        # The REAL production hook, and the REAL wire format. Rollouts send
        # extra_body["extra_args"], which healthy_rl.rollouts._transform_config
        # rewrites into vllm_xargs with each hook JSON-encoded as a string --
        # that rewrite is what the server plugin actually reads. Posting
        # extra_args straight through would be silently ignored and the "hooked"
        # arm would really be a second unhooked one.
        body["vllm_xargs"] = {"apply_hooks": json.dumps([hooks])}
    started = time.monotonic()
    try:
        r = requests.post(f"{base_url}/chat/completions", json=body, timeout=timeout_s)
        elapsed = time.monotonic() - started
        ok = r.status_code == 200
        n = 0
        if ok:
            data = r.json()
            n = (data.get("usage") or {}).get("completion_tokens", 0)
        if not ok:
            # Not a data point. A transport error counted as "did not hang"
            # turns a broken experiment into a clean-looking result.
            raise SystemExit(
                f"request failed with HTTP {r.status_code} -- the experiment is "
                f"invalid, not negative. Body: {r.text[:300]}"
            )
        return {"outcome": "ok", "seconds": round(elapsed, 1), "tokens": n}
    except requests.exceptions.Timeout:
        return {"outcome": "TIMEOUT", "seconds": round(time.monotonic() - started, 1),
                "tokens": 0}
    except Exception as exc:  # noqa: BLE001
        return {"outcome": f"{type(exc).__name__}",
                "seconds": round(time.monotonic() - started, 1), "tokens": 0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)  # serve.slurm compatibility
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--tasks", default="lcbhard_10,lcbhard_11,lcbhard_7,lcbhard_0,lcbhard_6",
                    help="comma-separated; include known-good ones as controls")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="requests in flight at once. Production runs 8 "
                         "(max_connections/max_num_seqs); at 1 the fault does "
                         "not reproduce, so this is the variable under test.")
    args = ap.parse_args()

    import pandas as pd

    from healthy_rl.server import base_url_from_env

    root = os.environ.get("HEALTHY_RL_ARTIFACT_ROOT") or os.environ.get("ARTIFACT_DIR")
    if not root:
        raise SystemExit("set ARTIFACT_DIR or HEALTHY_RL_ARTIFACT_ROOT")
    # MUST carry /v1: without it every request 404s with no useful message
    # (docs/infrastructure.md). The first run of this diagnostic 404'd ten times
    # and still printed "NO HANG REPRODUCED".
    from healthy_rl.rollouts import provider_base_url

    base_url = provider_base_url(args.base_url or base_url_from_env())
    frame = pd.read_parquet(f"{root}/bench/v1/conflicting.parquet").set_index("task_id")
    instruction = instruction_for(root, args.model)

    # The production hook, exactly as run_rollouts builds it.
    from safetensors.numpy import load_file

    from healthy_rl.rollouts import make_projection_hook

    meta = json.load(open(f"{root}/vectors/{args.model}/v1/vectors.json"))
    dirs = load_file(f"{root}/vectors/{args.model}/v1/vectors.safetensors")["directions"]
    capture = [int(x) for x in meta["capture_layers"]]
    probe = int(meta["probe_layer"])
    hook = make_projection_hook(dirs, capture, [probe]).model_dump()
    layers = capture

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"model={args.model} url={base_url} max_tokens={args.max_tokens} "
          f"timeout={args.timeout}s repeats={args.repeats} capture_layer={layers}")
    print(f"{'task':13s} {'hooks':6s} {'rep':>3s} {'outcome':>9s} {'secs':>7s} {'tok':>6s}")
    results = []
    for task in tasks:
        if task not in frame.index:
            print(f"{task:13s} -- not in parquet, skipping")
            continue
        text = build_input(frame.loc[task], instruction)
        for label, payload in (("no", None), ("yes", hook)):
            n = args.concurrency * args.repeats
            with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [pool.submit(one_request, base_url, args.model, text,
                                       args.max_tokens, payload, args.timeout)
                           for _ in range(n)]
                for rep, fut in enumerate(futures):
                    try:
                        r = fut.result()
                    except SystemExit as exc:
                        print(f"{task:13s} {label:6s} {rep:3d}  ABORT {exc}", flush=True)
                        raise
                    print(f"{task:13s} {label:6s} {rep:3d} {r['outcome']:>9s} "
                          f"{r['seconds']:7.1f} {r['tokens']:6d}", flush=True)
                    results.append({"task": task, "hooks": label == "yes",
                                    "rep": rep, **r})

    out = Path(os.environ.get("HEALTHY_RL_ARTIFACT_OUT") or root) / "stuck_request_diagnosis.json"
    out.write_text(json.dumps({"model": args.model, "max_tokens": args.max_tokens,
                               "results": results}, indent=2) + "\n")
    print(f"\nwrote {out}")

    timed_out = [r for r in results if r["outcome"] == "TIMEOUT"]
    if not timed_out:
        print("NO HANG REPRODUCED at this max_tokens — the trigger needs more than "
              "a single first turn, or a longer generation.")
    else:
        hooked = [r for r in timed_out if r["hooks"]]
        plain = [r for r in timed_out if not r["hooks"]]
        print(f"HANGS: {len(plain)} without hooks, {len(hooked)} with hooks")
        print("  both -> vLLM or the model, not our instrumentation"
              if plain and hooked else
              "  hooked only -> the projection hook is implicated"
              if hooked else
              "  plain only -> not the hook; the request itself stalls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
