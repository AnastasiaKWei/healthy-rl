#!/usr/bin/env python
"""Coherence check: chat-templated generation against a running vllm-lens server.

The smoke stage sends a raw ``/v1/completions`` prompt with no chat template and
only asserts the reply is non-empty, so a model that loads and hooks fine but
produces junk would still "pass". This asks a few questions through
``/v1/chat/completions`` and prints the replies for a human to read. It also
requests residual-stream capture on one chat request through vllm-lens's
``vllm_xargs`` path so the capture-with-chat-template route is exercised too.

Usage (via serve.slurm):
    --stage scripts/chat_sanity.py:configs/smoke.yaml
"""

from __future__ import annotations

import argparse
import json

import requests

from healthy_rl.server import LensClient, base_url_from_env

PROMPTS = [
    "What is 17 + 25? Answer with just the number.",
    "Write one sentence about a deadline that is about to be missed.",
    "Name the capital of France.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)  # accepted for serve.slurm compatibility
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    base_url = args.base_url or base_url_from_env()
    ok = True
    for p in PROMPTS:
        r = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": args.model,
                "messages": [{"role": "user", "content": p}],
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
            },
            timeout=600,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        print(f"Q: {p}\nA: {text!r}\n", flush=True)

    # capture through the lens client on a chat request
    client = LensClient(base_url, model=args.model)
    n_layers = None
    try:
        out = client.chat(
            [{"role": "user", "content": PROMPTS[0]}],
            max_tokens=8,
            temperature=0.0,
            capture_layers=[8, 16, 24],
        )
        acts = out.activations
        print("chat+capture text:", repr(out.text))
        rs = acts["residual_stream"] if isinstance(acts, dict) else acts
        print("captured residual_stream shape:", tuple(rs.shape), rs.dtype, flush=True)
    except Exception as exc:  # report, don't hide
        ok = False
        print("chat+capture FAILED:", type(exc).__name__, exc, flush=True)

    print(json.dumps({"model": args.model, "chat_capture_ok": ok}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
