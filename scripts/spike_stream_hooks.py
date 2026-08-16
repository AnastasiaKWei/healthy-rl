#!/usr/bin/env python
"""THROWAWAY SPIKE (dashboard task 11). Delete once docs/infrastructure.md records the answer.

This is not product code and nothing imports it. It exists to answer one
question the dashboard spec defers on: with ``stream: true`` on
``/v1/chat/completions``, can vllm-lens hook results be matched to the streamed
request, so that token-text streaming could be added later without losing the
per-token projections?

Three checks, run against a server already started by ``slurm/serve.slurm``:

A. **Per-request hooks on a streamed request.** Send one streamed chat request
   with the projection hook in ``vllm_xargs.apply_hooks`` and record whether any
   SSE chunk carries ``hook_results``. (vllm-lens 1.2.1 patches
   ``chat_completion_stream_generator`` to yield an extra ``data: {...}`` chunk
   with serialized hook results just before ``data: [DONE]``; whether that path
   actually fires end to end is what this measures.)

B. **Persistent hooks, matched by response id.** ``register_hooks`` the same
   hook, send a streamed request carrying no ``vllm_xargs``, then
   ``collect_hook_results`` and look for the streamed response ``id`` (the
   ``chatcmpl-...`` value in the SSE chunks) among the returned keys. vllm-lens
   keys results by the *internal* request id, and ``_worker_ext`` matches
   external to internal by an ``f"{external_id}-"`` prefix, so an exact key match
   is not the only useful outcome: a key of the form ``<id>-<suffix>`` is just as
   good for matching, and the verdict counts either. The notes record which.

C. **Non-streamed reference.** The same prompt, non-streamed, per-request hook --
   the path ``dashboard/engine.py`` already uses. Only here to say whether a row
   count seen in A/B is a streaming artefact or the normal convention.

``rows_match_tokens`` is true when at least one *streamed* path yielded decode
rows equal to that request's generated-token count; the notes give the raw
numbers for every path, which is the part worth reading -- an off-by-one
(prefill produces the first token, so decode passes can be n_tokens - 1) is a
different situation from rows being absent or garbage.

Never raises: every check is wrapped, the verdict JSON is the last line printed,
persistent hooks are always cleared, and the exit code is always 0.

Usage (via serve.slurm):
    sbatch --time=0:45:00 slurm/serve.slurm \
        --model Ministral-3-14B-Reasoning-2512 --config configs/dashboard.yaml \
        --stage scripts/spike_stream_hooks.py
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

# Long enough to need a good few decode steps, short enough to keep the prompt
# to one prefill chunk.
PROMPT = "Count from 1 to 12, separated by commas. Then stop."


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="spike: hook results on streamed requests")
    parser.add_argument("--config", type=Path, default=Path("configs/dashboard.yaml"))
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--vectors-dir", type=Path, default=None)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# SSE
# ---------------------------------------------------------------------------


def stream_chat(
    base_url: str,
    model: str,
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float,
    xargs: dict[str, str] | None = None,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """One streamed chat request, with the SSE ``data:`` lines parsed by hand.

    Returns what the probe needs to reason about: the response ``id``, how many
    chunks carried generated text, the usage block (asked for with
    ``stream_options.include_usage``, the authoritative token count), and any
    chunk that carried ``hook_results`` rather than ``choices``.
    """
    import requests

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if xargs:
        body["vllm_xargs"] = xargs

    info: dict[str, Any] = {
        "id": None,
        "n_chunks": 0,
        "n_text_deltas": 0,
        "text": "",
        "usage": None,
        "finish_reason": None,
        "hook_results_raw": None,
        "activations_in_stream": False,
        "hook_chunk_position": None,
        "saw_done": False,
        "unparsed_lines": 0,
    }

    with requests.post(
        f"{base_url}/v1/chat/completions", json=body, stream=True, timeout=timeout
    ) as resp:
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            payload = raw[len("data:") :].strip()
            if payload == "[DONE]":
                info["saw_done"] = True
                break
            try:
                chunk = json.loads(payload)
            except ValueError:
                info["unparsed_lines"] += 1
                continue
            if not isinstance(chunk, dict):
                info["unparsed_lines"] += 1
                continue
            info["n_chunks"] += 1

            if "hook_results" in chunk:
                info["hook_results_raw"] = chunk["hook_results"]
                info["hook_chunk_position"] = info["n_chunks"]
            if "activations" in chunk:
                info["activations_in_stream"] = True
            if info["id"] is None and isinstance(chunk.get("id"), str):
                info["id"] = chunk["id"]
            if chunk.get("usage"):
                info["usage"] = chunk["usage"]

            for choice in chunk.get("choices") or ():
                delta = choice.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    info["n_text_deltas"] += 1
                    info["text"] += piece
                if choice.get("finish_reason"):
                    info["finish_reason"] = choice["finish_reason"]
    return info


def streamed_token_count(info: dict[str, Any]) -> tuple[int, str]:
    """``(n_tokens, source)`` -- usage if the server sent it, else delta count."""
    usage = info.get("usage") or {}
    completion = usage.get("completion_tokens")
    if isinstance(completion, int):
        return completion, "usage.completion_tokens"
    return int(info["n_text_deltas"]), "text delta chunks"


# ---------------------------------------------------------------------------
# Hook payloads
# ---------------------------------------------------------------------------


def decode_rows(saved: dict[str, Any], capture_layers: list[int]) -> dict[str, Any]:
    """Prefill/decode row counts per capture layer, from the ``kind_L*`` markers.

    ``make_projection_hook`` writes 1.0 for a prefill row and 0.0 for a decode
    row, one row per forward pass position it keeps.
    """
    import numpy as np

    per_layer: dict[int, dict[str, int]] = {}
    for layer in capture_layers:
        kind = saved.get(f"kind_L{layer}")
        if kind is None:
            continue
        flat = np.asarray(kind, dtype=np.float32).reshape(-1)
        per_layer[layer] = {
            "decode": int((flat == 0.0).sum()),
            "prefill": int((flat == 1.0).sum()),
        }
    return per_layer


def probe_decode_rows(per_layer: dict[int, dict[str, int]], probe_layer: int) -> int | None:
    entry = per_layer.get(probe_layer)
    if entry is None:
        return None
    return entry["decode"]


def resolve_vectors_dir(config_path: Path, model: str, override: Path | None) -> Path:
    if override is not None:
        return override
    from healthy_rl.artifacts import artifact_dir
    from healthy_rl.config import load_config

    try:
        cfg = load_config(config_path)
    except (OSError, ValueError):
        cfg = {}
    dashboard = cfg.get("dashboard") if isinstance(cfg.get("dashboard"), dict) else {}
    configured = (dashboard or {}).get("vectors_dir")
    if configured:
        return Path(configured)
    return artifact_dir("vectors", model, "v1")


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    verdict: dict[str, Any] = {
        "per_request_stream_hooks": False,
        "persistent_keyed_by_response_id": False,
        "rows_match_tokens": False,
        "notes": [],
    }
    notes: list[str] = verdict["notes"]
    client = None

    try:
        from healthy_rl.dashboard.generation import merge_hook_results
        from healthy_rl.rollouts import load_vectors, make_projection_hook
        from healthy_rl.server import LensClient, base_url_from_env
        from vllm_lens._helpers._serialize import deserialize_hook_results

        base_url = args.base_url or base_url_from_env()
        vectors_dir = resolve_vectors_dir(args.config, args.model, args.vectors_dir)
        vectors = load_vectors(vectors_dir)
        notes.append(
            f"model={args.model} base_url={base_url} vectors={vectors_dir} "
            f"capture_layers={vectors.capture_layers} probe_layer={vectors.probe_layer} "
            f"n_emotions={vectors.n_emotions}"
        )

        def new_hook():
            return make_projection_hook(
                vectors.directions, vectors.capture_layers, [vectors.probe_layer]
            )

        messages = [{"role": "user", "content": PROMPT}]
        stream_kwargs = dict(
            max_tokens=args.max_tokens, temperature=args.temperature, timeout=args.timeout
        )

        # --- A: per-request hooks on a streamed request ----------------------
        try:
            xargs = {"apply_hooks": json.dumps([new_hook().model_dump()])}
            a = stream_chat(base_url, args.model, messages, xargs=xargs, **stream_kwargs)
            n_tokens_a, source_a = streamed_token_count(a)
            notes.append(
                f"A per-request+stream: id={a['id']} chunks={a['n_chunks']} "
                f"text_deltas={a['n_text_deltas']} tokens={n_tokens_a} ({source_a}) "
                f"finish={a['finish_reason']} done={a['saw_done']} "
                f"hook_chunk_position={a['hook_chunk_position']} "
                f"unparsed_lines={a['unparsed_lines']}"
            )
            if a["hook_results_raw"] is not None:
                verdict["per_request_stream_hooks"] = True
                saved_a = merge_hook_results(deserialize_hook_results(a["hook_results_raw"]))
                rows_a = decode_rows(saved_a, vectors.capture_layers)
                n_rows_a = probe_decode_rows(rows_a, vectors.probe_layer)
                notes.append(
                    f"A hook payload: keys={sorted(saved_a)} rows_per_layer={rows_a} "
                    f"probe_decode_rows={n_rows_a} vs tokens={n_tokens_a}"
                )
                if n_rows_a is not None and n_rows_a == n_tokens_a:
                    verdict["rows_match_tokens"] = True
            else:
                notes.append("A: no chunk carried hook_results")
        except BaseException as exc:  # noqa: BLE001 -- record, never raise
            notes.append(f"A FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()

        # --- B: persistent hooks, matched by streamed response id ------------
        try:
            client = LensClient(base_url, model=args.model)
            client.clear_hooks()  # start from a known-empty registry
            client.register_hooks([new_hook()])
            b = stream_chat(base_url, args.model, messages, xargs=None, **stream_kwargs)
            n_tokens_b, source_b = streamed_token_count(b)
            collected = client.collect_hook_results()
            keys = list(collected)
            response_id = b["id"]
            exact = response_id in collected if response_id else False
            prefixed = (
                [k for k in keys if k.startswith(f"{response_id}-")] if response_id else []
            )
            verdict["persistent_keyed_by_response_id"] = bool(exact or prefixed)
            notes.append(
                f"B persistent+stream: id={response_id} tokens={n_tokens_b} ({source_b}) "
                f"finish={b['finish_reason']} collect_keys={keys} exact_key={exact} "
                f"prefixed_keys={prefixed} "
                f"hook_results_in_stream={b['hook_results_raw'] is not None}"
            )
            entry = None
            if exact:
                entry = collected[response_id]
            elif prefixed:
                entry = collected[prefixed[0]]
            elif len(keys) == 1:
                entry = collected[keys[0]]
                notes.append(
                    f"B: response id did not match; falling back to the single "
                    f"collected key {keys[0]!r} for the row count"
                )
            if entry is not None:
                saved_b = merge_hook_results(entry)
                rows_b = decode_rows(saved_b, vectors.capture_layers)
                n_rows_b = probe_decode_rows(rows_b, vectors.probe_layer)
                notes.append(
                    f"B hook payload: keys={sorted(saved_b)} rows_per_layer={rows_b} "
                    f"probe_decode_rows={n_rows_b} vs tokens={n_tokens_b}"
                )
                if n_rows_b is not None and n_rows_b == n_tokens_b:
                    verdict["rows_match_tokens"] = True
            else:
                notes.append("B: no collected entry to count rows from")
        except BaseException as exc:  # noqa: BLE001
            notes.append(f"B FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()

        # --- C: non-streamed reference (what the dashboard does today) -------
        try:
            if client is None:
                client = LensClient(base_url, model=args.model)
            out = client.chat(
                messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                hooks=[new_hook()],
                logprobs=True,
            )
            usage = (out.raw or {}).get("usage") or {}
            n_tokens_c = usage.get("completion_tokens")
            n_logprob_tokens = len(((out.logprobs or {}).get("content") or []))
            saved_c = merge_hook_results(out.hook_results)
            rows_c = decode_rows(saved_c, vectors.capture_layers)
            notes.append(
                f"C non-streamed reference: tokens={n_tokens_c} "
                f"logprob_tokens={n_logprob_tokens} rows_per_layer={rows_c} "
                f"probe_decode_rows={probe_decode_rows(rows_c, vectors.probe_layer)} "
                f"hook_results_present={out.hook_results is not None}"
            )
        except BaseException as exc:  # noqa: BLE001
            notes.append(f"C FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    except BaseException as exc:  # noqa: BLE001 -- setup failure is still a result
        notes.append(f"SETUP FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        if client is not None:
            try:
                client.clear_hooks()
                notes.append("clear_hooks: ok")
            except BaseException as exc:  # noqa: BLE001
                notes.append(f"clear_hooks FAILED: {type(exc).__name__}: {exc}")
        for note in notes:
            print(f"[spike] {note}", flush=True)
        print(json.dumps(verdict), flush=True)

    # Always 0: a "no" is the answer this spike exists to produce, not a job failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
