#!/usr/bin/env python
"""Stage 0: verify vllm-lens can hook, capture from, and steer one architecture.

Runs on a compute node against an already-running vllm-lens server. **This stage never
raises.** Every check is caught and recorded as pass/fail with its exception text in
``smoke.json``, because the whole point is to find out which architectures work -- a
crash would destroy the result the stage exists to produce. The exit code is always 0.

Checks:
  1. ``generate``      -- 8 tokens come back non-empty.
  2. ``hook``          -- a persistent hook at ``capture_layers`` sees ``(seq_len, d_model)``
                          hidden states, with ``d_model`` matching the checkpoint config.
  3. ``steer``         -- a random ``SteeringVector`` at ``probe_layer`` (``norm_match``,
                          scale 0.5) changes the output for a fixed prompt and seed.
  4. ``capture_layers``-- the residual-stream capture path (what stage 3 uses) returns a
                          tensor whose last dimension is ``d_model``. Extra check, same
                          record-never-raise rule.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any, Callable

from healthy_rl.artifacts import write_manifest
from healthy_rl.config import load_config, load_env, repo_root

DEFAULT_CONFIG = repo_root() / "configs" / "smoke.yaml"
SMOKE_NAME = "smoke.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=None, help="override the config's model name")
    parser.add_argument("--base-url", default=None, help="vllm-lens server base URL")
    parser.add_argument("--out-dir", type=Path, default=None, help="where smoke.json goes")
    return parser.parse_args(argv)


def resolve_base_url(cli_value: str | None) -> str:
    """``--base-url``, else ``$HEALTHY_RL_SERVER_URL``, else the file it names."""
    if cli_value:
        return cli_value.strip()
    url = os.environ.get("HEALTHY_RL_SERVER_URL")
    if url and url.strip():
        return url.strip()
    url_file = os.environ.get("HEALTHY_RL_SERVER_URL_FILE")
    if url_file:
        path = Path(url_file)
        if not path.is_file():
            raise RuntimeError(
                f"HEALTHY_RL_SERVER_URL_FILE={url_file} does not exist; "
                "has the server job written its URL yet?"
            )
        text = path.read_text().strip()
        if not text:
            raise RuntimeError(f"HEALTHY_RL_SERVER_URL_FILE={url_file} is empty")
        return text
    raise RuntimeError(
        "no server URL: pass --base-url, or set HEALTHY_RL_SERVER_URL, "
        "or set HEALTHY_RL_SERVER_URL_FILE to a file containing it"
    )


def _shape_hook(ctx, hidden_states):
    """Record every hidden-state shape this hook sees. Runs on the server.

    Self-contained on purpose: it is cloudpickled to a process that has no
    ``healthy_rl`` on its path. Returns ``None`` so hidden states are unmodified,
    and is deterministic across TP ranks.
    """
    key = f"L{ctx.layer_idx}"
    record = ctx.saved.get(key)
    if record is None:
        record = {"shapes": [], "dtype": str(hidden_states.dtype)}
        ctx.saved[key] = record
    record["shapes"].append(list(hidden_states.shape))
    return None


class Smoke:
    """Holds the pieces the checks share; each check returns a JSON-able detail dict."""

    def __init__(self, client, spec, cfg: dict[str, Any]) -> None:
        self.client = client
        self.spec = spec
        self.cfg = cfg
        self.prompt = cfg["prompt"]
        self.max_tokens = int(cfg["max_tokens"])
        self.temperature = float(cfg.get("temperature", 0.0))
        self.seed = int(cfg["seed"])

    def check_generate(self) -> dict[str, Any]:
        out = self.client.generate(
            self.prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=self.seed,
        )
        text = out.text or ""
        if not text.strip():
            raise AssertionError(f"generation was empty (raw text {text!r})")
        return {"text": text, "n_chars": len(text)}

    def check_hook(self) -> dict[str, Any]:
        from vllm_lens import Hook

        capture_layers = list(self.spec.capture_layers)
        self.client.clear_hooks()
        try:
            self.client.register_hooks([Hook(fn=_shape_hook, layer_indices=capture_layers)])
            self.client.generate(
                self.prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                seed=self.seed,
            )
            results = self.client.collect_hook_results()
        finally:
            self.client.clear_hooks()

        if not results:
            raise AssertionError(
                f"no hook results returned for layers {capture_layers}; "
                "vllm-lens registered the hook but it never fired"
            )

        # {request_id: {hook_index: ctx.saved}} -> one merged {layer_key: record}
        saved: dict[str, Any] = {}
        for per_hook in results.values():
            for record in per_hook.values():
                saved.update(record)

        seen_layers = sorted(int(key[1:]) for key in saved if key.startswith("L"))
        missing = [layer for layer in capture_layers if layer not in seen_layers]
        if missing:
            raise AssertionError(
                f"hook never fired on layers {missing} (fired on {seen_layers})"
            )

        d_model = self.spec.d_model
        max_seq_len = 0
        for key, record in saved.items():
            for shape in record["shapes"]:
                if len(shape) != 2:
                    raise AssertionError(
                        f"{key}: hidden states are {len(shape)}-D {shape}, expected "
                        f"(seq_len, d_model)"
                    )
                if shape[1] != d_model:
                    raise AssertionError(
                        f"{key}: hidden states are {shape}, expected d_model={d_model} "
                        f"from {self.spec.path}/config.json"
                    )
                max_seq_len = max(max_seq_len, shape[0])
        if max_seq_len < 2:
            raise AssertionError(
                f"hook only ever saw seq_len {max_seq_len}; the prefill positions "
                "never reached it"
            )

        return {
            "capture_layers": capture_layers,
            "layers_fired": seen_layers,
            "d_model": d_model,
            "prefill_seq_len": max_seq_len,
            "dtype": next(iter(saved.values()))["dtype"],
            "shapes": {key: record["shapes"] for key, record in saved.items()},
        }

    def check_steer(self) -> dict[str, Any]:
        import torch
        from vllm_lens import SteeringVector

        max_tokens = int(self.cfg.get("steer_max_tokens", self.max_tokens))
        # Fixed generator: the same "random" direction every run, so a rerun of this
        # stage is comparable to the last one.
        generator = torch.Generator().manual_seed(self.seed)
        activations = torch.randn(1, self.spec.d_model, generator=generator)
        vector = SteeringVector(
            activations=activations,
            layer_indices=[self.spec.probe_layer],
            scale=float(self.cfg["steer_scale"]),
            norm_match=bool(self.cfg["norm_match"]),
        )

        self.client.clear_hooks()
        baseline = self.client.generate(
            self.prompt,
            max_tokens=max_tokens,
            temperature=self.temperature,
            seed=self.seed,
        )
        steered = self.client.generate(
            self.prompt,
            max_tokens=max_tokens,
            temperature=self.temperature,
            seed=self.seed,
            steering_vectors=[vector],
        )
        if steered.text == baseline.text:
            raise AssertionError(
                f"steering at layer {self.spec.probe_layer} (scale "
                f"{self.cfg['steer_scale']}, norm_match={self.cfg['norm_match']}) left "
                f"the output identical: {baseline.text!r}"
            )
        return {
            "probe_layer": self.spec.probe_layer,
            "scale": float(self.cfg["steer_scale"]),
            "norm_match": bool(self.cfg["norm_match"]),
            "baseline_text": baseline.text,
            "steered_text": steered.text,
        }

    def check_capture_layers(self) -> dict[str, Any]:
        capture_layers = list(self.spec.capture_layers)
        self.client.clear_hooks()
        out = self.client.generate(
            self.prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=self.seed,
            capture_layers=capture_layers,
        )
        if not out.activations:
            raise AssertionError(
                f"no activations returned for capture_layers={capture_layers}"
            )
        shapes = {name: list(t.shape) for name, t in out.activations.items()}
        for name, shape in shapes.items():
            if shape[-1] != self.spec.d_model:
                raise AssertionError(
                    f"{name}: activations are {shape}, expected last dim "
                    f"d_model={self.spec.d_model}"
                )
        return {"capture_layers": capture_layers, "shapes": shapes}


def run_check(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run one check; an exception is a recorded failure, never a raise."""
    try:
        detail = fn()
    except BaseException as exc:  # noqa: BLE001 - recording is the entire point
        return {
            "name": name,
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "detail": {},
        }
    return {"name": name, "passed": True, "error": None, "detail": detail}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result: dict[str, Any] = {
        "stage": "smoke",
        "model": None,
        "architecture": None,
        "base_url": None,
        "checks": [],
        "passed": False,
        "error": None,
    }
    out_dir: Path | None = None
    cfg: dict[str, Any] = {}

    try:
        load_env()
        cfg = load_config(args.config)
        model_name = args.model or cfg["model"]
        result["model"] = model_name

        out_dir = (
            Path(args.out_dir)
            if args.out_dir
            else _artifact_dir("smoke", model_name, str(cfg.get("version", "v1")))
        )

        base_url = resolve_base_url(args.base_url)
        result["base_url"] = base_url

        from healthy_rl.models import ModelSpec
        from vllm_lens.client import VLLMLensClient

        spec = ModelSpec.from_checkpoint(Path(cfg["model_dir"]) / model_name, name=model_name)
        result["architecture"] = spec.architecture
        result["spec"] = {
            "name": spec.name,
            "path": str(spec.path),
            "n_layers": spec.n_layers,
            "d_model": spec.d_model,
            "probe_layer": spec.probe_layer,
            "capture_layers": list(spec.capture_layers),
        }

        client = VLLMLensClient(
            base_url, timeout=float(cfg.get("request_timeout_s", 600.0))
        )
        result["served_model"] = _safe(lambda: client.model)

        smoke = Smoke(client, spec, cfg)
        result["checks"] = [
            run_check("generate", smoke.check_generate),
            run_check("hook", smoke.check_hook),
            run_check("steer", smoke.check_steer),
            run_check("capture_layers", smoke.check_capture_layers),
        ]
        result["passed"] = all(check["passed"] for check in result["checks"])
    except BaseException as exc:  # noqa: BLE001 - a setup failure is also a result
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()

    payload = json.dumps(result, indent=2, sort_keys=True, default=str)
    written = None
    if out_dir is not None:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / SMOKE_NAME
            target.write_text(payload + "\n")
            written = target
            # `cfg` holds only what the YAML declares -- no environment, no HF_TOKEN.
            # `model` is re-set so a `--model` override is what the manifest records.
            write_manifest(
                out_dir,
                stage="smoke",
                config={**cfg, "model": result["model"] or cfg.get("model")},
            )
        except BaseException as exc:  # noqa: BLE001
            result["error"] = (result["error"] or "") + f" | writing smoke.json: {exc}"

    # Always on stdout too, so the slurm log holds the result even if the write failed.
    print(payload)
    for check in result["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"{status} {check['name']}: {check['error'] or ''}")
    print(f"smoke {'PASSED' if result['passed'] else 'FAILED'} for {result['model']} "
          f"({result['architecture']}); wrote {written}")
    # Exit 0 unconditionally: a failed check is a recorded result, not a job failure.
    return 0


def _artifact_dir(kind: str, model: str, version: str) -> Path:
    from healthy_rl.artifacts import artifact_dir

    return artifact_dir(kind, model, version)


def _safe(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except BaseException as exc:  # noqa: BLE001
        return f"<unavailable: {type(exc).__name__}: {exc}>"


if __name__ == "__main__":
    raise SystemExit(main())
