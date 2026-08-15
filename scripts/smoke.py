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
                          scale 0.5) moves the *residual stream* by the amount
                          ``norm_match`` promises, at that layer and not below it.
  4. ``capture_layers``-- the residual-stream capture path (what stage 3 uses) returns a
                          tensor whose last dimension is ``d_model``. Extra check, same
                          record-never-raise rule.

The steering check measures the residual directly rather than comparing generated text.
Text is a downstream proxy and it lies in both directions: a model with a fixed preamble
generates identical text under real steering (false negative -- this happened to
Qwen3.6-27B), and a steered request forces ``skip_reading_prefix_cache`` while an
unsteered one may read the cache, so a text difference can come from the recompute rather
than from steering (false positive). Both capture requests here ask for activations, which
makes vllm-lens skip the prefix cache for both, so the two runs share a numeric path.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

from healthy_rl.artifacts import write_manifest
from healthy_rl.config import load_config, load_env, repo_root
from healthy_rl.server import LensClient, base_url_from_env

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
    """``--base-url``, else ``base_url_from_env()``.

    ``base_url_from_env`` reads ``$HEALTHY_RL_SERVER_URL`` then the file named by
    ``$HEALTHY_RL_ENDPOINT_FILE``, and adds the ``http://`` scheme that
    ``serve.slurm``'s ``host:port`` endpoint file lacks. Reading that file raw
    would hand ``requests`` a schemeless URL, and every check would then record
    a failure against a perfectly healthy server -- a false "this architecture
    cannot be hooked", which is the most expensive wrong answer this stage can
    produce. ``LensClient`` normalises the ``--base-url`` override the same way.
    """
    if cli_value:
        return cli_value.strip()
    return base_url_from_env()


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

    def _residuals(self, capture_layers: list[int], steering_vectors=None) -> np.ndarray:
        """``(n_layers, n_positions, d)`` float64 residuals for a prefill-only request.

        ``max_tokens=1`` so both the steered and unsteered call see exactly the prompt
        positions: no generated token can differ between them and change the shape or
        the meaning of a position-wise comparison.
        """
        kwargs: dict[str, Any] = {}
        if steering_vectors is not None:
            kwargs["steering_vectors"] = steering_vectors
        out = self.client.generate(
            self.prompt,
            max_tokens=1,
            temperature=self.temperature,
            seed=self.seed,
            capture_layers=capture_layers,
            **kwargs,
        )
        acts = (out.activations or {}).get("residual_stream")
        if acts is None:
            raise AssertionError(
                f"server returned no residual_stream for capture_layers={capture_layers} "
                f"(keys: {list(out.activations or {})})"
            )
        arr = acts.float().numpy().astype(np.float64)
        if arr.ndim != 3 or arr.shape[0] != len(capture_layers):
            raise AssertionError(
                f"residual_stream is {arr.shape}, expected "
                f"({len(capture_layers)}, n_positions, {self.spec.d_model})"
            )
        if arr.shape[2] != self.spec.d_model:
            raise AssertionError(
                f"residual_stream has d={arr.shape[2]}, expected d_model="
                f"{self.spec.d_model} from {self.spec.path}/config.json"
            )
        return arr

    def check_steer(self) -> dict[str, Any]:
        """Steering moves the residual stream, at the right layer, by the right amount.

        ``norm_match`` adds ``scale * ||h||`` per token (``vllm_lens/_helpers/types.py``),
        and vllm-lens captures the *post*-steering hidden states at the layer it steers
        (``_worker_ext._hook_inner``: apply steering, then capture), so
        ``||h_steered - h_base|| / ||h_base||`` at ``probe_layer`` must come out at
        ``scale``. Below the probe layer nothing may move -- that is what catches
        steering silently landing on the wrong layer. Generated text is recorded but is
        not a pass criterion; see the module docstring.
        """
        import torch
        from vllm_lens import SteeringVector

        probe = self.spec.probe_layer
        scale = float(self.cfg["steer_scale"])
        below = max(probe - 1, 0)
        above = min(probe + 1, self.spec.n_layers - 1)
        capture_layers = sorted({below, probe, above})

        # Fixed generator: the same "random" direction every run, so a rerun of this
        # stage is comparable to the last one.
        generator = torch.Generator().manual_seed(self.seed)
        vector = SteeringVector(
            activations=torch.randn(1, self.spec.d_model, generator=generator),
            layer_indices=[probe],
            scale=scale,
            norm_match=bool(self.cfg["norm_match"]),
        )

        self.client.clear_hooks()
        base = self._residuals(capture_layers)
        steered = self._residuals(capture_layers, steering_vectors=[vector])
        if base.shape != steered.shape:
            raise AssertionError(
                f"steered residuals are {steered.shape}, unsteered {base.shape}; "
                "the two requests did not see the same positions"
            )

        rel_change: dict[str, float] = {}
        cosine: dict[str, float] = {}
        for i, layer in enumerate(capture_layers):
            b, s = base[i], steered[i]
            b_norm = float(np.linalg.norm(b))
            if b_norm == 0.0:
                raise AssertionError(f"L{layer}: unsteered residual is all zeros")
            rel_change[f"L{layer}"] = float(np.linalg.norm(s - b) / b_norm)
            denom = np.linalg.norm(b, axis=1) * np.linalg.norm(s, axis=1)
            denom = np.where(denom > 0, denom, np.nan)
            cosine[f"L{layer}"] = float(np.nanmean((b * s).sum(axis=1) / denom))

        detail: dict[str, Any] = {
            "probe_layer": probe,
            "scale": scale,
            "norm_match": bool(self.cfg["norm_match"]),
            "capture_layers": capture_layers,
            "n_positions": int(base.shape[1]),
            "rel_change": rel_change,
            "cosine": cosine,
            "text": self._steer_text_info(vector),
        }

        rel_tol = float(self.cfg.get("steer_probe_rel_tol", 0.25))
        unchanged_tol = float(self.cfg.get("steer_unchanged_tol", 0.02))
        min_downstream = float(self.cfg.get("steer_min_downstream", 0.05))
        detail["thresholds"] = {
            "probe_rel_tol": rel_tol,
            "unchanged_tol": unchanged_tol,
            "min_downstream": min_downstream,
        }

        measured = rel_change[f"L{probe}"]
        if not abs(measured - scale) <= rel_tol * scale:
            raise AssertionError(
                f"steering at L{probe} moved the residual by {measured:.4f} of its norm, "
                f"expected {scale} +/- {rel_tol * scale:.4f} (norm_match adds "
                f"scale*||h|| per token); all layers: {rel_change}"
            )
        if below != probe:
            leaked = rel_change[f"L{below}"]
            if leaked > unchanged_tol:
                raise AssertionError(
                    f"the residual at L{below} moved by {leaked:.4f} although steering "
                    f"was applied at L{probe}; steering is landing on the wrong layer"
                )
        if above != probe:
            downstream = rel_change[f"L{above}"]
            if downstream < min_downstream:
                raise AssertionError(
                    f"steering at L{probe} did not propagate: L{above} moved by only "
                    f"{downstream:.4f} (need >= {min_downstream})"
                )
        return detail

    def _steer_text_info(self, vector) -> dict[str, Any]:
        """Generated text with and without steering. Informational only, never fatal.

        A model with a fixed preamble can emit identical text under real steering, so
        ``differs`` being False is not a failure -- it is a fact worth having in the
        record next to the residual measurement that decides the check.
        """
        max_tokens = int(self.cfg.get("steer_max_tokens", self.max_tokens))
        try:
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
        except Exception as exc:  # noqa: BLE001 - informational field only
            return {"error": f"{type(exc).__name__}: {exc}"}
        return {
            "max_tokens": max_tokens,
            "differs": steered.text != baseline.text,
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
    """Run one check; an exception is a recorded failure, never a raise.

    ``KeyboardInterrupt`` and ``SystemExit`` are re-raised: a Ctrl-C or a job
    cancellation says nothing about the architecture, and recording it as a failed
    check would be a fabricated result.
    """
    try:
        detail = fn()
    except (KeyboardInterrupt, SystemExit):
        raise
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
    interrupted: BaseException | None = None

    try:
        # Set before anything that can fail, so an explicit --out-dir still receives
        # the result when the config itself is what is broken.
        if args.out_dir:
            out_dir = Path(args.out_dir)
        load_env()
        cfg = load_config(args.config)
        model_name = args.model or cfg["model"]
        result["model"] = model_name

        if out_dir is None:
            out_dir = _artifact_dir("smoke", model_name, str(cfg.get("version", "v1")))

        base_url = resolve_base_url(args.base_url)
        result["base_url"] = base_url

        from healthy_rl.models import ModelSpec

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

        # LensClient normalises the URL and retries transient connection errors, so a
        # server that is briefly busy does not read out as an unhookable architecture.
        client = LensClient(
            base_url,
            timeout=float(cfg.get("request_timeout_s", 600.0)),
            max_attempts=int(cfg.get("client_max_attempts", 5)),
            backoff_s=float(cfg.get("client_backoff_s", 1.0)),
        )
        result["base_url"] = client.base_url
        result["served_model"] = _safe(lambda: client.model)

        smoke = Smoke(client, spec, cfg)
        result["checks"] = [
            run_check("generate", smoke.check_generate),
            run_check("hook", smoke.check_hook),
            run_check("steer", smoke.check_steer),
            run_check("capture_layers", smoke.check_capture_layers),
        ]
        result["passed"] = all(check["passed"] for check in result["checks"])
    except (KeyboardInterrupt, SystemExit) as exc:
        # Not an architecture result. Record what we have, then let it propagate.
        interrupted = exc
        result["error"] = f"interrupted: {type(exc).__name__}"
    except BaseException as exc:  # noqa: BLE001 - a setup failure is also a result
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()

    payload = json.dumps(result, indent=2, sort_keys=True, default=str)
    written = None
    manifest_dir = out_dir
    if out_dir is None:
        # ARTIFACT_DIR unset, or the config never loaded. Still write the result
        # somewhere on disk: a result that exists only in a slurm log is a result
        # nobody finds. No manifest here -- this is not an artifact directory.
        out_dir = repo_root() / "logs"
        smoke_name = f"smoke-{result['model'] or 'unknown'}.json"
    else:
        smoke_name = SMOKE_NAME
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / smoke_name
        target.write_text(payload + "\n")
        written = target
        if manifest_dir is not None:
            # `cfg` holds only what the YAML declares -- no environment, no HF_TOKEN.
            # `model` is re-set so a `--model` override is what the manifest records.
            write_manifest(
                manifest_dir,
                stage="smoke",
                config={**cfg, "model": result["model"] or cfg.get("model")},
            )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # noqa: BLE001
        result["error"] = (result["error"] or "") + f" | writing {smoke_name}: {exc}"

    # Always on stdout too, so the slurm log holds the result even if the write failed.
    print(payload)
    for check in result["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"{status} {check['name']}: {check['error'] or ''}")
    print(f"smoke {'PASSED' if result['passed'] else 'FAILED'} for {result['model']} "
          f"({result['architecture']}); wrote {written}")
    if interrupted is not None:
        raise interrupted
    # Exit 0 otherwise: a failed check is a recorded result, not a job failure.
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
