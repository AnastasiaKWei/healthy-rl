"""Client-side helpers for driving a vLLM + vllm-lens server over HTTP.

This module is importable on the login node: neither ``vllm`` nor ``torch`` is
imported at module scope. The vllm-lens client is imported lazily, the first
time a :class:`LensClient` is constructed.

Two halves:

* :func:`wait_for_health` and the endpoint-file helpers, used by
  ``slurm/serve.slurm`` and by driver scripts to find and wait on the server.
* :class:`LensClient`, a thin wrapper around ``vllm_lens.client.VLLMLensClient``
  that retries transient connection errors. A 60-70 GB model served over a
  multi-hour job will occasionally drop a socket; a dropped socket should not
  end a rollout sweep.
"""

from __future__ import annotations

import argparse
import functools
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "ServerNotReady",
    "DEFAULT_HEALTH_TIMEOUT_S",
    "wait_for_health",
    "read_endpoint",
    "base_url_from_env",
    "LensClient",
]

# A bf16 checkpoint of 60-70 GB loads off the network filesystem in roughly ten
# minutes; 30 minutes leaves room for a cold cache without hanging a job forever.
DEFAULT_HEALTH_TIMEOUT_S = 1800.0

ENDPOINT_FILE_ENV = "HEALTHY_RL_ENDPOINT_FILE"
# Alias exported by serve.slurm as well, because scripts/smoke.py reads this name.
ENDPOINT_FILE_ENV_ALIAS = "HEALTHY_RL_SERVER_URL_FILE"
SERVER_URL_ENV = "HEALTHY_RL_SERVER_URL"


class ServerNotReady(RuntimeError):
    """The server did not answer ``/health`` within the allotted time."""


def _normalise_base_url(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("empty server address")
    if "://" not in value:
        value = f"http://{value}"
    return value.rstrip("/")


def wait_for_health(
    base_url: str,
    timeout_s: float = DEFAULT_HEALTH_TIMEOUT_S,
    poll_interval_s: float = 2.0,
    is_alive: Callable[[], bool] | None = None,
) -> float:
    """Block until ``base_url/health`` returns 200. Return seconds waited.

    Args:
        base_url: Server root, with or without a scheme (``host:port`` is fine).
        timeout_s: Hard bound on the wait.
        poll_interval_s: Delay between probes.
        is_alive: Optional predicate checked between probes. If it returns
            False the server process has died and we fail immediately rather
            than burning the whole timeout.

    Raises:
        ServerNotReady: on timeout, or when ``is_alive`` reports the process is
            gone. The message names the URL and the elapsed time.
    """
    base_url = _normalise_base_url(base_url)
    url = f"{base_url}/health"
    deadline = time.monotonic() + timeout_s
    started = time.monotonic()
    last_error = "no response yet"

    while True:
        if is_alive is not None and not is_alive():
            raise ServerNotReady(
                f"server process exited before {url} became healthy "
                f"after {time.monotonic() - started:.1f}s"
            )
        try:
            with urllib.request.urlopen(url, timeout=poll_interval_s * 2) as resp:
                if 200 <= resp.status < 300:
                    return time.monotonic() - started
                last_error = f"HTTP {resp.status}"
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
        except (urllib.error.URLError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        if time.monotonic() >= deadline:
            raise ServerNotReady(
                f"{url} did not become healthy within {timeout_s:.0f}s "
                f"(waited {time.monotonic() - started:.1f}s, last error: {last_error})"
            )
        time.sleep(poll_interval_s)


def read_endpoint(path: str | os.PathLike[str], timeout_s: float = 0.0) -> str:
    """Read a ``host:port`` endpoint file and return it as a base URL.

    ``slurm/serve.slurm`` writes this file once the port is bound. Pass
    ``timeout_s`` to wait for the file to appear.
    """
    path = Path(path)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            text = path.read_text().strip()
            if text:
                return _normalise_base_url(text)
        except FileNotFoundError:
            pass
        if time.monotonic() >= deadline:
            raise ServerNotReady(
                f"endpoint file {path} did not appear (or stayed empty) "
                f"within {timeout_s:.0f}s"
            )
        time.sleep(0.5)


def base_url_from_env() -> str:
    """Resolve the server base URL from the environment ``serve.slurm`` sets."""
    url = os.environ.get(SERVER_URL_ENV)
    if url:
        return _normalise_base_url(url)
    endpoint_file = os.environ.get(ENDPOINT_FILE_ENV) or os.environ.get(
        ENDPOINT_FILE_ENV_ALIAS
    )
    if endpoint_file:
        return read_endpoint(endpoint_file)
    raise ServerNotReady(
        f"none of ${SERVER_URL_ENV}, ${ENDPOINT_FILE_ENV}, ${ENDPOINT_FILE_ENV_ALIAS} "
        "is set; driver scripts are meant to be launched by slurm/serve.slurm"
    )


@functools.lru_cache(maxsize=1)
def _transient_exceptions() -> tuple[type[BaseException], ...]:
    """Exception types worth retrying. Imported lazily: requests pulls in urllib3."""
    types: list[type[BaseException]] = [ConnectionError, TimeoutError]
    try:
        import requests.exceptions as rex
    except ImportError:
        return tuple(types)
    types.extend(
        [
            rex.ConnectionError,
            rex.Timeout,
            rex.ChunkedEncodingError,
            rex.ContentDecodingError,
        ]
    )
    return tuple(types)


class LensClient:
    """Retrying wrapper around ``vllm_lens.client.VLLMLensClient``.

    Every method of the underlying client is exposed unchanged; calls that fail
    with a transient connection error are retried with exponential backoff.
    Server-side errors (which the underlying client raises as ``RuntimeError``)
    are *not* retried, because retrying a rejected request just repeats it.

    Construct with :meth:`from_env` inside a driver launched by
    ``slurm/serve.slurm``.
    """

    def __init__(
        self,
        base_url: str,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = 600.0,
        max_attempts: int = 5,
        backoff_s: float = 1.0,
        backoff_max_s: float = 30.0,
    ) -> None:
        from vllm_lens.client import VLLMLensClient  # lazy: imports torch

        self.base_url = _normalise_base_url(base_url)
        self.max_attempts = max(1, max_attempts)
        self.backoff_s = backoff_s
        self.backoff_max_s = backoff_max_s
        self._client = VLLMLensClient(
            self.base_url, model=model, api_key=api_key, timeout=timeout
        )

    @classmethod
    def from_env(cls, **kwargs: Any) -> "LensClient":
        """Build a client against the server ``serve.slurm`` started."""
        return cls(base_url_from_env(), **kwargs)

    @property
    def raw(self) -> Any:
        """The unwrapped ``VLLMLensClient``, for anything not worth retrying."""
        return self._client

    @property
    def model(self) -> str:
        return self._call("model", lambda: self._client.model)

    def _call(self, name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        delay = self.backoff_s
        for attempt in range(1, self.max_attempts + 1):
            try:
                return fn(*args, **kwargs)
            except _transient_exceptions() as exc:
                if attempt == self.max_attempts:
                    raise ServerNotReady(
                        f"{name} failed against {self.base_url} after "
                        f"{self.max_attempts} attempts: {type(exc).__name__}: {exc}"
                    ) from exc
                sleep_s = min(delay, self.backoff_max_s) * (0.5 + random.random())
                print(
                    f"[healthy_rl.server] {name} attempt {attempt}/{self.max_attempts} "
                    f"failed ({type(exc).__name__}: {exc}); retrying in {sleep_s:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(sleep_s)
                delay = min(delay * 2, self.backoff_max_s)
        raise AssertionError("unreachable")

    def __getattr__(self, name: str) -> Any:
        if name == "_client":  # guards against recursion before __init__ finishes
            raise AttributeError(name)
        attr = getattr(self._client, name)
        if not callable(attr):
            return attr
        return functools.partial(self._call, name, attr)

    def __repr__(self) -> str:
        return f"LensClient(base_url={self.base_url!r})"


# ---------------------------------------------------------------------------
# CLI, used by slurm/serve.slurm
# ---------------------------------------------------------------------------


def _config_get(config: dict, key: str) -> Any:
    """Look up ``serve.<key>`` first, then top-level ``<key>``."""
    serve = config.get("serve")
    if isinstance(serve, dict) and key in serve:
        return serve[key]
    return config.get(key)


def _cmd_serve_args(args: argparse.Namespace) -> int:
    """Print shell assignments for the vLLM flags that must come from config.

    ``--max-model-len`` is mandatory: three of the four pilot checkpoints
    default to a 131k-262k context that no KV budget on these nodes supports,
    so a missing value is a hard error rather than a silent fallback.
    """
    from healthy_rl.config import load_config

    config = load_config(args.config)
    max_model_len = _config_get(config, "max_model_len")
    if max_model_len is None:
        print(
            f"ERROR: {args.config} sets neither 'max_model_len' nor "
            "'serve.max_model_len'. The checkpoint default (131k-262k tokens) "
            "does not fit in the KV budget of a 2-GPU node; set it explicitly.",
            file=sys.stderr,
        )
        return 1
    gpu_util = _config_get(config, "gpu_memory_utilization")
    if gpu_util is None:
        gpu_util = 0.90
    max_num_seqs = _config_get(config, "max_num_seqs")

    print(f"MAX_MODEL_LEN={int(max_model_len)}")
    print(f"GPU_MEMORY_UTILIZATION={float(gpu_util)}")
    if max_num_seqs is not None:
        print(f"MAX_NUM_SEQS={int(max_num_seqs)}")
    return 0


def _cmd_wait(args: argparse.Namespace) -> int:
    base_url = args.base_url or base_url_from_env()
    waited = wait_for_health(base_url, timeout_s=args.timeout)
    print(f"[healthy_rl.server] {base_url}/health ready after {waited:.1f}s")
    return 0


def _cmd_free_port(args: argparse.Namespace) -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((args.host, 0))
        print(sock.getsockname()[1])
    return 0


def _cmd_endpoint(args: argparse.Namespace) -> int:
    print(read_endpoint(args.path, timeout_s=args.timeout))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m healthy_rl.server")
    sub = parser.add_subparsers(dest="command", required=True)

    p_args = sub.add_parser("serve-args", help="emit vLLM flags from a stage config")
    p_args.add_argument("config")
    p_args.set_defaults(func=_cmd_serve_args)

    p_wait = sub.add_parser("wait", help="poll /health until ready")
    p_wait.add_argument("--base-url", default=None)
    p_wait.add_argument("--timeout", type=float, default=DEFAULT_HEALTH_TIMEOUT_S)
    p_wait.set_defaults(func=_cmd_wait)

    p_port = sub.add_parser("free-port", help="print an unused TCP port")
    p_port.add_argument("--host", default="127.0.0.1")
    p_port.set_defaults(func=_cmd_free_port)

    p_ep = sub.add_parser("endpoint", help="print the base URL from an endpoint file")
    p_ep.add_argument("path")
    p_ep.add_argument("--timeout", type=float, default=0.0)
    p_ep.set_defaults(func=_cmd_endpoint)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ServerNotReady as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
