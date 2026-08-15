"""Artifact directories and provenance manifests.

Every stage of the pipeline writes a ``manifest.json`` into its output directory
recording its config plus the sha256 of each upstream manifest it consumed. A
downstream stage can then detect that an upstream was rewritten underneath it.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from healthy_rl.config import load_env, repo_root

__all__ = [
    "MANIFEST_NAME",
    "StaleUpstreamError",
    "artifact_dir",
    "write_manifest",
    "check_upstream",
    "manifest_sha256",
    "verify_upstreams",
]

MANIFEST_NAME = "manifest.json"


class StaleUpstreamError(RuntimeError):
    """An upstream manifest no longer matches the sha256 recorded downstream."""


def _artifact_root() -> Path:
    root = os.environ.get("ARTIFACT_DIR")
    if not root:
        load_env()
        root = os.environ.get("ARTIFACT_DIR")
    if not root:
        raise RuntimeError(
            "ARTIFACT_DIR is not set; add it to the repo-root .env or export it "
            "(paths are never hardcoded in committed code)"
        )
    return Path(root)


def artifact_dir(kind: str, model: str, version: str) -> Path:
    """``$ARTIFACT_DIR/<kind>/<model>/<version>``, created if needed."""
    out = _artifact_root() / kind / model / version
    out.mkdir(parents=True, exist_ok=True)
    return out


def manifest_path(path: str | os.PathLike[str]) -> Path:
    """Accept either an artifact directory or the manifest file itself."""
    p = Path(path)
    return p if p.name == MANIFEST_NAME else p / MANIFEST_NAME


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _normalize_upstreams(
    upstreams: Mapping[str, Any] | Iterable[Any] | None,
) -> dict[str, Path]:
    if upstreams is None:
        return {}
    if isinstance(upstreams, Mapping):
        return {str(name): Path(path) for name, path in upstreams.items()}
    if isinstance(upstreams, (str, os.PathLike)):
        upstreams = [upstreams]

    resolved: dict[str, Path] = {}
    for path in upstreams:
        p = Path(path)
        name = check_upstream(p).get("stage") or manifest_path(p).parent.name
        resolved[str(name)] = p
    return resolved


def write_manifest(
    dir: str | os.PathLike[str],
    stage: str,
    config: Mapping[str, Any] | None = None,
    upstreams: Mapping[str, Any] | Iterable[Any] | None = None,
) -> Path:
    """Write ``<dir>/manifest.json`` describing this stage and its upstreams.

    ``upstreams`` may be a mapping ``name -> artifact dir`` or a sequence of
    artifact dirs (keyed by each upstream's own ``stage``). Each upstream must
    already have a manifest; its sha256 is recorded here.
    """
    out_dir = Path(dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recorded: dict[str, dict[str, Any]] = {}
    for name, path in _normalize_upstreams(upstreams).items():
        upstream = check_upstream(path)
        recorded[name] = {
            "path": str(Path(path).resolve()),
            "stage": upstream.get("stage"),
            "sha256": manifest_sha256(path),
        }

    manifest = {
        "stage": stage,
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config": dict(config) if config else {},
        "upstreams": recorded,
    }
    target = out_dir / MANIFEST_NAME
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    return target


def check_upstream(path: str | os.PathLike[str]) -> dict:
    """Read and return an artifact's manifest.

    Raises ``FileNotFoundError`` naming the path when the manifest is absent --
    the usual cause is a stage that was never run.
    """
    target = manifest_path(path)
    if not target.is_file():
        raise FileNotFoundError(
            f"no {MANIFEST_NAME} for upstream artifact {Path(path)} "
            f"(expected {target}); has that stage been run?"
        )
    return json.loads(target.read_text())


def manifest_sha256(dir: str | os.PathLike[str]) -> str:
    """sha256 of the manifest file bytes."""
    target = manifest_path(dir)
    if not target.is_file():
        raise FileNotFoundError(f"no {MANIFEST_NAME} to hash at {target}")
    return hashlib.sha256(target.read_bytes()).hexdigest()


def verify_upstreams(dir: str | os.PathLike[str]) -> dict:
    """Re-hash every upstream recorded in ``dir``'s manifest and compare.

    Raises ``StaleUpstreamError`` if any upstream manifest changed since this
    artifact was written, ``FileNotFoundError`` if one disappeared.
    """
    manifest = check_upstream(dir)
    stale = []
    for name, entry in manifest.get("upstreams", {}).items():
        current = manifest_sha256(entry["path"])
        if current != entry.get("sha256"):
            stale.append(f"{name} at {entry['path']}: {entry.get('sha256')} -> {current}")
    if stale:
        raise StaleUpstreamError(
            f"upstream manifest(s) changed since {manifest_path(dir)} was written: "
            + "; ".join(stale)
        )
    return manifest
