"""YAML config loading with ``${VAR}`` expansion, and ``.env`` loading.

Nothing here imports torch or vLLM: this module is safe on a login node.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

__all__ = ["repo_root", "load_env", "load_config", "expand_vars"]

# ${VAR} or ${VAR:-default}
_VAR_RE = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")


def repo_root() -> Path:
    """Directory containing ``pyproject.toml`` (walking up from this file)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return Path.cwd()


def load_env(path: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """Read a ``.env`` file into ``os.environ`` without overwriting set variables.

    Returns the parsed file contents (all of them, including the ones that were
    not applied because the variable was already set). A missing file is a no-op.
    """
    env_path = Path(path) if path is not None else repo_root() / ".env"
    if not env_path.is_file():
        return {}

    parsed: dict[str, str] = {}
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        parsed[key] = value
        os.environ.setdefault(key, value)
    return parsed


def expand_vars(obj: Any, env: dict[str, str] | None = None) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in strings.

    Raises ``KeyError`` naming the variable when it is undefined and has no
    default, rather than silently leaving an unusable literal in a path.
    """
    environ = os.environ if env is None else env

    if isinstance(obj, str):

        def _sub(match: re.Match[str]) -> str:
            name = match.group("name")
            if name in environ:
                return environ[name]
            default = match.group("default")
            if default is not None:
                return default
            raise KeyError(f"undefined environment variable ${{{name}}} in config value {obj!r}")

        return _VAR_RE.sub(_sub, obj)
    if isinstance(obj, dict):
        return {key: expand_vars(value, environ) for key, value in obj.items()}
    if isinstance(obj, list):
        return [expand_vars(value, environ) for value in obj]
    return obj


def load_config(path: str | os.PathLike[str]) -> dict:
    """Load a YAML config file and expand ``${VAR}`` references from the environment."""
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config file not found: {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text())
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"config file must contain a mapping at the top level: {cfg_path}")
    return expand_vars(data)
