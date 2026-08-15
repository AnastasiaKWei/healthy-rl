"""Host side of the task loop's code execution: apptainer --contain around sandbox_cli.

Model-generated code runs ONLY through ``Sandbox.run``, inside ``eval.sif``
with ``--contain`` (docs/infrastructure.md). Binds: the project read-only at
/project (code), the bench ROOT read-only at /bench (parquets), and a scratch
directory read-write at /scratch (the code file, cwd, TMPDIR). Nothing under
``$ARTIFACT_DIR`` other than the bench root is visible.

HOME is deliberately left alone: the image sets ``HOME=/work`` and apptainer
refuses ``--env HOME=...`` ("Overriding HOME environment variable with
APPTAINERENV_HOME is not permitted") on every invocation, which would put a
WARNING line in stderr of every call. Under ``--contain --writable-tmpfs`` the
in-image /work is a throwaway tmpfs, so it is already contained.
"""
from __future__ import annotations

import json
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

STARTUP_GRACE_S = 30

#: Split -> parquet path relative to the bench root. The two splits live in
#: separate fetch directories because ``fetch_bench`` writes fixed-name manifests
#: per directory, so one directory cannot hold both (see docs/runs.md).
SPLIT_PARQUETS = {"conflicting": "v1/conflicting.parquet", "original": "orig1/original.parquet"}


@dataclass
class SandboxResult:
    passed: bool
    stdout: str
    stderr: str
    feedback: str
    timed_out: bool = False
    seconds: float = 0.0
    error: str | None = None


class Sandbox:
    """Runs sandbox_cli inside eval.sif.

    ``bench_dir`` is the bench ROOT (``$ARTIFACT_DIR/bench``), not one fetch
    directory: the splits are fetched into separate subdirectories because
    ``fetch_bench`` writes fixed-name manifests per directory (docs/runs.md), so
    ``split_parquets`` maps a split name to its parquet relative to that root.
    """

    def __init__(self, *, sif: Path, project_dir: Path, bench_dir: Path, scratch_dir: Path,
                 timeout_s: int = 30, runner: Callable[..., Any] = subprocess.run,
                 split_parquets: dict[str, str] | None = None) -> None:
        self.sif, self.project_dir, self.bench_dir = Path(sif), Path(project_dir), Path(bench_dir)
        self.scratch_dir = Path(scratch_dir)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self._run = runner
        self.split_parquets = dict(SPLIT_PARQUETS if split_parquets is None else split_parquets)

    def parquet_for(self, split: str) -> str:
        """Container-side path of a split's parquet, under the /bench bind."""
        try:
            relative = self.split_parquets[split]
        except KeyError:
            raise ValueError(f"unknown split {split!r}; known: {sorted(self.split_parquets)}") from None
        return f"/bench/{relative}"

    def command(self, *cli_args: str) -> list[str]:
        return [
            "apptainer", "exec", "--contain", "--cleanenv", "--writable-tmpfs",
            "--bind", f"{self.project_dir}:/project:ro",
            "--bind", f"{self.bench_dir}:/bench:ro",
            "--bind", f"{self.scratch_dir}:/scratch:rw",
            "--env", "PYTHONPATH=/project/src", "--env", "TMPDIR=/scratch",
            "--pwd", "/scratch", str(self.sif),
            "python", "-m", "healthy_rl.dashboard.sandbox_cli", *cli_args,
        ]

    def _call(self, args: list[str], timeout: float) -> tuple[Any | None, str | None]:
        try:
            proc = self._run(self.command(*args), capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, "timeout"
        except OSError as exc:
            return None, f"could not exec apptainer: {exc}"
        if proc.returncode != 0:
            return None, f"sandbox_cli exited {proc.returncode}: {proc.stderr.strip()[-2000:]}"
        try:
            return json.loads(proc.stdout), None
        except json.JSONDecodeError:
            return None, f"sandbox_cli returned non-JSON: {proc.stdout[:200]!r} stderr={proc.stderr.strip()[-500:]!r}"

    def problems(self, split: str, affect: bool = False) -> dict:
        args = ["problems", "--parquet", self.parquet_for(split)] + (["--affect"] if affect else [])
        data, err = self._call(args, timeout=600)
        if err:
            raise RuntimeError(f"sandbox problems({split}) failed: {err}")
        return data

    def run(self, split: str, task_id: str, code: str, affect: bool = False) -> SandboxResult:
        parquet = self.parquet_for(split)
        name = f"sub_{uuid.uuid4().hex[:12]}.py"
        host_path = self.scratch_dir / name
        host_path.write_text(code, encoding="utf-8")
        started = time.monotonic()
        try:
            args = ["run", "--parquet", parquet, "--task-id", task_id,
                    "--code-file", f"/scratch/{name}", "--timeout", str(self.timeout_s)] + (["--affect"] if affect else [])
            data, err = self._call(args, timeout=self.timeout_s + STARTUP_GRACE_S)
        finally:
            try:
                host_path.unlink()
            except OSError:
                pass
        seconds = time.monotonic() - started
        if err == "timeout":
            # The container died with the wrapper, so its own ``finally`` never ran: sweep the
            # test file sandbox_cli writes into the cwd (/scratch) so scratch does not fill up.
            self._sweep_orphans()
            return SandboxResult(False, "", "", "", timed_out=True, seconds=seconds, error=f"sandbox exceeded {self.timeout_s + STARTUP_GRACE_S}s")
        if err:
            return SandboxResult(False, "", "", "", seconds=seconds, error=err)
        if not isinstance(data, dict) or "passed" not in data:
            return SandboxResult(False, "", "", "", seconds=seconds,
                                 error=f"sandbox_cli returned an unexpected payload: {str(data)[:200]!r}")
        return SandboxResult(bool(data["passed"]), data.get("stdout", ""), data.get("stderr", ""), data.get("feedback", ""),
                             timed_out=bool(data.get("timed_out")), seconds=seconds)

    def _sweep_orphans(self) -> None:
        """Remove ``t_*.py`` files sandbox_cli left behind when it was killed mid-run."""
        try:
            orphans = list(self.scratch_dir.glob("t_*.py"))
        except OSError:
            return
        for orphan in orphans:
            try:
                orphan.unlink()
            except OSError:
                pass
