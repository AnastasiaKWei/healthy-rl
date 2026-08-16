"""scripts/nemotron_pertok_cells.sh must generate six-shard configs that differ from
their Nemotron d6/aff6 template by exactly `shard`, `out_dir` and the appended arm
key, and print (never run) sbatch under --dry-run. Same stub-sbatch scheme as
test_mindset_cells.py."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "nemotron_pertok_cells.sh"
MODEL = "Nemotron-3-Nano-4B-BF16"
N = 6

pytestmark = pytest.mark.skipif(
    not (REPO / "slurm" / "serve.slurm").exists(),
    reason="slurm/ is environment-specific and untracked",
)

# version -> (template, extra keys, nice)
CELLS = {
    "d6r": ("d6", {}, 0),
    "aff6r": ("aff6", {}, 0),
    "inoc6": ("d6", {"inoculation": True}, 1000),
    "affinoc6": ("aff6", {"inoculation": True}, 1000),
    "growth6b": ("d6", {"mindset": ["growth"]}, 2000),
    "resil6b": ("d6", {"mindset": ["resilience"]}, 2000),
    "appr6b": ("d6", {"mindset": ["appraisal"]}, 2000),
    "affgrowth6b": ("aff6", {"mindset": ["growth"]}, 2000),
    "affresil6b": ("aff6", {"mindset": ["resilience"]}, 2000),
    "affappr6b": ("aff6", {"mindset": ["appraisal"]}, 2000),
}


@pytest.fixture(scope="module")
def dry_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("shards")
    stub = tmp_path_factory.mktemp("bin")
    (stub / "sbatch").write_text("#!/bin/sh\necho SBATCH_WAS_CALLED >&2\nexit 99\n")
    (stub / "sbatch").chmod(0o755)
    env = dict(os.environ, SHARD_DIR=str(out), PATH=f"{stub}:{os.environ['PATH']}")
    proc = subprocess.run([str(SCRIPT), "--dry-run"], cwd=REPO, env=env,
                          capture_output=True, text=True, check=True)
    return out, proc.stdout, proc.stderr


def _load(p: Path) -> dict:
    return yaml.safe_load(p.read_text())


@pytest.mark.parametrize("version", sorted(CELLS))
def test_configs_differ_from_template_by_exactly_the_varied_keys(dry_run, version):
    out, _, _ = dry_run
    template, extra, _ = CELLS[version]
    base = _load(REPO / "configs" / "shards" / f"rollouts-{MODEL}-{template}-s0of3.yaml")
    for i in range(N):
        new = _load(out / f"rollouts-{MODEL}-{version}-s{i}of{N}.yaml")
        assert new["shard"] == f"{i}/{N}"
        assert new["out_dir"] == f"/out/rollouts/{MODEL}/{version}"
        assert new["max_tokens"] == 24576
        for k in set(base) - {"shard", "out_dir"}:
            assert new[k] == base[k], f"{k} changed"
        assert {k: new[k] for k in set(new) - set(base)} == extra


def test_dry_run_prints_primary_and_two_continuations_per_shard(dry_run):
    _, stdout, stderr = dry_run
    lines = [l for l in stdout.splitlines() if l.startswith("sbatch")]
    assert len(lines) == 3 * N * len(CELLS)
    assert "SBATCH_WAS_CALLED" not in stderr
    prim = [l for l in lines if "-cont" not in l]
    conts = [l for l in lines if "-cont" in l]
    assert len(prim) == N * len(CELLS)
    assert all("--dependency=afterany:" in l for l in conts)
    assert all("--gres=gpu:2 " in l and "--time=2:00:00" in l for l in lines)
    for version, (_, _, nice) in CELLS.items():
        p = [l for l in prim if f"--job-name={MODEL}-{version}-s0 " in l]
        assert len(p) == 1 and f"--nice={nice} " in p[0]
        c = [l for l in conts if f"--job-name={MODEL}-{version}-s0-cont " in l]
        assert len(c) == 1 and f"--nice={nice + 10000} " in c[0]
