"""scripts/mindset_cells.sh must generate shard configs that differ from their base by
exactly the varied keys, and print (never run) sbatch under --dry-run."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "mindset_cells.sh"

CELLS = {
    ("Ministral-3-14B-Reasoning-2512", "growth6"): ("d6", ["growth"], 0),
    ("Ministral-3-14B-Reasoning-2512", "resil6"): ("d6", ["resilience"], 0),
    ("Ministral-3-14B-Reasoning-2512", "appr6"): ("d6", ["appraisal"], 0),
    ("Qwen3.5-9B", "growth6"): ("d6", ["growth"], 2000),
    ("Qwen3.5-9B", "resil6"): ("d6", ["resilience"], 2000),
    ("Qwen3.5-9B", "appr6"): ("d6", ["appraisal"], 2000),
    ("Ministral-3-14B-Reasoning-2512", "affgrowth6"): ("aff6", ["growth"], 4000),
    ("Ministral-3-14B-Reasoning-2512", "affresil6"): ("aff6", ["resilience"], 4000),
    ("Ministral-3-14B-Reasoning-2512", "affappr6"): ("aff6", ["appraisal"], 4000),
}


@pytest.fixture(scope="module")
def dry_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("shards")
    env = {**os.environ, "SHARD_DIR": str(out)}
    proc = subprocess.run(["bash", str(SCRIPT), "--dry-run"], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return out, proc.stdout


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize("model,version", sorted(CELLS))
def test_shard_configs_differ_from_base_by_exactly_the_varied_keys(dry_run, model, version):
    out, _ = dry_run
    base_version, mindset, _nice = CELLS[(model, version)]
    for i in range(3):
        new = _load(out / f"rollouts-{model}-{version}-s{i}of3.yaml")
        base = _load(REPO / "configs" / "shards" / f"rollouts-{model}-{base_version}-s{i}of3.yaml")
        assert new["mindset"] == mindset
        assert new["max_tokens"] == 24576
        assert new["out_dir"] == f"/out/rollouts/{model}/{version}"
        assert new["shard"] == f"{i}/3"
        for k in set(base) - {"max_tokens", "out_dir"}:
            assert new[k] == base[k], f"{k} changed"
        assert set(new) - set(base) == {"mindset"}


def test_dry_run_prints_primary_and_two_continuations_per_shard(dry_run):
    _, stdout = dry_run
    for (model, version), (_b, _m, nice) in CELLS.items():
        for i in range(3):
            name = f"{model}-{version}-s{i}"
            assert f"--job-name={name} " in stdout
            assert f"--job-name={name}-cont " in stdout
            assert f"--job-name={name}-cont2 " in stdout
            line = next(l for l in stdout.splitlines() if f"--job-name={name} " in l)
            assert "--time=4:00:00" in line and "slurm/serve.slurm" in line
            assert f"--config configs/shards/rollouts-{model}-{version}-s{i}of3.yaml" in line
            assert (f"--nice={nice}" in line) is (nice > 0)
    assert "sbatch --parsable" in stdout
