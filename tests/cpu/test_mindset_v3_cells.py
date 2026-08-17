"""scripts/mindset_v3_cells.sh must generate three-shard configs that differ from
their per-token base template by exactly `shard`, `out_dir` and `mindset`, and
print (never run) sbatch under --dry-run. Same stub-sbatch scheme as
test_nemotron_pertok_cells.py."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "mindset_v3_cells.sh"
GEMMA = "gemma-3-12b-it"
MINISTRAL = "Ministral-3-14B-Reasoning-2512"
N = 3

pytestmark = pytest.mark.skipif(
    not (REPO / "slurm" / "serve.slurm").exists(),
    reason="slurm/ is environment-specific and untracked",
)

# (model, version) -> (template, mindset name, nice)
CELLS = {
    (GEMMA, "spaffgrowth6v3"): ("spaff6", "growth", 0),
    (GEMMA, "spaffresil6v3"): ("spaff6", "resilience", 0),
    (GEMMA, "spaffctrl6v3"): ("spaff6", "control", 0),
    (GEMMA, "spaffcomp6v3"): ("spaff6", "compassion", 0),
    (MINISTRAL, "affgrowth6v3"): ("aff6r", "growth", 0),
    (MINISTRAL, "affresil6v3"): ("aff6r", "resilience", 0),
    (MINISTRAL, "affctrl6v3"): ("aff6r", "control", 0),
    (MINISTRAL, "affcomp6v3"): ("aff6r", "compassion", 0),
    (GEMMA, "spgrowth6v3"): ("sp6r", "growth", 1000),
    (GEMMA, "spresil6v3"): ("sp6r", "resilience", 1000),
    (GEMMA, "spctrl6v3"): ("sp6r", "control", 1000),
    (GEMMA, "spcomp6v3"): ("sp6r", "compassion", 1000),
    (MINISTRAL, "growth6v3"): ("d6r", "growth", 1000),
    (MINISTRAL, "resil6v3"): ("d6r", "resilience", 1000),
    (MINISTRAL, "ctrl6v3"): ("d6r", "control", 1000),
    (MINISTRAL, "comp6v3"): ("d6r", "compassion", 1000),
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


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_configs_differ_from_template_by_exactly_the_varied_keys(dry_run, cell):
    out, _, _ = dry_run
    model, version = cell
    template, arm, _ = CELLS[cell]
    base = _load(REPO / "configs" / "shards" / f"rollouts-{model}-{template}-s0of3.yaml")
    for i in range(N):
        new = _load(out / f"rollouts-{model}-{version}-s{i}of{N}.yaml")
        assert new["shard"] == f"{i}/{N}"
        assert new["out_dir"] == f"/out/rollouts/{model}/{version}"
        assert new["max_tokens"] == 24576
        assert new["request_timeout_s"] == 3600
        for k in set(base) - {"shard", "out_dir"}:
            assert new[k] == base[k], f"{k} changed"
        assert {k: new[k] for k in set(new) - set(base)} == {"mindset": [arm]}


def test_gemma_cells_keep_the_scratchpad_and_ministral_cells_do_not(dry_run):
    out, _, _ = dry_run
    for (model, version), _ in CELLS.items():
        new = _load(out / f"rollouts-{model}-{version}-s0of{N}.yaml")
        assert bool(new.get("scratchpad_reasoning")) is (model == GEMMA), version
        assert bool(new.get("affect_prompt")) is ("aff" in version), version


def test_committed_configs_match_a_fresh_dry_run(dry_run):
    out, _, _ = dry_run
    for (model, version) in CELLS:
        for i in range(N):
            name = f"rollouts-{model}-{version}-s{i}of{N}.yaml"
            committed = REPO / "configs" / "shards" / name
            assert committed.is_file(), f"{name} not committed"
            assert committed.read_text() == (out / name).read_text(), f"{name} drifted from the generator"


def test_config_comment_names_the_v3_mechanism_and_the_base(dry_run):
    out, _, _ = dry_run
    for (model, version), (template, arm, _) in CELLS.items():
        text = (out / f"rollouts-{model}-{version}-s0of{N}.yaml").read_text()
        assert f"# --- MINDSET ARM: {arm} (prompt v3" in text
        assert f"{model}/{template}" in text          # names its per-token base
        assert "reminder line" in text and "## Task" in text
        assert "77d558c" in text


def test_dry_run_prints_primary_and_two_continuations_per_shard(dry_run):
    _, stdout, stderr = dry_run
    lines = [l for l in stdout.splitlines() if l.startswith("sbatch")]
    assert len(lines) == 3 * N * len(CELLS)
    assert "SBATCH_WAS_CALLED" not in stderr
    prim = [l for l in lines if "-cont" not in l]
    conts = [l for l in lines if "-cont" in l]
    assert len(prim) == N * len(CELLS)
    assert all("--dependency=afterany:" in l for l in conts)
    assert all("--gres=gpu:A100-40G:2 " in l and "--time=4:00:00" in l for l in lines)
    for (model, version), (_, _, nice) in CELLS.items():
        p = [l for l in prim if f"--job-name={model}-{version}-s0 " in l]
        assert len(p) == 1 and f"--nice={nice} " in p[0]
        assert f"--stage scripts/run_rollouts.py:configs/shards/rollouts-{model}-{version}-s0of{N}.yaml" in p[0]
        c = [l for l in conts if f"--job-name={model}-{version}-s0-cont " in l]
        assert len(c) == 1 and f"--nice={nice + 10000} " in c[0]


def test_only_filters_restrict_the_scope(tmp_path):
    stub = tmp_path / "bin"; stub.mkdir()
    (stub / "sbatch").write_text("#!/bin/sh\nexit 99\n"); (stub / "sbatch").chmod(0o755)
    out = tmp_path / "shards"; out.mkdir()
    env = dict(os.environ, SHARD_DIR=str(out), PATH=f"{stub}:{os.environ['PATH']}",
               ONLY_MODEL=MINISTRAL, ONLY_VERSION="ctrl6v3")
    proc = subprocess.run([str(SCRIPT), "--dry-run"], cwd=REPO, env=env,
                          capture_output=True, text=True, check=True)
    assert sorted(p.name for p in out.iterdir()) == [
        f"rollouts-{MINISTRAL}-ctrl6v3-s{i}of{N}.yaml" for i in range(N)]
    assert len([l for l in proc.stdout.splitlines() if l.startswith("sbatch")]) == 3 * N
