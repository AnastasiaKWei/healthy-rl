"""scripts/mindset_cells.sh must generate shard configs that differ from their base by
exactly the varied keys, and print (never run) sbatch under --dry-run.

Both runs below put a stub `sbatch` first on PATH. Under --dry-run that stub proves a
negative -- the script must not invoke sbatch at all -- and under --submit it stands in
for the real thing so the chaining (`-cont` depends on the primary's returned id,
`-cont2` on `-cont`'s) can be checked without touching the cluster's queue.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "mindset_cells.sh"

# The script sbatch-es slurm/serve.slurm et al. `slurm/` is environment-specific
# and untracked, so a fresh clone has no way to run these; skip rather than fail.
pytestmark = pytest.mark.skipif(
    not (REPO / "slurm" / "serve.slurm").exists(),
    reason="slurm/ is environment-specific and untracked",
)

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

RESOURCES = "--gres=gpu:A100-40G:2 --mem=96G --cpus-per-task=16"
FIRST_STUB_ID = 900001

# A stub sbatch: record the invocation, hand back an id that increments with the log,
# so the id the script chains on is a function of call order and the test can predict it.
# $SBATCH_FAIL_AT makes the n-th call fail, which is how the partial-submission
# behaviour gets tested without a real queue.
STUB = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$SBATCH_LOG"
n=$(wc -l < "$SBATCH_LOG")
if [[ -n "${SBATCH_FAIL_AT:-}" && "$n" == "$SBATCH_FAIL_AT" ]]; then
  echo "stub sbatch: simulated failure on call $n" >&2
  exit 1
fi
printf '%d\\n' $((900000 + n))
"""


def _stubbed_env(tmp_path: Path) -> tuple[dict, Path]:
    """A stub sbatch on the front of PATH, plus the log it appends its argv to."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "sbatch"
    stub.write_text(STUB)
    stub.chmod(0o755)
    log = tmp_path / "sbatch.log"
    env = {
        **os.environ,
        "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
        "SBATCH_LOG": str(log),
    }
    # Prove the stub -- and nothing else -- is what `sbatch` resolves to before any
    # code that might submit for real gets to run.
    which = subprocess.run(["bash", "-c", "command -v sbatch"], env=env,
                           capture_output=True, text=True, timeout=30)
    assert which.stdout.strip() == str(stub), which.stdout
    return env, log


@pytest.fixture(scope="module")
def dry_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dry")
    out = tmp / "shards"
    env, log = _stubbed_env(tmp)
    env["SHARD_DIR"] = str(out)
    proc = subprocess.run(["bash", str(SCRIPT), "--dry-run"], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return out, proc.stdout, log


@pytest.fixture(scope="module")
def submit_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("submit")
    out = tmp / "shards"
    env, log = _stubbed_env(tmp)
    env["SHARD_DIR"] = str(out)
    proc = subprocess.run(["bash", str(SCRIPT), "--submit"], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout, log.read_text().splitlines()


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _primary_line(stdout: str, name: str) -> str:
    return next(l for l in stdout.splitlines() if f"--job-name={name} " in l)


@pytest.mark.parametrize("model,version", sorted(CELLS))
def test_shard_configs_differ_from_base_by_exactly_the_varied_keys(dry_run, model, version):
    out, _, _ = dry_run
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


@pytest.mark.parametrize("model,version", sorted(CELLS))
def test_generated_config_is_the_base_file_line_for_line(dry_run, model, version):
    """The YAML check above cannot see comments, and the comments carry the design
    rationale for these cells. Compare the files as lines instead."""
    out, _, _ = dry_run
    base_version, mindset, _nice = CELLS[(model, version)]
    for i in range(3):
        new_lines = (out / f"rollouts-{model}-{version}-s{i}of3.yaml").read_text().splitlines()
        base_lines = (REPO / "configs" / "shards"
                      / f"rollouts-{model}-{base_version}-s{i}of3.yaml").read_text().splitlines()
        assert len(new_lines) > len(base_lines)
        head, tail = new_lines[:len(base_lines)], new_lines[len(base_lines):]

        changed = [n for n, (a, b) in enumerate(zip(base_lines, head)) if a != b]
        assert all(head[n].startswith(("max_tokens:", "out_dir:")) for n in changed), \
            [(base_lines[n], head[n]) for n in changed]
        assert all(base_lines[n].startswith(("max_tokens:", "out_dir:")) for n in changed)
        # out_dir always moves; max_tokens only when the base is not already at 24576.
        assert any(head[n].startswith("out_dir:") for n in changed)

        assert tail[0] == ""
        assert tail[-1] == f"mindset: [{mindset[0]}]"
        assert all(l.startswith("#") for l in tail[1:-1]), tail[1:-1]


def test_dry_run_prints_primary_and_two_continuations_per_shard(dry_run):
    _, stdout, _ = dry_run
    for (model, version), (_b, _m, nice) in CELLS.items():
        for i in range(3):
            name = f"{model}-{version}-s{i}"
            assert f"--job-name={name} " in stdout
            assert f"--job-name={name}-cont " in stdout
            assert f"--job-name={name}-cont2 " in stdout
            line = _primary_line(stdout, name)
            assert "--time=4:00:00" in line and "slurm/serve.slurm" in line
            assert f"--config configs/shards/rollouts-{model}-{version}-s{i}of3.yaml" in line
            assert (f"--nice={nice}" in line) is (nice > 0)
    assert "sbatch --parsable" in stdout


def test_dry_run_chains_each_continuation_onto_the_one_before(dry_run):
    _, stdout, _ = dry_run
    for model, version in CELLS:
        for i in range(3):
            name = f"{model}-{version}-s{i}"
            primary = _primary_line(stdout, name)
            cont = _primary_line(stdout, f"{name}-cont")
            cont2 = _primary_line(stdout, f"{name}-cont2")
            assert "--dependency" not in primary
            assert "--dependency=afterany:PRIMARY" in cont
            assert "--dependency=afterany:CONT" in cont2


def test_dry_run_lines_all_carry_the_resource_and_stage_flags(dry_run):
    _, stdout, _ = dry_run
    for model, version in CELLS:
        for i in range(3):
            cfg = f"configs/shards/rollouts-{model}-{version}-s{i}of3.yaml"
            name = f"{model}-{version}-s{i}"
            for line in (_primary_line(stdout, name), _primary_line(stdout, f"{name}-cont"),
                         _primary_line(stdout, f"{name}-cont2")):
                assert RESOURCES in line
                assert "--gpu-memory-utilization 0.90" in line
                assert f"--stage scripts/run_rollouts.py:{cfg}" in line


def test_dry_run_never_invokes_sbatch(dry_run):
    """The stub is first on PATH, so an empty log is proof the script did not shell out
    to sbatch rather than proof that sbatch happened to be missing."""
    _, _, log = dry_run
    assert not log.exists() or log.read_text() == ""


def test_submit_records_one_sbatch_per_job(submit_run):
    _, calls = submit_run
    assert len(calls) == 3 * 3 * len(CELLS) == 81
    names = [re.search(r"--job-name=(\S+)", c).group(1) for c in calls]
    expected = {f"{model}-{version}-s{i}{suffix}"
                for model, version in CELLS for i in range(3)
                for suffix in ("", "-cont", "-cont2")}
    assert set(names) == expected
    assert len(set(names)) == len(names)


def test_submit_chains_continuations_onto_the_returned_job_ids(submit_run):
    _, calls = submit_run
    for n in range(0, len(calls), 3):
        primary, cont, cont2 = calls[n:n + 3]
        name = re.search(r"--job-name=(\S+)", primary).group(1)
        assert not name.endswith(("-cont", "-cont2"))
        assert "--dependency" not in primary
        # The stub's ids increment with the log, so call n got FIRST_STUB_ID + n.
        assert f"--job-name={name}-cont " in cont + " "
        assert f"--dependency=afterany:{FIRST_STUB_ID + n} " in cont + " "
        assert f"--job-name={name}-cont2 " in cont2 + " "
        assert f"--dependency=afterany:{FIRST_STUB_ID + n + 1} " in cont2 + " "


def test_submit_prints_a_table_row_per_shard(submit_run):
    stdout, _ = submit_run
    rows = [l for l in stdout.splitlines()
            if l.startswith("| ") and not l.startswith("| model ")]
    assert len(rows) == 3 * len(CELLS) == 27
    header = stdout.splitlines().index("| model | version | shard | primary | continuations |")
    assert stdout.splitlines()[header + 1] == "|---|---|---|---|---|"
    for row in rows:
        model, version, shard, primary, conts = [c.strip() for c in row.strip("|").split("|")]
        assert (model, version) in CELLS
        assert shard in {"s0", "s1", "s2"}
        assert primary.isdigit()
        a, b = conts.split(" / ")
        assert a.isdigit() and b.isdigit()
        assert int(a) == int(primary) + 1 and int(b) == int(primary) + 2


def test_preflight_failure_exits_before_submitting_anything(tmp_path):
    """A checkout without slurm/serve.slurm must abort in phase 1: no sbatch, and a
    message that says so, rather than a half-built DAG."""
    env, log = _stubbed_env(tmp_path)
    fake = tmp_path / "repo"
    (fake / "scripts").mkdir(parents=True)
    (fake / "scripts" / "mindset_cells.sh").write_text(SCRIPT.read_text())
    env["SHARD_DIR"] = str(tmp_path / "shards")
    proc = subprocess.run(["bash", str(fake / "scripts" / "mindset_cells.sh"), "--submit"],
                          cwd=fake, env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0
    assert "slurm/serve.slurm" in proc.stderr
    assert "nothing submitted" in proc.stderr
    assert not log.exists() or log.read_text() == ""
    assert proc.stdout == ""


def test_submit_leaves_the_ids_it_already_queued_on_screen_when_sbatch_fails(tmp_path):
    """An sbatch failure part way through aborts the run, and the rows printed so far are
    the only record of what is already queued -- so they have to be on stdout already,
    not held back for an end-of-run summary."""
    env, log = _stubbed_env(tmp_path)
    env["SHARD_DIR"] = str(tmp_path / "shards")
    env["SBATCH_FAIL_AT"] = "10"  # the primary of the 4th shard
    proc = subprocess.run(["bash", str(SCRIPT), "--submit"], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0
    assert len(log.read_text().splitlines()) == 10, "kept submitting past the failure"
    rows = [l for l in proc.stdout.splitlines()
            if l.startswith("| ") and not l.startswith("| model ")]
    assert len(rows) == 3, proc.stdout
    assert "| model | version | shard | primary | continuations |" in proc.stdout
    assert [r.split("|")[4].strip() for r in rows] == ["900001", "900004", "900007"]
