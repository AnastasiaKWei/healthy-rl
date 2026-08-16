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

# (model, version) -> (base version, mindset, nice, text). `text` is which prompt
# text the cell runs: `orig` for the 18 cells run overnight on 2026-08-16 with the
# v2 mindset blocks as of 9d6615a, `fixed` for the `…b` re-runs of the same 18 with
# the trigger sentence Anastasia repaired in 7d6fd07. Same model/base/mindset/nice
# in each mirrored pair -- the prompt text is the only thing that moves.
CELLS = {
    ("Ministral-3-14B-Reasoning-2512", "growth6"): ("d6", ["growth"], 0, "orig"),
    ("Ministral-3-14B-Reasoning-2512", "resil6"): ("d6", ["resilience"], 0, "orig"),
    ("Ministral-3-14B-Reasoning-2512", "appr6"): ("d6", ["appraisal"], 0, "orig"),
    ("Qwen3.5-9B", "growth6"): ("d6", ["growth"], 2000, "orig"),
    ("Qwen3.5-9B", "resil6"): ("d6", ["resilience"], 2000, "orig"),
    ("Qwen3.5-9B", "appr6"): ("d6", ["appraisal"], 2000, "orig"),
    ("Ministral-3-14B-Reasoning-2512", "affgrowth6"): ("aff6", ["growth"], 4000, "orig"),
    ("Ministral-3-14B-Reasoning-2512", "affresil6"): ("aff6", ["resilience"], 4000, "orig"),
    ("Ministral-3-14B-Reasoning-2512", "affappr6"): ("aff6", ["appraisal"], 4000, "orig"),
    ("gemma-3-12b-it", "spgrowth6"): ("sp6", ["growth"], 0, "orig"),
    ("gemma-3-12b-it", "spresil6"): ("sp6", ["resilience"], 0, "orig"),
    ("gemma-3-12b-it", "spappr6"): ("sp6", ["appraisal"], 0, "orig"),
    ("gemma-3-12b-it", "affgrowth6"): ("aff6", ["growth"], 2000, "orig"),
    ("gemma-3-12b-it", "affresil6"): ("aff6", ["resilience"], 2000, "orig"),
    ("gemma-3-12b-it", "affappr6"): ("aff6", ["appraisal"], 2000, "orig"),
    ("gemma-3-12b-it", "spaffgrowth6"): ("spaff6", ["growth"], 0, "orig"),
    ("gemma-3-12b-it", "spaffresil6"): ("spaff6", ["resilience"], 0, "orig"),
    ("gemma-3-12b-it", "spaffappr6"): ("spaff6", ["appraisal"], 0, "orig"),
    ("Ministral-3-14B-Reasoning-2512", "growth6b"): ("d6", ["growth"], 0, "fixed"),
    ("Ministral-3-14B-Reasoning-2512", "resil6b"): ("d6", ["resilience"], 0, "fixed"),
    ("Ministral-3-14B-Reasoning-2512", "appr6b"): ("d6", ["appraisal"], 0, "fixed"),
    ("Qwen3.5-9B", "growth6b"): ("d6", ["growth"], 2000, "fixed"),
    ("Qwen3.5-9B", "resil6b"): ("d6", ["resilience"], 2000, "fixed"),
    ("Qwen3.5-9B", "appr6b"): ("d6", ["appraisal"], 2000, "fixed"),
    ("Ministral-3-14B-Reasoning-2512", "affgrowth6b"): ("aff6", ["growth"], 4000, "fixed"),
    ("Ministral-3-14B-Reasoning-2512", "affresil6b"): ("aff6", ["resilience"], 4000, "fixed"),
    ("Ministral-3-14B-Reasoning-2512", "affappr6b"): ("aff6", ["appraisal"], 4000, "fixed"),
    ("gemma-3-12b-it", "spgrowth6b"): ("sp6", ["growth"], 0, "fixed"),
    ("gemma-3-12b-it", "spresil6b"): ("sp6", ["resilience"], 0, "fixed"),
    ("gemma-3-12b-it", "spappr6b"): ("sp6", ["appraisal"], 0, "fixed"),
    ("gemma-3-12b-it", "affgrowth6b"): ("aff6", ["growth"], 2000, "fixed"),
    ("gemma-3-12b-it", "affresil6b"): ("aff6", ["resilience"], 2000, "fixed"),
    ("gemma-3-12b-it", "affappr6b"): ("aff6", ["appraisal"], 2000, "fixed"),
    ("gemma-3-12b-it", "spaffgrowth6b"): ("spaff6", ["growth"], 0, "fixed"),
    ("gemma-3-12b-it", "spaffresil6b"): ("spaff6", ["resilience"], 0, "fixed"),
    ("gemma-3-12b-it", "spaffappr6b"): ("spaff6", ["appraisal"], 0, "fixed"),
}
# Every count below is derived from CELLS, so adding a row here is the only edit a
# new cell needs: 3 shards a cell, 3 jobs a shard.
SHARDS = 3 * len(CELLS)
SBATCH_LINES = 3 * SHARDS
ORIG = {k for k, v in CELLS.items() if v[3] == "orig"}
FIXED = {k for k, v in CELLS.items() if v[3] == "fixed"}
FIX_COMMIT = "7d6fd07"

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
    base_version, mindset, _nice, _text = CELLS[(model, version)]
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
    base_version, mindset, _nice, _text = CELLS[(model, version)]
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


def _appended_block(out: Path, model: str, version: str, base_version: str, i: int) -> str:
    """The text the script appended past the last line of the base config."""
    new_lines = (out / f"rollouts-{model}-{version}-s{i}of3.yaml").read_text().splitlines()
    base_lines = (REPO / "configs" / "shards"
                  / f"rollouts-{model}-{base_version}-s{i}of3.yaml").read_text().splitlines()
    return "\n".join(new_lines[len(base_lines):])


@pytest.mark.parametrize("model,version", sorted(ORIG))
def test_orig_cells_keep_the_comment_they_were_generated_with(dry_run, model, version):
    """The 18 orig configs are committed and the script refuses to overwrite a config
    that differs, so their appended block must not drift -- in particular it must not
    pick up the fixed cells' commit reference."""
    out, _, _ = dry_run
    base_version, mindset, _nice, text = CELLS[(model, version)]
    assert text == "orig"
    for i in range(3):
        block = _appended_block(out, model, version, base_version, i)
        assert f"# --- MINDSET ARM: {mindset[0]} (prompt v2) " in block
        assert FIX_COMMIT not in block
        assert "trigger" not in block
        assert block.endswith(f"\nmindset: [{mindset[0]}]")


@pytest.mark.parametrize("model,version", sorted(FIXED))
def test_fixed_cells_document_the_trigger_fix_and_their_comparison_cells(dry_run, model, version):
    """A fixed cell's appended comment is the only place the run itself records WHY
    there are two versions of each arm: which commit changed the prompt, what changed
    in it, which cell it re-runs, and what it is compared against."""
    out, _, _ = dry_run
    base_version, mindset, _nice, text = CELLS[(model, version)]
    assert text == "fixed"
    orig_version = version[:-1]
    assert (model, orig_version) in ORIG, "every fixed cell mirrors an orig one"
    pertok = f"{model}/spaff6" if base_version == "spaff6" else f"{model}/{base_version}r"
    for i in range(3):
        block = _appended_block(out, model, version, base_version, i)
        lines = block.splitlines()
        assert lines[0] == ""
        assert all(l.startswith("#") for l in lines[1:-1]), lines[1:-1]
        assert lines[-1] == f"mindset: [{mindset[0]}]"

        assert (f"# --- MINDSET ARM: {mindset[0]} (prompt v2, trigger phrasing fixed "
                f"2026-08-16, astwei {FIX_COMMIT})") in block
        # The fix itself: a standing rule, quoted, in place of the event-conditional
        # wording that went inert when the block stopped repeating every turn.
        assert ('"Your first attempt is just the code. Open every attempt after that '
                'with') in block
        assert '"whenever a test fails"' in block
        assert "inert" in block and "turn 1" in block
        # What this cell is: a re-run of the same arm, and what to compare it against.
        assert f"re-runs {model}/{orig_version} " in block
        assert "per-token comparison cell is " in block
        assert pertok in block
        if base_version == "spaff6":
            assert "already" in block, "spaff6 itself carries the per-token arrays"
        if model == "Qwen3.5-9B":
            assert "does not exist yet" in block, "no Qwen d6r has been built"
        # Unchanged from the orig cells: what the block is and where it is inserted.
        assert "healthy_rl.rollouts.MINDSET" in block
        assert "24576" in block


def test_dry_run_prints_primary_and_two_continuations_per_shard(dry_run):
    _, stdout, _ = dry_run
    for (model, version), (_b, _m, nice, _t) in CELLS.items():
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
    assert len(calls) == SBATCH_LINES
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
    assert len(rows) == SHARDS
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


def test_only_model_restricts_both_phases_to_that_model(tmp_path):
    """A re-run after some cells are already queued must not touch the others: with
    ONLY_MODEL set, the filter has to apply to the config writing as well as the
    submission, or the second run would rewrite (or refuse to rewrite) configs whose
    jobs are in the queue."""
    only = "gemma-3-12b-it"
    n_cells = sum(1 for model, _v in CELLS if model == only)
    assert n_cells == 18, "this test is about the gemma cells, orig and fixed"

    env, log = _stubbed_env(tmp_path)
    out = tmp_path / "shards"
    env["SHARD_DIR"] = str(out)
    env["ONLY_MODEL"] = only
    proc = subprocess.run(["bash", str(SCRIPT), "--dry-run"], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert only in proc.stderr.splitlines()[0]

    written = sorted(pth.name for pth in out.iterdir())
    assert len(written) == 3 * n_cells == 54, written
    assert all(name.startswith(f"rollouts-{only}-") for name in written), written
    expected = {f"rollouts-{only}-{version}-s{i}of3.yaml"
                for model, version in CELLS if model == only for i in range(3)}
    assert set(written) == expected

    lines = [l for l in proc.stdout.splitlines() if l.startswith("sbatch --parsable")]
    assert len(lines) == proc.stdout.count("\n") == 3 * 3 * n_cells == 162
    assert all(only in l for l in lines)
    assert not [l for l in lines if "Ministral" in l or "Qwen" in l]
    assert not log.exists() or log.read_text() == ""


def test_only_model_and_only_base_together_keep_only_the_rows_matching_both(tmp_path):
    """The two filters intersect: the gemma spaff6 cells are six of the eighteen gemma
    cells and the only six built on that base, so setting both has to leave exactly
    those -- neither every gemma cell nor every spaff6-named row of another model."""
    only_model, only_base = "gemma-3-12b-it", "spaff6"
    cells = [(m, v) for (m, v), (b, _m, _n, _t) in CELLS.items()
             if m == only_model and b == only_base]
    assert len(cells) == 6, cells

    env, log = _stubbed_env(tmp_path)
    out = tmp_path / "shards"
    env["SHARD_DIR"] = str(out)
    env["ONLY_MODEL"] = only_model
    env["ONLY_BASE"] = only_base
    proc = subprocess.run(["bash", str(SCRIPT), "--dry-run"], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    scope = proc.stderr.splitlines()[0]
    assert only_model in scope and only_base in scope, scope

    written = sorted(pth.name for pth in out.iterdir())
    assert len(written) == 18, written
    assert set(written) == {f"rollouts-{only_model}-{version}-s{i}of3.yaml"
                            for _m, version in cells for i in range(3)}
    assert all("spaff" in name for name in written), written

    lines = [l for l in proc.stdout.splitlines() if l.startswith("sbatch --parsable")]
    assert len(lines) == proc.stdout.count("\n") == 54
    assert all("spaff" in l for l in lines), lines
    # The aff6 and sp6 gemma cells share a model with these but not a base.
    assert not [l for l in lines if re.search(r"-(aff|sp)(growth|resil|appr)6-", l)]
    assert not log.exists() or log.read_text() == ""


def test_only_base_that_matches_nothing_is_an_error(tmp_path):
    """Same guard as ONLY_MODEL: a base that no row is built on, or a model/base pair
    that never co-occurs, must fail loudly rather than submit nothing quietly."""
    env, log = _stubbed_env(tmp_path)
    env["SHARD_DIR"] = str(tmp_path / "shards")
    env["ONLY_MODEL"] = "Qwen3.5-9B"
    env["ONLY_BASE"] = "spaff6"  # a real base, but not one Qwen has a row for
    proc = subprocess.run(["bash", str(SCRIPT), "--submit"], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0
    assert "matches no row" in proc.stderr
    assert "nothing submitted" in proc.stderr
    assert not log.exists() or log.read_text() == ""


def test_only_model_that_matches_nothing_is_an_error(tmp_path):
    """A typo'd model name would otherwise be a silent no-op run."""
    env, log = _stubbed_env(tmp_path)
    env["SHARD_DIR"] = str(tmp_path / "shards")
    env["ONLY_MODEL"] = "gemma-3-12b"  # a real name, minus the -it
    proc = subprocess.run(["bash", str(SCRIPT), "--submit"], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0
    assert "matches no row" in proc.stderr
    assert "nothing submitted" in proc.stderr
    assert not log.exists() or log.read_text() == ""



@pytest.mark.parametrize("only_text,expected", [("fixed", "FIXED"), ("orig", "ORIG")])
def test_only_text_restricts_both_phases_to_that_prompt_text(tmp_path, only_text, expected):
    """ONLY_TEXT is what makes it possible to queue the 18 re-runs without touching the
    18 configs whose jobs are already in the queue: like ONLY_MODEL it has to filter the
    config writing as well as the submission."""
    cells = {"FIXED": FIXED, "ORIG": ORIG}[expected]
    assert len(cells) == 18

    env, log = _stubbed_env(tmp_path)
    out = tmp_path / "shards"
    env["SHARD_DIR"] = str(out)
    env["ONLY_TEXT"] = only_text
    proc = subprocess.run(["bash", str(SCRIPT), "--dry-run"], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert only_text in proc.stderr.splitlines()[0]

    written = sorted(pth.name for pth in out.iterdir())
    assert len(written) == 54, written
    assert set(written) == {f"rollouts-{model}-{version}-s{i}of3.yaml"
                            for model, version in cells for i in range(3)}
    b_named = [name for name in written if re.search(r"6b-s\dof3\.yaml$", name)]
    assert len(b_named) == (54 if only_text == "fixed" else 0), b_named

    lines = [l for l in proc.stdout.splitlines() if l.startswith("sbatch --parsable")]
    assert len(lines) == proc.stdout.count("\n") == 162
    names = {re.search(r"--job-name=(\S+)", l).group(1) for l in lines}
    assert names == {f"{model}-{version}-s{i}{suffix}"
                     for model, version in cells for i in range(3)
                     for suffix in ("", "-cont", "-cont2")}
    assert not log.exists() or log.read_text() == ""


def test_only_text_and_only_model_together_keep_only_the_rows_matching_both(tmp_path):
    """The third filter composes with the other two the same way they compose with each
    other: gemma has 9 fixed cells out of 18 gemma cells and 18 fixed cells overall, so
    setting both leaves exactly those 9."""
    only_model, only_text = "gemma-3-12b-it", "fixed"
    cells = [(m, v) for m, v in FIXED if m == only_model]
    assert len(cells) == 9, cells

    env, log = _stubbed_env(tmp_path)
    out = tmp_path / "shards"
    env["SHARD_DIR"] = str(out)
    env["ONLY_MODEL"] = only_model
    env["ONLY_TEXT"] = only_text
    proc = subprocess.run(["bash", str(SCRIPT), "--dry-run"], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    scope = proc.stderr.splitlines()[0]
    assert only_model in scope and only_text in scope, scope

    written = sorted(pth.name for pth in out.iterdir())
    assert len(written) == 27, written
    assert set(written) == {f"rollouts-{only_model}-{version}-s{i}of3.yaml"
                            for _m, version in cells for i in range(3)}
    assert all(name.startswith(f"rollouts-{only_model}-") for name in written), written
    assert all(re.search(r"6b-s\dof3\.yaml$", name) for name in written), written

    lines = [l for l in proc.stdout.splitlines() if l.startswith("sbatch --parsable")]
    assert len(lines) == proc.stdout.count("\n") == 81
    assert all(only_model in l for l in lines)
    # The gemma orig cells share a model with these, the Ministral/Qwen fixed cells
    # share the text; neither may appear.
    assert not [l for l in lines if "Ministral" in l or "Qwen" in l]
    assert not [l for l in lines if re.search(r"-(sp|aff|spaff)?(growth|resil|appr)6-", l)]
    assert not log.exists() or log.read_text() == ""


def test_only_text_that_matches_nothing_is_an_error(tmp_path):
    """Same guard as the other two filters: a text name no row carries must fail loudly
    in phase 1 rather than submit nothing quietly."""
    env, log = _stubbed_env(tmp_path)
    env["SHARD_DIR"] = str(tmp_path / "shards")
    env["ONLY_TEXT"] = "new"  # the values are orig / fixed
    proc = subprocess.run(["bash", str(SCRIPT), "--submit"], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0
    assert "matches no row" in proc.stderr
    assert "ONLY_TEXT='new'" in proc.stderr
    assert "nothing submitted" in proc.stderr
    assert not log.exists() or log.read_text() == ""
    assert proc.stdout == ""
