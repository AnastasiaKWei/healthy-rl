"""CPU-only tests for the rollout harness's pure logic.

No GPU, no network, no ``impossiblebench``: only problem ordering (R11), sweep
selection (R4), the tier-ordered condition list, and the JSONL bookkeeping. The
Inspect integration is deliberately not tested here -- it cannot run without a
live vllm-lens server, and a mock of it would only test the mock.
"""

from __future__ import annotations

import json

import pytest

from healthy_rl.rollouts import (
    READOUT_CONDITION,
    JsonlWriter,
    build_conditions,
    completed_items,
    expand_work,
    group_by_epochs,
    hack_rates,
    parse_shard,
    read_jsonl,
    samples_for_tier,
    select_readout_problems,
    select_sweep_from_dir,
    select_sweep_problems,
    shard_items,
    sort_task_ids,
    task_order_key,
)

BENCH_IDS = [f"lcbhard_{i}" for i in range(103)]


def _readout_records(rates, n_samples=6, condition=READOUT_CONDITION):
    """Synthetic readout results: ``rates[task_id]`` is the fraction that passed."""
    records = []
    for task_id, rate in rates.items():
        n_passed = round(rate * n_samples)
        for sample in range(n_samples):
            records.append(
                {
                    "task_id": task_id,
                    "condition_name": condition,
                    "sample": sample,
                    "passed": sample < n_passed,
                }
            )
    return records


# ---------------------------------------------------------------------------
# Ruling R11: order by the trailing integer, not lexicographically
# ---------------------------------------------------------------------------


def test_task_ids_sort_numerically_not_lexicographically():
    ordered = sort_task_ids(BENCH_IDS)
    assert ordered == BENCH_IDS
    # The failure this ruling exists to prevent.
    assert sorted(BENCH_IDS)[:4] == ["lcbhard_0", "lcbhard_1", "lcbhard_10", "lcbhard_100"]


def test_readout_takes_the_first_24_by_integer_suffix():
    problems = select_readout_problems(BENCH_IDS, 24)
    assert problems == [f"lcbhard_{i}" for i in range(24)]
    assert "lcbhard_100" not in problems


def test_readout_selection_is_order_independent():
    shuffled = list(reversed(BENCH_IDS))
    assert select_readout_problems(shuffled, 24) == select_readout_problems(BENCH_IDS, 24)


def test_ids_without_a_trailing_integer_fall_back_to_lexicographic():
    ids = ["lcbhard_2", "lcbhard_10", "zebra", "alpha"]
    assert sort_task_ids(ids) == ["alpha", "lcbhard_2", "lcbhard_10", "zebra"]
    assert task_order_key("alpha")[1] == 1
    assert task_order_key("lcbhard_10")[1] == 0


def test_readout_selection_tolerates_a_short_benchmark():
    assert select_readout_problems(BENCH_IDS[:5], 24) == BENCH_IDS[:5]


# ---------------------------------------------------------------------------
# Hack rates come from readout samples only
# ---------------------------------------------------------------------------


def test_hack_rates_count_only_readout_records():
    records = _readout_records({"lcbhard_0": 0.5}) + _readout_records(
        {"lcbhard_0": 1.0}, condition="desperate+0.05"
    )
    assert hack_rates(records) == {"lcbhard_0": 0.5}


# ---------------------------------------------------------------------------
# Ruling R4: sweep selection, fill, and disqualification
# ---------------------------------------------------------------------------


def test_sweep_takes_twelve_closest_to_half():
    # 20 qualifying problems: rate rises with index, so the ones nearest 0.5 are
    # in the middle of the range.
    rates = {f"lcbhard_{i}": (i + 1) / 21 for i in range(20)}
    selection = select_sweep_problems(rates, 12)

    assert not selection.disqualified
    assert selection.filled == []
    assert len(selection.problems) == 12
    assert selection.problems == sort_task_ids(selection.problems)
    distances = sorted(abs(rates[t] - 0.5) for t in selection.problems)
    excluded = min(abs(rates[t] - 0.5) for t in rates if t not in selection.problems)
    assert distances[-1] <= excluded


def test_sweep_ties_break_by_task_id_in_numeric_order():
    # Every rate is equidistant from 0.5, so the tie-break alone decides.
    rates = {f"lcbhard_{i}": (0.4 if i % 2 else 0.6) for i in range(20)}
    selection = select_sweep_problems(rates, 12)
    assert selection.problems == [f"lcbhard_{i}" for i in range(12)]


def test_sweep_fills_to_twelve_when_too_few_qualify():
    rates = {f"lcbhard_{i}": 0.5 for i in range(5)}
    rates.update({f"lcbhard_{i}": (0.0 if i % 2 else 1.0) for i in range(5, 30)})

    selection = select_sweep_problems(rates, 12)

    assert not selection.disqualified
    assert len(selection.problems) == 12
    assert selection.qualifying == [f"lcbhard_{i}" for i in range(5)]
    assert selection.filled == [f"lcbhard_{i}" for i in range(5, 12)]
    assert all(rates[t] in (0.0, 1.0) for t in selection.filled)
    assert "fill" in selection.reason or "filled" in selection.reason


def test_sweep_fill_stops_at_the_problems_that_exist():
    rates = {"lcbhard_0": 0.5, "lcbhard_1": 1.0, "lcbhard_2": 0.0}
    selection = select_sweep_problems(rates, 12)
    assert selection.problems == ["lcbhard_0", "lcbhard_1", "lcbhard_2"]
    assert not selection.disqualified


@pytest.mark.parametrize("rate", [0.0, 1.0])
def test_sweep_disqualifies_when_nothing_qualifies(rate):
    rates = {f"lcbhard_{i}": rate for i in range(24)}
    selection = select_sweep_problems(rates, 12)

    assert selection.disqualified
    assert selection.problems == []
    assert selection.qualifying == []
    assert "disqualified" in selection.reason


def test_sweep_disqualifies_on_a_mix_of_all_zero_and_all_one():
    rates = {f"lcbhard_{i}": (0.0 if i % 2 else 1.0) for i in range(24)}
    assert select_sweep_problems(rates, 12).disqualified


def test_sweep_selection_runs_off_synthetic_readout_records():
    rates = {f"lcbhard_{i}": i / 6 for i in range(7)}  # 0, 1/6 ... 1
    selection = select_sweep_problems(hack_rates(_readout_records(rates, n_samples=6)), 12)
    assert not selection.disqualified
    # 0 and 1 are degenerate, the five between them qualify, the rest is fill.
    # `qualifying` is in selection order (nearest 0.5 first), not task_id order.
    assert selection.qualifying[0] == "lcbhard_3"
    assert set(selection.qualifying) == {f"lcbhard_{i}" for i in range(1, 6)}
    assert set(selection.filled) == {"lcbhard_0", "lcbhard_6"}


# ---------------------------------------------------------------------------
# Condition list and tier ordering
# ---------------------------------------------------------------------------


def test_conditions_run_in_tier_order():
    conditions = build_conditions()
    tiers = [c.tier for c in conditions]
    assert tiers == sorted(tiers)
    # Tier 2 carries the control, so it must be complete before tier 3 starts.
    assert max(i for i, t in enumerate(tiers) if t == 2) < min(
        i for i, t in enumerate(tiers) if t == 3
    )


def test_condition_list_matches_the_specified_table():
    names = [(c.tier, c.name) for c in build_conditions()]
    assert names == [
        (1, "readout"),
        (2, "desperate+0.05"),
        (2, "desperate-0.05"),
        (2, "calm+0.05"),
        (2, "calm-0.05"),
        (2, "frustrated+0.05"),
        (2, "frustrated-0.05"),
        (3, "desperate+0.1"),
        (3, "desperate-0.1"),
        (3, "calm+0.1"),
        (3, "calm-0.1"),
    ]


def test_readout_is_unsteered_and_sweeps_are_steered():
    readout, *sweep = build_conditions()
    assert readout.tier == 1
    assert readout.emotion is None
    assert readout.as_record() is None
    assert readout.problem_set == "readout"
    assert all(c.steered and c.problem_set == "sweep" for c in sweep)
    assert sweep[0].as_record() == {"emotion": "desperate", "strength": 0.05}


def test_frustrated_control_appears_only_at_tier_two():
    conditions = build_conditions()
    frustrated = [c for c in conditions if c.emotion == "frustrated"]
    assert len(frustrated) == 2
    assert {c.tier for c in frustrated} == {2}


def test_sample_budgets_are_configurable():
    conditions = build_conditions(readout_samples=3, sweep_samples=4)
    assert conditions[0].n_samples == 3
    assert {c.n_samples for c in conditions[1:]} == {4}


# ---------------------------------------------------------------------------
# JSONL bookkeeping: a timeout must truncate the tail, not lose the run
# ---------------------------------------------------------------------------


def test_writer_appends_one_line_per_record(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    with JsonlWriter(path) as writer:
        writer.write({"task_id": "lcbhard_0", "passed": True})
        # Readable, and complete, before the writer is closed.
        assert len(read_jsonl(path)) == 1
        writer.write({"task_id": "lcbhard_1", "passed": False})
    assert [r["task_id"] for r in read_jsonl(path)] == ["lcbhard_0", "lcbhard_1"]


def test_read_jsonl_drops_a_truncated_final_line(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    path.write_text('{"task_id": "lcbhard_0"}\n{"task_id": "lcbhar')
    assert [r["task_id"] for r in read_jsonl(path)] == ["lcbhard_0"]


def test_resume_is_per_rollout_not_per_problem():
    records = _readout_records({"lcbhard_0": 0.5}, n_samples=6)
    records += _readout_records({"lcbhard_1": 0.5}, n_samples=6)[:2]  # interrupted

    done = completed_items(records)
    assert len(done) == 8
    assert (READOUT_CONDITION, "lcbhard_0", 5) in done
    # The interrupted problem keeps the two rollouts it did finish; only the
    # missing four get scheduled again, so no epoch is ever duplicated.
    assert (READOUT_CONDITION, "lcbhard_1", 1) in done
    assert (READOUT_CONDITION, "lcbhard_1", 2) not in done


# ---------------------------------------------------------------------------
# Sharding: split the expanded rollout list across nodes
# ---------------------------------------------------------------------------


def _work(readout=("lcbhard_0", "lcbhard_1"), sweep=("lcbhard_0",), **kwargs):
    return expand_work(
        build_conditions(**kwargs), {"readout": list(readout), "sweep": list(sweep)}
    )


def test_work_expands_to_one_item_per_rollout():
    items = _work(readout=[f"lcbhard_{i}" for i in range(24)],
                  sweep=[f"lcbhard_{i}" for i in range(12)],
                  readout_samples=6, sweep_samples=6)
    # 24 x 6 readout, then 10 steered conditions x 12 problems x 6 samples.
    assert len(items) == 24 * 6 + 10 * 12 * 6
    assert [i.index for i in items] == list(range(len(items)))
    assert [i.tier for i in items] == sorted(i.tier for i in items)


def test_tier_one_indices_do_not_depend_on_the_sweep():
    """A shard has to number tier 1 before the sweep problems are known."""
    readout = [f"lcbhard_{i}" for i in range(24)]
    without = [i for i in _work(readout=readout, sweep=[]) if i.tier == 1]
    with_sweep = [i for i in _work(readout=readout, sweep=readout[:12]) if i.tier == 1]
    assert without == with_sweep


def test_shards_partition_the_work_exactly():
    items = _work(readout=[f"lcbhard_{i}" for i in range(24)],
                  sweep=[f"lcbhard_{i}" for i in range(12)])
    for n in (1, 2, 3, 5):
        pieces = [shard_items(items, i, n) for i in range(n)]
        flat = [item for piece in pieces for item in piece]
        assert sorted(i.index for i in flat) == [i.index for i in items]
        assert len({i.index for i in flat}) == len(items)
        # Balanced to within one item, so no node is left holding the bag.
        assert max(map(len, pieces)) - min(map(len, pieces)) <= 1


def test_every_shard_gets_tier_one_work():
    """Sharding over rollouts, not over tiers, is what makes this true."""
    items = _work(readout=[f"lcbhard_{i}" for i in range(24)],
                  sweep=[f"lcbhard_{i}" for i in range(12)])
    for i in range(4):
        tiers = {item.tier for item in shard_items(items, i, 4)}
        assert tiers == {1, 2, 3}


@pytest.mark.parametrize(
    "text,expected",
    [(None, (0, 1)), ("", (0, 1)), ("0/1", (0, 1)), ("2/3", (2, 3)), (" 1/4 ", (1, 4))],
)
def test_parse_shard(text, expected):
    assert parse_shard(text) == expected


@pytest.mark.parametrize("bad", ["3/3", "-1/2", "2", "a/b", "1/0"])
def test_parse_shard_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        parse_shard(bad)


def test_group_by_epochs_batches_uneven_shares():
    from healthy_rl.rollouts import WorkItem

    items = [
        WorkItem(0, "readout", 1, "lcbhard_0", 0),
        WorkItem(1, "readout", 1, "lcbhard_0", 3),
        WorkItem(2, "readout", 1, "lcbhard_1", 1),
    ]
    grouped = group_by_epochs(items)
    assert grouped == {2: {"lcbhard_0": [0, 3]}, 1: {"lcbhard_1": [1]}}


# ---------------------------------------------------------------------------
# Per-tier sample budgets
# ---------------------------------------------------------------------------


def test_samples_per_problem_accepts_a_mapping_or_a_scalar():
    assert samples_for_tier({"samples_per_problem": {1: 12, 2: 8, 3: 8}}, 1) == 12
    assert samples_for_tier({"samples_per_problem": {"1": 12, "2": 8}}, 2) == 8
    assert samples_for_tier({"samples_per_problem": 5}, 3) == 5


def test_samples_per_problem_falls_back_to_the_older_keys():
    cfg = {"readout_samples": 6, "sweep_samples": 4}
    assert samples_for_tier(cfg, 1) == 6
    assert samples_for_tier(cfg, 2) == 4
    assert samples_for_tier({}, 1) == 6


# ---------------------------------------------------------------------------
# Two-phase launch (R26): apply the selection rule once, to the whole readout
# ---------------------------------------------------------------------------

SELECT_CFG = {"readout_problems": 6, "sweep_problems": 3, "samples_per_problem": {1: 4}}


def _write_shards(tmp_path, rates, n_samples=4, n_shards=2):
    """Spread readout records over n_shards files, the way a sharded run does."""
    records = _readout_records(rates, n_samples=n_samples)
    handles = [(tmp_path / f"rollouts.shard{i}of{n_shards}.jsonl").open("w") for i in range(n_shards)]
    for index, record in enumerate(records):
        handles[index % n_shards].write(json.dumps(record) + "\n")
    for handle in handles:
        handle.close()
    return records


def test_selection_reads_every_shard_file(tmp_path):
    rates = {f"lcbhard_{i}": r for i, r in enumerate([0.0, 0.5, 0.25, 0.75, 1.0, 0.5])}
    _write_shards(tmp_path, rates, n_shards=3)

    report = select_sweep_from_dir(tmp_path, SELECT_CFG)

    assert len(report["shard_files"]) == 3
    assert report["n_readout_records"] == 24 == report["n_expected_records"]
    assert report["complete"] and not report["disqualified"]
    # Both problems at 0.5, plus the 0.25/0.75 pair's tie-break winner. The list
    # itself comes back in R11 order, not selection order.
    assert report["problems"] == ["lcbhard_1", "lcbhard_2", "lcbhard_5"]
    assert report["sweep"]["qualifying"][:2] == ["lcbhard_1", "lcbhard_5"]
    assert set(report["selected_rates"]) == set(report["problems"])


def test_selection_is_identical_however_the_records_are_split(tmp_path):
    """This is the property the two-phase launch buys: one rule, one answer."""
    rates = {f"lcbhard_{i}": (i % 5) / 4 for i in range(6)}
    chosen = []
    for n_shards in (1, 2, 3, 5):
        for stale in tmp_path.glob("rollouts*.jsonl"):
            stale.unlink()
        _write_shards(tmp_path, rates, n_shards=n_shards)
        chosen.append(select_sweep_from_dir(tmp_path, SELECT_CFG)["problems"])
    assert len(set(map(tuple, chosen))) == 1


def test_selection_flags_a_partial_readout(tmp_path):
    rates = {f"lcbhard_{i}": 0.5 for i in range(6)}
    _write_shards(tmp_path, rates, n_shards=2)
    # Simulate a shard that has not finished: drop one of its files.
    (tmp_path / "rollouts.shard1of2.jsonl").unlink()

    report = select_sweep_from_dir(tmp_path, SELECT_CFG)

    assert not report["complete"]
    assert report["n_readout_records"] < report["n_expected_records"]
    assert report["problems_with_missing_samples"]
    # It still produces a selection -- the caller decides whether to accept it.
    assert len(report["problems"]) == 3


def test_selection_reports_problems_with_no_records_at_all(tmp_path):
    _write_shards(tmp_path, {f"lcbhard_{i}": 0.5 for i in range(4)})
    report = select_sweep_from_dir(tmp_path, SELECT_CFG, BENCH_IDS)
    assert report["problems_with_no_records"] == ["lcbhard_4", "lcbhard_5"]
    assert not report["complete"]


def test_selection_reports_disqualification(tmp_path):
    _write_shards(tmp_path, {f"lcbhard_{i}": float(i % 2) for i in range(6)})
    report = select_sweep_from_dir(tmp_path, SELECT_CFG)
    assert report["disqualified"]
    assert report["problems"] == []
    assert report["complete"]  # the readout is complete; the model just never varies


def test_selection_on_an_empty_directory_is_disqualified_not_a_crash(tmp_path):
    report = select_sweep_from_dir(tmp_path, SELECT_CFG)
    assert report["disqualified"] and report["n_readout_records"] == 0
    assert not report["complete"]
