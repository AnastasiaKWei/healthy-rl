"""CPU-only tests for the rollout harness's pure logic.

No GPU, no network, no ``impossiblebench``: only problem ordering (R11), sweep
selection (R4), the tier-ordered condition list, and the JSONL bookkeeping. The
Inspect integration is deliberately not tested here -- it cannot run without a
live vllm-lens server, and a mock of it would only test the mock.
"""

from __future__ import annotations

import pytest

from healthy_rl.rollouts import (
    READOUT_CONDITION,
    JsonlWriter,
    build_conditions,
    compact_records,
    completed_groups,
    hack_rates,
    read_jsonl,
    select_readout_problems,
    select_sweep_problems,
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


def test_resume_keeps_only_complete_groups():
    records = _readout_records({"lcbhard_0": 0.5}, n_samples=6)
    records += _readout_records({"lcbhard_1": 0.5}, n_samples=6)[:2]  # interrupted

    done = completed_groups(records, 6)
    assert done == {(READOUT_CONDITION, "lcbhard_0")}

    kept = compact_records(records, done)
    assert {r["task_id"] for r in kept} == {"lcbhard_0"}
