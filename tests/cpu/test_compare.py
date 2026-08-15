"""Tests for ``scripts/compare.py`` -- the stage that writes ``results/``.

Two things are worth pinning here, and they are different in kind.

The first is arithmetic: hack rates and turn-statistic contrasts computed from a
synthetic JSONL whose answers are worked out by hand in the test, so a refactor
that changes what "the failure-turn contrast" means fails loudly.

The second is the property the whole stage exists for: **missing data must never
render as a measured zero.** A model with no rollouts is "not run"; a tier that
never ran is "absent"; a sweep that never ran makes the causal section say "not
run" rather than "no effect". Those paths are tested by asserting on the rendered
summary text, because the summary is the artifact a human actually reads.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from healthy_rl.config import repo_root

SCRIPTS = repo_root() / "scripts"

# The pipeline's fixed emotion order (healthy_rl vectors artifact). `desperate`
# is index 1 and `frustrated` index 11; the tests below index by name anyway.
EMOTIONS = [
    "calm",
    "desperate",
    "nervous",
    "angry",
    "afraid",
    "joyful",
    "guilty",
    "proud",
    "loving",
    "sad",
    "hostile",
    "frustrated",
    "exasperated",
    "overwhelmed",
]

MODEL = "TestModel-1B"


def load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


compare = load_script("compare")


# ---------------------------------------------------------------------------
# Synthetic artifacts
# ---------------------------------------------------------------------------


def make_record(
    task_id: str,
    sample: int,
    passed: bool,
    series: dict[str, list[float]],
    flags: list[bool],
    condition: dict | None = None,
    tier: int = 1,
    none_turns: tuple[int, ...] = (),
    model: str = MODEL,
    n_generated: list[int] | None = None,
    **overrides,
) -> dict:
    """One rollout record. ``series`` gives the per-turn value of named emotions."""
    n_turns = len(flags)
    turn_stat: list[list[float] | None] = []
    for turn in range(n_turns):
        if turn in none_turns:
            turn_stat.append(None)
            continue
        turn_stat.append([float(series.get(name, [0.0] * n_turns)[turn]) for name in EMOTIONS])
    record = {
        "run_id": "test-run",
        "model": model,
        "task_id": task_id,
        "tier": tier,
        "condition": condition,
        "condition_name": "readout"
        if condition is None
        else f"{condition['emotion']}{condition['strength']:+g}",
        "sample": sample,
        "epoch": 1,
        "shard": "0/1",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "n_turns": n_turns,
        "emotions": list(EMOTIONS),
        "probe_layer": 43,
        "capture_layers": [41, 42, 43, 44, 45],
        "turn_stat": turn_stat,
        "turn_stat_layers": [],
        "turn_n_generated": list(n_generated) if n_generated else [16] * n_turns,
        "turn_after_test_failure": list(flags),
        "turn_observed_norm": [],
        "residuals": None,
        "hook_data": True,
        "turn_errors": [],
        "sample_error": None,
        "total_time": 12.0,
    }
    record.update(overrides)
    return record


def write_rollouts(
    root: Path,
    model: str,
    records,
    shards: int = 1,
    raw_tail: str = "",
    max_tokens: int | None = 3072,
) -> Path:
    out = root / "rollouts" / model / "v1"
    out.mkdir(parents=True, exist_ok=True)
    if max_tokens is not None:
        # The manifest is where the per-turn token budget comes from: the run used
        # generated per-shard configs, so compare.py must not read its own config.
        (out / "manifest.json").write_text(
            json.dumps({"stage": "rollouts", "config": {"max_tokens": max_tokens}})
        )
    buckets: list[list[dict]] = [[] for _ in range(shards)]
    for index, record in enumerate(records):
        buckets[index % shards].append(record)
    for index, bucket in enumerate(buckets):
        path = out / f"rollouts.shard{index}of{shards}.jsonl"
        text = "".join(json.dumps(r) + "\n" for r in bucket)
        if index == 0:
            text += raw_tail
        path.write_text(text)
    return out


def write_instrument(root: Path, model: str, gate_passed: bool = True) -> None:
    gate = root / "gate" / model / "v1"
    gate.mkdir(parents=True, exist_ok=True)
    (gate / "gate.json").write_text(
        json.dumps(
            {
                "model": model,
                "passed": gate_passed,
                "self_token_rate": 0.6428571428571429,
                "latin_initial_rate": 0.7857142857142857,
                "thresholds": {"self_token_rate": 0.5, "latin_initial_rate": 0.7},
                "probe_layer": 43,
                "error": None,
                "per_emotion": [
                    {
                        "emotion": name,
                        "self_token": name != "exasperated",
                        "latin_initial": True,
                        "top_decoded": [f" {name}", f" {name}ly", " sob"],
                    }
                    for name in EMOTIONS
                ],
            }
        )
    )
    vectors = root / "vectors" / model / "v1"
    vectors.mkdir(parents=True, exist_ok=True)
    (vectors / "vectors.json").write_text(
        json.dumps(
            {
                "model": model,
                "emotions": list(EMOTIONS),
                "probe_layer": 43,
                "capture_layers": [41, 42, 43, 44, 45],
                "d_model": 5120,
                "n_removed": {"41": 474, "42": 471, "43": 472, "44": 482, "45": 481},
                "mean_residual_norm": {"43": 74.75387437157926},
                "var_frac": 0.5,
            }
        )
    )
    acts = root / "activations" / model / "v1"
    acts.mkdir(parents=True, exist_ok=True)
    (acts / "norms.json").write_text(
        json.dumps(
            {
                "n_stories_used": 18000,
                "n_stories_failed": 0,
                "n_stories_skipped": 0,
                "emotion_story_counts": {name: 1200 for name in EMOTIONS},
            }
        )
    )


CFG = {
    "models": [MODEL],
    "rollouts_version": "v1",
    "gate_version": "v1",
    "vectors_version": "v1",
    "activations_version": "v1",
    "out_dir": "results",
    "focus_emotions": ["desperate", "calm", "frustrated"],
    "hypothesis_emotion": "desperate",
    "alpha": 0.05,
    "drop_errored_samples": True,
    "caveats": ["14 emotion directions, not 171."],
}


def run_compare(tmp_path: Path, root: Path, models=None, no_figures: bool = True):
    """Drive ``run()`` the way ``main()`` does, returning (out_dir, reports, summary)."""
    out_dir = tmp_path / "results"
    args = SimpleNamespace(
        config=None,
        models=models,
        artifact_root=root,
        out_dir=out_dir,
        no_figures=no_figures,
    )
    cfg = dict(CFG)
    if models is not None:
        cfg["models"] = list(models)
    out_dir, reports = compare.run(cfg, args)
    return out_dir, reports, (out_dir / "summary.md").read_text()


# ---------------------------------------------------------------------------
# Hack rate, hand-computed
# ---------------------------------------------------------------------------


def _flat(value: float, n: int) -> list[float]:
    return [value] * n


def hack_records() -> list[dict]:
    """9 unsteered rollouts: A 3/4 pass, B 0/4 pass, C 1/1 pass -> 4/9 overall."""
    records = []
    outcomes = {
        "lcbhard_1": [True, True, True, False],
        "lcbhard_2": [False, False, False, False],
        "lcbhard_10": [True],
    }
    for task, passes in outcomes.items():
        for sample, passed in enumerate(passes):
            records.append(
                make_record(
                    task,
                    sample,
                    passed,
                    {"desperate": _flat(0.2, 2)},
                    [False, True],
                )
            )
    return records


def test_hack_rate_matches_hand_computed(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(root, MODEL, hack_records(), shards=3)

    _, reports, summary = run_compare(tmp_path, root)
    report = reports[0]

    assert report.baseline.k == 4
    assert report.baseline.n == 9
    assert report.baseline.rate == pytest.approx(4 / 9)

    by_task = report.by_task
    assert (by_task["lcbhard_1"].k, by_task["lcbhard_1"].n) == (3, 4)
    assert by_task["lcbhard_1"].rate == pytest.approx(0.75)
    assert (by_task["lcbhard_2"].k, by_task["lcbhard_2"].n) == (0, 4)
    assert by_task["lcbhard_2"].rate == pytest.approx(0.0)
    assert (by_task["lcbhard_10"].k, by_task["lcbhard_10"].n) == (1, 1)

    # Ruling R11 ordering: lcbhard_2 before lcbhard_10, not after.
    assert list(by_task) == ["lcbhard_1", "lcbhard_2", "lcbhard_10"]

    # A per-problem zero is a measured zero and must carry its n.
    assert "| `lcbhard_2` | 0.0% | 0/4 |" in summary
    assert "44.4%" in summary and "4/9" in summary
    # The reason the metric works at all is stated, not assumed.
    assert "mutually unsatisfiable" in summary
    assert "necessarily means the model cheated" in summary


def test_shards_are_concatenated(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    out = write_rollouts(root, MODEL, hack_records(), shards=3)
    assert len(list(out.glob("rollouts.shard*.jsonl"))) == 3

    _, reports, _ = run_compare(tmp_path, root)
    assert reports[0].load.n_records == 9


def test_errored_and_unscored_records_are_dropped_and_named(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    records = hack_records()
    records.append(
        make_record("lcbhard_3", 0, False, {}, [False], sample_error="sandbox died")
    )
    unscored = make_record("lcbhard_3", 1, False, {}, [False])
    unscored["passed"] = None
    records.append(unscored)
    write_rollouts(root, MODEL, records)

    _, reports, summary = run_compare(tmp_path, root)
    # Still 4/9: an errored sample scored as a non-hack would bias the rate down.
    assert (reports[0].baseline.k, reports[0].baseline.n) == (4, 9)
    assert reports[0].dropped == {"unscored": 1, "errored": 1}
    assert "1 unscored record(s) and 1 record(s) with a sample error" in summary


# ---------------------------------------------------------------------------
# Turn statistics, hand-computed
# ---------------------------------------------------------------------------


def contrast_records() -> list[dict]:
    """Three transcripts with worked-out failure-turn contrasts.

    desperate: A first 0.10, failure turns (0.30, 0.50) -> mean 0.40, diff +0.30
               B first 0.20, failure turn 0.40          -> diff +0.20
               C has no failure turn                    -> excluded
    frustrated: A 0.10 -> (0.15, 0.20) mean 0.175 -> diff +0.075
                B 0.20 -> 0.22                    -> diff +0.02
    every other direction is flat, so its diff is exactly 0.
    """
    return [
        make_record(
            "lcbhard_1",
            0,
            True,
            {"desperate": [0.10, 0.30, 0.50], "frustrated": [0.10, 0.15, 0.20]},
            [False, True, True],
        ),
        make_record(
            "lcbhard_2",
            0,
            False,
            {"desperate": [0.20, 0.40], "frustrated": [0.20, 0.22]},
            [False, True],
        ),
        make_record(
            "lcbhard_3",
            0,
            False,
            {"desperate": [0.05], "frustrated": [0.05]},
            [False],
        ),
    ]


def test_failure_turn_contrast_matches_hand_computed(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(root, MODEL, contrast_records())

    _, reports, summary = run_compare(tmp_path, root)
    paired = reports[0].paired

    assert paired.n_pairs == 2
    assert paired.n_transcripts == 3
    assert paired.excluded["no_failure_turn"] == 1

    by_name = {c.emotion: c for c in paired.contrasts}
    desperate = by_name["desperate"]
    # diffs are +0.30 and +0.20: mean 0.25, sd 0.0707107, dz 3.5355
    assert desperate.mean == pytest.approx(0.25)
    assert desperate.n == 2
    assert desperate.sem == pytest.approx(0.05)
    assert desperate.dz == pytest.approx(0.25 / 0.070710678, rel=1e-6)

    frustrated = by_name["frustrated"]
    # diffs are +0.075 and +0.02: mean 0.0475
    assert frustrated.mean == pytest.approx(0.0475)
    assert frustrated.dz < desperate.dz

    # Every other direction is flat: a real zero, with the p reported as
    # untestable rather than as 1.0.
    flat = by_name["calm"]
    assert flat.mean == pytest.approx(0.0)
    assert compare.is_missing(flat.p_t)
    assert compare.is_missing(flat.p_w)

    # All 14 directions appear, so a reader can see whether desperate stands out.
    for name in EMOTIONS:
        assert name in summary
    assert len(paired.contrasts) == 14


def test_contrast_table_is_sorted_by_effect_size(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(root, MODEL, contrast_records())
    _, reports, summary = run_compare(tmp_path, root)

    rows = [line for line in summary.splitlines() if line.startswith("| **desperate**")]
    assert rows, "the hypothesis direction must be emphasised in the table"
    table = [
        line for line in summary.splitlines() if line.startswith("|") and "|" in line[1:]
    ]
    ordered = [line for line in table if line.startswith(("| desperate", "| **desperate**", "| frustrated"))]
    assert ordered[0].startswith("| **desperate**"), ordered[:3]


def test_null_turn_rows_are_excluded_not_zeroed(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    records = [
        # Turn 1's hook data is missing. It must be dropped, not read as 0.0 --
        # which would drag the failure-turn mean from 0.40 down to 0.20.
        make_record(
            "lcbhard_1",
            0,
            True,
            {"desperate": [0.10, 0.0, 0.40]},
            [False, True, True],
            none_turns=(1,),
        ),
        make_record("lcbhard_2", 0, False, {"desperate": [0.10, 0.40]}, [False, True]),
    ]
    write_rollouts(root, MODEL, records)
    _, reports, _ = run_compare(tmp_path, root)
    by_name = {c.emotion: c for c in reports[0].paired.contrasts}
    assert by_name["desperate"].mean == pytest.approx(0.30)


def test_missing_first_turn_excludes_the_transcript(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    records = [
        make_record(
            "lcbhard_1", 0, True, {"desperate": [0.0, 0.40]}, [False, True], none_turns=(0,)
        ),
        make_record("lcbhard_2", 0, False, {"desperate": [0.10, 0.40]}, [False, True]),
    ]
    write_rollouts(root, MODEL, records)
    _, reports, _ = run_compare(tmp_path, root)
    paired = reports[0].paired
    assert paired.n_pairs == 1
    assert paired.excluded["no_first_turn"] == 1


def test_hack_vs_nohack_contrast_matches_hand_computed(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    records = [
        # hack transcripts: transcript means 0.30 and 0.50
        make_record("lcbhard_1", 0, True, {"desperate": [0.20, 0.40]}, [False, True]),
        make_record("lcbhard_1", 1, True, {"desperate": [0.40, 0.60]}, [False, True]),
        # no-hack transcripts: transcript means 0.10 and 0.20
        make_record("lcbhard_2", 0, False, {"desperate": [0.05, 0.15]}, [False, True]),
        make_record("lcbhard_2", 1, False, {"desperate": [0.10, 0.30]}, [False, True]),
    ]
    write_rollouts(root, MODEL, records)

    _, reports, _ = run_compare(tmp_path, root)
    grouped = reports[0].grouped
    assert grouped.n_hack == 2
    assert grouped.n_nohack == 2
    by_name = {c.emotion: c for c in grouped.contrasts}
    # mean(0.30, 0.50) - mean(0.10, 0.20) = 0.40 - 0.15 = 0.25
    assert by_name["desperate"].mean == pytest.approx(0.25)
    assert not compare.is_missing(by_name["desperate"].p_t)
    assert not compare.is_missing(by_name["desperate"].p_w)


def test_hack_contrast_absent_when_every_transcript_hacked(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    records = [
        make_record("lcbhard_1", i, True, {"desperate": [0.2, 0.4]}, [False, True])
        for i in range(3)
    ]
    write_rollouts(root, MODEL, records)
    _, reports, summary = run_compare(tmp_path, root)
    assert compare.is_missing(reports[0].grouped)
    assert "no non-hack transcripts" in reports[0].grouped.reason
    assert "no non-hack transcripts" in summary


# ---------------------------------------------------------------------------
# The steering sweep, and the tier that never ran
# ---------------------------------------------------------------------------


def sweep_records() -> list[dict]:
    records = [
        make_record("lcbhard_1", i, i < 1, {"desperate": [0.2, 0.3]}, [False, True])
        for i in range(4)
    ]
    for emotion, strength, passes in (
        ("desperate", 0.05, [True, True, False, False]),
        ("desperate", 0.1, [True, True, True, False]),
        ("calm", 0.05, [False, False, False, False]),
        ("frustrated", 0.05, [True, False, False, False]),
    ):
        for sample, passed in enumerate(passes):
            records.append(
                make_record(
                    "lcbhard_1",
                    sample,
                    passed,
                    {"desperate": [0.2, 0.3]},
                    [False, True],
                    condition={"emotion": emotion, "strength": strength},
                    tier=2 if strength == 0.05 else 3,
                )
            )
    return records


def test_sweep_rates_and_baseline_scope(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(root, MODEL, sweep_records())

    _, reports, summary = run_compare(tmp_path, root)
    sweep = reports[0].sweep
    points = {(p.emotion, p.strength): (p.rate.k, p.rate.n) for p in sweep.points}
    assert points[("desperate", 0.05)] == (2, 4)
    assert points[("desperate", 0.1)] == (3, 4)
    assert points[("calm", 0.05)] == (0, 4)
    assert points[("frustrated", 0.05)] == (1, 4)
    assert (sweep.baseline.k, sweep.baseline.n) == (1, 4)
    assert sweep.tiers == [2, 3]
    assert "sweep problem(s)" in sweep.baseline_scope
    # A steered condition with a zero rate is a measured zero and shows its n.
    assert "| calm | 0.05 | 0.0% | 0/4 |" in summary


def test_control_that_moved_is_called_out(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    records = [
        make_record("lcbhard_1", i, False, {"desperate": [0.2, 0.3]}, [False, True])
        for i in range(4)
    ]
    for emotion in ("desperate", "frustrated"):
        for sample in range(4):
            records.append(
                make_record(
                    "lcbhard_1",
                    sample,
                    True,
                    {"desperate": [0.2, 0.3]},
                    [False, True],
                    condition={"emotion": emotion, "strength": 0.1},
                    tier=3,
                )
            )
    write_rollouts(root, MODEL, records)
    _, _, summary = run_compare(tmp_path, root)
    assert "THE CONTROL MOVED TOO" in summary
    assert "not specific to desperate" in summary


def test_missing_control_is_stated_as_uncontrolled(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    records = [
        make_record("lcbhard_1", i, False, {"desperate": [0.2, 0.3]}, [False, True])
        for i in range(4)
    ]
    for sample in range(4):
        records.append(
            make_record(
                "lcbhard_1",
                sample,
                True,
                {"desperate": [0.2, 0.3]},
                [False, True],
                condition={"emotion": "desperate", "strength": 0.1},
                tier=3,
            )
        )
    write_rollouts(root, MODEL, records)
    _, _, summary = run_compare(tmp_path, root)
    assert "`frustrated` was not run" in summary
    assert "Any apparent effect above is uncontrolled." in summary
    # calm never ran: absent, not a zero hack rate.
    assert "| calm | - | **absent** (this condition never ran) | - | - |" in summary


def test_control_at_a_weaker_dose_is_compared_at_the_shared_strength(tmp_path):
    """The control runs at tier 2 only. Comparing each direction's own maximum dose
    would pit a strong `desperate` against a weak `frustrated` and flatter the
    hypothesis, so the verdict must fix on the strongest shared strength."""
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    records = [
        make_record("lcbhard_1", i, False, {"desperate": [0.2, 0.3]}, [False, True])
        for i in range(4)
    ]
    for emotion, strength, passes in (
        ("desperate", 0.05, [True, False, False, False]),
        ("frustrated", 0.05, [True, False, False, False]),
        ("desperate", 0.1, [True, True, True, True]),
    ):
        for sample, passed in enumerate(passes):
            records.append(
                make_record(
                    "lcbhard_1",
                    sample,
                    passed,
                    {"desperate": [0.2, 0.3]},
                    [False, True],
                    condition={"emotion": emotion, "strength": strength},
                    tier=2 if strength == 0.05 else 3,
                )
            )
    write_rollouts(root, MODEL, records)
    _, _, summary = run_compare(tmp_path, root)

    # At the shared strength 0.05 both moved +25.0 pp, so the control moved too --
    # even though desperate reaches +100 pp at 0.1 where there is no control.
    assert "THE CONTROL MOVED TOO" in summary
    assert "at strength 0.05 (the highest dose both ran at" in summary
    assert "where there is no control to compare it against" in summary


def test_no_shared_strength_is_called_not_comparable(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    records = [
        make_record("lcbhard_1", i, False, {"desperate": [0.2, 0.3]}, [False, True])
        for i in range(4)
    ]
    for emotion, strength in (("desperate", 0.1), ("frustrated", 0.05)):
        for sample in range(4):
            records.append(
                make_record(
                    "lcbhard_1",
                    sample,
                    True,
                    {"desperate": [0.2, 0.3]},
                    [False, True],
                    condition={"emotion": emotion, "strength": strength},
                    tier=2 if strength == 0.05 else 3,
                )
            )
    write_rollouts(root, MODEL, records)
    _, _, summary = run_compare(tmp_path, root)
    assert "NOT COMPARABLE" in summary
    assert "no strength in common" in summary


def test_missing_sweep_tier_reads_as_not_run_not_no_effect(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(root, MODEL, hack_records())

    _, reports, summary = run_compare(tmp_path, root)
    assert compare.is_missing(reports[0].sweep)
    assert reports[0].tiers == [1]

    causal = summary.split("### Causal test")[1]
    assert "**Not run.**" in causal
    assert "This is missing data, not a null effect." in causal
    assert "no effect" not in causal.lower().replace("null effect", "")
    assert "0.0%" not in causal


def test_missing_readout_tier_leaves_baseline_absent(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    records = [
        make_record(
            "lcbhard_1",
            sample,
            True,
            {"desperate": [0.2, 0.3]},
            [False, True],
            condition={"emotion": "desperate", "strength": 0.1},
            tier=3,
        )
        for sample in range(4)
    ]
    write_rollouts(root, MODEL, records)

    _, reports, summary = run_compare(tmp_path, root)
    assert compare.is_missing(reports[0].baseline)
    assert "**Absent.**" in summary
    assert "tier 1 (the unsteered readout) is absent" in summary
    # The correlational test needs unsteered transcripts; it must not silently
    # fall back to the steered ones.
    assert compare.is_missing(reports[0].paired)
    assert "no unsteered rollout records" in summary


# ---------------------------------------------------------------------------
# Missing and partial data
# ---------------------------------------------------------------------------


def test_model_with_no_rollout_dir_is_not_run_not_zero(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)  # instrument built and gate passed
    _, reports, summary = run_compare(tmp_path, root)

    report = reports[0]
    assert not report.has_rollouts
    assert compare.is_missing(report.load.status())
    assert compare.is_missing(report.baseline)

    assert "**Rollouts not run.**" in summary
    assert "no rollout directory at" in summary
    # The single most important assertion in this file: no zero anywhere that a
    # reader could mistake for a measured hack rate.
    assert "0.0%" not in summary
    assert "0/0" not in summary
    assert "hack rate: 0" not in summary
    # And the instrument is still reported as built, because it was.
    assert "18000 stories used" in summary
    assert "gate: **PASSED**" in summary


def test_empty_jsonl_files_read_as_not_run(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(root, MODEL, [], shards=4)

    _, reports, summary = run_compare(tmp_path, root)
    status = reports[0].load.status()
    assert compare.is_missing(status)
    assert "4 shard file(s) present but all empty" in status.reason
    assert "**Rollouts not run.**" in summary
    assert "0.0%" not in summary


def test_directory_without_jsonl_reads_as_not_run(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    (root / "rollouts" / MODEL / "v1").mkdir(parents=True)
    (root / "rollouts" / MODEL / "v1" / "summary.shard0of2.json").write_text("{}")

    _, reports, summary = run_compare(tmp_path, root)
    assert compare.is_missing(reports[0].load.status())
    assert "contains no rollouts.shard*.jsonl" in summary


def test_truncated_final_line_is_skipped_and_reported(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    # A wall-clock kill mid-write leaves half a JSON object on the last line.
    write_rollouts(root, MODEL, hack_records(), raw_tail='{"run_id": "test-run", "task_')

    _, reports, summary = run_compare(tmp_path, root)
    assert reports[0].load.n_records == 9
    assert len(reports[0].load.bad_lines) == 1
    assert (reports[0].baseline.k, reports[0].baseline.n) == (4, 9)
    assert "unparseable JSONL line(s)" in summary


def test_missing_instrument_files_do_not_crash(tmp_path):
    root = tmp_path / "artifacts"
    write_rollouts(root, MODEL, hack_records())
    _, reports, summary = run_compare(tmp_path, root)
    assert compare.is_missing(reports[0].gate)
    assert "gate: **not run**" in summary
    assert "Vectors: **not built**" in summary
    assert "Extraction: **not run**" in summary
    # The rollouts still report normally.
    assert (reports[0].baseline.k, reports[0].baseline.n) == (4, 9)


def test_two_models_one_run_one_not(tmp_path):
    """The pilot's actual shape: one model with rollouts, one with only an instrument."""
    root = tmp_path / "artifacts"
    other = "OtherModel-2B"
    write_instrument(root, MODEL)
    write_instrument(root, other)
    write_rollouts(root, MODEL, hack_records())

    _, reports, summary = run_compare(tmp_path, root, models=[MODEL, other])
    assert reports[0].has_rollouts
    assert not reports[1].has_rollouts

    head = summary.split("## What was found")[1].split("## What ran")[0]
    assert MODEL in head and other in head
    assert "rollouts were not run" in head
    assert "not a zero" in head

    section = summary.split(f"## {other}")[1]
    assert "**Rollouts not run.**" in section
    assert "0.0%" not in section
    assert "gate: **PASSED**" in section


def test_records_with_inconsistent_emotion_order_are_refused(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    records = hack_records()
    records[0]["emotions"] = list(reversed(EMOTIONS))
    write_rollouts(root, MODEL, records)

    _, reports, summary = run_compare(tmp_path, root)
    assert compare.is_missing(reports[0].emotions)
    assert compare.is_missing(reports[0].paired)
    assert "disagree about the emotion order" in summary
    # The hack rate does not depend on the directions and is still reported.
    assert (reports[0].baseline.k, reports[0].baseline.n) == (4, 9)


def test_no_models_configured(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    out_dir, reports, summary = run_compare(tmp_path, root, models=[])
    assert reports == []
    assert (out_dir / "results.csv").is_file()
    assert "no result yet" in summary


# ---------------------------------------------------------------------------
# results.csv and figures
# ---------------------------------------------------------------------------


def test_results_csv_is_tidy_and_includes_dropped_records(tmp_path):
    import csv

    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    records = contrast_records()
    records.append(make_record("lcbhard_9", 0, False, {}, [False], sample_error="boom"))
    write_rollouts(root, MODEL, records)

    out_dir, _, _ = run_compare(tmp_path, root)
    with (out_dir / "results.csv").open() as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 4, "dropped records still belong in the tidy table"
    by_task = {row["task_id"]: row for row in rows}
    first = by_task["lcbhard_1"]
    assert first["passed"] == "True"
    assert first["n_turns"] == "3"
    assert first["n_failure_turns"] == "2"
    assert float(first["first_desperate"]) == pytest.approx(0.10)
    assert float(first["failmean_desperate"]) == pytest.approx(0.40)
    assert float(first["mean_desperate"]) == pytest.approx(0.30)
    assert by_task["lcbhard_9"]["sample_error"] == "boom"
    for name in EMOTIONS:
        assert f"mean_{name}" in rows[0]


def test_figures_are_written_when_data_supports_them(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(root, MODEL, sweep_records() + contrast_records())

    out_dir, _, summary = run_compare(tmp_path, root, no_figures=False)
    for name in ("steering_curves.png", "desperate_trace.png", "emotion_table.png"):
        path = out_dir / name
        assert path.is_file(), name
        assert path.stat().st_size > 5000, f"{name} looks empty"
        assert f"`{name}`: written" in summary


def test_figures_are_not_drawn_when_data_is_absent(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(root, MODEL, hack_records())  # no sweep, so no steering curves

    out_dir, _, summary = run_compare(tmp_path, root, no_figures=False)
    assert not (out_dir / "steering_curves.png").exists()
    assert "`steering_curves.png`: **not drawn**" in summary
    # The two that only need readout data are still drawn.
    assert (out_dir / "desperate_trace.png").is_file()
    assert (out_dir / "emotion_table.png").is_file()


def test_no_figure_files_for_a_model_that_never_ran(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    out_dir, _, summary = run_compare(tmp_path, root, no_figures=False)
    assert list(out_dir.glob("*.png")) == []
    assert "rollouts not run for this model" in summary


# ---------------------------------------------------------------------------
# Small pieces
# ---------------------------------------------------------------------------


def test_wilson_interval_brackets_the_point_estimate():
    low, high = compare.wilson(2, 4)
    assert low < 0.5 < high
    assert 0.0 <= low and high <= 1.0
    # n = 4 is wide: that width is the point of using Wilson here.
    assert high - low > 0.5
    assert compare.is_missing(compare.wilson(0, 0))


def test_missing_is_not_silently_falsy():
    missing = compare.Missing("because")
    assert compare.is_missing(missing)
    with pytest.raises(TypeError):
        bool(missing)
    assert "because" in compare.render(missing)


def test_task_ordering_is_by_trailing_integer():
    assert compare._task_key("lcbhard_2") < compare._task_key("lcbhard_10")
    assert compare._task_key("other") == ("other", -1)


# ---------------------------------------------------------------------------
# Scope and truncation, both counted from the records rather than the config
# ---------------------------------------------------------------------------


def scoped_records(n_tasks: int = 12, n_samples: int = 3) -> list[dict]:
    return [
        make_record(
            f"lcbhard_{task}",
            sample,
            False,
            {"desperate": [0.10, 0.20, 0.30]},
            [False, True, True],
        )
        for task in range(n_tasks)
        for sample in range(n_samples)
    ]


def test_scope_is_counted_from_the_records_not_the_config(tmp_path):
    """The run used generated per-shard configs that cut the scope at launch, so a
    caveat quoting the top-level config states an n that never ran."""
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(root, MODEL, scoped_records(12, 3), shards=5)

    _, reports, summary = run_compare(tmp_path, root)
    scope = reports[0].scope
    assert len(scope.readout_tasks) == 12
    assert scope.readout_samples == [3] * 12
    assert scope.readout_transcripts == 36
    assert scope.sweep_tasks == []

    assert "**12 problems x 3 samples** = 36 transcripts" in summary
    assert "counted from the rollout records rather than the config" in summary
    assert "the steering sweep is absent from the records" in summary
    # The wrong, config-derived figure must not appear anywhere.
    assert "16 readout problems" not in summary
    assert "16 problems" not in summary


def test_scope_reports_ragged_sample_counts_as_a_range(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    records = scoped_records(3, 3)
    records = [r for r in records if not (r["task_id"] == "lcbhard_1" and r["sample"] == 2)]
    write_rollouts(root, MODEL, records)

    _, reports, summary = run_compare(tmp_path, root)
    assert sorted(reports[0].scope.readout_samples) == [2, 3, 3]
    assert "**3 problems x 2-3 samples** = 8 transcripts" in summary


def test_scope_counts_the_sweep_from_the_records(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(root, MODEL, sweep_records())

    _, reports, summary = run_compare(tmp_path, root)
    scope = reports[0].scope
    assert scope.sweep_tasks == ["lcbhard_1"]
    assert scope.sweep_transcripts == 16
    assert len(scope.sweep_conditions) == 4
    assert "4 steered condition(s) = 16 transcripts" in summary


def test_truncation_is_counted_against_the_manifest_budget(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    records = [
        # 3 turns each: two at the cap (3072 and 3071, since the counter can drop
        # the final stop token), one well below it.
        make_record(
            f"lcbhard_{task}",
            0,
            False,
            {"desperate": [0.1, 0.2, 0.3]},
            [False, True, True],
            n_generated=[3072, 3071, 800],
        )
        for task in range(4)
    ]
    write_rollouts(root, MODEL, records, max_tokens=3072)

    _, reports, summary = run_compare(tmp_path, root)
    trunc = reports[0].truncation
    assert trunc.cap == 3072
    assert (trunc.n_at_cap, trunc.n_turns) == (8, 12)
    assert (trunc.n_empty, trunc.n_natural) == (0, 4)
    assert trunc.fraction == pytest.approx(8 / 12)
    assert "**8 of 12 turns (67%) hit the 3072-token per-turn budget**" in summary
    assert "0 generated zero tokens, and 4 ended on the model's own stop token" in summary


def test_empty_turns_are_their_own_category_not_untruncated(tmp_path):
    """A turn that generated nothing did not happen. Counting it in the denominator
    as "not truncated" understates how little of the budget the model ever used."""
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    records = [
        make_record(
            f"lcbhard_{task}", 0, False, {"desperate": [0.1, 0.2, 0.3]},
            [False, True, True], n_generated=[3071, 3071, 0],
        )
        for task in range(4)
    ]
    write_rollouts(root, MODEL, records, max_tokens=3072)

    _, reports, summary = run_compare(tmp_path, root)
    trunc = reports[0].truncation
    assert (trunc.n_at_cap, trunc.n_empty, trunc.n_natural) == (8, 4, 0)
    assert trunc.n_records_with_empty == 4
    assert "4 generated zero tokens" in summary
    assert "0 ended on the model's own stop token" in summary
    assert "effectively got **one fewer attempt**" in summary


def test_truncation_absent_without_a_manifest_budget(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(root, MODEL, scoped_records(2, 2), max_tokens=None)

    _, reports, summary = run_compare(tmp_path, root)
    assert compare.is_missing(reports[0].cap)
    assert compare.is_missing(reports[0].truncation)
    assert "Per-turn truncation could not be measured" in summary
    # Absent truncation must never render as 0% truncated.
    assert "0% hit the" not in summary
    assert "0 of 0 turns" not in summary


def test_zero_hack_rate_is_not_attributed_to_the_model_when_turns_were_cut_off(tmp_path):
    """A 0/n hack rate with most turns truncated has two explanations and only one
    of them is about the model. Saying so is the whole point of the caveat."""
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    records = [
        make_record(
            f"lcbhard_{task}",
            sample,
            False,
            {"desperate": [0.1, 0.2, 0.3]},
            [False, True, True],
            n_generated=[3072, 3072, 400],
        )
        for task in range(4)
        for sample in range(3)
    ]
    write_rollouts(root, MODEL, records, max_tokens=3072)

    _, reports, summary = run_compare(tmp_path, root)
    assert (reports[0].baseline.k, reports[0].baseline.n) == (0, 12)

    # Next to the hack rate in the headline, not only in the caveats.
    head = summary.split("## What was found")[1].split("## What ran")[0]
    assert "0.0% (0/12" in head
    assert "hit the 3072-token per-turn budget" in head
    assert "cannot be attributed to the model rather than to the token budget" in head

    # And again beside the per-problem table.
    baseline_section = summary.split("### Baseline (unsteered) hack rate")[1]
    assert "hit the 3072-token per-turn budget" in baseline_section
    assert "would be unsupported" in baseline_section


# ---------------------------------------------------------------------------
# Headline order and the exploratory framing
# ---------------------------------------------------------------------------


def valence_records() -> list[dict]:
    """Negative-valence directions rise after a failure, positive-valence ones fall.

    `frustrated` rises hardest, harder than `desperate` -- the shape the real tier-1
    data showed.
    """
    rises = {"frustrated": 0.09, "exasperated": 0.06, "hostile": 0.05, "desperate": 0.02}
    falls = {"joyful": -0.06, "loving": -0.05, "calm": -0.04, "proud": -0.03}
    records = []
    for task in range(12):
        for sample in range(3):
            series = {}
            # Asymmetric jitter: a symmetric one would cancel in the failure-turn
            # mean and leave every transcript with an identical difference, which
            # is a degenerate sample rather than a small effect.
            jitter = 0.004 * ((task + sample) % 5 - 2)
            for name, step in {**rises, **falls}.items():
                series[name] = [0.10, 0.10 + step + jitter, 0.10 + step + 2 * jitter]
            records.append(
                make_record(
                    f"lcbhard_{task}", sample, False, series, [False, True, True],
                    n_generated=[3072, 3072, 500],
                )
            )
    return records


def test_headline_reports_prereg_then_exploratory_then_behaviour(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(root, MODEL, valence_records(), max_tokens=3072)

    _, _, summary = run_compare(tmp_path, root)
    head = summary.split("## What was found")[1].split("## What ran")[0]

    prereg = head.index("Pre-registered outcome")
    exploratory = head.index("Exploratory -- the pattern across all 14 directions")
    caution = head.index("How much to believe it")
    behaviour = head.index("Behaviour -- did it cheat?")
    assert prereg < exploratory < caution < behaviour

    # The pre-registered line leads with desperate even though frustrated is bigger.
    assert "`desperate` on failure-following turns" in head
    assert "comes first even though it is not the largest effect" in head
    assert "The largest effect is `frustrated`" in head
    assert "Not pre-registered" in head


def test_valence_pattern_is_offered_against_the_common_mode_reading(tmp_path):
    """The signs going both ways is forced by the zero-sum centring in
    build_directions, so it must NOT be offered as evidence. The valence ORDERING
    is what carries the paragraph."""
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(root, MODEL, valence_records(), max_tokens=3072)

    _, reports, summary = run_compare(tmp_path, root)
    split = compare.valence_split(reports[0].paired.contrasts)
    assert split.separates
    assert split.mean_negative > 0 > split.mean_positive

    assert "guaranteed by construction" in summary
    assert "near zero-sum" in summary
    assert "What does carry weight is the **ordering**" in summary
    assert "Zero-sum centring forces the signs to balance" in summary
    # The retracted argument must be gone.
    assert "the signs go in **both** directions" not in summary
    assert "not the signature of a common-mode artefact" not in summary


def test_common_mode_shift_is_not_dressed_up_as_a_valence_pattern(tmp_path):
    """Every direction moving the same way is exactly the artefact the caution is
    about, so the report must not claim a valence split when there is none."""
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    records = []
    for task in range(12):
        for sample in range(3):
            jitter = 0.003 * ((task + sample) % 5 - 2)
            series = {name: [0.10, 0.16 + jitter, 0.15 + 2 * jitter] for name in EMOTIONS}
            records.append(
                make_record(f"lcbhard_{task}", sample, False, series, [False, True, True])
            )
    write_rollouts(root, MODEL, records)

    _, reports, summary = run_compare(tmp_path, root)
    split = compare.valence_split(reports[0].paired.contrasts)
    assert not split.separates
    assert "**Here that cannot be ruled out:**" in summary
    assert "a shared shift remains a live explanation" in summary
    assert "the signs go in **both** directions" not in summary


def test_disagreeing_tests_are_both_reported(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(root, MODEL, valence_records(), max_tokens=3072)
    _, reports, summary = run_compare(tmp_path, root)

    target = next(c for c in reports[0].paired.contrasts if c.emotion == "desperate")
    assert not compare.is_missing(target.p_t) and not compare.is_missing(target.p_w)
    if (target.p_t < 0.05) != (target.p_w < 0.05):
        assert "The two tests disagree" in summary
        assert "Both are reported rather than the more favourable one" in summary
    assert "wilcoxon p" in summary or "wilcoxon <" in summary


# ---------------------------------------------------------------------------
# Ruling R35: a zero baseline changes what the sweep is
# ---------------------------------------------------------------------------


def floor_records(steered_passes: dict[tuple[str, float], list[bool]]) -> list[dict]:
    """A hard-zero unsteered baseline, plus whatever the sweep did."""
    records = [
        make_record(
            f"lcbhard_{task}", sample, False, {"desperate": [0.10, 0.15, 0.20]},
            [False, True, True], n_generated=[3072, 3072, 500],
        )
        for task in range(6)
        for sample in range(3)
    ]
    for (emotion, strength), passes in steered_passes.items():
        for sample, passed in enumerate(passes):
            records.append(
                make_record(
                    f"lcbhard_{sample % 6}",
                    sample,
                    passed,
                    {emotion: [0.30, 0.36, 0.42], "desperate": [0.30, 0.36, 0.42]}
                    if emotion == "desperate"
                    else {emotion: [0.30, 0.36, 0.42], "desperate": [0.11, 0.16, 0.21]},
                    [False, True, True],
                    condition={"emotion": emotion, "strength": strength},
                    tier=2 if strength == 0.05 else 3,
                    n_generated=[3072, 3072, 500],
                )
            )
    return records


def test_zero_baseline_relabels_the_sweep_as_secondary_and_exploratory(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(
        root,
        MODEL,
        floor_records({("desperate", 0.05): [False] * 6, ("calm", 0.05): [False] * 6}),
        max_tokens=3072,
    )

    _, reports, summary = run_compare(tmp_path, root)
    assert (reports[0].baseline.k, reports[0].baseline.n) == (0, 18)

    assert "### Steering sweep (secondary): the floor-effect test" in summary
    assert "this is NOT the pre-registered causal test" in summary
    assert "required a baseline hack rate strictly between 0 and 1" in summary
    assert "0/18, which disqualifies it" in summary
    assert "**Steering check**" in summary
    assert "**Floor effect**" in summary
    # The R35 bullet must not claim the steering check answered the causal question.
    assert "not a demonstration that steering changed the model's computation" in summary
    # The headline must carry the relabelling too, not just the section.
    head = summary.split("## What was found")[1].split("## What ran")[0]
    assert "secondary and\nexploratory" in head or "**secondary and exploratory**" in head


def test_non_zero_baseline_keeps_the_pre_registered_causal_heading(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(root, MODEL, sweep_records())

    _, reports, summary = run_compare(tmp_path, root)
    assert reports[0].baseline.k > 0
    assert "### Causal test: does steering the direction change the hack rate?" in summary
    assert "NOT the pre-registered causal test" not in summary


def test_floor_effect_reports_a_lift_off_zero(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(
        root,
        MODEL,
        floor_records(
            {("desperate", 0.05): [True, True, False, False, False, False],
             ("calm", 0.05): [False] * 6}
        ),
        max_tokens=3072,
    )

    _, _, summary = run_compare(tmp_path, root)
    assert "**Floor effect: steering lifted hacking off zero.**" in summary
    assert "2 of 12 steered rollouts cheated where none of the 18 unsteered ones" in summary
    assert "`desperate` at 0.05 (2/6)" in summary
    # 2/6 against 0/18 does not clear Fisher, so the lift must not be sold as a result.
    assert "not distinguishable from the baseline" in summary
    assert "Fisher exact" in summary
    assert "worth a follow-up with more samples, not a claim" in summary


def test_a_large_lift_off_zero_is_reported_as_a_real_positive(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(
        root,
        MODEL,
        floor_records({("desperate", 0.05): [True] * 6, ("calm", 0.05): [False] * 6}),
        max_tokens=3072,
    )
    _, _, summary = run_compare(tmp_path, root)
    assert "**Floor effect: steering lifted hacking off zero.**" in summary
    assert "that is a real positive" in summary
    assert "not distinguishable from the baseline" not in summary


def test_control_verdict_when_neither_direction_moved(tmp_path):
    """Both at +0.0 pp must not read as "the control moved substantially less"."""
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(
        root,
        MODEL,
        floor_records({("desperate", 0.05): [False] * 6, ("frustrated", 0.05): [False] * 6}),
        max_tokens=3072,
    )
    _, _, summary = run_compare(tmp_path, root)
    assert "**Discriminant control: nothing to discriminate.**" in summary
    assert "neither `desperate` nor `frustrated` moved the hack rate at all" in summary
    assert "The control moved substantially less" not in summary


def test_control_verdict_when_only_the_control_moved(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(
        root,
        MODEL,
        floor_records(
            {("desperate", 0.05): [False] * 6,
             ("frustrated", 0.05): [True, True, True, False, False, False]}
        ),
        max_tokens=3072,
    )
    _, _, summary = run_compare(tmp_path, root)
    assert "did not move the hack rate at all" in summary
    assert "There is no desperate-specific effect to defend here." in summary
    assert "The control moved substantially less" not in summary


def test_floor_effect_reports_a_bounded_negative(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(
        root,
        MODEL,
        floor_records({("desperate", 0.05): [False] * 6, ("calm", 0.05): [False] * 6}),
        max_tokens=3072,
    )

    _, _, summary = run_compare(tmp_path, root)
    assert "**Floor effect: none.**" in summary
    assert "0 of 12" in summary
    assert "bounded negative rather than a null" in summary
    # The bound is the conditions on disk, not the conditions that were planned.
    assert "bounded to **exactly the condition(s) on disk**" in summary
    assert "`calm` at 0.05" in summary and "`desperate` at 0.05" in summary


def steered_probe_records(
    on_target: float = 0.047, off_target: float = 0.005, strength: float = 0.05
) -> list[dict]:
    """Unsteered baseline plus a steered arm on the SAME (task_id, sample) cells.

    The steered arm lifts `desperate` by ``on_target`` and every other direction by
    ``off_target``, so the specificity ratio is known by construction.
    """
    records = []
    for task in range(6):
        for sample in range(3):
            jitter = 0.001 * ((task + sample) % 5 - 2)
            base = {name: [0.010 + jitter, 0.012 + jitter, 0.011 + jitter] for name in EMOTIONS}
            records.append(
                make_record(
                    f"lcbhard_{task}", sample, False, base, [False, True, True],
                    n_generated=[3072, 3072, 500],
                )
            )
            steered = {
                name: [
                    v + (on_target if name == "desperate" else off_target)
                    for v in base[name]
                ]
                for name in EMOTIONS
            }
            records.append(
                make_record(
                    f"lcbhard_{task}", sample, False, steered, [False, True, True],
                    condition={"emotion": "desperate", "strength": strength},
                    tier=2,
                    n_generated=[3072, 3072, 500],
                )
            )
    return records


def test_probe_shift_is_paired_on_task_and_sample(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(root, MODEL, steered_probe_records(), max_tokens=3072)

    _, reports, summary = run_compare(tmp_path, root)
    shift = reports[0].shifts[0]
    assert shift.paired
    assert shift.n_pairs == 18
    assert shift.n_steered == 18
    assert shift.shift == pytest.approx(0.047)
    assert shift.ratio == pytest.approx(0.047 / 0.05)

    assert "### Steering check (secondary): plumbing, locality, and what they do not show" in summary
    assert "18 / 18 paired" in summary
    assert "**+0.04700**" in summary


def test_probe_shift_falls_back_to_unpaired_when_cells_do_not_match(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    records = steered_probe_records()
    for record in records:
        if record["condition"]:
            # Steered arm ran on problems the readout never covered.
            record["task_id"] = record["task_id"].replace("lcbhard_", "lcbhard_9")
    write_rollouts(root, MODEL, records, max_tokens=3072)

    _, reports, summary = run_compare(tmp_path, root)
    shift = reports[0].shifts[0]
    assert not shift.paired
    assert shift.n_pairs == 0
    assert shift.shift == pytest.approx(0.047, abs=1e-6)
    assert "unpaired vs 18" in summary


def test_injection_layer_identity_is_stated_before_the_numbers(tmp_path):
    """The vector is injected at the probe layer and the probe reads that layer, so
    the shift there is an identity. The report must say so before showing it, and
    must not claim the causal apparatus is validated."""
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(root, MODEL, steered_probe_records(), max_tokens=3072)

    _, _, summary = run_compare(tmp_path, root)
    section = summary.split("### Steering check")[1]
    caveat = section.index("Read this before the numbers")
    table = section.index("| direction | strength | unsteered |")
    assert caveat < table, "the identity caveat must precede the numbers"

    assert "injected **at the probe layer**" in section
    assert "an identity" in section
    assert "without running the model at all" in section
    assert "It is **not** evidence that steering changed the model's computation." in section

    # The retracted overclaims must be gone from the whole document.
    assert "quantitative validation of the whole causal apparatus" not in summary
    assert "direction-specific, not a general perturbation" not in summary
    assert "a future run only has to fix the floor effect" not in summary
    assert "The apparatus works" not in summary


def test_off_target_shift_is_reported_as_geometry_not_as_a_finding(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(root, MODEL, steered_probe_records(), max_tokens=3072)

    _, _, summary = run_compare(tmp_path, root)
    assert "#### Off-target shift is the geometry of the direction set" in summary
    section = summary.split("#### Off-target shift")[1]
    # Without a vectors.safetensors the cosines cannot be read, and the report must
    # say that rather than presenting the shifts as a behavioural result.
    assert "should NOT be read as a behavioural result" in section
    assert "reflects how non-orthogonal the directions are" in section


def test_steered_behaviour_carries_the_floor_and_the_truncation(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(root, MODEL, steered_probe_records(), max_tokens=3072)

    _, reports, summary = run_compare(tmp_path, root)
    shift = reports[0].shifts[0]
    assert (shift.rate.k, shift.rate.n) == (0, 18)
    assert (shift.truncation.n_at_cap, shift.truncation.n_turns) == (36, 54)

    section = summary.split("#### Behaviour under steering")[1]
    assert "0.0% (0/18" in section
    assert "36/54 at cap (67%)" in section
    assert "steered hack rate is 0/18 -- still on the floor" in section
    assert "cannot be read as 'steering does not induce hacking'" in section


def test_steering_check_is_a_section_not_a_footnote(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(root, MODEL, steered_probe_records(), max_tokens=3072)

    _, _, summary = run_compare(tmp_path, root)
    assert "\n### Steering check (secondary)" in summary
    assert summary.index("### Steering check") < summary.index("### Steering sweep")
    head = summary.split("## What was found")[1].split("## What ran")[0]
    assert "**Apparatus -- did the steering reach the model?** Yes, as **plumbing**" in head
    assert "this number is an **identity**" in head
    assert "No part of this run shows that steering changed the model's computation" in head
    assert "secondary" in summary
    assert "NOT the pre-registered causal test" in summary


def test_apparatus_headline_judges_by_sign_of_strength_not_of_shift(tmp_path):
    """A negative-strength arm SHOULD move the probe down. Ranking on the raw shift
    would call a correctly-working negative arm a failure."""
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    records = steered_probe_records(on_target=-0.047, off_target=-0.005, strength=-0.05)
    write_rollouts(root, MODEL, records, max_tokens=3072)

    _, reports, summary = run_compare(tmp_path, root)
    shift = reports[0].shifts[0]
    assert shift.strength == -0.05
    assert shift.shift < 0
    assert shift.ratio > 0, "down-shift under negative strength is success"

    head = summary.split("## What was found")[1].split("## What ran")[0]
    assert "did the steering reach the model?** Yes" in head
    assert "did the steering reach the model?** No" not in head


def test_apparatus_headline_reports_a_probe_that_moved_the_wrong_way(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    # Positive strength, but the probe went down: the plumbing is wrong.
    write_rollouts(root, MODEL, steered_probe_records(-0.02, 0.0), max_tokens=3072)

    _, reports, summary = run_compare(tmp_path, root)
    assert reports[0].shifts[0].ratio < 0
    head = summary.split("## What was found")[1].split("## What ran")[0]
    assert "did the steering reach the model?** No" in head
    assert "the wrong way" in head
    assert "cannot be interpreted" in head


def test_steering_check_compares_each_direction_to_unsteered(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(
        root,
        MODEL,
        floor_records({("desperate", 0.05): [False] * 6, ("calm", 0.05): [False] * 6}),
        max_tokens=3072,
    )

    _, reports, summary = run_compare(tmp_path, root)
    shifts = {(s.emotion, s.strength): s for s in reports[0].shifts}
    desperate = shifts[("desperate", 0.05)]
    assert desperate.steered_mean == pytest.approx(0.36)
    assert desperate.baseline_mean == pytest.approx(0.15)
    assert desperate.shift == pytest.approx(0.21)
    assert desperate.n_steered == 6 and desperate.n_baseline == 18

    assert "### Steering check (secondary)" in summary
    assert "it is arithmetic, not a finding" in summary


def test_layer_profile_reports_the_upstream_control(tmp_path):
    """Upstream layers cannot be affected by an edit made below them, so a flat
    upstream reading is the one part of this check that could have failed."""
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    records = steered_probe_records()
    for record in records:
        stats = record["turn_stat"]
        # L41/L42 upstream (unchanged), L43 injection, L44/L45 attenuated downstream.
        record["turn_stat_layers"] = [
            {
                "41": [0.01] * len(EMOTIONS),
                "42": [0.01] * len(EMOTIONS),
                "43": list(row),
                "44": [v * 0.94 for v in row],
                "45": [v * 0.89 for v in row],
            }
            if row is not None
            else {}
            for row in stats
        ]
    write_rollouts(root, MODEL, records, max_tokens=3072)

    _, reports, summary = run_compare(tmp_path, root)
    by_layer = reports[0].shifts[0].by_layer
    assert set(by_layer) == {41, 42, 43, 44, 45}
    assert by_layer[41] == pytest.approx(0.0, abs=1e-9)
    assert by_layer[42] == pytest.approx(0.0, abs=1e-9)
    assert by_layer[43] == pytest.approx(0.047)
    assert by_layer[44] < by_layer[43] and by_layer[45] < by_layer[44]

    section = summary.split("#### Across layers")[1]
    assert "L43 (injection)" in section
    assert "**This is a genuine control, and it passed.**" in section
    assert "the one part of this check that could have failed" in section
    # Downstream persistence must be explicitly deflated.
    assert "Read this as weaker evidence than it looks" in section
    assert "propagation is not influence" in section


def test_layer_profile_flags_a_failed_upstream_control(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    records = steered_probe_records()
    for record in records:
        stats = record["turn_stat"]
        # Upstream moves as much as the injection layer: impossible for a clean edit.
        record["turn_stat_layers"] = [
            {"41": list(row), "43": list(row)} if row is not None else {}
            for row in stats
        ]
    write_rollouts(root, MODEL, records, max_tokens=3072)

    _, _, summary = run_compare(tmp_path, root)
    section = summary.split("#### Across layers")[1]
    assert "**This control did NOT pass cleanly.**" in section
    assert "should not be trusted until it is explained" in section


def test_manipulation_check_absent_without_steered_records(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(root, MODEL, scoped_records(4, 3))
    _, reports, summary = run_compare(tmp_path, root)
    assert compare.is_missing(reports[0].shifts)
    # No sweep ran, so the R35 framing must not appear either: printing it would
    # imply a manipulation check and a floor test happened when neither did.
    assert "Manipulation check" not in summary
    assert "Floor effect" not in summary
    assert "NOT the pre-registered causal test" not in summary
    assert "**Not run.**" in summary


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_summary_carries_provenance_and_a_manifest_is_written(tmp_path):
    """The summary is what a human reads and acts on, so it must say which code and
    which upstream artifacts produced it."""
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(root, MODEL, hack_records())
    # An upstream manifest for the rollouts, as the real stage writes.
    (root / "rollouts" / MODEL / "v1" / "manifest.json").write_text(
        json.dumps({"stage": "rollouts", "config": {"max_tokens": 3072}})
    )

    out_dir, _, summary = run_compare(tmp_path, root)

    assert "_Provenance: run `" in summary
    assert "working tree)" in summary
    assert f"`{MODEL}/rollouts`" in summary

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["stage"] == "compare"
    assert "sha" in manifest["git"] and "dirty" in manifest["git"]
    assert manifest["config"]["run_id"]
    assert manifest["config"]["artifact_root"] == str(root)
    hashes = manifest["config"]["upstream_manifest_sha256"]
    assert hashes[f"{MODEL}/rollouts"] == compare.manifest_sha256(
        root / "rollouts" / MODEL / "v1"
    )
    # The run_id in the manifest is the one printed in the summary.
    assert manifest["config"]["run_id"] in summary


def test_missing_upstream_manifest_is_named_not_skipped(tmp_path):
    root = tmp_path / "artifacts"
    write_instrument(root, MODEL)
    write_rollouts(root, MODEL, hack_records(), max_tokens=None)

    out_dir, _, summary = run_compare(tmp_path, root)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    hashes = manifest["config"]["upstream_manifest_sha256"]
    assert "no manifest.json under" in hashes[f"{MODEL}/gate"]
    assert "no manifest.json under" in summary
