#!/usr/bin/env python
"""Stages 6-7: run the ImpossibleBench rollouts for one model.

Runs INSIDE ``apptainer/eval.sif`` on a compute node, against a vllm-lens server
that is already up. The container binds ``$PROJECT_DIR`` read-only at
``/project``, ``$ARTIFACT_DIR`` read-only at ``/artifacts``, and a writable
scratch at ``/work``; results therefore go to ``$HEALTHY_RL_ARTIFACT_OUT`` and
the job copies them out afterwards. Nothing here ever writes into ``/artifacts``.

    apptainer exec ... apptainer/eval.sif \
        python /project/scripts/run_rollouts.py --config /project/configs/rollouts.yaml

Outputs, under ``$HEALTHY_RL_ARTIFACT_OUT/rollouts/<model>/<version>/``:

    rollouts.jsonl   one record per rollout, appended as it completes
    residuals/       event-position residuals, one .npz per rollout
    summary.json     conditions run, sweep selection, preflight, error counts
    manifest.json    provenance, including the sha256 of every upstream manifest

Exit status is 1 if the run did not finish every condition, because a partial
overnight run must not look like a successful one -- but whatever did finish is
already durable on disk before the exception propagates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from healthy_rl.artifacts import check_upstream, verify_upstreams, write_manifest
from healthy_rl.config import load_config, load_env, repo_root
from healthy_rl.rollouts import (
    output_dir,
    parse_shard,
    run_rollouts,
    select_sweep_from_dir,
)

DEFAULT_CONFIG = repo_root() / "configs" / "rollouts.yaml"
SUMMARY_NAME = "summary.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=None, help="override $HEALTHY_RL_MODEL_NAME")
    parser.add_argument("--base-url", default=None, help="vllm-lens server base URL")
    parser.add_argument("--out-dir", type=Path, default=None, help="where results go")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="root holding vectors/ and bench/ (default: /artifacts in the container)",
    )
    parser.add_argument(
        "--shard",
        default=None,
        metavar="I/N",
        help=(
            "run only work items where index %% N == I, over the fully expanded "
            "(tier, condition, task_id, sample) list. Each shard writes its own JSONL."
        ),
    )
    parser.add_argument(
        "--tiers",
        default=None,
        help="comma-separated tiers to run (default: all). e.g. --tiers 1 or --tiers 2,3",
    )
    parser.add_argument(
        "--sweep-problems",
        default=None,
        help=(
            "comma-separated task_ids to pin the sweep to, instead of deriving them "
            "from the readout. Use this when tier 1 ran as separate shard jobs whose "
            "JSONLs this job cannot see, so every shard sweeps the same problems."
        ),
    )
    parser.add_argument(
        "--select-sweep-only",
        action="store_true",
        help=(
            "apply the sweep-selection rule to the completed tier-1 JSONLs, print the "
            "chosen task_ids and the rates they came from, and exit without running "
            "any rollout or contacting the server (Ruling R26, phase 1.5)"
        ),
    )
    parser.add_argument(
        "--allow-partial-readout",
        action="store_true",
        help="let --select-sweep-only succeed on an incomplete readout (it warns and exits 2 otherwise)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="discard any existing rollouts.jsonl instead of continuing it",
    )
    return parser.parse_args(argv)


def resolve_base_url(cli_value: str | None) -> str:
    """``--base-url``, else the environment ``slurm/serve.slurm`` exports.

    ``healthy_rl.server`` owns that chain -- ``$HEALTHY_RL_SERVER_URL``, then
    ``$HEALTHY_RL_ENDPOINT_FILE`` or its ``$HEALTHY_RL_SERVER_URL_FILE`` alias --
    so both spellings work and the endpoint file is parsed the same way here as
    everywhere else.
    """
    if cli_value:
        return cli_value.strip()
    from healthy_rl.server import base_url_from_env

    return base_url_from_env()


CONTAINER_ARTIFACT_ROOT = Path("/artifacts")


def resolve_artifact_root(cli_value: Path | None, cfg: dict) -> Path:
    """Where ``vectors/`` and ``bench/`` live.

    ``/artifacts`` wins whenever it exists, because it only exists inside the
    rollout container -- and inside the container the repo's ``.env`` is still
    readable at ``/project/.env``, so ``$ARTIFACT_DIR`` would otherwise resolve
    to a host path that is not bound in and does not exist.
    """
    if cli_value is not None:
        return cli_value
    configured = cfg.get("artifact_root") or os.environ.get("HEALTHY_RL_ARTIFACT_ROOT")
    if configured:
        return Path(configured)
    if CONTAINER_ARTIFACT_ROOT.is_dir():
        return CONTAINER_ARTIFACT_ROOT
    root = os.environ.get("ARTIFACT_DIR")
    if not root:
        raise RuntimeError(
            "no artifact root: pass --artifact-root, set HEALTHY_RL_ARTIFACT_ROOT or "
            "ARTIFACT_DIR, or run inside the container where /artifacts is bound"
        )
    return Path(root)


def resolve_model(cli_value: str | None, cfg: dict) -> str:
    model = cli_value or os.environ.get("HEALTHY_RL_MODEL_NAME") or cfg.get("model")
    if not model:
        raise RuntimeError(
            "no model name: pass --model, or set HEALTHY_RL_MODEL_NAME, "
            "or set `model:` in the config"
        )
    return str(model)


def summary_name(shard: tuple[int, int]) -> str:
    """Shard-specific, matching what ``run_rollouts`` writes."""
    index, count = shard
    return SUMMARY_NAME if count == 1 else f"summary.shard{index}of{count}.json"


SWEEP_SELECTION_NAME = "sweep_selection.json"


def report_sweep_selection(
    out_dir: Path, cfg: dict, bench_parquet: Path, allow_partial: bool
) -> int:
    """Print and record the sweep problems selected from the completed readout.

    Phase 1.5 of the two-phase launch (**Ruling R26**). The selection rule is
    applied exactly once, to every tier-1 rollout across every shard file, and the
    result is written to ``sweep_selection.json`` before any sweep rollout runs --
    that file is the pre-registration record.

    Exit codes: 0 selected, 2 the readout is incomplete (so the selection would
    not be the pre-registered one), 3 the model is disqualified from the causal
    test under R4.
    """
    task_ids = None
    if bench_parquet.is_file():
        import pandas as pd

        task_ids = [str(v) for v in pd.read_parquet(bench_parquet, columns=["task_id"])["task_id"]]

    report = select_sweep_from_dir(out_dir, cfg, task_ids)
    (out_dir / SWEEP_SELECTION_NAME).write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    )

    print(f"readout: {report['n_readout_records']}/{report['n_expected_records']} rollouts "
          f"from {report['shard_files'] or ['(no shard files found)']}")
    print("per-problem hack rate:")
    for task_id, rate in report["rates"].items():
        mark = "*" if task_id in report["problems"] else " "
        print(f"  {mark} {task_id:<16} {rate:.3f}")

    if report["disqualified"]:
        print(f"\nDISQUALIFIED: {report['sweep']['reason']}")
        print(f"wrote {out_dir / SWEEP_SELECTION_NAME}")
        return 3

    if report["sweep"]["filled"]:
        print(f"\nR4 fill: {report['sweep']['reason']}")

    print(f"\nsweep problems ({len(report['problems'])}):")
    print("  " + ",".join(report["problems"]))
    print("\nlaunch tiers 2-3 with:")
    print(f"  --tiers 2,3 --sweep-problems {','.join(report['problems'])}")
    print(f"\nwrote {out_dir / SWEEP_SELECTION_NAME}")

    if not report["complete"]:
        detail = []
        if report["problems_with_no_records"]:
            detail.append(f"no records for {report['problems_with_no_records']}")
        if report["problems_with_missing_samples"]:
            detail.append(f"short of samples: {report['problems_with_missing_samples']}")
        print(
            "\nINCOMPLETE READOUT: " + "; ".join(detail or ["fewer records than expected"]),
            file=sys.stderr,
        )
        if not allow_partial:
            print(
                "Selecting from a partial readout is exactly the race the two-phase "
                "launch exists to avoid -- finish tier 1 first, or pass "
                "--allow-partial-readout to accept this selection anyway.",
                file=sys.stderr,
            )
            return 2
        print("--allow-partial-readout given; using the selection anyway", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_env()
    cfg = load_config(args.config)

    model = resolve_model(args.model, cfg)
    version = str(cfg.get("version", "v1"))

    artifact_root = resolve_artifact_root(args.artifact_root, cfg)
    vectors_dir = Path(cfg.get("vectors_dir") or artifact_root / "vectors" / model / version)
    bench_dir = Path(cfg.get("bench_dir") or artifact_root / "bench" / version)
    bench_parquet = Path(cfg.get("bench_parquet") or bench_dir / "conflicting.parquet")

    shard = parse_shard(args.shard)
    tiers = [int(t) for t in args.tiers.replace(",", " ").split()] if args.tiers else None
    sweep_problems = (
        args.sweep_problems.replace(",", " ").split() if args.sweep_problems else None
    )

    out_dir = Path(args.out_dir) if args.out_dir else output_dir("rollouts", model, version)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.select_sweep_only:
        # Phase 1.5 of the two-phase launch: no server, no vectors, no rollouts.
        return report_sweep_selection(out_dir, cfg, bench_parquet, args.allow_partial_readout)

    # Upstreams must exist and carry a manifest before anything runs: a missing
    # one means a stage was skipped, which is a setup error, not a data finding.
    check_upstream(vectors_dir)
    check_upstream(bench_dir)

    base_url = resolve_base_url(args.base_url)
    print(
        f"rollouts: model={model} url={base_url} shard={shard[0]}/{shard[1]}\n"
        f"  vectors  {vectors_dir}\n"
        f"  bench    {bench_parquet}\n"
        f"  out      {out_dir}",
        flush=True,
    )

    summary: dict = {"stage": "rollouts", "model": model, "complete": False}
    failure: BaseException | None = None
    # Drop any summary from an earlier run: below, "the file exists" is taken to
    # mean this run wrote it. Resume state lives in the JSONL, not here.
    (out_dir / summary_name(shard)).unlink(missing_ok=True)

    try:
        summary = run_rollouts(
            cfg,
            base_url=base_url,
            model_name=model,
            vectors_dir=vectors_dir,
            bench_parquet=bench_parquet,
            out_dir=out_dir,
            resume=not args.no_resume,
            shard=shard,
            tiers=tiers,
            sweep_problems=sweep_problems,
        )
    except BaseException as exc:  # noqa: BLE001 - record what got done, then re-raise loudly
        failure = exc
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()

    # run_rollouts() rewrites summary.json after every condition, so on failure the
    # file on disk already records how far the run got. Keep that detail and add
    # the traceback to it rather than replacing it with this thinner dict.
    summary_path = out_dir / summary_name(shard)
    if failure is not None and summary_path.is_file():
        on_disk = json.loads(summary_path.read_text())
        on_disk.update(
            {"error": summary["error"], "traceback": summary["traceback"], "complete": False}
        )
        summary = on_disk
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
    )
    # `cfg` holds only what the YAML declares -- no environment, no HF_TOKEN.
    write_manifest(
        out_dir,
        stage="rollouts",
        config={**cfg, "model": model, "version": version},
        upstreams={"vectors": vectors_dir, "bench": bench_dir},
    )
    verify_upstreams(out_dir)

    if failure is not None:
        print(f"rollouts FAILED for {model}: {summary['error']}", file=sys.stderr, flush=True)
        print(summary.get("traceback", ""), file=sys.stderr, flush=True)
        return 1

    for entry in summary.get("conditions", []):
        note = entry.get("skipped") or f"{entry.get('n_written', 0)} records"
        print(f"  tier {entry['tier']} {entry['name']}: {note}", flush=True)
    print(
        f"rollouts {'COMPLETE' if summary.get('complete') else 'PARTIAL'} for {model}: "
        f"{summary.get('n_records', 0)} records, "
        f"{summary.get('samples_without_hook', 0)}/{summary.get('samples_seen', 0)} "
        f"rollouts without emotion data; wrote {out_dir}",
        flush=True,
    )
    if summary.get("disqualified"):
        print(f"  DISQUALIFIED from the causal test: {summary['sweep']['reason']}", flush=True)
    return 0 if summary.get("complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
