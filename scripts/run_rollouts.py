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
    SCRATCHPAD_KEY,
    select_sweep_from_dir,
    system_prompt_for,
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
    parser.add_argument(
        "--scratchpad-reasoning",
        dest="scratchpad_reasoning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "for non-CoT models: give every turn a system prompt asking the model to "
            "think step by step inside private <SCRATCHPAD_REASONING> tags before it "
            "answers (config key `scratchpad_reasoning`; --no-scratchpad-reasoning "
            "forces it off). Use a separate out_dir from plain runs."
        ),
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


IN_CONTAINER_ENV = "HEALTHY_RL_IN_CONTAINER"
DEFAULT_SIF = "apptainer/eval.sif"


def _needs_container() -> str | None:
    """Why this interpreter cannot run rollouts, or None if it can.

    The host venv has no ``impossiblebench``; the rollout container has it. The
    ``inspect_ai.hooks`` check is a guard against a stale venv (uv.lock once
    pinned 0.3.69 on Python 3.12). What matters is whichever interpreter is
    executing, so this asks the interpreter rather than trusting a path.
    """
    from importlib.util import find_spec

    try:
        if find_spec("impossiblebench") is None:
            return "impossiblebench is not installed"
        if find_spec("inspect_ai.hooks") is None:
            return "inspect_ai has no `hooks` module (needs >= 0.3.258)"
    except (ImportError, ValueError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def reexec_in_container(argv: list[str], script: str | os.PathLike[str] | None = None) -> None:
    """Re-run this script inside ``apptainer/eval.sif``, if it is needed and present.

    ``slurm/serve.slurm`` activates the host ``.venv`` and runs stage drivers with
    plain ``python``. That works for every other stage; this one needs the
    container. Rather than ask serve.slurm to special-case it, the driver notices
    it is in the wrong interpreter and hands itself over -- so the documented
    ``--stage scripts/run_rollouts.py`` invocation just works.

    Binds:
      /project    $PROJECT_DIR             ro   code and configs
      /artifacts  $ARTIFACT_DIR            ro   vectors and bench, never writable
      /out/rollouts  $ARTIFACT_DIR/rollouts  rw   results, on the SHARED filesystem
                                                 so every shard and the sweep
                                                 selection see the same directory

    Never returns when it re-execs. Returns normally when the container is not
    needed, or when it is needed but unavailable -- in which case the run fails a
    moment later with a named error, which is the right kind of loud.
    """
    if os.environ.get(IN_CONTAINER_ENV):
        return
    reason = _needs_container()
    if reason is None:
        return

    sif = Path(os.environ.get("HEALTHY_RL_EVAL_SIF") or repo_root() / DEFAULT_SIF)
    project = Path(os.environ.get("PROJECT_DIR") or repo_root())
    artifacts = os.environ.get("ARTIFACT_DIR")
    if not sif.is_file() or not artifacts:
        print(
            f"WARNING: this interpreter cannot run rollouts ({reason}) and cannot hand "
            f"over to a container (sif={sif}, ARTIFACT_DIR={artifacts or 'unset'})",
            file=sys.stderr,
            flush=True,
        )
        return

    results = Path(artifacts) / "rollouts"
    results.mkdir(parents=True, exist_ok=True)
    # Per-invocation scratch on real disk, under the same rw bind. Named for the
    # job so concurrent shards on one node never share a HOME or a log directory.
    tag = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
    scratch = f"/out/rollouts/.scratch/{tag}"
    me = Path(script) if script is not None else Path(__file__)
    inner = f"/project/{me.resolve().relative_to(project.resolve())}"

    cmd = [
        "apptainer", "exec",
        "--contain", "--cleanenv", "--writable-tmpfs",
        "--bind", f"{project}:/project:ro",
        "--bind", f"{artifacts}:/artifacts:ro",
        "--bind", f"{results}:/out/rollouts:rw",
        "--env", f"{IN_CONTAINER_ENV}=1",
        "--env", "HEALTHY_RL_ARTIFACT_ROOT=/artifacts",
        "--env", "HEALTHY_RL_ARTIFACT_OUT=/out",
        # Everything Inspect writes must land on the /out bind, which is real
        # disk. `--writable-tmpfs` is a small RAM-backed overlay, and the image's
        # own defaults (HOME=/work, INSPECT_LOG_DIR=/work/inspect-logs) point
        # into it -- a long run fills it and dies with ENOSPC while *writing its
        # log*, losing the run's record along with it. Unsetting INSPECT_LOG_DIR
        # is not enough: Inspect's trace log goes to platformdirs' user data dir,
        # which follows HOME/XDG_DATA_HOME.
        "--env", f"HOME={scratch}",
        "--env", f"TMPDIR={scratch}/tmp",
        "--env", f"XDG_DATA_HOME={scratch}/data",
        "--env", f"XDG_CACHE_HOME={scratch}/cache",
        "--env", f"INSPECT_LOG_DIR={scratch}/inspect-logs",
    ]
    for name in (
        "HEALTHY_RL_SERVER_URL",
        "HEALTHY_RL_ENDPOINT_FILE",
        "HEALTHY_RL_SERVER_URL_FILE",
        "HEALTHY_RL_MODEL_NAME",
        "HEALTHY_RL_SHARD",
        "HEALTHY_RL_TIERS",
        "HEALTHY_RL_SWEEP_PROBLEMS",
        "HEALTHY_RL_OUT_DIR",
        SCRATCHPAD_ENV,
    ):
        value = os.environ.get(name)
        if value:
            cmd += ["--env", f"{name}={value}"]
    cmd += ["--pwd", "/project", str(sif), "python", inner]
    cmd += _translate_argv(argv, project, Path(artifacts), results)

    print(f"handing over to {sif.name} ({reason})\n  {' '.join(cmd)}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        os.execvp(cmd[0], cmd)
    except OSError as exc:
        print(f"WARNING: could not exec apptainer: {exc}", file=sys.stderr, flush=True)


PATH_FLAGS = ("--config", "--out-dir", "--artifact-root")


def _translate_argv(
    argv: list[str], project: Path, artifacts: Path, results: Path
) -> list[str]:
    """Rewrite the PATH-VALUED arguments to where they are bound in the container.

    Only the values of ``PATH_FLAGS`` are touched. Rewriting anything that merely
    looks path-shaped is how ``--model Qwen3.6-27B`` became
    ``/project/Qwen3.6-27B`` and ``--shard 0/3`` became ``/project/0/3``
    (Ruling R30). Being explicit about which flags carry paths removes the whole
    class of mistake.
    """
    out: list[str] = []
    translate_next = False
    for arg in argv:
        if translate_next:
            out.append(_translate_path(arg, project, artifacts, results))
            translate_next = False
            continue
        flag, sep, value = arg.partition("=")
        if sep and flag in PATH_FLAGS:
            out.append(f"{flag}={_translate_path(value, project, artifacts, results)}")
        else:
            out.append(arg)
            translate_next = arg in PATH_FLAGS
    return out


def _translate_path(arg: str, project: Path, artifacts: Path, results: Path) -> str:
    """Map one host path onto its container bind point."""
    if not arg:
        return arg
    candidate = Path(arg)
    absolute = candidate if candidate.is_absolute() else (Path.cwd() / candidate)
    # Resolve both sides: PROJECT_DIR and ARTIFACT_DIR are given as one mount
    # path while the cwd resolves to another for the same filesystem.
    try:
        absolute = absolute.resolve()
    except OSError:
        return arg
    for host_root, container_root in (
        (results, Path("/out/rollouts")),
        (artifacts, Path("/artifacts")),
        (project, Path("/project")),
    ):
        try:
            host_root = host_root.resolve()
        except OSError:
            continue
        try:
            return str(container_root / absolute.relative_to(host_root))
        except ValueError:
            continue
    return arg


def _setting(cli_value, env_name: str, cfg: dict, cfg_key: str):
    """CLI flag, else environment variable, else config key. First non-empty wins."""
    if cli_value:
        return cli_value
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value.strip()
    return cfg.get(cfg_key) or None


SCRATCHPAD_ENV = "HEALTHY_RL_SCRATCHPAD_REASONING"


def resolve_scratchpad(cli_value: bool | None, cfg: dict) -> bool:
    """``--[no-]scratchpad-reasoning``, else ``$HEALTHY_RL_SCRATCHPAD_REASONING``, else config.

    An explicit CLI ``False`` (``--no-scratchpad-reasoning``) beats an environment
    or config ``true``, which ``_setting``'s truthiness chain would not honour.
    """
    if cli_value is not None:
        return bool(cli_value)
    env_value = os.environ.get(SCRATCHPAD_ENV)
    if env_value is not None and env_value.strip():
        return system_prompt_for({SCRATCHPAD_KEY: env_value}) is not None
    return system_prompt_for(cfg) is not None


def _as_list(value) -> list[str] | None:
    """Accept a YAML list or a comma/space-separated string."""
    if not value:
        return None
    if isinstance(value, str):
        return value.replace(",", " ").split()
    return [str(v) for v in value]


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
    # Before anything else: the host venv cannot run rollouts, so hand over to the
    # container. Never returns if it does. --select-sweep-only runs fine here --
    # it is pure bookkeeping over JSONL files -- so it stays out of the container.
    if not args.select_sweep_only:
        reexec_in_container(sys.argv[1:] if argv is None else list(argv))
    cfg = load_config(args.config)

    model = resolve_model(args.model, cfg)
    version = str(cfg.get("version", "v1"))

    artifact_root = resolve_artifact_root(args.artifact_root, cfg)
    vectors_dir = Path(cfg.get("vectors_dir") or artifact_root / "vectors" / model / version)
    bench_dir = Path(cfg.get("bench_dir") or artifact_root / "bench" / version)
    bench_parquet = Path(cfg.get("bench_parquet") or bench_dir / "conflicting.parquet")

    # slurm/serve.slurm invokes drivers as `python DRIVER --config CONFIG --model
    # NAME` and forwards nothing else, so every flag below also has an environment
    # and a config fallback. sbatch propagates the submitting environment
    # (--export=ALL is the default), which makes `HEALTHY_RL_SHARD=0/3 sbatch ...`
    # work with no change to serve.slurm.
    shard = parse_shard(_setting(args.shard, "HEALTHY_RL_SHARD", cfg, "shard"))
    tiers_value = _setting(args.tiers, "HEALTHY_RL_TIERS", cfg, "tiers")
    tiers = [int(t) for t in str(tiers_value).replace(",", " ").split()] if tiers_value else None
    sweep_value = _setting(
        args.sweep_problems, "HEALTHY_RL_SWEEP_PROBLEMS", cfg, "sweep_problems_override"
    )
    sweep_problems = _as_list(sweep_value)

    # The flag lands in cfg so that run_rollouts() and the manifest agree on it.
    cfg = {**cfg, SCRATCHPAD_KEY: resolve_scratchpad(args.scratchpad_reasoning, cfg)}

    out_setting = _setting(args.out_dir, "HEALTHY_RL_OUT_DIR", cfg, "out_dir")
    out_dir = Path(out_setting) if out_setting else output_dir("rollouts", model, version)
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
        f"  out      {out_dir}\n"
        f"  scratchpad_reasoning={cfg[SCRATCHPAD_KEY]}",
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
