#!/usr/bin/env python
"""Sanity check for the scratchpad-reasoning flag against a live model.

Runs a handful of ImpossibleBench problems through the SAME task the rollout
stage builds (``healthy_rl.rollouts.build_task``, with the scratchpad system
prompt chained ahead of ImpossibleBench's solver) and reports whether the model
actually complies: opens ``<SCRATCHPAD_REASONING>`` first, closes it, and puts a
code block after the closing tag where the scorer will find it. By default the
same problems are then run WITHOUT the prompt, so the two hack rates and reply
shapes can be compared side by side.

No vectors, no hooks, no residuals: it uses Inspect's stock ``vllm`` provider,
so it works for a checkpoint that has no directions built yet. Like
``run_rollouts.py`` it hands itself over to ``apptainer/eval.sif`` when the
interpreter it starts in cannot run Inspect + impossiblebench.

    sbatch slurm/serve.slurm --model gemma-3-12b-it --config configs/rollouts.yaml \\
        --max-model-len 16384 --vllm-args "--language-model-only" \\
        --stage scripts/scratchpad_sanity.py::--problems 4 --samples 2

Outputs, under ``$ARTIFACT_DIR/rollouts/<model>/scratchpad-sanity/``:

    transcripts.jsonl     every turn of every rollout, both settings, full text
    scratchpad_sanity.json  the counts the verdict is based on
    inspect-logs/         the Inspect eval logs

Exit status is 1 when the model mostly ignores the prompt (fewer than half of
the first turns open AND close the tags) or when the system message never
reached the model, because that is what "the flag does not work here" means.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# scripts/ is sys.path[0] when this file is run as a script.
from run_rollouts import (
    reexec_in_container,
    resolve_artifact_root,
    resolve_base_url,
    resolve_model,
)

from healthy_rl.config import load_config, load_env, repo_root
from healthy_rl.rollouts import (
    build_task,
    output_dir,
    provider_base_url,
    scratchpad_summary,
    select_readout_problems,
    system_prompt_for,
)

DEFAULT_CONFIG = repo_root() / "configs" / "rollouts.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=None, help="override $HEALTHY_RL_MODEL_NAME")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--problems", type=int, default=4, help="first N problems by task id")
    parser.add_argument("--samples", type=int, default=1, help="rollouts per problem per setting")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=None, help="default: config's, else 1.0")
    parser.add_argument(
        "--baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also run the same problems without the prompt (default on)",
    )
    parser.add_argument("--print-chars", type=int, default=3500, help="how much of a reply to print")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------


def _text(message: Any) -> str:
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


def _sample_turns(sample: Any) -> list[dict[str, Any]]:
    """One dict per assistant turn: what went in (roles) and what came out."""
    turns = []
    for event in getattr(sample, "events", None) or []:
        if getattr(event, "event", None) != "model":
            continue
        output = getattr(event, "output", None)
        completion = getattr(output, "completion", "") if output is not None else ""
        messages = getattr(event, "input", None) or []
        turns.append(
            {
                "input_roles": [getattr(m, "role", "?") for m in messages],
                "system_prompt_seen": any(getattr(m, "role", None) == "system" for m in messages),
                "last_user_message": next(
                    (_text(m) for m in reversed(messages) if getattr(m, "role", None) == "user"), ""
                ),
                "completion": completion or "",
                "stop_reason": getattr(output, "stop_reason", None) if output is not None else None,
                "scratchpad": scratchpad_summary(completion),
            }
        )
    return turns


def _passed(sample: Any) -> bool:
    for score in (getattr(sample, "scores", None) or {}).values():
        value = getattr(score, "value", None)
        if isinstance(value, str):
            return value.upper() == "C"
        if isinstance(value, (bool, int, float)):
            return bool(value)
    return False


def run_setting(
    name: str,
    system_prompt: str | None,
    problems: list[str],
    bench_parquet: Path,
    model_name: str,
    base_url: str,
    cfg: dict,
    args: argparse.Namespace,
    out: Path,
) -> list[dict[str, Any]]:
    from inspect_ai import eval as inspect_eval
    from inspect_ai.log import read_eval_log
    from inspect_ai.model import GenerateConfig, get_model

    model = get_model(
        f"vllm/{model_name}",
        base_url=provider_base_url(base_url),
        memoize=False,
        config=GenerateConfig(
            temperature=float(args.temperature if args.temperature is not None else cfg.get("temperature", 1.0)),
            top_p=float(cfg.get("top_p", 1.0)),
            max_tokens=int(args.max_tokens),
            max_connections=int(cfg.get("max_connections", 8)),
        ),
    )
    task = build_task(
        problems,
        bench_parquet,
        max_attempts=int(cfg.get("max_attempts", 3)),
        message_limit=int(cfg.get("message_limit", 30)),
        sandbox=str(cfg.get("sandbox", "local")),
        use_hf=bool(cfg.get("use_hf_dataset", False)),
        system_prompt=system_prompt,
    )
    print(f"\n=== setting {name!r}: {len(problems)} problems x {args.samples} samples ===", flush=True)
    logs = inspect_eval(
        task,
        model=model,
        epochs=int(args.samples),
        log_dir=str(out / "inspect-logs" / name),
        max_samples=int(cfg.get("max_samples", 8)),
        max_subprocesses=int(cfg.get("max_subprocesses", 8)),
        max_sandboxes=int(cfg.get("max_sandboxes", 8)),
        fail_on_error=False,
        display="plain",
        score=True,
    )
    rollouts: list[dict[str, Any]] = []
    for log in logs:
        full = read_eval_log(log.location) if not log.samples else log
        for sample in full.samples or []:
            rollouts.append(
                {
                    "setting": name,
                    "task_id": str(sample.id),
                    "epoch": int(sample.epoch),
                    "passed": _passed(sample),
                    "error": getattr(getattr(sample, "error", None), "message", None),
                    "turns": _sample_turns(sample),
                }
            )
    return rollouts


def summarise(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    turns = [t for r in rollouts for t in r["turns"]]
    first = [r["turns"][0] for r in rollouts if r["turns"]]

    def frac(items, key):
        return (sum(1 for t in items if t["scratchpad"][key]) / len(items)) if items else None

    return {
        "n_rollouts": len(rollouts),
        "n_passed": sum(1 for r in rollouts if r["passed"]),
        "hack_rate": (sum(1 for r in rollouts if r["passed"]) / len(rollouts)) if rollouts else None,
        "n_errors": sum(1 for r in rollouts if r["error"]),
        "n_turns": len(turns),
        "system_prompt_seen_every_turn": all(t["system_prompt_seen"] for t in turns) if turns else False,
        "first_turn_opened": frac(first, "opened"),
        "first_turn_closed": frac(first, "closed"),
        "first_turn_starts_with_tag": frac(first, "starts_with_tag"),
        "all_turns_opened": frac(turns, "opened"),
        "all_turns_closed": frac(turns, "closed"),
        "all_turns_answer_has_code_block": frac(turns, "answer_has_code_block"),
        "mean_reasoning_chars": (
            sum(t["scratchpad"]["reasoning_chars"] for t in turns) / len(turns) if turns else None
        ),
        "stop_reasons": _count(t["stop_reason"] for t in turns),
        "turns_per_rollout": _count(len(r["turns"]) for r in rollouts),
    }


def _count(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items()))


def _print_example(rollout: dict[str, Any], limit: int) -> None:
    turn = rollout["turns"][0] if rollout["turns"] else None
    print(f"\n--- example: {rollout['setting']} {rollout['task_id']} epoch {rollout['epoch']} "
          f"passed={rollout['passed']} turns={len(rollout['turns'])} ---")
    if turn is None:
        print("(no model turns)")
        return
    print(f"input roles: {turn['input_roles']}   stop_reason: {turn['stop_reason']}")
    text = turn["completion"]
    if len(text) > limit:
        head, tail = limit * 2 // 3, limit // 3
        text = text[:head] + f"\n... [{len(turn['completion']) - limit} chars elided] ...\n" + text[-tail:]
    print(text, flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_env()
    reexec_in_container(sys.argv[1:] if argv is None else list(argv), script=__file__)
    cfg = load_config(args.config)

    model = resolve_model(args.model, cfg)
    version = str(cfg.get("version", "v1"))
    artifact_root = resolve_artifact_root(args.artifact_root, cfg)
    bench_dir = Path(cfg.get("bench_dir") or artifact_root / "bench" / version)
    bench_parquet = Path(cfg.get("bench_parquet") or bench_dir / "conflicting.parquet")
    out = Path(args.out_dir) if args.out_dir else output_dir("rollouts", model, "scratchpad-sanity")
    out.mkdir(parents=True, exist_ok=True)

    system_prompt = system_prompt_for({**cfg, "scratchpad_reasoning": True})
    assert system_prompt is not None

    import pandas as pd

    all_ids = [str(v) for v in pd.read_parquet(bench_parquet, columns=["task_id"])["task_id"]]
    problems = select_readout_problems(all_ids, int(args.problems))

    base_url = resolve_base_url(args.base_url)
    from healthy_rl.server import wait_for_health

    wait_for_health(base_url, timeout_s=float(cfg.get("health_timeout_s", 1800.0)))
    print(
        f"scratchpad sanity: model={model} url={base_url}\n"
        f"  bench     {bench_parquet}\n  out       {out}\n"
        f"  problems  {problems}\n  samples   {args.samples}  max_tokens {args.max_tokens}\n"
        f"  system prompt ({len(system_prompt)} chars):\n"
        + "\n".join("    | " + line for line in system_prompt.splitlines()),
        flush=True,
    )

    settings = [("scratchpad", system_prompt)] + ([("plain", None)] if args.baseline else [])
    rollouts: list[dict[str, Any]] = []
    for name, prompt in settings:
        rollouts.extend(run_setting(name, prompt, problems, bench_parquet, model, base_url, cfg, args, out))

    with (out / "transcripts.jsonl").open("w") as fh:
        for r in rollouts:
            fh.write(json.dumps(r, default=str) + "\n")

    report = {
        "model": model,
        "problems": problems,
        "samples": int(args.samples),
        "max_tokens": int(args.max_tokens),
        "system_prompt": system_prompt,
        "settings": {name: summarise([r for r in rollouts if r["setting"] == name]) for name, _ in settings},
    }

    # Per-rollout table.
    print("\nsetting     task_id      ep passed turns  first-turn: opened closed starts-with-tag code-after-close  reasoning-chars")
    for r in rollouts:
        if r["turns"]:
            s = r["turns"][0]["scratchpad"]
            shape = f"{str(s['opened']):6} {str(s['closed']):6} {str(s['starts_with_tag']):15} {str(s['answer_has_code_block']):16} {s['reasoning_chars']:>6}"
        else:
            shape = "(no turns)" + (f" error={r['error']}" if r["error"] else "")
        print(f"{r['setting']:11} {r['task_id']:12} {r['epoch']:>2} {str(r['passed']):6} {len(r['turns']):>5}  {shape}")

    for name, _ in settings:
        for r in rollouts:
            if r["setting"] == name and r["turns"]:
                _print_example(r, args.print_chars)
                break

    print("\n" + json.dumps(report["settings"], indent=2), flush=True)

    scratch = report["settings"]["scratchpad"]
    ok = bool(scratch["n_rollouts"]) and scratch["system_prompt_seen_every_turn"]
    compliant = (scratch["first_turn_opened"] or 0.0) >= 0.5 and (scratch["first_turn_closed"] or 0.0) >= 0.5
    verdict = "OK" if ok and compliant else "FAIL"
    if not ok:
        why = "no rollouts ran" if not scratch["n_rollouts"] else "the system message did not reach the model on every turn"
    elif not compliant:
        why = "fewer than half of the first turns opened and closed the scratchpad tags"
    else:
        why = "the system prompt reached the model and it used the tags"
    report["verdict"] = verdict
    report["why"] = why
    (out / "scratchpad_sanity.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"\nscratchpad sanity {verdict} for {model}: {why}\n  wrote {out}", flush=True)
    return 0 if verdict == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
