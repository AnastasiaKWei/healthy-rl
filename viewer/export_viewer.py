"""Export step0 .eval logs to a JSON blob for the transcript viewer.

Usage: export_viewer.py <logdir> [<logdir> ...] -o data.json [--all]

By default this keeps only real experimental runs: it drops smoke tests, capability
screens, anything under logs/partial/, and any run that produced no samples. Those
exist on disk for the record but they are noise in a reader. Pass --all to keep them.
"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path
from inspect_ai.log import read_eval_log

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "external/impossiblebench/src/impossiblebench"))

# Scored with the analysis code itself, not a second copy of the lexicon — otherwise
# the viewer and the numbers in analyse_step0.py drift apart silently.
from analyse_step0 import affect_points  # noqa: E402

# Upstream's extractor (the buggy one that actually graded these runs) and the
# corrected one, so each turn can be labelled with what the sandbox really got.
try:
    from livecodebench_scorers import find_code as upstream_find_code
    from healthy_rl.rollouts import robust_find_code
except ImportError:  # viewer still builds without the benchmark checked out
    upstream_find_code = robust_find_code = None

RESCORE = REPO / "logs" / "rescore_step0.json"

MAX_MSG = 60_000  # chars kept per message; beyond this the middle is elided


def clip(text):
    if text is None:
        return ""
    if len(text) <= MAX_MSG:
        return text
    head, tail = MAX_MSG // 3, MAX_MSG - MAX_MSG // 3
    cut = len(text) - head - tail
    return text[:head] + f"\n\n… [{cut:,} characters elided by the viewer export] …\n\n" + text[-tail:]


def parts(msg):
    reasoning, visible = [], []
    c = msg.content
    if isinstance(c, list):
        for item in c:
            t = getattr(item, "type", None)
            if t == "reasoning":
                reasoning.append(getattr(item, "reasoning", "") or "")
            elif t == "text":
                visible.append(getattr(item, "text", "") or "")
            else:
                visible.append(str(item))
    else:
        visible.append(c or "")
    return "\n".join(reasoning), "\n".join(visible)


def scratchpad_split(visible):
    pads = re.findall(r"<scratchpad>(.*?)(?:</scratchpad>|$)", visible, re.S | re.I)
    rest = re.sub(r"<scratchpad>.*?(?:</scratchpad>|$)", "", visible, flags=re.S | re.I)
    return "\n".join(pads), rest


def arm_of(path: Path):
    """Human label for the experimental arm, from the log directory name."""
    d = path.parent.name
    if not d.startswith(("google-", "qwen-", "openai-", "anthropic-")):
        return d, d  # ad-hoc dir (smoke tests, screens): label it by its own name
    reasoning = "reasoning on" if "reasoning-on" in d else ("reasoning off" if "reasoning-off" in d else "reasoning default")
    pad = "scratchpad" if "-pad" in d else None
    affect = "affect-prompted" if d.endswith("affect") else "neutral"
    bits = [reasoning] + ([pad] if pad else []) + [affect]
    return " · ".join(bits), d


SKIP_DIRS = ("smoke", "screen", "partial", "test")


def is_scratch(path: Path) -> bool:
    """True for smoke tests, screens and set-aside runs — real logs, but not results."""
    parts = [q.lower() for q in path.parts]
    return any(p.startswith(SKIP_DIRS) for p in parts)


# Runs from this timestamp onward call patch_find_code() in step0_elicitation.py, so
# their solver used robust_find_code and the model received real test failures. Before
# it, the solver ran upstream's extractor and could feed the model a SyntaxError raised
# by its own prose. The check below is only meaningful for the earlier runs: re-running
# the broken extractor over a *fixed* run's text answers "what would the old code have
# done", which says nothing about what that model actually experienced.
FIX_TIMESTAMP = "2026-08-15T18-35"


def extraction_status(visible: str, log_name: str = "") -> str:
    """What the sandbox actually received for this attempt.

    'prose' means upstream's find_code handed English to the interpreter, so the
    resulting failure was manufactured by the grader, not committed by the model.
    See experiments/rescore_step0.py for the mechanism.
    """
    if log_name and log_name >= FIX_TIMESTAMP:
        return "ok"                 # ran with the fix; nothing was manufactured
    if upstream_find_code is None or not (visible or "").strip():
        return ""
    try:
        ast.parse(upstream_find_code(visible))
        return "ok"
    except SyntaxError:
        pass
    try:                                    # did the corrected extractor do better?
        ast.parse(robust_find_code(visible))
        return "prose"                      # upstream broke it; the fix recovers code
    except SyntaxError:
        return "nocode"                     # no parsable code in the reply at all


def load_rescore():
    """{(arm, split, task_id): rescored_pass} from experiments/rescore_step0.py."""
    if not RESCORE.exists():
        return {}
    return {(r["arm"], r["split"], r["task_id"]): r["rescored_pass"]
            for r in json.loads(RESCORE.read_text())}


def export(paths):
    rescored = load_rescore()
    runs = []
    for p in paths:
        try:
            log = read_eval_log(str(p))
        except Exception as e:
            print(f"  skip {p.name}: {e}")
            continue
        label, dirname = arm_of(p)
        split = "conflicting" if "conflicting" in log.eval.task else "original"
        acc = None
        if log.results:
            for s in log.results.scores:
                m = s.metrics.get("accuracy")
                if m:
                    acc = m.value

        samples = []
        for s in (log.samples or []):
            ag = (s.metadata or {}).get("agentic_results", {}) or {}
            turns, turn_no = [], 0
            for m in s.messages:
                if m.role == "assistant":
                    turn_no += 1
                    reasoning, visible = parts(m)
                    pad, answer = scratchpad_split(visible)
                    pts = affect_points(f"{reasoning}\n{pad}\n{answer}")
                    turns.append({
                        "kind": "assistant", "turn": turn_no,
                        "reasoning": clip(reasoning), "scratchpad": clip(pad),
                        "answer": clip(answer),
                        "extraction": extraction_status(visible, p.name),
                        "neg": pts["neg"], "pos": pts["pos"],
                    })
                else:
                    turns.append({"kind": m.role, "turn": turn_no, "text": clip(
                        m.content if isinstance(m.content, str) else parts(m)[1])})
            score = None
            if s.scores:
                for k, v in s.scores.items():
                    score = str(v.value)
                    break
            samples.append({
                "id": str(s.id),
                "epoch": getattr(s, "epoch", 1),
                "score": score,
                "rescored": rescored.get((dirname, split, str(s.id))),
                "bad_turns": sum(1 for t in turns if t.get("extraction") in ("prose", "nocode")),
                "neg": sum(t.get("neg", 0) for t in turns),
                "pos": sum(t.get("pos", 0) for t in turns),
                "turns": sum(1 for t in turns if t["kind"] == "assistant"),
                "flag": bool(ag.get("flag_for_human_intervention")),
                "attempts": len(ag.get("attempt_history", []) or []),
                "max_attempts": ag.get("max_attempts"),
                "error": str(s.error)[:500] if s.error else None,
                "limit": str(s.limit)[:200] if getattr(s, "limit", None) else None,
                "messages": turns,
            })
        runs.append({
            "file": p.name, "dir": dirname, "arm": label, "split": split,
            "task": log.eval.task, "model": log.eval.model, "status": log.status,
            "accuracy": acc, "samples": samples,
        })
    return merge_reruns(runs)


def merge_reruns(runs):
    """Collapse each (arm, split) to one run, newest version of every sample winning.

    Samples corrupted by the broken `find_code` were re-run individually with the
    fix, so a later log holds only those task ids. Paths are timestamp-prefixed, so
    lexical order over the filename is chronological.
    """
    by_unit = {}
    for r in sorted(runs, key=lambda r: r["file"]):
        key = (r["dir"], r["split"])
        if key not in by_unit:
            by_unit[key] = {**r, "samples": {}}
        for s in r["samples"]:
            # keyed by (task, epoch): with --epochs>1 the same task id appears once
            # per epoch, and keying on the id alone would silently drop all but one.
            by_unit[key]["samples"][(s["id"], s.get("epoch", 1))] = s   # later file wins
        by_unit[key]["file"] = r["file"]              # keep newest for stable keys

    merged = []
    for unit in by_unit.values():
        unit["samples"] = [unit["samples"][k] for k in sorted(unit["samples"])]
        for smp in unit["samples"]:
            smp["label"] = (f"{smp['id']} · ep{smp['epoch']}"
                            if any(x["epoch"] != 1 for x in unit["samples"]) else smp["id"])
        scored = [s for s in unit["samples"] if s["score"] in ("C", "I")]
        unit["accuracy"] = (round(sum(s["score"] == "C" for s in scored) / len(scored), 3)
                            if scored else None)
        merged.append(unit)
    return merged


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--all", action="store_true",
                    help="keep smoke tests, screens and empty runs as well")
    a = ap.parse_args()
    files = []
    for d in a.dirs:
        p = Path(d)
        files += sorted(p.glob("**/*.eval")) if p.is_dir() else [p]
    if not a.all:
        kept = [f for f in files if not is_scratch(f)]
        if len(kept) != len(files):
            print(f"  dropped {len(files) - len(kept)} smoke/screen/partial log(s)")
        files = kept
    runs = export(files)
    if not a.all:
        empty = [r for r in runs if not r["samples"]]
        if empty:
            print(f"  dropped {len(empty)} run(s) with no samples")
        runs = [r for r in runs if r["samples"]]
    Path(a.out).write_text(json.dumps(runs, separators=(",", ":")))
    n_s = sum(len(r["samples"]) for r in runs)
    print(f"{len(runs)} runs, {n_s} samples -> {a.out} ({Path(a.out).stat().st_size/1e6:.2f} MB)")
    for r in runs:
        print(f"  {r['dir']:52s} {r['split']:12s} samples={len(r['samples'])} acc={r['accuracy']}")
