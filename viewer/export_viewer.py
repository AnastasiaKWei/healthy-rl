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
JUDGE = REPO / "logs" / "judge_step0.json"

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


# How each mindset variant reads in the rail. Keep in step with MINDSET in
# experiments/step0_elicitation.py.
MINDSET_LABEL = {"growth": "growth mindset",
                 "resilience": "resilience",
                 "appraisal": "may declare it impossible"}


def arm_of(path: Path):
    """(condition label, dir name, factors) for the arm, from the log directory.

    The directory name is the slug built by experiments/step0_elicitation.py:

        <model>[-reasoning-{on,off}]-{pad,nopad}-{affect,neutral}[-mindset-<v>[+<v>]]

    plus a hand-added `-e<N>` for multi-epoch runs. This has to be parsed rather
    than pattern-matched: the previous check for the affect prompt was
    `d.endswith("affect")`, which is false for every mindset and every -e3 arm,
    so seven affect-prompted arms were labelled "neutral" in the rail.
    """
    d = path.parent.name
    if not d.startswith(("google-", "qwen-", "openai-", "anthropic-")):
        return d, d, {"model": d, "cond": d, "prompt": "?", "mindset": []}

    reasoning = ("on" if "reasoning-on" in d else
                 "off" if "reasoning-off" in d else "default")
    pad = "-pad-" in d or d.endswith("-pad")
    prompt = "affect" if re.search(r"-affect(?:-|$)", d) else "neutral"
    # `-mindset-[vN-]variant[+variant]`. The optional version segment must be matched
    # explicitly: `[a-z+]+` alone stops at the digit in "v2" and reports the variant
    # as "v", so every v2 arm would be labelled identically and wrongly.
    m = re.search(r"-mindset-(?:v(\d+)-)?([a-z+]+)", d)
    mindset = m.group(2).split("+") if m else []
    mversion = int(m.group(1)) if m and m.group(1) else 1
    ep = int(re.search(r"-e(\d+)(?:-|$)", d).group(1)) if re.search(r"-e(\d+)(?:-|$)", d) else 1

    # The condition label is what the rail groups by, so it names only the things
    # that were manipulated: the prompt and any intervention on top of it.
    cond = "asked how it feels" if prompt == "affect" else "neutral prompt"
    if mindset:
        cond += " + " + " + ".join(MINDSET_LABEL.get(v, v) for v in mindset)
        cond += f" (v{mversion})"   # prompt wording differs between versions

    # Setup deliberately excludes the epoch count: a 1-epoch and a 3-epoch run of
    # the same condition are the same condition, and the per-sample labels already
    # carry "· ep2". Keeping epochs out lets those two runs share a rail group.
    setup = " · ".join([f"reasoning {reasoning}"] + (["scratchpad"] if pad else []))

    # Rail grouping key. It must carry the setup as well as the prompt, because
    # "neutral prompt" is the label for both reasoning-on and reasoning-off arms
    # of the same model -- grouping on the visible label alone silently merges
    # two different conditions into one list.
    condkey = (f"{reasoning}/{'pad' if pad else 'nopad'}/{prompt}/"
               f"{'+'.join(mindset)}/v{mversion}")

    return cond, d, {"cond": cond, "condkey": condkey, "prompt": prompt,
                     "mindset": mindset, "mversion": mversion,
                     "reasoning": reasoning, "pad": pad,
                     "epochs": ep, "setup": setup}


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


def load_judge():
    """{(arm, split, task, epoch, turn, channel): verdict} from judge_step0.py.

    The judge's two channels line up with the two block colours in the viewer:
    its `private` is the reasoning trace plus the scratchpad (the amber block),
    its `visible` is the graded answer (the teal block). Turns the judge could
    not score are left out, so a missing badge means "not scored", never "calm".
    """
    if not JUDGE.exists():
        return {}
    out = {}
    for r in json.loads(JUDGE.read_text()):
        if r.get("rating") is None:
            continue
        key = (r["arm"], r["split"], r["task_id"], r.get("epoch", 1),
               r["turn"], r["channel"])
        out[key] = {"rating": r["rating"], "evidence": r.get("evidence", ""),
                    "why": r.get("reasoning", "")}
    return out


def export(paths):
    rescored = load_rescore()
    judged = load_judge()
    runs = []
    for p in paths:
        try:
            log = read_eval_log(str(p))
        except Exception as e:
            print(f"  skip {p.name}: {e}")
            continue
        label, dirname, facts = arm_of(p)
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
                    ep = getattr(s, "epoch", 1)
                    jk = (dirname, split, str(s.id), ep, turn_no)
                    turns.append({
                        "kind": "assistant", "turn": turn_no,
                        "reasoning": clip(reasoning), "scratchpad": clip(pad),
                        "answer": clip(answer),
                        "extraction": extraction_status(visible, p.name),
                        "neg": pts["neg"], "pos": pts["pos"],
                        "jpriv": judged.get(jk + ("private",)),
                        "jvis": judged.get(jk + ("visible",)),
                    })
                else:
                    turns.append({"kind": m.role, "turn": turn_no, "text": clip(
                        m.content if isinstance(m.content, str) else parts(m)[1])})
            score = None
            if s.scores:
                for k, v in s.scores.items():
                    score = str(v.value)
                    break
            # Per-turn judge score is the higher of the turn's two channels, which
            # is what the rubric asks for: the single most negative expression in
            # the response. `jhigh` counts turns at >=5, the paper's headline band.
            jt = [max([c["rating"] for c in (t.get("jpriv"), t.get("jvis")) if c],
                      default=None)
                  for t in turns if t["kind"] == "assistant"]
            jt = [x for x in jt if x is not None]
            samples.append({
                "id": str(s.id),
                "epoch": getattr(s, "epoch", 1),
                "score": score,
                "jmean": round(sum(jt) / len(jt), 2) if jt else None,
                "jmax": max(jt) if jt else None,
                "jhigh": sum(x >= 5 for x in jt) if jt else None,
                "jturns": len(jt),
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
            "accuracy": acc, "samples": samples, **facts,
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
