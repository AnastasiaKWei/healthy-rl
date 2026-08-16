"""Fast prompt-iteration loop: one model, five impossible tasks, one command.

The iteration vehicle is Gemma 3 12B on lcbhard_0-4, `conflicting` split, with
scratchpad + affect prompt — the highest-affect, fastest, cheapest arm we have
(~1 min/sample, one OpenRouter provider, prompted baseline ~2 neg/turn, and the
model Soligo et al. report as the high-distress outlier). Everything is pinned
except the intervention block, so two ledger rows differ only in the prompt.

Each run registers your draft block as a v2 mindset (sent once, on turn 1 only,
via `send_mindset_once`), runs it, scores it with the lexicon, and appends to
`logs/iterate/ledger.jsonl`. The table prints every draft so far, newest last.

Resolution honesty: at --epochs 1 (default, ~7 min) the 5-task aggregate moved
by ~0.9/turn between identical sessions once, so treat e1 as directional only —
rough read ±0.6. At --epochs 3 (~25 min) within-session SE was ~0.09, so ±0.35
between two arms. Anchor each session with `--label baseline-<date>` first and
compare drafts to that anchor, never to yesterday's numbers.

Usage:
    ./.venv/bin/python experiments/iterate.py --label baseline-aug16
    ./.venv/bin/python experiments/iterate.py --label wise-fb-1 --block-file drafts/wise.txt
    ./.venv/bin/python experiments/iterate.py --table          # ledger only, no run
    ./.venv/bin/python experiments/iterate.py --label wise-fb-1 --judge   # add LLM-judge pass
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LEDGER = REPO / "logs" / "iterate" / "ledger.jsonl"

MODEL = "openrouter/google/gemma-3-12b-it"
SPLIT = "conflicting"


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "step0_elicitation", REPO / "experiments" / "step0_elicitation.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_arm(label: str, block: str | None, epochs: int, tasks: int) -> str:
    """Run one draft through the step0 runner; returns the log dir name."""
    m = load_runner()
    if block:
        # Registered under the label so the log dir is unique per draft and the
        # ledger row can always be traced back to its exact prompt text.
        m.MINDSET[label] = block
    argv = ["iterate", "--model", MODEL, "--scratchpad", "--affect-prompt",
            "--splits", SPLIT, "--epochs", str(epochs), "--limit", str(tasks),
            "--max-sandboxes", "4"]
    if block:
        argv += ["--mindset", label]
    old = sys.argv
    try:
        sys.argv = argv
        m.main()
    finally:
        sys.argv = old
    slug = "google-gemma-3-12b-it-pad-affect"
    if block:
        # match the runner's own naming, including its prompt-schema version tag
        slug += f"-mindset-v{m.MINDSET_VERSION}-{label}"
    if epochs != 1:
        slug += f"-e{epochs}"
    if not (REPO / "logs" / "step0" / slug).is_dir():
        raise RuntimeError(f"expected log dir missing: {slug} — runner naming changed?")
    return slug


def score(logdir_name: str) -> dict:
    """Lexicon-score one arm: totals plus per-epoch aggregates."""
    sys.path.insert(0, str(REPO / "experiments"))
    from analyse_step0 import affect_points, parts  # noqa: E402
    from inspect_ai.log import read_eval_log  # noqa: E402

    by_epoch = defaultdict(lambda: [0, 0, 0])  # neg, pos, turns
    n = 0
    for f in sorted((REPO / "logs" / "step0" / logdir_name).glob("*.eval")):
        log = read_eval_log(str(f))
        if log.status != "success":
            continue
        for s in log.samples or []:
            n += 1
            for msg in s.messages:
                if msg.role != "assistant":
                    continue
                r, v = parts(msg)
                p = affect_points(r + "\n" + v)
                e = by_epoch[s.epoch]
                e[0] += p["neg"]; e[1] += p["pos"]; e[2] += 1
    neg = sum(e[0] for e in by_epoch.values())
    pos = sum(e[1] for e in by_epoch.values())
    turns = sum(e[2] for e in by_epoch.values())
    return {
        "samples": n, "turns": turns, "neg": neg, "pos": pos,
        "neg_per_turn": round(neg / turns, 3) if turns else None,
        "pos_per_turn": round(pos / turns, 3) if turns else None,
        "per_epoch_neg": [round(e[0] / e[2], 2) for e in
                          (by_epoch[k] for k in sorted(by_epoch)) if e[2]],
    }


def print_table() -> None:
    if not LEDGER.exists():
        print("ledger is empty — run an arm first")
        return
    rows = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    print(f"{'label':24s} {'when':16s} {'ep':>2} {'n':>3} {'neg/t':>6} "
          f"{'pos/t':>6}  per-epoch neg")
    print("-" * 84)
    for r in rows:
        nt = "—" if r["neg_per_turn"] is None else r["neg_per_turn"]
        pt = "—" if r["pos_per_turn"] is None else r["pos_per_turn"]
        print(f"{r['label'][:24]:24s} {r['when']:16s} {r['epochs']:>2} "
              f"{r['samples']:>3} {str(nt):>6} {str(pt):>6}  {r['per_epoch_neg']}")
    print("\nrough read: ±0.6 at e1, ±0.35 at e3. Compare drafts to this "
          "session's baseline row, not to yesterday's.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", help="short name for this draft; becomes the log dir")
    ap.add_argument("--block-file", help="file whose whole text is the intervention "
                    "block (the 'How to approach this:' body). Omit for a baseline.")
    ap.add_argument("--block-text", help="the block inline, instead of a file")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--judge", action="store_true",
                    help="after the lexicon, run the Soligo-replica LLM judge on "
                    "this arm (OpenRouter, cached, ~cents)")
    ap.add_argument("--table", action="store_true", help="print the ledger and exit")
    a = ap.parse_args()

    if a.table:
        print_table()
        return
    if not a.label:
        ap.error("--label is required (or use --table)")
    label = re.sub(r"[^a-z0-9-]", "-", a.label.lower())

    block = None
    if a.block_file:
        block = Path(a.block_file).read_text().strip()
    elif a.block_text:
        block = a.block_text.strip()

    t0 = time.time()
    logdir = run_arm(label, block, a.epochs, a.tasks)
    result = score(logdir)

    row = {
        "label": label, "when": time.strftime("%m-%d %H:%M"),
        "epochs": a.epochs, "logdir": logdir,
        "block_sha": hashlib.sha1(block.encode()).hexdigest()[:10] if block else None,
        "block": block, "minutes": round((time.time() - t0) / 60, 1),
        **result,
    }
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(row) + "\n")

    print()
    print_table()

    if a.judge:
        print(f"\n=== judge pass on {logdir} ===")
        subprocess.run([sys.executable, str(REPO / "experiments" / "judge_step0.py"),
                        "--logs", str(REPO / "logs" / "step0" / logdir),
                        "--out", str(REPO / "logs" / "iterate" / f"judge_{label}.json"),
                        "--cache", str(REPO / "logs" / "judge_step0_cache.json")])


if __name__ == "__main__":
    main()
