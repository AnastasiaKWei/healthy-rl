"""Emotion-vector trajectories from rollouts still in flight. No GPU, no waiting.

Each rollout record carries `turn_stat` -- the 14 emotion projections at the
probe layer, one vector per turn -- and records are appended and fsynced as each
rollout completes. So the trajectory is readable continuously; nothing here has
to wait for a run to finish.

Reports, per model and output version:
  * projection as a function of TURN INDEX, per direction. This is the shape the
    source paper's transcript figures show, and it is the one a 3-turn run could
    not support: whether affect BUILDS over turns rather than merely differing
    between the first turn and later ones.
  * the first-vs-last turn contrast, paired within transcript, ranked by effect.
  * a repetition flag: turns whose generated length is identical to the previous
    turn's, which in practice marks a model re-emitting the same answer rather
    than exploring. "Stuck" and "making progress" are plausibly different
    affective states and this separates them cheaply.

Usage:
    scripts/live_trajectory.py --model Qwen3.5-9B --version d6
    scripts/live_trajectory.py --model Qwen3.5-9B --version d6 --emotions desperate,frustrated
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
from pathlib import Path

DEFAULT_SHOW = ["desperate", "frustrated", "exasperated", "overwhelmed",
                "nervous", "angry", "calm", "joyful"]


def load(model: str, version: str) -> list[dict]:
    base = Path(os.environ["ARTIFACT_DIR"]) / "rollouts" / model / version
    return [json.loads(line) for f in sorted(base.glob("*.jsonl")) for line in f.open()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--emotions", default=None, help="comma-separated subset")
    args = ap.parse_args()

    rows = load(args.model, args.version)
    if not rows:
        print(f"no records yet for {args.model}/{args.version}")
        return 0

    emotions = rows[0]["emotions"]
    show = [e.strip() for e in args.emotions.split(",")] if args.emotions else \
        [e for e in DEFAULT_SHOW if e in emotions]

    depths = [r["n_turns"] for r in rows]
    hacks = sum(r["passed"] for r in rows)
    print(f"{args.model}/{args.version}: {len(rows)} rollouts in flight or done, "
          f"turns/rollout {min(depths)}-{max(depths)}, hack {hacks}/{len(rows)}")

    # --- trajectory: projection by turn index -------------------------------
    max_turns = max(depths)
    print(f"\nPROJECTION BY TURN INDEX (mean over rollouts; n per turn in brackets)")
    header = "direction     " + "".join(f"{'t'+str(i):>9s}" for i in range(max_turns))
    print(header)
    # Index by position among NON-EMPTY turns, not by raw turn index. Some
    # rollouts begin with turns that generated zero tokens (an errored or retried
    # request), and those carry no projection. Aligning on the raw index would
    # average one rollout's first real turn against another's fourth.
    seqs = [[t for t in (r.get("turn_stat") or []) if t] for r in rows]
    seqs = [s for s in seqs if s]
    max_turns = max((len(s) for s in seqs), default=0)
    counts = [sum(1 for s in seqs if len(s) > i) for i in range(max_turns)]
    for e in show:
        idx = emotions.index(e)
        cells = []
        for i in range(max_turns):
            vals = [s[i][idx] for s in seqs if len(s) > i]
            cells.append(f"{st.mean(vals):+9.4f}" if vals else f"{'-':>9s}")
        print(f"{e:14s}" + "".join(cells))
    print(f"{'n':14s}" + "".join(f"{c:>9d}" for c in counts))
    empties = sum(1 for r in rows for g in (r.get("turn_n_generated") or []) if g == 0)
    if empties:
        print(f"  ({empties} turn(s) generated zero tokens and are excluded; "
              f"columns are position among non-empty turns)")

    # --- first vs last turn, paired within transcript -----------------------
    print(f"\nFIRST vs LAST TURN, paired within transcript")
    deltas: dict[str, list[float]] = {e: [] for e in emotions}
    for r in rows:
        ts = [t for t in (r.get("turn_stat") or []) if t]
        if len(ts) < 2:
            continue
        for j, e in enumerate(emotions):
            deltas[e].append(ts[-1][j] - ts[0][j])
    n = len(next(iter(deltas.values()))) if deltas else 0
    if n < 2:
        print("  not enough multi-turn transcripts yet")
    else:
        try:
            from scipy import stats as sps
        except ImportError:
            sps = None
        ranked = []
        for e in emotions:
            d = deltas[e]
            p = sps.wilcoxon(d).pvalue if (sps and len(d) >= 6 and any(d)) else float("nan")
            ranked.append((e, st.mean(d), st.pstdev(d) / max(len(d) ** 0.5, 1), p))
        ranked.sort(key=lambda r: -r[1])
        print(f"  {'direction':14s} {'last-first':>11s} {'sem':>9s} {'wilcoxon':>9s}   (n={n})")
        for e, m, sem, p in ranked:
            star = " *" if p == p and p < 0.05 else ""
            pf = f"{p:9.4f}" if p == p else f"{'-':>9s}"
            print(f"  {e:14s} {m:+11.5f} {sem:9.5f} {pf}{star}")

    # --- repetition: same generated length as the previous turn -------------
    rep_turns = rep_rollouts = tot_turns = 0
    for r in rows:
        gens = r.get("turn_n_generated") or []
        hit = False
        for a, b in zip(gens, gens[1:]):
            tot_turns += 1
            if a == b and a > 0:
                rep_turns += 1
                hit = True
        rep_rollouts += hit
    if tot_turns:
        print(f"\nREPETITION  {rep_turns}/{tot_turns} turn-pairs identical in length, "
              f"in {rep_rollouts}/{len(rows)} rollouts")
        print("  (a turn generating exactly as many tokens as the previous one usually means the "
              "model re-emitted its answer rather than exploring)")

    print("\nExploratory: no correction across 14 directions, and turn-index means mix "
          "rollouts of differing depth. Read the n row.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
