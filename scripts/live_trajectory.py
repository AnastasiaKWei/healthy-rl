"""Emotion-vector trajectories from rollouts still in flight. No GPU, no waiting.

MEASUREMENT GRANULARITY -- read this before comparing against the paper.

Two statistics are available and they differ by 2-3x:

  --stat token  (DEFAULT)  cosine similarity at a SINGLE token, computed from the
                           residual stored at each turn boundary. This is what
                           Anthropic's emotion-vectors paper reports: it reads its
                           probes at one position (the ":" after "Assistant") and
                           plots cosine similarity. Its observational contrasts --
                           Tylenol 500mg vs 16000mg on `afraid`, hours without food
                           on `afraid`, exam results on `happy` -- span 0.05-0.08 in
                           these units, and its behaviourally effective steering was
                           0.05-0.10.

  --stat mean              mean projection over ALL generated tokens in the turn,
                           divided by the layer's mean residual norm. Averaging over
                           ~900 tokens washes out anything localised: measured here
                           it shrinks the same effects by 2-3x (Ministral
                           `frustrated` across six turns is 0.021 at a single token
                           and 0.009 averaged). Kept for comparison, and because it
                           is what the rollout records store directly.

Both are dimensionless and near-identically constructed -- directions are unit
norm, so `mean` divides by the layer's mean residual norm where `token` divides by
that token's own norm. They agree to within a token's deviation from average.

ALSO REPORTED
  * projection by TURN INDEX, indexed by position among NON-EMPTY turns: some
    rollouts open with turns that generated zero tokens, and raw indexing would
    average one rollout's first real turn against another's fourth.
  * first-vs-last turn, paired within transcript, ranked by effect size.
  * a repetition flag -- turns generating exactly as many tokens as the previous
    turn, which marks a model re-emitting rather than exploring.

Usage:
    scripts/live_trajectory.py --model Ministral-3-14B-Reasoning-2512 --version d6
    scripts/live_trajectory.py --model Qwen3.5-9B --version d6 --stat mean
    scripts/live_trajectory.py --model gemma-3-12b-it --version aff6 --position end
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
from pathlib import Path

import numpy as np

DEFAULT_SHOW = ["desperate", "frustrated", "exasperated", "overwhelmed",
                "nervous", "angry", "calm", "joyful"]


def load_rows(base: Path) -> list[dict]:
    return [json.loads(line) for f in sorted(base.glob("*.jsonl")) for line in f.open()]


def load_directions(root: str, model: str):
    from safetensors.numpy import load_file

    meta = json.load(open(f"{root}/vectors/{model}/v1/vectors.json"))
    dirs = load_file(f"{root}/vectors/{model}/v1/vectors.safetensors")["directions"]
    probe = int(meta["probe_layer"])
    li = meta["capture_layers"].index(probe)
    return dirs[:, li, :].astype(np.float64), list(meta["emotions"]), probe


def token_sequences(rows, base: Path, D, probe: int, position: str):
    """Per-rollout list of per-turn cosine vectors, from the boundary residuals.

    Skips residuals that are not finite. Some checkpoints emit inf/NaN at
    individual positions -- gemma-3-12b-it does so on 4/288 stored residuals in
    its plain run and 144/288 in the affect-prompt run -- and a single NaN would
    otherwise poison the mean for that whole turn index. The count of skipped
    vectors is returned so it can be reported rather than silently absorbed.
    """
    seqs, skipped, seen = [], 0, 0
    for r in rows:
        rel = r.get("residuals")
        if not rel:
            continue
        path = base / rel
        if not path.exists():
            continue
        z = np.load(path)
        turns = []
        for t in range(r.get("n_turns", 0)):
            key = f"t{t}_res_{position}_L{probe}"
            if key not in z:
                continue
            seen += 1
            h = z[key].astype(np.float64)
            if not np.isfinite(h).all():
                skipped += 1
                continue
            n = float(np.linalg.norm(h))
            if not np.isfinite(n) or n == 0.0:
                skipped += 1
                continue
            turns.append(list((D @ h) / n))
        if turns:
            seqs.append(turns)
    return seqs, skipped, seen


def mean_sequences(rows):
    """Per-rollout list of per-turn mean-projection vectors, skipping empty turns."""
    return [s for s in ([t for t in (r.get("turn_stat") or []) if t] for r in rows) if s]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--emotions", default=None, help="comma-separated subset")
    ap.add_argument("--stat", choices=["token", "mean"], default="token",
                    help="token = single-token cosine, paper-comparable (default); "
                         "mean = turn-averaged projection")
    ap.add_argument("--position", choices=["start", "end"], default="start",
                    help="which boundary token for --stat token (default start: the first "
                         "generated token of the turn, closest analogue to the paper's "
                         "Assistant-colon readout)")
    args = ap.parse_args()

    root = os.environ["ARTIFACT_DIR"]
    base = Path(root) / "rollouts" / args.model / args.version
    rows = load_rows(base)
    if not rows:
        print(f"no records yet for {args.model}/{args.version}")
        return 0

    emotions = rows[0]["emotions"]
    if args.stat == "token":
        D, dir_emotions, probe = load_directions(root, args.model)
        if dir_emotions != emotions:
            print("ERROR: direction order differs from the records' emotion order; "
                  "refusing to mix them.")
            return 2
        seqs, skipped, seen = token_sequences(rows, base, D, probe, args.position)
        unit = f"single-token cosine at turn {args.position}, layer {probe}"
        if skipped:
            unit += f"  [{skipped}/{seen} residuals skipped as non-finite]"
    else:
        seqs = mean_sequences(rows)
        skipped = seen = 0
        unit = "turn-mean projection / layer mean residual norm"

    if not seqs:
        print(f"no usable {args.stat} data yet (residuals may not be written)")
        return 0

    show = [e.strip() for e in args.emotions.split(",")] if args.emotions else \
        [e for e in DEFAULT_SHOW if e in emotions]
    depths = [r["n_turns"] for r in rows]
    passed = sum(r["passed"] for r in rows)
    # `passed` inverts across ImpossibleBench splits: on `conflicting` it means the
    # model satisfied mutually unsatisfiable tests (a hack), on `original` it means
    # it solved the problem. Printing "hack 5/24" for a solvable run would be a
    # fabricated finding, so the label comes from the record, not from a default.
    split = str(rows[0].get("bench_split") or "conflicting")
    label = "hack" if split == "conflicting" else "solved"
    print(f"{args.model}/{args.version}: {len(rows)} rollouts, {len(seqs)} with {args.stat} "
          f"data, turns/rollout {min(depths)}-{max(depths)}, split {split}, "
          f"{label} {passed}/{len(rows)}")
    print(f"statistic: {unit}")

    max_turns = max(len(s) for s in seqs)
    counts = [sum(1 for s in seqs if len(s) > i) for i in range(max_turns)]
    print("\nBY TURN INDEX (position among non-empty turns)")
    print("direction     " + "".join(f"{'t'+str(i):>9s}" for i in range(max_turns))
          + f"{'range':>9s}")
    for e in show:
        idx = emotions.index(e)
        means = []
        for i in range(max_turns):
            vals = [s[i][idx] for s in seqs if len(s) > i]
            means.append(st.mean(vals) if vals else float("nan"))
        finite = [m for m in means if m == m]
        rng = (max(finite) - min(finite)) if finite else float("nan")
        print(f"{e:14s}" + "".join(f"{m:+9.4f}" for m in means) + f"{rng:+9.4f}")
    print(f"{'n':14s}" + "".join(f"{c:>9d}" for c in counts))

    print("\nFIRST vs LAST TURN, paired within transcript")
    deltas = {e: [s[-1][i] - s[0][i] for s in seqs if len(s) >= 2]
              for i, e in enumerate(emotions)}
    n = len(next(iter(deltas.values()), []))
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

    rep = tot = rep_rollouts = 0
    for r in rows:
        gens = r.get("turn_n_generated") or []
        hit = False
        for a, b in zip(gens, gens[1:]):
            tot += 1
            if a == b and a > 0:
                rep += 1
                hit = True
        rep_rollouts += hit
    if tot:
        print(f"\nREPETITION  {rep}/{tot} turn-pairs identical in length, in "
              f"{rep_rollouts}/{len(rows)} rollouts")

    if args.stat == "token":
        print("\nPaper reference, same units: observational contrasts span 0.05-0.08 "
              "(Tylenol 500mg->16000mg on `afraid` +0.058, hours-without-food +0.059, "
              "exam results on `happy` +0.079); behaviourally effective steering 0.05-0.10.")
    print("Exploratory: no correction across 14 directions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
