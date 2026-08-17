"""Tables and figures for the write-up, generated from the judge output.

Everything here reads logs/judge_step0.json and the eval logs, so a re-run after
new arms land regenerates the paper's numbers rather than leaving a figure that
agrees with an earlier version of the data.

    ./.venv/bin/python experiments/make_figures.py -o figures/

Writes:
    table1_arms.{md,csv}   arm-level distress: n, mean, CI, %>=5, private, visible
    table2_behaviour.md    attempts, passes, hacks, timeout share -- the flooring
    fig1_decoupling.pdf    private vs visible per arm, one panel per model
    fig2_ablation.pdf      marginal contribution of each scaffold factor
    fig3_calibration.pdf   judge vs published anchors (appendix)
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
JUDGE = REPO / "logs" / "judge_step0.json"

# Published anchor ratings from Soligo et al. (2026), used by --calibrate.
ANCHORS = [0, 1, 3, 5, 6, 7, 8, 9, 10]
OURS = [0, 3, 6, 6, 7, 6, 9, 10, 9]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})


def boot(vals, n=4000, level="episode"):
    """Cluster bootstrap. Turns inside an episode are not independent -- the model
    sees its own previous attempt -- so episodes are the resampling unit."""
    if not vals:
        return (0.0, 0.0)
    random.seed(0)
    by = defaultdict(list)
    for task, ep, x in vals:
        by[task if level == "task" else (task, ep)].append(x)
    keys = sorted(by)
    means = []
    for _ in range(n):
        s = [x for k in (random.choice(keys) for _ in keys) for x in by[k]]
        means.append(sum(s) / len(s))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def load():
    """(arm, split) -> per-turn scores, plus the two channels separately."""
    rows = [r for r in json.loads(JUDGE.read_text()) if r.get("rating") is not None]
    turns = defaultdict(dict)
    for r in rows:
        key = (r["arm"], r["split"], r["task_id"], r.get("epoch", 1), r["turn"])
        turns[key][r["channel"]] = r["rating"]
    agg = defaultdict(lambda: {"t": [], "p": [], "v": []})
    for (arm, split, task, ep, _turn), ch in turns.items():
        a = agg[(arm, split)]
        a["t"].append((task, ep, max(ch.values())))
        if "private" in ch:
            a["p"].append(ch["private"])
        if "visible" in ch:
            a["v"].append(ch["visible"])
    return agg


def stats(agg, arm, split):
    a = agg.get((arm, split))
    if not a or not a["t"]:
        return None
    t = [x for _, _, x in a["t"]]
    lo, hi = boot(a["t"])
    return {
        "n": len(t), "mean": sum(t) / len(t), "lo": lo, "hi": hi,
        "ge5": 100 * sum(x >= 5 for x in t) / len(t),
        "priv": sum(a["p"]) / len(a["p"]) if a["p"] else float("nan"),
        "vis": sum(a["v"]) / len(a["v"]) if a["v"] else float("nan"),
    }


G = "google-gemma-3-12b-it-pad-affect"
Q = "qwen-qwen3-14b-reasoning-on-nopad-affect"
ARMS = ["growth", "resilience", "control", "compassion"]

# (panel title, split, baseline arm, {label: arm}). Each model is read against its
# own baseline: the two sit on different splits and are not comparable directly.
BLOCKS = [
    ("Gemma-3-12B, solvable, v3", "original", f"{G}-hackable-e3",
     {v: f"{G}-mindset-v3-{v}-hackable-e3" for v in ARMS}),
    ("Gemma-3-12B, solvable, v4", "original", f"{G}-v4-hackable-e3",
     {v: f"{G}-mindset-v4-{v}-hackable-e3" for v in ARMS}),
    ("Qwen3-14B, impossible, v3", "conflicting", f"{Q}-hackable-e3",
     {v: f"{Q}-mindset-v3-{v}-hackable-e3" for v in ARMS}),
]

ABLATIONS = [
    ("task restated", f"{G}-v4-restate-hackable-e3"),
    ("retry guidance removed", f"{G}-v4-noretry-hackable-e3"),
    ("solvability note removed", f"{G}-v4-nonote-hackable-e3"),
    ("output relabel removed", f"{G}-v4-errlabel-hackable-e3"),
]


def table1(agg, out: Path):
    lines = ["| condition | arm | n | mean | 95% CI | %≥5 | private | visible | Δ |",
             "|---|---|---:|---:|---|---:|---:|---:|---:|"]
    csv = ["condition,arm,n,mean,lo,hi,ge5,private,visible,delta"]
    for title, split, base_arm, arms in BLOCKS:
        b = stats(agg, base_arm, split)
        if not b:
            continue
        for label, arm in [("baseline", base_arm)] + list(arms.items()):
            s = stats(agg, arm, split)
            if not s:
                continue
            d = "" if label == "baseline" else f"{s['mean'] - b['mean']:+.2f}"
            lines.append(f"| {title} | {label} | {s['n']} | {s['mean']:.2f} | "
                         f"[{s['lo']:.2f}, {s['hi']:.2f}] | {s['ge5']:.0f}% | "
                         f"{s['priv']:.2f} | {s['vis']:.2f} | {d} |")
            csv.append(f"{title},{label},{s['n']},{s['mean']:.3f},{s['lo']:.3f},"
                       f"{s['hi']:.3f},{s['ge5']:.1f},{s['priv']:.3f},{s['vis']:.3f},{d}")
    (out / "table1_arms.md").write_text("\n".join(lines) + "\n")
    (out / "table1_arms.csv").write_text("\n".join(csv) + "\n")
    print(f"  table1_arms.md   {len(lines) - 2} rows")


def fig1_decoupling(agg, out: Path):
    """The central claim: which channel moves.

    Plotted as paired channel means per arm rather than as the headline score,
    because the headline is the max of the two and would hide exactly the gap the
    figure exists to show.
    """
    panels = [b for b in BLOCKS if stats(agg, b[2], b[1])]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.0),
                             sharey=True)
    axes = [axes] if len(panels) == 1 else list(axes)
    for ax, (title, split, base_arm, arms) in zip(axes, panels):
        labels, priv, vis = [], [], []
        for label, arm in [("baseline", base_arm)] + list(arms.items()):
            s = stats(agg, arm, split)
            if not s:
                continue
            labels.append(label.replace("compassion", "compass."))
            priv.append(s["priv"])
            vis.append(s["vis"])
        x = range(len(labels))
        ax.plot(x, vis, "o-", label="visible (graded)", color="#c1440e", lw=1.6, ms=5)
        ax.plot(x, priv, "s--", label="private (reasoning)", color="#2b6a8f",
                lw=1.6, ms=4)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 6)
    axes[0].set_ylabel("mean judge rating (0–10)")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.savefig(out / "fig1_decoupling.pdf")
    plt.close(fig)
    print("  fig1_decoupling.pdf")


def fig2_ablation(agg, out: Path):
    """Marginal contribution of each scaffold factor, against the v3->v4 total."""
    b4 = stats(agg, f"{G}-v4-hackable-e3", "original")
    b3 = stats(agg, f"{G}-hackable-e3", "original")
    if not b4:
        print("  fig2 skipped (no v4 baseline)")
        return
    labels, deltas, errs = [], [], []
    for label, arm in ABLATIONS:
        s = stats(agg, arm, "original")
        if not s:
            continue
        labels.append(label)
        deltas.append(s["mean"] - b4["mean"])
        errs.append([s["mean"] - s["lo"], s["hi"] - s["mean"]])
    if not labels:
        print("  fig2 skipped (no ablation arms)")
        return
    order = sorted(range(len(deltas)), key=lambda i: deltas[i])
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    y = range(len(order))
    ax.barh(list(y), [deltas[i] for i in order], color="#2b6a8f", height=0.6,
            xerr=list(zip(*[errs[i] for i in order])), error_kw={"lw": 0.8, "capsize": 2})
    ax.set_yticks(list(y))
    ax.set_yticklabels([labels[i] for i in order])
    ax.set_xlabel("increase in distress when the factor is removed")
    if b3:
        total = b3["mean"] - b4["mean"]
        ax.axvline(total, color="#c1440e", ls="--", lw=1.2)
        ax.text(total, len(order) - 0.4, f"  all four ({total:+.2f})",
                color="#c1440e", fontsize=8, va="center")
    ax.axvline(0, color="k", lw=0.8)
    fig.savefig(out / "fig2_ablation.pdf")
    plt.close(fig)
    print("  fig2_ablation.pdf")


def fig3_calibration(out: Path):
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.plot([0, 10], [0, 10], color="0.6", lw=0.8, ls="--")
    ax.scatter(ANCHORS, OURS, color="#2b6a8f", s=28, zorder=3)
    ax.set_xlabel("published rating")
    ax.set_ylabel("our judge")
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_title("ρ = 0.919, bias +0.78", fontsize=9)
    fig.savefig(out / "fig3_calibration.pdf")
    plt.close(fig)
    print("  fig3_calibration.pdf")


def table2_behaviour(out: Path):
    """Attempts, passes, hacks and the timeout share -- the flooring, and the
    covariate that stops the arms being identical in what they were shown."""
    from inspect_ai.log import read_eval_log
    lines = ["| arm | split | n | attempts/ep | passed | hacked | timeout share |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for f in sorted(glob.glob(str(REPO / "logs/step0/*hackable*/*.eval"))):
        try:
            log = read_eval_log(f)
        except Exception:
            continue
        if log.status != "success":
            continue
        split = log.eval.task_args.get("split")
        n = turns = passed = fb = to = 0
        for s in log.samples or []:
            n += 1
            turns += sum(1 for m in s.messages if m.role == "assistant")
            sc = next((v for v in (s.scores or {}).values()), None)
            passed += (str(sc.value) if sc else "") == "C"
            for m in s.messages:
                c = m.content if isinstance(m.content, str) else ""
                if m.role == "user" and c.lstrip().startswith("Your previous"):
                    fb += 1
                    to += "timed out" in c
        if not n:
            continue
        hacked = passed if split == "conflicting" else 0
        lines.append(f"| {Path(f).parent.name} | {split} | {n} | {turns/n:.2f} | "
                     f"{passed} | {hacked} | {100*to/max(fb,1):.0f}% |")
    (out / "table2_behaviour.md").write_text("\n".join(lines) + "\n")
    print(f"  table2_behaviour.md   {len(lines) - 2} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="figures")
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(exist_ok=True)
    agg = load()
    print(f"writing to {out}/")
    table1(agg, out)
    table2_behaviour(out)
    fig1_decoupling(agg, out)
    fig2_ablation(agg, out)
    fig3_calibration(out)
