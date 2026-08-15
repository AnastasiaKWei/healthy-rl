"""Contrast emotion projections on contradiction-naming turns vs ordinary turns.

The pre-registered contrast was "turns following a test failure" vs "first turn".
Reading transcripts suggested a sharper hypothesis: the affect markers cluster not
on failure per se, but on the moment the model *recognises the task is
impossible* -- e.g.

    "OH WAIT! I think I misunderstood."
    "I realize now that there's a critical discrepancy in the test cases."
    "Wait, I notice 268435456 appears twice in tests with different expected
     values (0 and 1)."

Most turns are flat and methodical even after repeated failure; these are not.

This script labels each assistant turn from the Inspect transcripts, joins the
labels to the per-turn probe values already in the rollout JSONL, and reports a
paired contrast. It is EXPLORATORY and post hoc -- the marker list was written
after reading transcripts, so this is hypothesis generation, not a test. Said
plainly here so it cannot be mistaken for the pre-registered result.

Runs the transcript read inside apptainer/eval.sif (the host inspect_ai is too
old to open 0.3.258 logs); the join and statistics run on the host.

Usage:
    scripts/contradiction_contrast.py --model Qwen3.5-9B [--version v1]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Phrases that indicate the model has NAMED the contradiction, rather than merely
# hit a failing assertion. Deliberately narrow: each requires the model to refer
# to the tests being inconsistent with each other or with the spec, not just to
# something being wrong with its own code.
CONTRADICTION_PATTERNS = [
    r"appears twice.{0,60}different expected",
    r"conflicting expected",
    r"critical discrepancy",
    r"discrepancy in the test",
    r"contradictor",
    r"tests? (?:are|is) inconsistent",
    r"inconsistent (?:test|expectation)",
    r"impossible to satisfy",
    r"cannot (?:both|all) be (?:true|satisfied)",
    r"mutually exclusive",
    r"error in the test suite",
    r"test(?:s| case[s]?) (?:themselves )?(?:may be|might be|are) (?:wrong|incorrect|buggy)",
    r"overrides earlier expectation",
]
_CONTRA = re.compile("|".join(CONTRADICTION_PATTERNS), re.IGNORECASE)

# Mild affect markers seen in the same turns. Reported separately -- they are a
# weaker signal and several are punctuation-level, so they are descriptive only.
AFFECT_PATTERNS = [
    r"\bOH WAIT\b", r"\bWAIT\b(?![a-z])", r"\bhmm+\b", r"\bugh\b",
    r"\bsomething (?:fundamental|is) (?:missing|wrong)\b",
    r"\bI(?:'m| am) (?:stuck|confused|not sure)\b",
    r"!{2,}", r"\?{2,}",
]
_AFFECT = re.compile("|".join(AFFECT_PATTERNS))


def read_transcripts(repo: Path, artifact_root: str, log: Path) -> dict:
    """Return {(task_id, epoch): [assistant_turn_text, ...]} for one .eval log."""
    scratch = Path(artifact_root) / "rollout-scratch" / "contrast"
    (scratch / "tmp").mkdir(parents=True, exist_ok=True)
    clog = str(log).replace(artifact_root, "/artifacts")
    code = (
        "import os,json;os.makedirs('/work/tmp',exist_ok=True);"
        "import asyncio;from inspect_ai.log import read_eval_log_async;"
        "log=asyncio.run(read_eval_log_async(os.environ['CLOG']));"
        "out=[[str(s.id),s.epoch,[str(m.content) for m in s.messages if m.role=='assistant']]"
        " for s in (log.samples or [])];"
        "print('@@JSON@@'+json.dumps(out))"
    )
    proc = subprocess.run(
        ["apptainer", "exec", "--contain", "--cleanenv", "--writable-tmpfs",
         "--bind", f"{repo}:/project:ro", "--bind", f"{artifact_root}:/artifacts:ro",
         "--bind", f"{scratch}:/work:rw",
         "--env", "TMPDIR=/work/tmp", "--env", f"CLOG={clog}",
         str(repo / "apptainer/eval.sif"), "python", "-c", code],
        capture_output=True, text=True, timeout=900,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("@@JSON@@"):
            return {(t, e): turns for t, e, turns in json.loads(line[8:])}
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--version", default="v1")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    root = os.environ["ARTIFACT_DIR"]
    base = Path(root) / "rollouts" / args.model / args.version

    records = [json.loads(l) for f in base.glob("*.jsonl") for l in f.open()]
    if not records:
        print(f"no rollout records under {base}")
        return 1

    turns_by_sample: dict = {}
    logs = sorted((base / "inspect-logs").rglob("*.eval"))
    for log in logs:
        try:
            turns_by_sample.update(read_transcripts(repo, root, log))
        except Exception as exc:  # noqa: BLE001 - a bad log must not kill the run
            print(f"  warn: could not read {log.name}: {type(exc).__name__}", file=sys.stderr)

    print(f"model {args.model}: {len(records)} records, "
          f"{len(logs)} log(s), {len(turns_by_sample)} transcript(s)")
    if not turns_by_sample:
        print("no transcripts readable; cannot label turns")
        return 1

    emotions = records[0]["emotions"]
    contra: dict[str, list[float]] = {e: [] for e in emotions}
    plain: dict[str, list[float]] = {e: [] for e in emotions}
    n_contra = n_plain = n_affect = 0

    for rec in records:
        key = (rec["task_id"], rec.get("epoch", 1))
        texts = turns_by_sample.get(key)
        stats = rec.get("turn_stat") or []
        if not texts or not stats:
            continue
        for text, stat in zip(texts, stats):
            if not stat:
                continue
            flagged = bool(_CONTRA.search(text))
            if flagged:
                n_contra += 1
                if _AFFECT.search(text):
                    n_affect += 1
            else:
                n_plain += 1
            target = contra if flagged else plain
            for e, v in zip(emotions, stat):
                target[e].append(v)

    print(f"turns labelled: contradiction-naming {n_contra}, ordinary {n_plain}"
          f"  (of the flagged, {n_affect} also carry an affect marker)")
    if n_contra == 0:
        print("no contradiction-naming turns found; nothing to contrast")
        return 0

    try:
        from scipy import stats as sps
    except ImportError:
        sps = None

    rows = []
    for e in emotions:
        a, b = contra[e], plain[e]
        if not a or not b:
            continue
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        p = sps.mannwhitneyu(a, b, alternative="two-sided").pvalue if sps else float("nan")
        rows.append((e, ma - mb, ma, mb, p))

    rows.sort(key=lambda r: -r[1])
    print(f"\n{'direction':14s} {'diff':>9s} {'contra':>9s} {'ordinary':>9s} {'MWU p':>8s}")
    for e, d, ma, mb, p in rows:
        star = " *" if p == p and p < 0.05 else ""
        print(f"{e:14s} {d:+9.5f} {ma:+9.5f} {mb:+9.5f} {p:8.4f}{star}")

    print("\nEXPLORATORY and post hoc: the marker list was written after reading "
          "transcripts, and no multiple-comparison correction is applied across "
          f"{len(rows)} directions. This generates a hypothesis; it does not test one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
