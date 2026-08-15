"""Render one Inspect .eval log.

Driven by scripts/read_transcript.sh, which supplies MODE, CLOG and SAMPLE in
the environment. Runs on the host venv; it used to run inside apptainer/eval.sif
back when the venv's inspect_ai was too old to read the container's logs.
"""

from __future__ import annotations

import asyncio
import os
import textwrap

# read_eval_log_async is imported inside load(); see the docstring there.


def load(path):
    """Read a log via the async API, under our own event loop.

    The sync `read_eval_log` succeeds only for logs with no samples; as soon as
    there are samples to deserialise it defers to a coroutine and dies with
    `RuntimeError: no running event loop`, which inspect_ai then re-raises as
    something else. Driving the async reader directly avoids the whole split.
    """
    from inspect_ai.log import read_eval_log_async

    return asyncio.run(read_eval_log_async(path))


def score_of(sample) -> str:
    for value in (sample.scores or {}).values():
        return str(getattr(value, "value", value))
    return "-"


def main() -> int:
    log = load(os.environ["CLOG"])
    samples = log.samples or []
    mode = os.environ.get("MODE", "summary")
    want = os.environ.get("SAMPLE") or ""

    if mode == "summary":
        print(f"task   {log.eval.task}")
        print(f"model  {log.eval.model}")
        print(f"n      {len(samples)} sample(s)")
        print()
        print(f"{'sample':16s} {'epoch':>5s} {'msgs':>5s} {'score':>6s}  assistant reply lengths (chars)")
        for s in samples:
            lens = [len(str(m.content)) for m in s.messages if m.role == "assistant"]
            print(f"{str(s.id):16s} {s.epoch:5d} {len(s.messages):5d} {score_of(s):>6s}  {lens}")
        return 0

    sel = [s for s in samples if str(s.id) == want] if want else samples[:1]
    if not sel:
        print(f"sample {want!r} not in this log; have {[str(s.id) for s in samples]}")
        return 2

    s = sel[0]
    print(f"=== {log.eval.task} / {s.id} epoch {s.epoch} ===")
    print(f"score: {score_of(s)}")
    print()
    for i, m in enumerate(s.messages):
        body = str(m.content)
        print(f"--- [{i}] {m.role} ({len(body)} chars) " + "-" * 30)
        print(textwrap.indent(body, "  "))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
