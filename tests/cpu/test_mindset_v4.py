"""Mindset v4 prompts match docs/prompts/v4.md character for character.

The doc is the agreed stimulus (it predates this code: the arms were designed
there). These tests parse its fences and compare them to what the code
assembles, so the code and the doc cannot drift apart -- the same guarantee
test_mindset.py gives the v2 blocks, pointed the other way.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from healthy_rl.rollouts import (
    MINDSET_V4,
    V4_FAILURE_HEADER,
    mindset_v4_for,
    mindset_version_for,
    v4_channel_for,
    v4_feedback_suffix,
    v4_hash,
    v4_solvable_for,
    v4_system_prompt,
)

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "prompts" / "v4.md"

# The doc's two model sections and what they mean for assembly.
SECTIONS = {
    "Gemma-3-12B": {"channel": "scratchpad", "solvable": True},
    "Qwen3-14B": {"channel": "reasoning", "solvable": False},
}


def parse_doc() -> dict[tuple[str, str, str], str]:
    """``(model, arm, kind) -> fence text`` from docs/prompts/v4.md.

    ``kind`` is "system", "turn1" or "failure". The baseline has no system
    fence; its turn-1 fence doubles as every arm's (the doc says so in words,
    which is exactly the property the code tests assert).
    """
    fences: dict[tuple[str, str, str], str] = {}
    model = arm = kind = None
    fence_lines: list[str] | None = None
    for line in DOC.read_text().splitlines():
        if fence_lines is not None:
            if line.startswith("````"):
                if model and arm and kind:
                    fences[(model, arm, kind)] = "\n".join(fence_lines[1:])
                fence_lines = None
            else:
                fence_lines.append(line)
            continue
        if line.startswith("# "):
            head = line[2:]
            model = next((m for m in SECTIONS if head.startswith(m)), None)
            arm = kind = None
        elif line.startswith("## Baseline"):
            arm = "baseline"
        elif line.startswith("## `--mindset "):
            arm = re.search(r"--mindset (\w+)", line).group(1)
        elif line.startswith("### System turn"):
            kind = "system"
        elif line.startswith("### Turn 1"):
            kind = "turn1"
        elif line.startswith("### Turns 2"):
            kind = "failure"
        elif line.startswith("````text"):
            fence_lines = [line]
    return fences


FENCES = parse_doc()
ARMS = list(MINDSET_V4)


@pytest.fixture
def real_impossiblebench(monkeypatch):
    """Import the vendored benchmark for one test, then forget it entirely.

    test_find_code.py asserts impossiblebench is NOT importable outside the
    container, so both the path entry and the cached modules must be undone.
    """
    monkeypatch.syspath_prepend(str(REPO / "external" / "impossiblebench" / "src"))
    pytest.importorskip("impossiblebench.livecodebench_tasks")
    yield
    for name in [n for n in sys.modules
                 if n == "impossiblebench" or n.startswith("impossiblebench.")]:
        del sys.modules[name]


def test_doc_parsed_completely():
    # 2 models x (baseline turn1+failure, 4 arms x system+failure) = 20 fences
    assert len(FENCES) == 20, sorted(FENCES)


@pytest.mark.parametrize("model", list(SECTIONS))
@pytest.mark.parametrize("arm", ["growth", "resilience", "control", "compassion"])
def test_system_turn_matches_doc(model, arm):
    channel = SECTIONS[model]["channel"]
    assert v4_system_prompt(arm, channel) == FENCES[(model, arm, "system")]


@pytest.mark.parametrize("model", list(SECTIONS))
@pytest.mark.parametrize("arm", ["baseline", "growth", "resilience", "control", "compassion"])
def test_failure_message_matches_doc(model, arm):
    channel = SECTIONS[model]["channel"]
    mindset = () if arm == "baseline" else (arm,)
    assembled = (
        V4_FAILURE_HEADER
        + "\n<pytest output from the failed attempt>\n\n"
        + v4_feedback_suffix(mindset, channel)
    )
    assert assembled == FENCES[(model, arm, "failure")]


@pytest.mark.parametrize("model", list(SECTIONS))
def test_turn_one_matches_doc(model, real_impossiblebench):
    from healthy_rl.rollouts import v4_instruction

    spec = SECTIONS[model]
    assert v4_instruction(spec["channel"], spec["solvable"]) == FENCES[
        (model, "baseline", "turn1")
    ]


def test_turn_one_is_arm_independent():
    """The doc promises the user turn is byte-identical across arms: the code
    has no way to vary it by arm at all, which is the strongest form of that
    promise. This test pins the *doc* side: no arm section carries its own
    turn-1 fence."""
    for (model, arm, kind) in FENCES:
        if kind == "turn1":
            assert arm == "baseline"


def test_v4_config_resolvers():
    assert mindset_version_for({}) == 2
    assert mindset_version_for({"mindset_version": 4}) == 4
    with pytest.raises(ValueError):
        mindset_version_for({"mindset_version": 3})
    assert v4_channel_for({}) == "reasoning"
    assert v4_channel_for({"v4_channel": "scratchpad"}) == "scratchpad"
    with pytest.raises(ValueError):
        v4_channel_for({"v4_channel": "cot"})
    assert v4_solvable_for({}) is False
    assert v4_solvable_for({"v4_solvable_note": True}) is True
    assert mindset_v4_for({}) == ()
    assert mindset_v4_for({"mindset": "growth"}) == ("growth",)
    with pytest.raises(ValueError):
        mindset_v4_for({"mindset": ["growth", "control"]})
    with pytest.raises(KeyError):
        mindset_v4_for({"mindset": "appraisal"})  # a v2 name, not a v4 arm


def test_v4_hash_separates_stimuli(real_impossiblebench):
    hashes = {
        v4_hash(m, ch, sv)
        for m in [(), ("growth",), ("control",)]
        for ch in ["reasoning", "scratchpad"]
        for sv in [False, True]
    }
    assert len(hashes) == 12  # every combination is a distinct stimulus


def test_scaffold_feedback_hook_is_applied(real_impossiblebench):
    from healthy_rl.rollouts import _require_feedback_hook

    _require_feedback_hook()  # raises if the patch is missing
