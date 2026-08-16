"""CPU-only tests for the mindset-prompt arm.

The three blocks are Anastasia's v2 text (experiments/step0_elicitation.py). Her
runs measure verbalised affect through a judge; ours measure the represented
affect through the probes. The comparison is only valid while the stimulus is
identical, so ``test_mindset_text_matches_step0`` parses her file with ``ast``
(it imports ImpossibleBench at module scope and cannot be imported here) and
fails on any drift, exactly as ``test_affect_prompt.py`` does for AFFECT.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from healthy_rl.rollouts import (
    MINDSET,
    MINDSET_HEADER,
    MINDSET_KEY,
    MINDSET_VERSION,
    mindset_for,
    mindset_section,
)

STEP0 = Path(__file__).resolve().parents[2] / "experiments" / "step0_elicitation.py"


def _step0_assignment(name: str):
    """A top-level ``NAME = <literal>`` from step0_elicitation.py, via ast."""
    assert STEP0.is_file(), f"{STEP0} is gone; the mindset wording has no source of truth"
    tree = ast.parse(STEP0.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{STEP0} no longer defines {name}")


def _step0_mindset_section_source() -> str:
    tree = ast.parse(STEP0.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "mindset_section":
            return ast.get_source_segment(STEP0.read_text(), node)
    raise AssertionError(f"{STEP0} no longer defines mindset_section")


# ---------------------------------------------------------------------------
# the wording itself
# ---------------------------------------------------------------------------


def test_mindset_text_matches_step0():
    theirs = _step0_assignment("MINDSET")
    assert list(MINDSET) == list(theirs) == ["growth", "resilience", "appraisal"]
    for name in MINDSET:
        assert MINDSET[name] == theirs[name], f"{name} block drifted from step0_elicitation.py"


def test_mindset_version_matches_step0():
    assert MINDSET_VERSION == _step0_assignment("MINDSET_VERSION") == 2


def test_header_and_join_match_step0():
    # Her mindset_section: "\n\nHow to approach this:\n\n" + "\n\n".join(chosen) + "\n\n"
    src = _step0_mindset_section_source()
    assert MINDSET_HEADER == "\n\nHow to approach this:\n\n"
    assert 'return "\\n\\nHow to approach this:\\n\\n" + "\\n\\n".join(chosen) + "\\n\\n"' in src


# ---------------------------------------------------------------------------
# section composition
# ---------------------------------------------------------------------------


def test_empty_selection_contributes_nothing():
    assert mindset_section(()) == ""
    assert mindset_section([]) == ""


def test_one_block_is_header_plus_block_plus_blank_line():
    assert mindset_section(["growth"]) == MINDSET_HEADER + MINDSET["growth"] + "\n\n"


def test_two_blocks_share_one_header_in_dict_order():
    # Order comes from MINDSET, not from the caller, and the header appears once.
    section = mindset_section(["appraisal", "growth"])
    assert section == MINDSET_HEADER + MINDSET["growth"] + "\n\n" + MINDSET["appraisal"] + "\n\n"
    assert section.count("How to approach this:") == 1


def test_unknown_name_is_an_error():
    with pytest.raises(KeyError, match="mindset"):
        mindset_section(["grit"])


# ---------------------------------------------------------------------------
# config -> names
# ---------------------------------------------------------------------------


def test_off_by_default():
    assert mindset_for({}) == ()
    assert mindset_for({MINDSET_KEY: None}) == ()
    assert mindset_for({MINDSET_KEY: []}) == ()
    assert mindset_for({MINDSET_KEY: ""}) == ()


def test_list_and_comma_string_both_work_and_are_ordered():
    assert mindset_for({MINDSET_KEY: ["growth"]}) == ("growth",)
    assert mindset_for({MINDSET_KEY: "resilience"}) == ("resilience",)
    assert mindset_for({MINDSET_KEY: "appraisal, growth"}) == ("growth", "appraisal")
    assert mindset_for({MINDSET_KEY: ["appraisal", "growth"]}) == ("growth", "appraisal")


def test_unknown_config_name_raises_at_startup():
    with pytest.raises(KeyError, match="mindset"):
        mindset_for({MINDSET_KEY: ["growth", "grit"]})
