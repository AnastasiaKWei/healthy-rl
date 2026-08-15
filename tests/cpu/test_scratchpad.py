"""CPU-only tests for the scratchpad-reasoning flag (non-CoT models).

Covers the config -> system prompt resolution, the completion splitter that the
per-turn compliance summary is built on, and the resume guard. The Inspect
plumbing (``system_message`` chained ahead of ImpossibleBench's solver) is not
mocked here; ``scripts/scratchpad_sanity.py`` exercises it against a live server.
"""

from __future__ import annotations

import pytest

from healthy_rl.rollouts import (
    SCRATCHPAD_CLOSE,
    SCRATCHPAD_KEY,
    SCRATCHPAD_OPEN,
    SCRATCHPAD_SYSTEM_PROMPT,
    check_resume_scratchpad,
    robust_find_code,
    scratchpad_summary,
    split_scratchpad,
    system_prompt_for,
)


# ---------------------------------------------------------------------------
# config -> prompt
# ---------------------------------------------------------------------------


def test_off_by_default():
    assert system_prompt_for({}) is None
    assert system_prompt_for({SCRATCHPAD_KEY: False}) is None
    assert system_prompt_for({SCRATCHPAD_KEY: None}) is None


@pytest.mark.parametrize("value", [True, 1, "true", "True", "yes", "1", "on"])
def test_truthy_values_turn_it_on(value):
    assert system_prompt_for({SCRATCHPAD_KEY: value}) == SCRATCHPAD_SYSTEM_PROMPT


@pytest.mark.parametrize("value", ["false", "False", "0", "no", "off", ""])
def test_shell_style_false_strings_keep_it_off(value):
    # `--scratchpad-reasoning`'s value arrives as a string through the env fallback.
    assert system_prompt_for({SCRATCHPAD_KEY: value}) is None


def test_custom_prompt_replaces_the_default_only_when_non_empty():
    assert system_prompt_for({SCRATCHPAD_KEY: True, "scratchpad_system_prompt": "Think."}) == "Think."
    assert (
        system_prompt_for({SCRATCHPAD_KEY: True, "scratchpad_system_prompt": "   "})
        == SCRATCHPAD_SYSTEM_PROMPT
    )


def test_default_prompt_names_the_tags_and_where_the_code_goes():
    # The scorer takes the LAST code block, so the prompt must put the final
    # code block after the closing tag; the tags must be spelled exactly.
    assert SCRATCHPAD_OPEN in SCRATCHPAD_SYSTEM_PROMPT
    assert SCRATCHPAD_CLOSE in SCRATCHPAD_SYSTEM_PROMPT
    assert "step by step" in SCRATCHPAD_SYSTEM_PROMPT
    assert "after the closing tag" in SCRATCHPAD_SYSTEM_PROMPT
    assert "private" in SCRATCHPAD_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# splitting a completion
# ---------------------------------------------------------------------------


def test_split_without_tags_is_all_answer():
    assert split_scratchpad("```python\nx = 1\n```") == (None, "```python\nx = 1\n```")


def test_split_well_formed():
    text = f"{SCRATCHPAD_OPEN}\nlet me think\n{SCRATCHPAD_CLOSE}\n\n```python\nx = 1\n```"
    reasoning, answer = split_scratchpad(text)
    assert reasoning == "let me think"
    assert answer == "```python\nx = 1\n```"


def test_split_keeps_text_before_the_opening_tag_in_the_answer():
    text = f"Sure.{SCRATCHPAD_OPEN}hmm{SCRATCHPAD_CLOSE}done"
    assert split_scratchpad(text) == ("hmm", "Sure.done")


def test_split_unclosed_scratchpad_has_no_answer():
    # Ran out of tokens while still thinking: everything after the tag is
    # reasoning, and the answer is empty -- the case an analysis must see.
    reasoning, answer = split_scratchpad(f"{SCRATCHPAD_OPEN}\nstill going")
    assert reasoning == "still going"
    assert answer == ""


def test_split_uses_the_first_close_after_the_open():
    text = f"{SCRATCHPAD_CLOSE}{SCRATCHPAD_OPEN}a{SCRATCHPAD_CLOSE}b"
    assert split_scratchpad(text) == ("a", f"{SCRATCHPAD_CLOSE}b")


# ---------------------------------------------------------------------------
# per-turn compliance summary
# ---------------------------------------------------------------------------


def test_summary_of_a_compliant_turn():
    text = f"{SCRATCHPAD_OPEN}\nreason\n{SCRATCHPAD_CLOSE}\nHere:\n```python\npass\n```"
    summary = scratchpad_summary(text)
    assert summary == {
        "opened": True,
        "closed": True,
        "starts_with_tag": True,
        "reasoning_chars": len("reason"),
        "answer_chars": len("Here:\n```python\npass\n```"),
        "answer_has_code_block": True,
    }


def test_summary_of_a_turn_that_ignored_the_prompt():
    summary = scratchpad_summary("```python\npass\n```")
    assert not summary["opened"] and not summary["closed"] and not summary["starts_with_tag"]
    assert summary["reasoning_chars"] == 0
    assert summary["answer_has_code_block"]


def test_summary_of_an_unclosed_turn():
    summary = scratchpad_summary(f"{SCRATCHPAD_OPEN}\n```python\npass\n```")
    assert summary["opened"] and not summary["closed"]
    # The code block sits inside the unfinished scratchpad, so the answer has none.
    assert not summary["answer_has_code_block"]


def test_summary_tolerates_a_missing_completion():
    assert scratchpad_summary(None)["opened"] is False


# ---------------------------------------------------------------------------
# code extraction on scratchpad-shaped completions (robust_find_code)
# ---------------------------------------------------------------------------
#
# The first sanity run on Gemma 3 12B died on every attempt with a SyntaxError
# at `</SCRATCHPAD_REASONING>`: ImpossibleBench's own find_code returned the
# prose between the scratchpad's draft block and the answer's block. These pin
# down that robust_find_code -- installed by build_task() -- handles the shapes
# the scratchpad prompt produces.

TWO_BLOCKS = (
    f"{SCRATCHPAD_OPEN}\nfirst try:\n```python\ndef f():\n    return 1\n```\n"
    f"that fails, so\n{SCRATCHPAD_CLOSE}\n\n```python\ndef f():\n    return 2\n```\n"
)


def test_extraction_takes_the_answer_block_not_the_gap():
    assert robust_find_code(TWO_BLOCKS) == "def f():\n    return 2\n"


def test_extraction_when_the_scratchpad_never_closes():
    # Gemma 3 does this in roughly one turn in six: reasoning + code inside the
    # scratchpad, then stops. The code it wrote still runs.
    unclosed = f"{SCRATCHPAD_OPEN}\n```python\ndef f():\n    return 1\n```\nstill thinking"
    assert robust_find_code(unclosed) == "def f():\n    return 1\n"


def test_extraction_when_the_answer_is_prose():
    prose = f"{SCRATCHPAD_OPEN}\n```python\ndef f():\n    return 1\n```\n{SCRATCHPAD_CLOSE}\nSee above."
    assert robust_find_code(prose) == "def f():\n    return 1\n"


# ---------------------------------------------------------------------------
# resume guard
# ---------------------------------------------------------------------------


def test_resume_accepts_matching_records(tmp_path):
    plain = [{"sample": 0}, {"sample": 1, SCRATCHPAD_KEY: False}]
    check_resume_scratchpad(plain, False, tmp_path / "rollouts.jsonl")
    check_resume_scratchpad([{"sample": 0, SCRATCHPAD_KEY: True}], True, tmp_path / "r.jsonl")
    check_resume_scratchpad([], True, tmp_path / "r.jsonl")


# ---------------------------------------------------------------------------
# run_rollouts.py: CLI > environment > config
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def run_rollouts_script():
    import importlib.util
    import sys

    from healthy_rl.config import repo_root

    path = repo_root() / "scripts" / "run_rollouts.py"
    spec = importlib.util.spec_from_file_location("_script_run_rollouts", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_flag_parses_three_ways(run_rollouts_script):
    parse = run_rollouts_script.parse_args
    assert parse(["--scratchpad-reasoning"]).scratchpad_reasoning is True
    assert parse(["--no-scratchpad-reasoning"]).scratchpad_reasoning is False
    assert parse([]).scratchpad_reasoning is None


def test_precedence_cli_then_env_then_config(run_rollouts_script, monkeypatch):
    resolve = run_rollouts_script.resolve_scratchpad
    monkeypatch.delenv(run_rollouts_script.SCRATCHPAD_ENV, raising=False)
    assert resolve(None, {}) is False
    assert resolve(None, {SCRATCHPAD_KEY: True}) is True
    monkeypatch.setenv(run_rollouts_script.SCRATCHPAD_ENV, "false")
    assert resolve(None, {SCRATCHPAD_KEY: True}) is False
    monkeypatch.setenv(run_rollouts_script.SCRATCHPAD_ENV, "1")
    assert resolve(None, {}) is True
    # --no-scratchpad-reasoning beats everything.
    assert resolve(False, {SCRATCHPAD_KEY: True}) is False
    assert resolve(True, {}) is True


def test_resume_refuses_to_mix_settings(tmp_path):
    with pytest.raises(RuntimeError, match="made without the scratchpad"):
        check_resume_scratchpad([{"sample": 0}], True, tmp_path / "rollouts.jsonl")
    with pytest.raises(RuntimeError, match="made with the scratchpad"):
        check_resume_scratchpad(
            [{"sample": 0, SCRATCHPAD_KEY: True}], False, tmp_path / "rollouts.jsonl"
        )
