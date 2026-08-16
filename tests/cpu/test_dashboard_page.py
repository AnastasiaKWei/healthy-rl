"""The page is one self-contained file: no external URLs, both themes defined, every API route referenced."""
import re
from pathlib import Path

PAGE = Path("src/healthy_rl/dashboard/static/index.html").read_text()


def test_self_contained():
    assert not re.search(r'(src|href)="https?://', PAGE), "no CDN/external assets"
    assert "<!doctype html>" in PAGE.lower() and "<title>Affect Scope</title>" in PAGE


def test_theme_tokens_defined_for_all_three_states():
    assert ":root{" in PAGE.replace(" ", "") and 'prefers-color-scheme: dark' in PAGE and ':root[data-theme="dark"]' in PAGE


def test_every_route_is_referenced():
    for route in ["/api/session", "/api/conversations", "/api/chat/", "/api/task/start", "/continue", "/stop", "/tokens", "/api/aggregate", "/api/problems", "/api/health"]:
        assert route in PAGE, route


def test_javascript_parses(tmp_path):
    import shutil, subprocess
    node = shutil.which("node")
    if not node:
        import pytest; pytest.skip("node not on PATH")
    scripts = re.findall(r"<script>(.*?)</script>", PAGE, flags=re.S)
    assert scripts
    js = tmp_path / "page.js"; js.write_text("\n".join(scripts))
    subprocess.run([node, "--check", str(js)], check=True)


def test_rollouts_mode_strings_present():
    for s in ["S.rollouts", "applyModelLayers", "no per-token arrays", "session.models", "railFilter",
              "renderAggPicker", "aggDrawGroups", "--g1", "with base"]:
        assert s in PAGE, s


def test_turn_body_separates_an_empty_turn_from_an_old_record():
    """A zero-token turn in a new cell has no arrays either, so the "written before the mindset
    merge" note must not claim it, and its "no tokenizer" error is not a generation failure."""
    body = PAGE.split("function renderTurnBody(")[1].split("\nasync function")[0]
    assert "This turn generated nothing" in body
    assert body.index("This turn generated nothing") < body.index("Generation failed")
    assert body.index("This turn generated nothing") < body.index("written before the mindset merge")
    # the old-record note is gated on the turn having generated something
    assert "turn.has_token_arrays === false && !turn.misaligned && !empty" in body


def test_turn_header_reports_recovered_text_and_a_mismatched_emotion_order():
    head = PAGE.split("function assistantMessage(")[1].split("function renderTurnBody(")[0]
    assert 'turn.text_source === "eval"' in head and "text from .eval" in head
    assert "turn.emotion_order_mismatch" in head and "emotion order differs" in head
