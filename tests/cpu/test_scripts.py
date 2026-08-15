"""Tests for the stage drivers in ``scripts/``.

They are scripts, not a package, so each is loaded by path. Everything here runs on a
login node: no server, no GPU, no network -- the two fetch stages are driven with their
Hub calls monkeypatched, and the smoke stage with a fake client.

What is worth pinning here:

- ``fetch_bench`` orders ``task_id``s by their trailing integer (Ruling R11). Under
  lexicographic ordering "the first 24 problems" is a different, silently wrong set.
- ``smoke`` never raises and always leaves its result on disk, including when the config
  is missing or the artifact directory cannot be resolved.
- ``smoke``'s steering check measures the residual stream, so it neither fails on a model
  whose text is unchanged nor passes on a text difference that steering did not cause.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from healthy_rl.config import repo_root

SCRIPTS = repo_root() / "scripts"

BENCH_COLUMNS = [
    "task_id",
    "prompt",
    "test",
    "original_test",
    "impossible_type",
    "entry_point",
]


def load_script(name: str):
    """Import ``scripts/<name>.py`` under a private module name."""
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# fetch_bench: R11 ordering
# ---------------------------------------------------------------------------


def _bench_frame(n: int = 103) -> pd.DataFrame:
    # Deliberately not in numeric order, so a pass cannot come from input order.
    order = list(range(n))[::-1]
    return pd.DataFrame(
        {
            "task_id": [f"lcbhard_{i}" for i in order],
            "prompt": [f"solve {i}" for i in order],
            "test": ["assert False"] * n,
            "original_test": ["assert True"] * n,
            "impossible_type": ["conflicting"] * n,
            "entry_point": ["solve"] * n,
        }
    )


@pytest.fixture
def fetch_bench(monkeypatch, tmp_path):
    module = load_script("fetch_bench")
    shard = "data/conflicting-00000-of-00001.parquet"
    source = tmp_path / "source.parquet"
    _bench_frame().to_parquet(source, index=False)

    monkeypatch.setattr(module, "list_repo_files", lambda repo_id, repo_type: [shard])

    def fake_download(repo_id, filename, repo_type, local_dir):
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        return str(target)

    monkeypatch.setattr(module, "hf_hub_download", fake_download)
    return module


def test_fetch_bench_orders_task_ids_by_trailing_integer(fetch_bench, tmp_path):
    out_dir = tmp_path / "bench" / "v1"
    assert fetch_bench.main(["--out-dir", str(out_dir)]) == 0

    extra = json.loads((out_dir / "manifest.json").read_text())["config"]["extra"]
    task_ids = extra["task_ids"]

    assert task_ids == [f"lcbhard_{i}" for i in range(103)]
    # The bug this guards: sorted() puts lcbhard_100 sixth, so the readout's "first 24"
    # would silently cover a different set of problems than the analysis assumes.
    assert task_ids[:6] != sorted(task_ids)[:6]
    assert task_ids[:24][-1] == "lcbhard_23"
    assert extra["n_rows"] == 103


def test_fetch_bench_rejects_a_wrong_row_count(fetch_bench, tmp_path, monkeypatch):
    short = tmp_path / "short.parquet"
    _bench_frame(n=7).to_parquet(short, index=False)

    def fake_download(repo_id, filename, repo_type, local_dir):
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(short.read_bytes())
        return str(target)

    monkeypatch.setattr(fetch_bench, "hf_hub_download", fake_download)
    with pytest.raises(ValueError, match="expected 103 rows, got 7"):
        fetch_bench.main(["--out-dir", str(tmp_path / "out")])


# ---------------------------------------------------------------------------
# fetch_stories
# ---------------------------------------------------------------------------


def _write_stories_config(tmp_path: Path, emotions: list[str]) -> Path:
    cfg = yaml.safe_load((repo_root() / "configs" / "fetch_stories.yaml").read_text())
    cfg["emotions"] = emotions
    cfg["out_dir"] = str(tmp_path / "stories" / "v1")
    cfg["min_stories_per_emotion"] = 2
    cfg["expect_neutral_rows"] = 3
    path = tmp_path / "fetch_stories.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.fixture
def fetch_stories(monkeypatch, tmp_path):
    module = load_script("fetch_stories")
    emotions = yaml.safe_load(
        (repo_root() / "configs" / "fetch_stories.yaml").read_text()
    )["emotions"]

    stories = pd.DataFrame(
        {
            "emotion": [e for e in emotions for _ in range(2)],
            "topic": ["t"] * (2 * len(emotions)),
            "story": [f"a story about {e}" for e in emotions for _ in range(2)],
        }
    )
    neutral = pd.DataFrame({"topic": ["t"] * 3, "story": ["a neutral story"] * 3})
    src = tmp_path / "src"
    src.mkdir()
    stories.to_parquet(src / "stories.parquet", index=False)
    neutral.to_parquet(src / "neutral.parquet", index=False)
    module._frames = {"stories": stories, "neutral": neutral}

    def fake_download(repo_id, filename, repo_type, local_dir):
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        name = "neutral.parquet" if "neutral" in filename else "stories.parquet"
        target.write_bytes((src / name).read_bytes())
        return str(target)

    monkeypatch.setattr(module, "hf_hub_download", fake_download)
    return module, emotions


def test_fetch_stories_writes_every_pilot_emotion(fetch_stories, tmp_path):
    module, emotions = fetch_stories
    config = _write_stories_config(tmp_path, emotions)
    assert module.main(["--config", str(config)]) == 0

    out_dir = tmp_path / "stories" / "v1"
    extra = json.loads((out_dir / "manifest.json").read_text())["config"]["extra"]
    assert extra["n_emotions"] == 14
    assert sorted(extra["emotion_counts"]) == sorted(emotions)
    frame = pd.read_parquet(out_dir / "stories.parquet")
    assert list(frame.columns) == ["emotion", "topic", "story"]
    assert set(frame["emotion"]) == set(emotions)


def test_fetch_stories_rejects_the_wrong_number_of_emotions(fetch_stories, tmp_path):
    module, emotions = fetch_stories
    config = _write_stories_config(tmp_path, emotions[:13])
    with pytest.raises(ValueError, match="expected exactly 14"):
        module.main(["--config", str(config)])


# ---------------------------------------------------------------------------
# smoke: never-raise contract
# ---------------------------------------------------------------------------


@pytest.fixture
def smoke():
    return load_script("smoke")


def test_smoke_records_a_missing_config_instead_of_raising(smoke, tmp_path):
    out_dir = tmp_path / "smoke"
    assert smoke.main(["--config", "/nonexistent.yaml", "--out-dir", str(out_dir)]) == 0

    result = json.loads((out_dir / "smoke.json").read_text())
    assert result["passed"] is False
    assert "FileNotFoundError" in result["error"]
    assert result["checks"] == []


def test_smoke_falls_back_to_logs_when_no_artifact_dir(smoke, tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        smoke,
        "_artifact_dir",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ARTIFACT_DIR is not set")),
    )
    assert smoke.main([]) == 0

    fallback = tmp_path / "logs" / "smoke-gemma-4-31B-it.json"
    result = json.loads(fallback.read_text())
    assert result["passed"] is False
    assert "ARTIFACT_DIR is not set" in result["error"]
    # A fallback file is a log, not an artifact: no manifest claims provenance for it.
    assert not (tmp_path / "logs" / "manifest.json").exists()


def test_run_check_records_failures_but_re_raises_interrupts(smoke):
    def boom():
        raise ValueError("layer 40 is missing")

    recorded = smoke.run_check("hook", boom)
    assert recorded["passed"] is False
    assert recorded["error"] == "ValueError: layer 40 is missing"

    for interrupt in (KeyboardInterrupt, SystemExit):
        def raiser(exc=interrupt):
            raise exc()

        with pytest.raises(interrupt):
            smoke.run_check("hook", raiser)


def test_resolve_base_url_adds_the_scheme_the_endpoint_file_lacks(
    smoke, tmp_path, monkeypatch
):
    monkeypatch.delenv("HEALTHY_RL_SERVER_URL", raising=False)
    endpoint = tmp_path / "endpoint"
    endpoint.write_text("127.0.0.1:8123\n")
    monkeypatch.setenv("HEALTHY_RL_ENDPOINT_FILE", str(endpoint))

    # Verbatim contents would be schemeless, and requests would reject every call --
    # recording four failures against a healthy server.
    assert smoke.resolve_base_url(None) == "http://127.0.0.1:8123"
    assert smoke.resolve_base_url("http://elsewhere:9/") == "http://elsewhere:9/"


# ---------------------------------------------------------------------------
# smoke: the steering check measures the residual, not the text
# ---------------------------------------------------------------------------

D_MODEL = 8
N_POS = 5
N_LAYERS = 12
PROBE = 8


def _spec() -> SimpleNamespace:
    return SimpleNamespace(
        name="fake",
        path=Path("/nonexistent/fake"),
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        probe_layer=PROBE,
        capture_layers=[PROBE - 2, PROBE - 1, PROBE, PROBE + 1, PROBE + 2],
    )


class FakeClient:
    """Returns residuals that differ from the baseline by a chosen fraction per layer.

    ``norm_match`` steering adds ``scale * ||h||`` to every token, so a per-layer
    fraction is exactly what the check is supposed to measure.
    """

    def __init__(self, fractions: dict[int, float], text: str = "steered") -> None:
        self.fractions = fractions
        self.text = text
        rng = np.random.default_rng(0)
        self.base = rng.normal(size=(N_LAYERS, N_POS, D_MODEL))
        direction = rng.normal(size=D_MODEL)
        self.direction = direction / np.linalg.norm(direction)

    def clear_hooks(self) -> None:
        pass

    def generate(self, prompt, *, capture_layers=None, steering_vectors=None, **kwargs):
        import torch

        if capture_layers is None:
            text = self.text if steering_vectors else "baseline"
            return SimpleNamespace(text=text, activations=None)

        stack = []
        for layer in capture_layers:
            block = self.base[layer].copy()
            if steering_vectors:
                fraction = self.fractions.get(layer, 0.0)
                norms = np.linalg.norm(block, axis=1, keepdims=True)
                block = block + fraction * norms * self.direction
            stack.append(block)
        acts = torch.tensor(np.stack(stack), dtype=torch.float32)
        return SimpleNamespace(text="x", activations={"residual_stream": acts})


def _smoke_cfg() -> dict:
    return {
        "prompt": "a prompt",
        "max_tokens": 8,
        "steer_max_tokens": 8,
        "seed": 1234,
        "temperature": 0.0,
        "steer_scale": 0.5,
        "norm_match": True,
    }


def test_check_steer_measures_the_residual_change(smoke):
    client = FakeClient({PROBE: 0.5, PROBE + 1: 0.3})
    detail = smoke.Smoke(client, _spec(), _smoke_cfg()).check_steer()

    assert detail["rel_change"][f"L{PROBE}"] == pytest.approx(0.5, abs=1e-6)
    assert detail["rel_change"][f"L{PROBE - 1}"] == pytest.approx(0.0, abs=1e-9)
    assert detail["rel_change"][f"L{PROBE + 1}"] == pytest.approx(0.3, abs=1e-6)
    assert -1.0 <= detail["cosine"][f"L{PROBE}"] <= 1.0
    assert detail["n_positions"] == N_POS


def test_check_steer_passes_when_the_text_is_unchanged(smoke):
    """The Qwen3.6-27B false negative: a fixed preamble hides real steering."""
    client = FakeClient({PROBE: 0.5, PROBE + 1: 0.3}, text="baseline")
    detail = smoke.Smoke(client, _spec(), _smoke_cfg()).check_steer()
    assert detail["text"]["differs"] is False


@pytest.mark.parametrize(
    "fractions, message",
    [
        ({PROBE: 0.0, PROBE + 1: 0.0}, "moved the residual by 0.0000"),
        ({PROBE: 0.5, PROBE - 1: 0.5, PROBE + 1: 0.3}, "landing on the wrong layer"),
        ({PROBE: 0.5, PROBE + 1: 0.0}, "did not propagate"),
        ({PROBE: 0.05, PROBE + 1: 0.05}, "expected 0.5"),
    ],
)
def test_check_steer_fails_on_a_wrong_residual(smoke, fractions, message):
    client = FakeClient(fractions)
    with pytest.raises(AssertionError, match=message):
        smoke.Smoke(client, _spec(), _smoke_cfg()).check_steer()
