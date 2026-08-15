"""CPU-only tests for config, model introspection, and artifact manifests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from healthy_rl.artifacts import (
    StaleUpstreamError,
    artifact_dir,
    check_upstream,
    manifest_sha256,
    verify_upstreams,
    write_manifest,
)
from healthy_rl.config import load_config, load_env
from healthy_rl.models import ModelSpec

# (name, architecture, nested_under_text_config, n_layers, d_model, probe_layer)
# Architectures and nesting match the real checkpoints under $MODEL_DIR: the three
# ConditionalGeneration wrappers keep the language config under `text_config`,
# Olmo has it at the top level.
PILOT_MODELS = [
    ("gemma-4-31B-it", "Gemma4ForConditionalGeneration", True, 60, 5376, 40),
    ("Qwen3.6-27B", "Qwen3_5ForConditionalGeneration", True, 64, 5120, 43),
    ("Muse-Glimmer-30B", "MuseGlimmerForConditionalGeneration", True, 52, 6656, 35),
    ("Olmo-3.1-32B-Think", "Olmo3ForCausalLM", False, 64, 5120, 43),
]


def _write_checkpoint(root, name, architecture, nested, n_layers, d_model):
    ckpt = Path(root) / name
    ckpt.mkdir(parents=True)
    inner = {"num_hidden_layers": n_layers, "hidden_size": d_model}
    if nested:
        cfg = {"architectures": [architecture], "model_type": "multimodal", "text_config": inner}
    else:
        cfg = {"architectures": [architecture], **inner}
    (ckpt / "config.json").write_text(json.dumps(cfg))
    return ckpt


@pytest.mark.parametrize(
    "name,architecture,nested,n_layers,d_model,probe_layer", PILOT_MODELS
)
def test_from_checkpoint_pilot_table(
    tmp_path, name, architecture, nested, n_layers, d_model, probe_layer
):
    ckpt = _write_checkpoint(tmp_path, name, architecture, nested, n_layers, d_model)
    spec = ModelSpec.from_checkpoint(ckpt)

    assert spec.name == name
    assert spec.path == ckpt
    assert spec.architecture == architecture
    assert spec.n_layers == n_layers
    assert spec.d_model == d_model
    assert spec.probe_layer == probe_layer
    assert spec.capture_layers == [
        probe_layer - 2,
        probe_layer - 1,
        probe_layer,
        probe_layer + 1,
        probe_layer + 2,
    ]


def test_capture_layers_clipped_at_both_ends(tmp_path):
    low = _write_checkpoint(tmp_path, "tiny", "TinyForCausalLM", False, 3, 8)
    spec = ModelSpec.from_checkpoint(low)
    # probe_layer = round(2*3/3) = 2, window 0..4 clipped to 0..2
    assert spec.probe_layer == 2
    assert spec.capture_layers == [0, 1, 2]
    assert all(0 <= layer < spec.n_layers for layer in spec.capture_layers)


def test_from_checkpoint_missing_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.json"):
        ModelSpec.from_checkpoint(tmp_path / "nope")


def test_from_checkpoint_explicit_name(tmp_path):
    ckpt = _write_checkpoint(tmp_path, "snapshot-abc", "Olmo3ForCausalLM", False, 64, 5120)
    spec = ModelSpec.from_checkpoint(ckpt, name="Olmo-3.1-32B-Think")
    assert spec.name == "Olmo-3.1-32B-Think"


def test_load_config_expands_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_DIR", "/artifacts")
    monkeypatch.setenv("MODEL_DIR", "/models")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "out: ${ARTIFACT_DIR}/probes",
                "models:",
                "  - ${MODEL_DIR}/gemma",
                "nested:",
                "  key: ${MODEL_DIR}/qwen/config.json",
                "missing_ok: ${NOT_SET:-fallback}",
                "n: 3",
                "flag: true",
            ]
        )
    )
    cfg = load_config(cfg_path)
    assert cfg["out"] == "/artifacts/probes"
    assert cfg["models"] == ["/models/gemma"]
    assert cfg["nested"]["key"] == "/models/qwen/config.json"
    assert cfg["missing_ok"] == "fallback"
    assert cfg["n"] == 3
    assert cfg["flag"] is True


def test_load_config_undefined_var_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("DEFINITELY_NOT_SET", raising=False)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("out: ${DEFINITELY_NOT_SET}/x")
    with pytest.raises(KeyError, match="DEFINITELY_NOT_SET"):
        load_config(cfg_path)


def test_load_env_does_not_overwrite(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "# a comment",
                "",
                "MODEL_DIR=/from/file",
                'ARTIFACT_DIR="/quoted/artifacts"',
                "export EXPORTED=yes",
            ]
        )
    )
    monkeypatch.setenv("MODEL_DIR", "/already/set")
    monkeypatch.delenv("ARTIFACT_DIR", raising=False)
    monkeypatch.delenv("EXPORTED", raising=False)

    loaded = load_env(env_path)

    assert os.environ["MODEL_DIR"] == "/already/set"
    assert os.environ["ARTIFACT_DIR"] == "/quoted/artifacts"
    assert os.environ["EXPORTED"] == "yes"
    assert loaded["MODEL_DIR"] == "/from/file"


def test_load_env_missing_file_is_noop(tmp_path):
    assert load_env(tmp_path / "absent.env") == {}


def test_artifact_dir_creates_path(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    out = artifact_dir("probes", "gemma-4-31B-it", "v1")
    assert out == tmp_path / "probes" / "gemma-4-31B-it" / "v1"
    assert out.is_dir()
    # idempotent
    assert artifact_dir("probes", "gemma-4-31B-it", "v1") == out


def test_artifact_dir_requires_env(tmp_path, monkeypatch):
    monkeypatch.delenv("ARTIFACT_DIR", raising=False)
    monkeypatch.setattr("healthy_rl.artifacts.load_env", lambda *a, **k: {})
    with pytest.raises(RuntimeError, match="ARTIFACT_DIR"):
        artifact_dir("probes", "m", "v1")


def test_manifest_round_trip(tmp_path):
    cfg = {"model": "gemma-4-31B-it", "layers": [38, 39, 40], "var_frac": 0.5}
    path = write_manifest(tmp_path, stage="probes", config=cfg, upstreams=None)

    assert path == tmp_path / "manifest.json"
    manifest = check_upstream(tmp_path)
    assert manifest["stage"] == "probes"
    assert manifest["config"] == cfg
    assert manifest["upstreams"] == {}
    # Schema fixed by the plan: created_at, git.sha, git.dirty.
    assert "created_at" in manifest
    assert set(manifest["git"]) == {"sha", "dirty"}
    assert isinstance(manifest["git"]["dirty"], bool)
    # readable from the manifest file path directly as well
    assert check_upstream(path) == manifest
    assert manifest_sha256(tmp_path) == manifest_sha256(path)
    assert len(manifest_sha256(tmp_path)) == 64


def test_check_upstream_missing_manifest_names_path(tmp_path):
    missing = tmp_path / "activations" / "v1"
    missing.mkdir(parents=True)
    with pytest.raises(FileNotFoundError) as excinfo:
        check_upstream(missing)
    assert str(missing) in str(excinfo.value)


def test_manifest_chain_detects_rewritten_upstream(tmp_path):
    upstream = tmp_path / "activations" / "v1"
    upstream.mkdir(parents=True)
    write_manifest(upstream, stage="activations", config={"n_prompts": 128}, upstreams=None)
    sha_before = manifest_sha256(upstream)

    downstream = tmp_path / "probes" / "v1"
    downstream.mkdir(parents=True)
    write_manifest(
        downstream,
        stage="probes",
        config={"var_frac": 0.5},
        upstreams={"activations": upstream},
    )

    recorded = check_upstream(downstream)["upstreams"]["activations"]
    assert recorded["manifest_sha256"] == sha_before
    assert recorded["stage"] == "activations"
    verify_upstreams(downstream)  # clean chain

    # Rewriting the upstream changes its hash, so the downstream check must fail.
    write_manifest(upstream, stage="activations", config={"n_prompts": 256}, upstreams=None)
    assert manifest_sha256(upstream) != sha_before
    with pytest.raises(StaleUpstreamError) as excinfo:
        verify_upstreams(downstream)
    assert "activations" in str(excinfo.value)


def test_write_manifest_rejects_missing_upstream(tmp_path):
    downstream = tmp_path / "probes"
    downstream.mkdir()
    with pytest.raises(FileNotFoundError):
        write_manifest(
            downstream, stage="probes", config={}, upstreams={"activations": tmp_path / "gone"}
        )


def test_write_manifest_accepts_upstream_sequence(tmp_path):
    upstream = tmp_path / "activations"
    upstream.mkdir()
    write_manifest(upstream, stage="activations", config={}, upstreams=None)

    downstream = tmp_path / "probes"
    downstream.mkdir()
    write_manifest(downstream, stage="probes", config={}, upstreams=[upstream])

    manifest = check_upstream(downstream)
    assert set(manifest["upstreams"]) == {str(upstream.resolve())}
    verify_upstreams(downstream)


def test_upstream_sequence_keeps_both_same_stage_upstreams(tmp_path):
    """The comparison stage takes two models' activation dirs; they must not collide."""
    first = tmp_path / "activations" / "gemma-4-31B-it"
    second = tmp_path / "activations" / "Olmo-3.1-32B-Think"
    for path, model in ((first, "gemma-4-31B-it"), (second, "Olmo-3.1-32B-Think")):
        path.mkdir(parents=True)
        write_manifest(path, stage="activations", config={"model": model}, upstreams=None)

    downstream = tmp_path / "compare"
    downstream.mkdir()
    write_manifest(downstream, stage="compare", config={}, upstreams=[first, second])

    upstreams = check_upstream(downstream)["upstreams"]
    assert len(upstreams) == 2
    assert {entry["path"] for entry in upstreams.values()} == {
        str(first.resolve()),
        str(second.resolve()),
    }
    verify_upstreams(downstream)

    # Either one drifting must be caught, not just the last one written.
    write_manifest(first, stage="activations", config={"model": "changed"}, upstreams=None)
    with pytest.raises(StaleUpstreamError) as excinfo:
        verify_upstreams(downstream)
    assert str(first.resolve()) in str(excinfo.value)


def test_manifest_records_dirty_worktree(tmp_path, monkeypatch):
    """git.dirty must follow `git status --porcelain` being non-empty."""
    calls = {}

    def fake_git(*args):
        calls[args] = calls.get(args, 0) + 1
        if args == ("rev-parse", "HEAD"):
            return "deadbeef" + "0" * 32 + "\n"
        if args == ("status", "--porcelain"):
            return " M src/healthy_rl/server.py\n"
        return None

    monkeypatch.setattr("healthy_rl.artifacts._git", fake_git)
    write_manifest(tmp_path, stage="probes", config={}, upstreams=None)
    manifest = check_upstream(tmp_path)
    assert manifest["git"]["sha"] == "deadbeef" + "0" * 32
    assert manifest["git"]["dirty"] is True


def test_manifest_records_clean_worktree(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "healthy_rl.artifacts._git",
        lambda *args: "abc123\n" if args == ("rev-parse", "HEAD") else "",
    )
    write_manifest(tmp_path, stage="probes", config={}, upstreams=None)
    assert check_upstream(tmp_path)["git"] == {"sha": "abc123", "dirty": False}


def test_manifest_git_unknown_is_not_reported_clean(tmp_path, monkeypatch):
    """No git available must read as unknown, never as a clean tree."""
    monkeypatch.setattr("healthy_rl.artifacts._git", lambda *args: None)
    write_manifest(tmp_path, stage="probes", config={}, upstreams=None)
    assert check_upstream(tmp_path)["git"] == {"sha": None, "dirty": None}


def test_verify_upstreams_reads_legacy_sha256_key(tmp_path):
    """Manifests written before the schema fix must still verify."""
    upstream = tmp_path / "activations"
    upstream.mkdir()
    write_manifest(upstream, stage="activations", config={}, upstreams=None)

    downstream = tmp_path / "probes"
    downstream.mkdir()
    write_manifest(downstream, stage="probes", config={}, upstreams={"activations": upstream})

    legacy_path = downstream / "manifest.json"
    legacy = json.loads(legacy_path.read_text())
    entry = legacy["upstreams"]["activations"]
    entry["sha256"] = entry.pop("manifest_sha256")
    legacy_path.write_text(json.dumps(legacy, indent=2, sort_keys=True))

    verify_upstreams(downstream)
    write_manifest(upstream, stage="activations", config={"changed": True}, upstreams=None)
    with pytest.raises(StaleUpstreamError):
        verify_upstreams(downstream)
