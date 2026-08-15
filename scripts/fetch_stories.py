#!/usr/bin/env python
"""Stage 1: download the emotion story corpus into ``$ARTIFACT_DIR/stories/v1``.

Login-node stage. Compute nodes have no DNS, so everything the pipeline needs from
the Hub is pulled here. Nothing in this file imports torch or vLLM.

The raw downloads land in ``raw/`` and are left untouched; the filtered pilot subset
is written next to them as ``stories.parquet`` / ``neutral_stories.parquet``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from healthy_rl.artifacts import write_manifest
from healthy_rl.config import load_config, load_env, repo_root

DEFAULT_CONFIG = repo_root() / "configs" / "fetch_stories.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="override the config's out_dir"
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(repo_id: str, filename: str, raw_dir: Path) -> Path:
    """Fetch one file by name (never ``load_dataset``: see configs/fetch_stories.yaml)."""
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(raw_dir),
        )
    )


def check_columns(frame: pd.DataFrame, expected: list[str], what: str) -> None:
    actual = list(frame.columns)
    if actual != list(expected):
        raise ValueError(f"{what}: expected columns {list(expected)}, got {actual}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_env()
    cfg = load_config(args.config)

    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["out_dir"])
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    repo_id = cfg["repo_id"]
    emotion_col = cfg["emotion_column"]
    text_col = cfg["text_column"]
    emotions = list(cfg["emotions"])

    stories_raw = download(repo_id, cfg["stories_file"], raw_dir)
    neutral_raw = download(repo_id, cfg["neutral_file"], raw_dir)

    stories = pd.read_parquet(stories_raw)
    neutral = pd.read_parquet(neutral_raw)
    check_columns(stories, cfg["expect_columns"], cfg["stories_file"])
    check_columns(neutral, cfg["expect_neutral_columns"], cfg["neutral_file"])

    # Guard against the deflection table sneaking in under the same filename.
    if text_col not in stories.columns:
        raise ValueError(
            f"{cfg['stories_file']} has no {text_col!r} column (got {list(stories.columns)}); "
            "this is the wrong table"
        )

    expect_n = int(cfg["expect_n_emotions"])
    if len(emotions) != expect_n or len(set(emotions)) != expect_n:
        raise ValueError(
            f"config lists {len(emotions)} emotions ({len(set(emotions))} distinct), "
            f"expected exactly {expect_n}: {emotions}"
        )

    available = set(stories[emotion_col].unique())
    missing = [emotion for emotion in emotions if emotion not in available]
    if missing:
        raise ValueError(
            f"pilot emotions missing from {repo_id}/{cfg['stories_file']}: {missing}"
        )

    subset = stories[stories[emotion_col].isin(emotions)].reset_index(drop=True)
    counts = {
        emotion: int((subset[emotion_col] == emotion).sum()) for emotion in emotions
    }

    minimum = int(cfg["min_stories_per_emotion"])
    short = {e: n for e, n in counts.items() if n < minimum}
    if short:
        raise ValueError(f"emotions with fewer than {minimum} stories: {short}")

    expect_neutral = int(cfg["expect_neutral_rows"])
    if len(neutral) != expect_neutral:
        raise ValueError(
            f"{cfg['neutral_file']}: expected {expect_neutral} rows, got {len(neutral)}"
        )

    empty = int((subset[text_col].fillna("").str.strip() == "").sum())
    if empty:
        raise ValueError(f"{empty} pilot stories have empty {text_col!r} text")

    # Neutral stories feed the covariance and the per-layer norms, so an empty one
    # would corrupt every projection downstream just as surely as an empty emotion story.
    empty_neutral = int((neutral[text_col].fillna("").str.strip() == "").sum())
    if empty_neutral:
        raise ValueError(f"{empty_neutral} neutral stories have empty {text_col!r} text")

    stories_out = out_dir / "stories.parquet"
    neutral_out = out_dir / "neutral_stories.parquet"
    subset.to_parquet(stories_out, index=False)
    neutral.to_parquet(neutral_out, index=False)

    extra = {
        "n_emotions": len(emotions),
        "emotion_counts": counts,
        "n_stories": int(len(subset)),
        "n_stories_upstream": int(len(stories)),
        "n_neutral": int(len(neutral)),
        "columns": list(subset.columns),
        "neutral_columns": list(neutral.columns),
        "files": {
            "stories": {
                "path": stories_out.name,
                "sha256": sha256_file(stories_out),
            },
            "neutral_stories": {
                "path": neutral_out.name,
                "sha256": sha256_file(neutral_out),
            },
        },
    }
    (out_dir / "stories.json").write_text(json.dumps(extra, indent=2, sort_keys=True) + "\n")

    # `cfg` holds only what the YAML declares -- no environment, no HF_TOKEN.
    write_manifest(out_dir, stage=cfg["stage"], config={**cfg, "extra": extra})

    print(f"wrote {stories_out} ({len(subset)} stories over {len(emotions)} emotions)")
    print(f"wrote {neutral_out} ({len(neutral)} neutral stories)")
    for emotion in emotions:
        print(f"  {emotion:<14} {counts[emotion]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
