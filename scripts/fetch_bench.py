#!/usr/bin/env python
"""Stage 2: download the impossible-LiveCodeBench split into ``$ARTIFACT_DIR/bench/v1``.

Login-node stage: no torch, no vLLM, and it must run here because compute nodes have
no DNS. The sorted ``task_id`` list goes into the manifest, because downstream problem
selection ("the first 24 in sorted order") is only reproducible if that order is pinned.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

from healthy_rl.artifacts import write_manifest
from healthy_rl.config import load_config, load_env, repo_root

# Ruling R11: ordering is by the trailing integer of the task_id, not lexicographic.
# Imported rather than reimplemented so this manifest and the rollout stage cannot
# disagree about which problems "the first 24" are. `healthy_rl.rollouts` keeps its
# heavy imports inside functions, so this stays a login-node stage.
from healthy_rl.rollouts import sort_task_ids

DEFAULT_CONFIG = repo_root() / "configs" / "fetch_bench.yaml"


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


def resolve_shards(repo_id: str, pattern: str) -> list[str]:
    """Repo files matching ``pattern``, sorted, so a re-shard upstream is visible."""
    files = list_repo_files(repo_id, repo_type="dataset")
    shards = sorted(fnmatch.filter(files, pattern))
    if not shards:
        raise ValueError(f"no files in {repo_id} match {pattern!r}; found {files}")
    return shards


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_env()
    cfg = load_config(args.config)

    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["out_dir"])
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    repo_id = cfg["repo_id"]
    split = cfg["split"]
    id_col = cfg["id_column"]

    shards = resolve_shards(repo_id, cfg["shard_glob"].format(split=split))
    frames = []
    for shard in shards:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=shard,
            repo_type="dataset",
            local_dir=str(raw_dir),
        )
        frames.append(pd.read_parquet(local))
    bench = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    expected_cols = list(cfg["expect_columns"])
    if list(bench.columns) != expected_cols:
        raise ValueError(
            f"{repo_id} split {split}: expected columns {expected_cols}, "
            f"got {list(bench.columns)}"
        )

    expect_rows = int(cfg["expect_rows"])
    if len(bench) != expect_rows:
        raise ValueError(
            f"{repo_id} split {split}: expected {expect_rows} rows, got {len(bench)}"
        )

    task_ids = sort_task_ids(str(task_id) for task_id in bench[id_col])
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"duplicate {id_col} values in {repo_id} split {split}")

    bench_out = out_dir / f"{split}.parquet"
    bench.to_parquet(bench_out, index=False)

    extra = {
        "split": split,
        "n_rows": int(len(bench)),
        "columns": list(bench.columns),
        "shards": shards,
        "task_ids": task_ids,
        "task_id_order": "R11: by trailing integer (healthy_rl.rollouts.sort_task_ids)",
        "impossible_types": sorted(bench["impossible_type"].astype(str).unique().tolist()),
        "files": {
            split: {"path": bench_out.name, "sha256": sha256_file(bench_out)},
        },
    }
    (out_dir / "bench.json").write_text(json.dumps(extra, indent=2, sort_keys=True) + "\n")

    # `cfg` holds only what the YAML declares -- no environment, no HF_TOKEN.
    write_manifest(out_dir, stage=cfg["stage"], config={**cfg, "extra": extra})

    print(f"wrote {bench_out} ({len(bench)} rows, split {split})")
    print(f"first 5 task_ids in R11 order: {task_ids[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
