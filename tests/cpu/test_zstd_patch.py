"""The vllm-lens zstd transport must survive concurrent use.

Guards `patches/vllm_lens_zstd_threadsafe.py`. Without it, the shared
ZstdCompressor/ZstdDecompressor singletons corrupt frames under concurrency,
which showed up as `ZstdError: Data corruption detected` on 13-19% of
activation requests during stage 3 — at batch_size 4 as well as 32, so
throttling alone does not fix it.

This test fails on an unpatched venv, which is the point: reinstalling
vllm-lens silently reverts the patch, and a silently reverted patch means
silently dropped stories in the emotion means.
"""

from __future__ import annotations

import concurrent.futures
import os
import pickle

import pytest

zstd_pair = pytest.importorskip("vllm_lens._worker_ext", reason="vllm_lens not installed")


def _pair():
    from vllm_lens._activations_plugin import _ZSTD_DECOMPRESSOR
    from vllm_lens._worker_ext import _ZSTD_COMPRESSOR

    return _ZSTD_COMPRESSOR, _ZSTD_DECOMPRESSOR


def test_patch_is_applied():
    compressor, decompressor = _pair()
    for obj in (compressor, decompressor):
        assert type(obj).__name__ == "_PerCallZstd", (
            "vllm-lens zstd singletons are unpatched; run "
            "`.venv/bin/python patches/vllm_lens_zstd_threadsafe.py`"
        )


@pytest.mark.parametrize("payload_bytes", [64_000, 2_000_000])
def test_concurrent_roundtrip_is_lossless(payload_bytes):
    compressor, decompressor = _pair()
    # Incompressible bytes: the corruption tracks context reuse, and random
    # data keeps zstd from trivially shrinking the frame to nothing.
    blob = pickle.dumps({"activations": os.urandom(payload_bytes)})

    def roundtrip(_):
        return decompressor.decompress(compressor.compress(blob)) == blob

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(roundtrip, range(128)))

    assert all(results), (
        f"{results.count(False)}/{len(results)} concurrent zstd roundtrips corrupted "
        f"at {payload_bytes} bytes"
    )
