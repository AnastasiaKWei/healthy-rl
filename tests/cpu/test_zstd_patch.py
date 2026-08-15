"""The vllm-lens zstd transport must survive concurrent use.

Guards `patches/vllm_lens_zstd_threadsafe.py`. vllm-lens keeps THREE pairs of
process-global zstd objects and calls them from concurrent handlers. Each pair
reuses one ZSTD_CCtx/ZSTD_DCtx, so concurrent calls interleave and produce
corrupt frames:

    vllm_lens/_worker_ext.py            server, per collective_rpc response
    vllm_lens/_activations_plugin.py    server, rank-payload decode
    vllm_lens/_helpers/_serialize.py    server compress + CLIENT decompress

The `_serialize.py` pair is the one that actually broke stage 3, and it has a
SILENT failure mode: at the real ~9.5 MB payload size with 8+ threads it
produced 12 raised errors AND 5 payloads that decompressed without raising and
returned the wrong bytes. A failure count of zero does not prove a run was
clean, so these tests compare bytes rather than merely asserting no exception.

These fail on an unpatched venv, which is the point: `uv sync` silently reverts
the patch, and a silently reverted patch means silently wrong emotion means.
"""

from __future__ import annotations

import concurrent.futures
import os
import pickle

import pytest

pytest.importorskip("vllm_lens", reason="vllm_lens not installed")

# ~9.5 MB matches a real capture_layers response (5 layers x ~200 positions x
# 5120 x 2 bytes). The corruption is size-sensitive, so testing small payloads
# only would pass on a broken venv.
REAL_PAYLOAD_BYTES = 9_500_000
THREADS = 16
ROUNDTRIPS = 128


def _pairs():
    """Every (name, compressor, decompressor) the patch is responsible for."""
    from vllm_lens import _activations_plugin, _worker_ext
    from vllm_lens._helpers import _serialize

    return [
        ("worker_ext/activations_plugin",
         _worker_ext._ZSTD_COMPRESSOR, _activations_plugin._ZSTD_DECOMPRESSOR),
        ("_helpers/_serialize",
         _serialize._ZSTD_COMPRESSOR, _serialize._ZSTD_DECOMPRESSOR),
    ]


def test_patch_is_applied_to_all_three_files():
    from vllm_lens import _activations_plugin, _worker_ext
    from vllm_lens._helpers import _serialize

    objs = {
        "_worker_ext._ZSTD_COMPRESSOR": _worker_ext._ZSTD_COMPRESSOR,
        "_activations_plugin._ZSTD_DECOMPRESSOR": _activations_plugin._ZSTD_DECOMPRESSOR,
        "_serialize._ZSTD_COMPRESSOR": _serialize._ZSTD_COMPRESSOR,
        "_serialize._ZSTD_DECOMPRESSOR": _serialize._ZSTD_DECOMPRESSOR,
    }
    unpatched = [n for n, o in objs.items() if type(o).__name__ != "_PerCallZstd"]
    assert not unpatched, (
        f"unpatched vllm-lens zstd singletons: {unpatched}; run "
        "`.venv/bin/python patches/vllm_lens_zstd_threadsafe.py`"
    )


@pytest.mark.parametrize("name,compressor,decompressor", _pairs(), ids=lambda v: v if isinstance(v, str) else "")
def test_concurrent_roundtrip_is_byte_exact(name, compressor, decompressor):
    """No raised errors AND no silent mismatches."""
    # Incompressible bytes: random data keeps zstd from shrinking the frame to
    # nothing, so the concurrent path is actually exercised.
    blob = pickle.dumps({"activations": os.urandom(REAL_PAYLOAD_BYTES)})

    def roundtrip(_):
        try:
            return "ok" if decompressor.decompress(compressor.compress(blob)) == blob else "SILENT"
        except Exception as exc:  # noqa: BLE001 - the failure mode under test
            return f"RAISED:{type(exc).__name__}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as pool:
        results = list(pool.map(roundtrip, range(ROUNDTRIPS)))

    silent = results.count("SILENT")
    raised = sum(1 for r in results if r.startswith("RAISED"))
    assert silent == 0 and raised == 0, (
        f"{name}: {raised} raised, {silent} SILENT mismatches out of {ROUNDTRIPS} "
        f"at {REAL_PAYLOAD_BYTES} bytes across {THREADS} threads"
    )
