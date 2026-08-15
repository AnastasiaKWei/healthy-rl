#!/usr/bin/env python
"""Make vllm-lens 1.2.1's zstd transport thread-safe.

vllm-lens keeps THREE pairs of PROCESS-GLOBAL zstd objects and calls them from
concurrent request handlers:

    vllm_lens/_worker_ext.py:52          _ZSTD_COMPRESSOR   = zstd.ZstdCompressor(level=1)
    vllm_lens/_activations_plugin.py:35  _ZSTD_DECOMPRESSOR = zstd.ZstdDecompressor()
    vllm_lens/_helpers/_serialize.py:14  _ZSTD_COMPRESSOR    (serialize_tensor, server side)
    vllm_lens/_helpers/_serialize.py:15  _ZSTD_DECOMPRESSOR  (deserialize_tensor, CLIENT side)

The `_serialize.py` pair is the one that actually broke stage 3: the client
decompresses every activation response there, and the observed exception was a
bare ZstdError raised inside the client's own fetch, which only happens at
`_serialize.py:70`.

THE SILENT VARIANT IS THE DANGEROUS ONE. Measured at the real 9.5 MB payload
size with 8+ threads, unpatched `_serialize` produced 12 raised errors AND
**5 silent mismatches** — payloads that decompress without raising and return
the WRONG BYTES. Those fold into the emotion means undetected; a failure count
of zero does not mean a run was clean. Any `capture_layers` data taken on an
unpatched venv must be treated as suspect, not merely incomplete.

A ZstdCompressor/ZstdDecompressor reuses one internal ZSTD_CCtx/ZSTD_DCtx, so
concurrent compress()/decompress() calls on the same instance interleave and
produce corrupt frames. Observed on this cluster while extracting activations:

    ZstdError: decompression error: Data corruption detected
    Olmo 9/64 (14%) and Qwen 12/64 (19%) requests at batch_size=32
    Olmo 9/68 (13%) still failing at batch_size=4  <- NOT fixed by throttling

This replaces both singletons with a proxy that builds a fresh context per call.
Constructing a context is negligible next to compressing a multi-MB payload.

Idempotent. Reversible with `uv sync --reinstall-package vllm-lens`.
The upstream fix would be thread-local contexts or a lock; report accordingly.
"""
from __future__ import annotations

import sys
from pathlib import Path

SHIM = '''
class _PerCallZstd:  # healthy-rl patch: fresh context per call (thread-safe)
    """Drop-in for a shared ZstdCompressor/ZstdDecompressor.

    A shared instance reuses one ZSTD_CCtx/ZSTD_DCtx and corrupts frames when
    used concurrently; building a context per call costs microseconds against
    multi-MB payloads.
    """

    __slots__ = ("_factory",)

    def __init__(self, factory):
        self._factory = factory

    def compress(self, data):
        return self._factory().compress(data)

    def decompress(self, data, *a, **kw):
        return self._factory().decompress(data, *a, **kw)
'''

TARGETS = [
    (
        "_worker_ext.py",
        "_ZSTD_COMPRESSOR = zstd.ZstdCompressor(level=1)",
        "_ZSTD_COMPRESSOR = _PerCallZstd(lambda: zstd.ZstdCompressor(level=1))",
    ),
    (
        "_activations_plugin.py",
        "_ZSTD_DECOMPRESSOR = zstd.ZstdDecompressor()",
        "_ZSTD_DECOMPRESSOR = _PerCallZstd(lambda: zstd.ZstdDecompressor())",
    ),
    # The pair that actually broke stage 3. `deserialize_tensor` runs in the
    # CLIENT process, which decompresses one multi-MB activation payload per
    # response across many threads.
    (
        "_helpers/_serialize.py",
        "_ZSTD_COMPRESSOR = zstd.ZstdCompressor(level=1)",
        "_ZSTD_COMPRESSOR = _PerCallZstd(lambda: zstd.ZstdCompressor(level=1))",
    ),
    (
        "_helpers/_serialize.py",
        "_ZSTD_DECOMPRESSOR = zstd.ZstdDecompressor()",
        "_ZSTD_DECOMPRESSOR = _PerCallZstd(lambda: zstd.ZstdDecompressor())",
    ),
]


def main() -> int:
    import vllm_lens

    pkg = Path(vllm_lens.__file__).parent
    changed = []
    for fname, old, new in TARGETS:
        path = pkg / fname
        src = path.read_text()
        if new in src:
            print(f"  already patched: {fname}")
            continue
        if old not in src:
            print(f"  ERROR: anchor not found in {fname}: {old!r}", file=sys.stderr)
            return 2
        # _serialize.py carries two anchors; the shim class goes in once per file.
        prefix = "" if "_PerCallZstd" in src else SHIM + "\n"
        path.write_text(src.replace(old, prefix + new, 1))
        changed.append(f"{fname}:{old.split(' =')[0]}")
        print(f"  patched: {path} ({old.split(' =')[0]})")
    print(f"patched {len(changed)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
