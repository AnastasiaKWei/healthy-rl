#!/usr/bin/env python
"""Stop vllm-lens 1.2.1 logging one traceback per layer per forward pass.

`_make_pre_hook` reads the layer's hidden states from `args[1]`, documenting
that "vLLM decoder layers have signature forward(positions, hidden_states,
residual)". When a model's layers are called with fewer positional arguments,
`args[1]` raises IndexError. vllm-lens CATCHES it and returns None -- the
pre-hook is simply skipped -- and then logs the full traceback at WARNING with
`exc_info=True`. On every forward pass. Of every layer. Of every request.

MEASURED COST. One 3-hour Qwen3.5-9B rollout job logged it 12.1 million times
and produced a 5.6 GB server log; the serve directory reached 99 GB out of a
108 GB artifact tree, and roughly 95% of every large log was these six lines.

WHAT IT IS NOT. It is not a measurement failure. This project's captures come
from POST-hooks, not pre-hooks; every record written while the log was filling
carries `hook_data: true`, a residual file and zero `turn_errors`. Verified on
Qwen3.5-9B pos6/aff6/affpos6 and Nemotron d6 -- 81 records, all clean. The
patch changes nothing that any measurement reads.

WHAT IT DOES. Warns once per layer index, then stays silent for that layer.
The first traceback survives, so the diagnostic is not lost -- only its
repetition. If a NEW pre-hook failure mode appears on some other layer, it
still gets its own warning.

Idempotent. Reversible with `uv sync --reinstall-package vllm-lens`, which is
also what silently reverts it -- see tests/cpu/test_pre_hook_log_spam.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

REGISTRY = (
    "\n# healthy-rl patch: layer indices already warned about, so a pre-hook that\n"
    "# fails on every forward pass logs its traceback once instead of millions of\n"
    "# times. Module-global deliberately: the hook closures are per-layer.\n"
    "_HEALTHY_RL_PRE_HOOK_WARNED: set[int] = set()\n"
)

OLD = """        except Exception:
            logger.warning(
                "vllm-lens pre-hook error on layer %d, skipping",
                layer_idx,
                exc_info=True,
            )
            return None
"""

NEW = """        except Exception:
            # healthy-rl patch: once per layer, not once per forward pass.
            if layer_idx not in _HEALTHY_RL_PRE_HOOK_WARNED:
                _HEALTHY_RL_PRE_HOOK_WARNED.add(layer_idx)
                logger.warning(
                    "vllm-lens pre-hook error on layer %d, skipping "
                    "(healthy-rl patch: later failures on this layer are silent)",
                    layer_idx,
                    exc_info=True,
                )
            return None
"""

MARK = "_HEALTHY_RL_PRE_HOOK_WARNED"


def main() -> int:
    import vllm_lens

    target = Path(vllm_lens.__file__).parent / "_worker_ext.py"
    source = target.read_text()

    if MARK in source:
        print(f"already patched: {target}")
        return 0
    if OLD not in source:
        print(
            f"ERROR: {target} does not contain the expected pre-hook except block; "
            "vllm-lens has changed and this patch needs rewriting.",
            file=sys.stderr,
        )
        return 1

    # The registry goes after the imports but before any use: anchor it on the
    # function that uses it rather than on a line number.
    anchor = "def _make_pre_hook("
    patched = source.replace(OLD, NEW, 1).replace(anchor, REGISTRY.lstrip("\n") + "\n\n" + anchor, 1)
    target.write_text(patched)
    print(f"patched {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
