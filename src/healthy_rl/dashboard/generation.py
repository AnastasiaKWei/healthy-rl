"""Turn one chat response plus its projection-hook payload into a ``Generation``.

Pure: no HTTP, no torch import at module scope. ``Engine`` (engine.py) does the
request and hands the pieces here; tests feed canned payloads.

Hook payload (from ``healthy_rl.rollouts.make_projection_hook``), per capture
layer ``l``: ``proj_L{l}`` ``(P, E)``, ``norm_L{l}`` ``(P,)``, ``kind_L{l}``
``(P,)`` where 1.0 marks a prefill row and 0.0 a decode row. Under chunked
prefill several prefill rows precede the decode rows; only the LAST prefill
row is the residual that produced the first generated token.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

TAG_PAIRS: tuple[tuple[str, str], ...] = (
    ("<think>", "</think>"),
    ("[THINK]", "[/THINK]"),
    ("<SCRATCHPAD_REASONING>", "</SCRATCHPAD_REASONING>"),
)


def split_reasoning(text: str) -> tuple[str | None, str, int]:
    """``(reasoning, answer, think_end_char)`` for text that may carry a reasoning span.

    ``think_end_char`` is the offset in ``text`` at which the answer begins:
    just after the closing tag, ``len(text)`` for an unclosed span, 0 when
    there is no reasoning. Tokens starting before it are ``think`` tokens.
    """
    for open_tag, close_tag in TAG_PAIRS:
        start = text.find(open_tag)
        end = text.find(close_tag)
        if start < 0 and end < 0:
            continue
        if start >= 0:
            body_start = start + len(open_tag)
            end = text.find(close_tag, body_start)
            if end < 0:
                return text[body_start:].strip(), "", len(text)
            answer = (text[:start] + text[end + len(close_tag):]).strip()
            return text[body_start:end].strip(), answer, end + len(close_tag)
        # close tag only: some templates emit the opening tag as part of the prompt
        return text[:end].strip(), text[end + len(close_tag):].strip(), end + len(close_tag)
    return None, text, 0


def token_kinds(tokens: Sequence[str], think_end_char: int) -> list[str]:
    """Label each token ``think``/``answer`` by where its span starts in the text."""
    kinds: list[str] = []
    pos = 0
    for tok in tokens:
        kinds.append("think" if pos < think_end_char else "answer")
        pos += len(tok)
    return kinds


def _to_numpy(value: Any) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if detach is not None:
        v = detach().cpu()
        if str(v.dtype) == "torch.bfloat16":
            v = v.float()
        return v.numpy()
    return np.asarray(value)


def merge_hook_results(hook_results: dict | None) -> dict[str, np.ndarray]:
    """Flatten ``{hook_index: {key: tensor}}`` into ``{key: ndarray}``."""
    merged: dict[str, np.ndarray] = {}
    for per_hook in (hook_results or {}).values():
        for key, value in per_hook.items():
            merged[key] = _to_numpy(value)
    return merged


@dataclass
class Generation:
    text: str
    reasoning: str | None
    answer: str
    tokens: list[str]
    token_kind: list[str]
    proj: np.ndarray            # (T, L, E) decode rows
    norm: np.ndarray            # (T, L)
    proj_prefill: np.ndarray    # (L, E)
    norm_prefill: np.ndarray    # (L,)
    res_start: np.ndarray | None
    res_end: np.ndarray | None
    n_generated: int
    n_think: int
    at_cap: bool
    finish_reason: str | None
    misaligned: bool = False
    error: str | None = None
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def arrays(self, probe_layer: int) -> dict[str, np.ndarray]:
        """What ``SessionStore.append`` stores for this generation."""
        out = {"proj": self.proj.astype(np.float32), "norm": self.norm.astype(np.float32),
               "proj_prefill": self.proj_prefill.astype(np.float32),
               "norm_prefill": self.norm_prefill.astype(np.float32)}
        if self.res_start is not None:
            out[f"res_start_L{probe_layer}"] = self.res_start.astype(np.float16)
        if self.res_end is not None:
            out[f"res_end_L{probe_layer}"] = self.res_end.astype(np.float16)
        return out


def assemble_generation(
    *,
    text: str,
    reasoning_content: str | None,
    tokens: Sequence[str],
    finish_reason: str | None,
    hook_saved: dict[str, np.ndarray],
    capture_layers: Sequence[int],
    probe_layer: int,
    n_emotions: int,
    max_tokens: int,
    seconds: float,
) -> Generation:
    tokens = list(tokens)
    problems: list[str] = []
    warns: list[str] = []
    L = len(capture_layers)

    # --- reasoning / answer split -------------------------------------------
    if reasoning_content:
        # A server-side reasoning parser hands back the two halves already split,
        # so the think/answer boundary has to be located in the token stream: find
        # where the answer starts inside the joined tokens. When that fails the
        # fallback is an offset into `reasoning_content`, which the tokens may not
        # cover -- every token then gets labelled "think". Say so rather than
        # reporting a confident n_think.
        reasoning, answer = reasoning_content.strip(), text.strip()
        joined = "".join(tokens)
        idx = joined.find(text.strip()) if text.strip() else -1
        if idx > 0:
            think_end_char = idx
        else:
            think_end_char = len(reasoning_content)
            warns.append(
                "reasoning_content offset is a guess: answer text not found in token stream"
            )
        full_text = reasoning_content + text
    else:
        reasoning, answer, think_end_char = split_reasoning(text)
        full_text = text
    kinds = token_kinds(tokens, think_end_char)

    # --- hook rows -----------------------------------------------------------
    n_decode: int | None = None
    per_layer_proj: list[np.ndarray | None] = []
    per_layer_norm: list[np.ndarray | None] = []
    per_layer_pp: list[np.ndarray | None] = []
    per_layer_pn: list[float] = []
    for layer in capture_layers:
        proj = hook_saved.get(f"proj_L{layer}")
        norm = hook_saved.get(f"norm_L{layer}")
        kind = hook_saved.get(f"kind_L{layer}")
        if proj is None or norm is None or kind is None:
            problems.append(f"layer {layer} missing from hook results")
            per_layer_proj.append(None); per_layer_norm.append(None); per_layer_pp.append(None); per_layer_pn.append(np.nan)
            continue
        proj = np.asarray(proj, dtype=np.float32); norm = np.asarray(norm, dtype=np.float32).reshape(-1)
        kind = np.asarray(kind, dtype=np.float32).reshape(-1)
        if (proj.ndim != 2 or proj.shape[0] != kind.shape[0] or proj.shape[1] != n_emotions
                or norm.shape[0] != kind.shape[0]):
            problems.append(
                f"layer {layer}: proj shape {proj.shape}, norm {norm.shape} vs kind {kind.shape}, E={n_emotions}"
            )
            per_layer_proj.append(None); per_layer_norm.append(None); per_layer_pp.append(None); per_layer_pn.append(np.nan)
            continue
        decode = kind == 0.0
        prefill_rows = np.flatnonzero(kind == 1.0)
        n_here = int(decode.sum())
        if n_decode is None:
            n_decode = n_here
        elif n_here != n_decode:
            problems.append(f"layer {layer}: {n_here} decode rows, layer {capture_layers[0]} has {n_decode}")
        per_layer_proj.append(proj[decode]); per_layer_norm.append(norm[decode])
        if prefill_rows.size:
            per_layer_pp.append(proj[prefill_rows[-1]]); per_layer_pn.append(float(norm[prefill_rows[-1]]))
        else:
            problems.append(f"layer {layer}: no prefill row")
            per_layer_pp.append(None); per_layer_pn.append(np.nan)

    T = n_decode or 0
    proj_out = np.full((T, L, n_emotions), np.nan, np.float32)
    norm_out = np.full((T, L), np.nan, np.float32)
    pp_out = np.full((L, n_emotions), np.nan, np.float32)
    pn_out = np.asarray(per_layer_pn, np.float32) if L else np.zeros(0, np.float32)
    for li in range(L):
        p, n, pp = per_layer_proj[li], per_layer_norm[li], per_layer_pp[li]
        if p is not None and p.shape[0] == T:
            proj_out[:, li, :] = p; norm_out[:, li] = n
        if pp is not None:
            pp_out[li] = pp

    misaligned = T != len(tokens)
    if misaligned:
        problems.append(f"{len(tokens)} logprob tokens but {T} decode rows in hook results")

    res_start = hook_saved.get(f"res_start_L{probe_layer}")
    res_end = hook_saved.get(f"res_end_L{probe_layer}")
    return Generation(
        text=full_text, reasoning=reasoning, answer=answer, tokens=tokens, token_kind=kinds,
        proj=proj_out, norm=norm_out, proj_prefill=pp_out, norm_prefill=pn_out,
        res_start=None if res_start is None else np.asarray(res_start, np.float32),
        res_end=None if res_end is None else np.asarray(res_end, np.float32),
        n_generated=T, n_think=sum(k == "think" for k in kinds),
        at_cap=(T >= max_tokens) or (finish_reason == "length"),
        finish_reason=finish_reason, misaligned=misaligned,
        error="; ".join(problems) if problems else None, seconds=seconds, warnings=warns,
    )
