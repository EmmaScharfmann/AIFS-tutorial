"""
Import this module *before* importing anything from ``anemoi`` to allow the compatibility with available GPU / CPU
"""

import sys
import time
import types

import torch
import torch.nn.functional as F


# ── SDPA-based attention replacement ─────────────────────────────────────────

def _window_bounds(seq_len, left, right, causal, device):
    """
    Build the per-query (lower, upper) inclusive key bounds implied by
    ``window_size=(left, right)`` and ``causal``, following flash-attn semantics:

    - left  < 0  -> no lower-bound restriction from the window
    - right < 0  -> no upper-bound restriction from the window
    - causal=True additionally forces key <= query, regardless of `right`
    """
    idx = torch.arange(seq_len, device=device)
    lower = torch.zeros(seq_len, dtype=torch.long, device=device) if left < 0 else torch.clamp(idx - left, min=0)
    if causal:
        upper = idx.clone()
    elif right < 0:
        upper = torch.full((seq_len,), seq_len - 1, dtype=torch.long, device=device)
    else:
        upper = torch.clamp(idx + right, max=seq_len - 1)
    return lower, upper


def _masked_attention(q, k, v, lower, upper, dropout_p, softmax_scale):
    """
    Full (non-chunked) windowed/causal attention via an explicit boolean mask.
    q, k, v: (B, H, S, D). lower/upper: (S,) per-query inclusive key bounds.
    Suitable when S is small enough that an S x S bool mask is affordable.
    """
    S = q.shape[-2]
    key_idx = torch.arange(S, device=q.device).view(1, S)          # (1, S)
    allowed = (key_idx >= lower.view(S, 1)) & (key_idx <= upper.view(S, 1))  # (S, S)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=allowed, dropout_p=dropout_p, scale=softmax_scale
    )


def _chunked_windowed_attention(q, k, v, left, right, causal, dropout_p, softmax_scale, chunk_size):
    """
    Memory-friendly windowed/causal attention: iterate over query chunks and
    only materialize the (small) slice of keys/values each chunk can attend
    to, with a per-row mask inside that slice to get exact bounds right.
    """
    B, H, S, D = q.shape
    out = torch.empty_like(q)
    lower_full, upper_full = _window_bounds(S, left, right, causal, q.device)

    for qs in range(0, S, chunk_size):
        qe = min(S, qs + chunk_size)

        # Superset of keys any query in [qs, qe) could need.
        k_start = int(lower_full[qs:qe].min().item())
        k_end = int(upper_full[qs:qe].max().item()) + 1

        q_chunk = q[:, :, qs:qe]
        k_chunk = k[:, :, k_start:k_end]
        v_chunk = v[:, :, k_start:k_end]

        # Per-row mask within this (small) chunk to enforce exact bounds.
        key_idx = torch.arange(k_start, k_end, device=q.device).view(1, -1)
        lower_c = lower_full[qs:qe].view(-1, 1)
        upper_c = upper_full[qs:qe].view(-1, 1)
        allowed = (key_idx >= lower_c) & (key_idx <= upper_c)

        out[:, :, qs:qe] = F.scaled_dot_product_attention(
            q_chunk, k_chunk, v_chunk, attn_mask=allowed, dropout_p=dropout_p, scale=softmax_scale
        )

    return out


def _sdpa_compat(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """
    Drop-in replacement for ``flash_attn_func``.

    Signature mirrors flash-attn 2.x. Input tensors are shaped (batch, seq, heads, dim).
    """
    if softcap not in (None, 0.0):
        raise NotImplementedError("softcap is not supported by the SDPA compatibility shim")
    if alibi_slopes is not None:
        raise NotImplementedError("alibi_slopes is not supported by the SDPA compatibility shim")
    if return_attn_probs:
        raise NotImplementedError("return_attn_probs is not supported by the SDPA compatibility shim")

    t0 = time.perf_counter()

    # flash-attn layout: (B, S, H, D)  →  SDPA layout: (B, H, S, D)
    q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))

    if isinstance(window_size, (tuple, list)):
        left, right = window_size
    else:
        left = right = int(window_size)

    S = q.shape[-2]
    no_window = left < 0 and right < 0

    if q.device.type == "cuda":
        if no_window:
            # Full attention; SDPA dispatches to a flash-attn kernel when available.
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale
            )
        else:
            # Windowed: chunk to bound peak memory, even though CUDA could
            # often afford a full S x S mask.
            out = _chunked_windowed_attention(
                q, k, v, left, right, causal, dropout_p, softmax_scale, chunk_size=2048
            )

    elif q.device.type == "mps":
        if no_window:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale
            )
        else:
            # MPS: chunked to avoid OOM on large sequences.
            chunk = max(left if left > 0 else 0, right if right > 0 else 0) or 512
            out = _chunked_windowed_attention(
                q, k, v, left, right, causal, dropout_p, softmax_scale, chunk_size=chunk
            )

    else:
        # CPU fallback — move to CPU in case tensors are on an unsupported device.
        q_cpu, k_cpu, v_cpu = q.cpu(), k.cpu(), v.cpu()
        if no_window:
            out = F.scaled_dot_product_attention(
                q_cpu, k_cpu, v_cpu, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale
            )
        else:
            out = _chunked_windowed_attention(
                q_cpu, k_cpu, v_cpu, left, right, causal, dropout_p, softmax_scale, chunk_size=1024
            )
        out = out.to(q.device)

    if q.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"  [compat] attn {elapsed:.3f}s  device={q.device.type}  S={S}  window=({left},{right})  causal={causal}")

    return out.permute(0, 2, 1, 3)


# ── Build stub modules ────────────────────────────────────────────────────────

def _patch():
    """Install the flash_attn stub into ``sys.modules``."""
    if "flash_attn" in sys.modules:
        return

    flash_attn = types.ModuleType("flash_attn")
    flash_attn.__version__ = "2.6.0"  # version Anemoi checks against
    flash_attn.flash_attn_func = _sdpa_compat  # top-level re-export, matches real package

    # flash_attn.layers.rotary  (imported but only used on specific GPU paths)
    layers_mod = types.ModuleType("flash_attn.layers")
    rotary_mod = types.ModuleType("flash_attn.layers.rotary")

    def _rotary_not_implemented(*args, **kwargs):
        raise NotImplementedError(
            "flash_attn.layers.rotary.RotaryEmbedding is not available in the SDPA "
            "compatibility shim; this code path requires real flash-attn on CUDA."
        )

    rotary_mod.RotaryEmbedding = _rotary_not_implemented
    layers_mod.rotary = rotary_mod
    flash_attn.layers = layers_mod

    # flash_attn.flash_attn_interface  (the one Anemoi actually calls)
    interface_mod = types.ModuleType("flash_attn.flash_attn_interface")
    interface_mod.flash_attn_func = _sdpa_compat
    flash_attn.flash_attn_interface = interface_mod

    sys.modules["flash_attn"] = flash_attn
    sys.modules["flash_attn.layers"] = layers_mod
    sys.modules["flash_attn.layers.rotary"] = rotary_mod
    sys.modules["flash_attn.flash_attn_interface"] = interface_mod


_patch()