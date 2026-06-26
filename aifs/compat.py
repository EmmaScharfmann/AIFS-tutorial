"""
Import this module *before* importing anything from ``anemoi`` to allow the compatibility with available GPU / CPU
"""

import sys
import time
import types

import torch
import torch.nn.functional as F


# ── SDPA-based attention replacement ─────────────────────────────────────────

def _sdpa_compat(q, k, v, causal=False, window_size=(-1, -1), dropout_p=0.0, softcap=None, alibi_slopes=None):
    """
    Drop-in replacement for ``flash_attn_func``.

    Parameters mirror the flash-attn 2.x signature that Anemoi calls.
    Input tensors are shaped ``(batch, seq, heads, dim)``.
    """
    t0 = time.perf_counter()

    # flash-attn layout: (B, S, H, D)  →  SDPA layout: (B, H, S, D)
    q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))
    ws = window_size[0] if isinstance(window_size, (tuple, list)) else int(window_size)

    if q.device.type == "cuda":
        # Full global attention; SDPA dispatches to flash-attn kernel when available
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

    elif ws > 0:
        # MPS: chunked sliding-window attention (avoids OOM on large sequences)
        B, H, S, D = q.shape
        out = torch.zeros_like(q)
        for i in range(0, S, ws):
            k_start = max(0, i - ws)
            k_end   = min(S, i + ws + ws)
            out[:, :, i : i + ws] = F.scaled_dot_product_attention(
                q[:, :, i : i + ws],
                k[:, :, k_start:k_end],
                v[:, :, k_start:k_end],
                dropout_p=dropout_p,
            )

    else:
        # CPU fallback — move to CPU in case tensors are on an unsupported device
        out = F.scaled_dot_product_attention(
            q.cpu(), k.cpu(), v.cpu(), dropout_p=dropout_p
        ).to(q.device)

    elapsed = time.perf_counter() - t0
    print(f"  [compat] attn {elapsed:.2f}s  device={q.device.type}  ws={ws}")

    return out.permute(0, 2, 1, 3)


# ── Build stub modules ────────────────────────────────────────────────────────

def _patch():
    """Install the flash_attn stub into ``sys.modules``."""
    if "flash_attn" in sys.modules:
        return

    flash_attn = types.ModuleType("flash_attn")
    flash_attn.__version__ = "2.6.0"  # version Anemoi checks against

    # flash_attn.layers.rotary  (imported but only used on specific GPU paths)
    layers_mod = types.ModuleType("flash_attn.layers")
    rotary_mod = types.ModuleType("flash_attn.layers.rotary")
    rotary_mod.RotaryEmbedding = None
    layers_mod.rotary = rotary_mod
    flash_attn.layers = layers_mod

    # flash_attn.flash_attn_interface  (the one Anemoi actually calls)
    interface_mod = types.ModuleType("flash_attn.flash_attn_interface")
    interface_mod.flash_attn_func = _sdpa_compat
    flash_attn.flash_attn_interface = interface_mod

    sys.modules["flash_attn"]                    = flash_attn
    sys.modules["flash_attn.layers"]             = layers_mod
    sys.modules["flash_attn.layers.rotary"]      = rotary_mod
    sys.modules["flash_attn.flash_attn_interface"] = interface_mod


_patch()
