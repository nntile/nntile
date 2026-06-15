"""Contiguous tensor payloads for graph bind and safetensors."""

from __future__ import annotations

import numpy as np


def as_bind_float32(arr: np.ndarray) -> np.ndarray:
    """Return a contiguous float32 buffer for graph tensor bind."""
    return np.ascontiguousarray(arr, dtype=np.float32)


def linear_from_conv1d(conv_weight: np.ndarray) -> np.ndarray:
    """HF Conv1D ``(in, out)`` → graph Linear ``(out, in)``."""
    return as_bind_float32(conv_weight.T)


def linear_weight(weight: np.ndarray) -> np.ndarray:
    """PyTorch Linear weight already ``(out, in)``."""
    return as_bind_float32(weight)


def gpt2_attn_qkv_weight(
    c_attn_col: np.ndarray, hidden: int, n_heads: int, head_size: int,
) -> np.ndarray:
    """HF ``c_attn`` Q/K/V column → graph ``(hidden, head_size, n_heads)``."""
    w = np.asarray(c_attn_col, dtype=np.float32).T.reshape(
        n_heads, head_size, hidden)
    return as_bind_float32(w.transpose(2, 1, 0))


def gpt2_attn_o_weight(
    c_proj_weight: np.ndarray, hidden: int, n_heads: int, head_size: int,
) -> np.ndarray:
    """HF ``c_proj`` → graph ``(head_size, n_heads, hidden)``."""
    w = c_proj_weight.T.reshape(hidden, n_heads, head_size)
    return as_bind_float32(w.transpose(2, 1, 0))


def linear_attn_qkv_weight(
    weight: np.ndarray, hidden: int, n_heads: int, head_size: int,
) -> np.ndarray:
    """HF Linear ``(out, in)`` → graph ``(hidden, head_size, n_heads)``."""
    w = np.asarray(weight, dtype=np.float32).reshape(n_heads, head_size, hidden)
    return as_bind_float32(w.transpose(2, 1, 0))


def linear_attn_o_weight(
    weight: np.ndarray, hidden: int, n_heads: int, head_size: int,
) -> np.ndarray:
    """HF Linear ``(out, in)`` → graph ``(head_size, n_heads, hidden)``."""
    w = np.asarray(weight, dtype=np.float32).reshape(hidden, n_heads, head_size)
    return as_bind_float32(w.transpose(2, 1, 0))


def gptneox_attn_qkv_weight(qkv_slice: np.ndarray) -> np.ndarray:
    """``(n_heads, head_size, hidden)`` QKV slice → graph layout."""
    return as_bind_float32(
        np.asarray(qkv_slice, dtype=np.float32).transpose(2, 1, 0))


def gptneox_attn_o_weight(
    weight: np.ndarray, n_heads: int, head_size: int, hidden: int,
) -> np.ndarray:
    """HF ``dense`` → graph ``(head_size, n_heads, hidden)``."""
    w = weight.reshape(hidden, n_heads, head_size)
    return as_bind_float32(w.transpose(2, 1, 0))


def _rotate_tensor_in(x: np.ndarray, axis: int) -> np.ndarray:
    """Interleave RoPE pairs on ``axis`` (Llama Q/K layout)."""
    if axis == 0:
        new_shape = (1, x.shape[0], int(np.prod(x.shape[1:])))
    elif axis == x.ndim - 1:
        new_shape = (int(np.prod(x.shape[:-1])), x.shape[-1], 1)
    else:
        new_shape = (
            int(np.prod(x.shape[:axis])),
            x.shape[axis],
            int(np.prod(x.shape[axis + 1:])),
        )
    x_reshaped = x.reshape(new_shape)
    mid = x.shape[axis] // 2
    y_reshaped = np.empty_like(x_reshaped)
    y_reshaped[:, 0::2, :] = x_reshaped[:, :mid, :]
    y_reshaped[:, 1::2, :] = x_reshaped[:, mid:, :]
    return y_reshaped.reshape(x.shape)


def _reverse_axes(arr: np.ndarray) -> np.ndarray:
    """Map head layout to graph labels by reversing axes."""
    axes = tuple(range(arr.ndim - 1, -1, -1))
    return as_bind_float32(np.asarray(arr, dtype=np.float32).transpose(axes))


def llama_attention_weights(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    o: np.ndarray,
    *,
    hidden: int,
    n_heads: int,
    kv_heads: int,
    head_size: int,
    kv_group_size: int,
    use_gqa: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """HF Q/K/V/O → graph ``LlamaAttention`` layouts."""
    if use_gqa:
        q_arr = q.reshape(kv_heads, kv_group_size, head_size, hidden).transpose(
            1, 0, 2, 3)
        o_arr = np.moveaxis(
            o.reshape(hidden, kv_heads, kv_group_size, head_size), 1, 2)
    else:
        q_arr = q.reshape(n_heads, head_size, hidden)
        o_arr = o.reshape(hidden, n_heads, head_size)

    k_arr = k.reshape(kv_heads, head_size, hidden)
    v_arr = v.reshape(kv_heads, head_size, hidden)

    q_arr = _rotate_tensor_in(
        np.asarray(q_arr, dtype=np.float32), q_arr.ndim - 2)
    k_arr = _rotate_tensor_in(np.asarray(k_arr, dtype=np.float32), 1)

    return (
        _reverse_axes(q_arr),
        _reverse_axes(k_arr),
        _reverse_axes(v_arr),
        _reverse_axes(o_arr),
    )
