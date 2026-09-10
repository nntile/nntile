# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/models/dit.py
# DiT (AdaLN-Zero) for device="nntile" via torch_nntile.nn only.

"""Diffusers-style DiT on classic NNTile kernels (no new low-level ops).

Patchify / unpatchify and Fourier timestep tables are host layout helpers
(same idea as Llama RoPE ``sin``/``cos`` tables). The TensorGraph is
Linear / Embedding / LayerNorm / SiLU / GELU / SDPA / add / mul /
``scale_slice``. AdaLN-Zero uses six Linears (classic ``narrow`` is
wrong for ``start != 0``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from torch_nntile.nn import Embedding, GELU, LayerNorm, SiLU
from torch_nntile.nn.functional import add, mul, scale_slice
from torch_nntile.nn.linear import (
    NntileAttentionOutput,
    NntileLinear,
    NntileQKVProjection,
)
from torch_nntile.nn.sdpa import sdpa_kernel

_TIMESTEP_FREQ_DIM = 256


def timestep_embedding_table(
    num_embeds: int,
    dim: int = _TIMESTEP_FREQ_DIM,
    *,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 1.0,
    max_period: int = 10000,
) -> Tensor:
    """Host Fourier table for integer timesteps ``0 .. num_embeds-1``."""
    timesteps = torch.arange(num_embeds, dtype=torch.float32)
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(half, dtype=torch.float32)
    exponent = exponent / (half - downscale_freq_shift)
    freqs = torch.exp(exponent)
    args = timesteps[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half:], emb[:, :half]], dim=-1)
    return emb


def sincos_2d_pos_embed(embed_dim: int, grid_size: int) -> Tensor:
    """``[grid_size**2, embed_dim]`` 2D sin/cos (Diffusers / MAE layout)."""
    if embed_dim % 4 != 0:
        raise ValueError("sincos_2d_pos_embed: embed_dim must be 4k")
    half = embed_dim // 2
    grid = torch.arange(grid_size, dtype=torch.float32)
    # numpy meshgrid default ``xy``: first return is width (x).
    wx, wy = torch.meshgrid(grid, grid, indexing="xy")

    def _1d(pos: Tensor, dim: int) -> Tensor:
        omega = torch.arange(dim // 2, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / (dim / 2)))
        out = pos.reshape(-1, 1) * omega[None, :]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    return torch.cat([_1d(wx, half), _1d(wy, half)], dim=1)


def patchify_nchw(images: Tensor, patch_size: int) -> Tensor:
    """``[B, C, H, W]`` → ``[B, T, C*p*p]`` (host layout helper)."""
    if images.dim() != 4:
        raise ValueError("patchify_nchw expects NCHW")
    batch, channels, height, width = images.shape
    if height % patch_size or width % patch_size:
        raise ValueError("H and W must be divisible by patch_size")
    gh, gw = height // patch_size, width // patch_size
    patches = images.reshape(batch, channels, gh, patch_size, gw, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
    return patches.reshape(batch, gh * gw, channels * patch_size * patch_size)


def unpatchify_nchw(
    tokens: Tensor,
    *,
    patch_size: int,
    out_channels: int,
    grid_h: int,
    grid_w: int,
) -> Tensor:
    """``[B, T, p*p*C]`` → ``[B, C, H, W]`` (host layout helper)."""
    batch = tokens.shape[0]
    patches = tokens.reshape(
        batch,
        grid_h,
        grid_w,
        patch_size,
        patch_size,
        out_channels,
    )
    images = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
    return images.reshape(
        batch,
        out_channels,
        grid_h * patch_size,
        grid_w * patch_size,
    )


def nchw_to_unpatchify_tokens(images: Tensor, patch_size: int) -> Tensor:
    """Inverse of ``unpatchify_nchw`` (layout ``p, p, C``). Host helper."""
    if images.dim() != 4:
        raise ValueError("nchw_to_unpatchify_tokens expects NCHW")
    batch, channels, height, width = images.shape
    if height % patch_size or width % patch_size:
        raise ValueError("H and W must be divisible by patch_size")
    gh, gw = height // patch_size, width // patch_size
    return (
        images.reshape(batch, channels, gh, patch_size, gw, patch_size)
        .permute(0, 2, 4, 3, 5, 1)
        .contiguous()
        .reshape(batch, gh * gw, patch_size * patch_size * channels)
    )


@dataclass
class DiTConfig:
    sample_size: int = 16
    patch_size: int = 2
    in_channels: int = 3
    out_channels: int | None = None
    num_layers: int = 2
    num_attention_heads: int = 2
    attention_head_dim: int = 8
    attention_bias: bool = True
    activation_fn: str = "gelu-approximate"
    num_embeds_ada_norm: int = 1000
    norm_eps: float = 1e-5
    mlp_ratio: float = 4.0

    @property
    def hidden_size(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def patch_dim(self) -> int:
        channels = self.in_channels
        return channels * self.patch_size * self.patch_size

    @property
    def grid_size(self) -> int:
        return self.sample_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_size * self.grid_size

    def validate(self) -> None:
        if self.sample_size % self.patch_size != 0:
            raise ValueError("sample_size must be divisible by patch_size")
        if self.activation_fn not in ("gelu-approximate", "gelu"):
            raise ValueError(
                "DiTConfig: activation_fn must be gelu-approximate or gelu"
            )


def _identity_layer_norm(hidden_size: int, eps: float) -> LayerNorm:
    """HF AdaLN uses ``elementwise_affine=False``.

    Classic ``_C.layer_norm`` cannot take undefined weight/bias (autograd
    inspects every Tensor argument). Frozen ones/zeros match no-affine.
    """
    norm = LayerNorm(hidden_size, eps=eps, elementwise_affine=True)
    nn.init.ones_(norm.weight)
    nn.init.zeros_(norm.bias)
    norm.weight.requires_grad_(False)
    norm.bias.requires_grad_(False)
    return norm


def _ada_modulate(x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    """``x * (1 + scale) + shift`` with ``scale``/``shift`` ``[B, H]``."""
    seq = int(x.size(1))
    scale_bt = scale_slice(scale, 1, seq)
    shift_bt = scale_slice(shift, 1, seq)
    return add(add(x, mul(x, scale_bt)), shift_bt)


def _ada_gate(x: Tensor, gate: Tensor) -> Tensor:
    """``gate[:, None] * x`` with ``gate`` ``[B, H]``."""
    return mul(x, scale_slice(gate, 1, int(x.size(1))))


_ADA_MOD_NAMES = (
    "shift_msa",
    "scale_msa",
    "gate_msa",
    "shift_mlp",
    "scale_mlp",
    "gate_mlp",
)


class CombinedTimestepLabelEmbeddings(nn.Module):
    """Class Embedding + Fourier-timestep MLP (host sin/cos table)."""

    def __init__(self, num_classes: int, embedding_dim: int) -> None:
        super().__init__()
        self.time_freq = Embedding(num_classes, _TIMESTEP_FREQ_DIM)
        with torch.no_grad():
            self.time_freq.weight.copy_(timestep_embedding_table(num_classes))
        self.time_freq.weight.requires_grad_(False)
        self.time_linear_1 = NntileLinear(_TIMESTEP_FREQ_DIM, embedding_dim)
        self.time_act = SiLU()
        self.time_linear_2 = NntileLinear(embedding_dim, embedding_dim)
        self.class_embed = Embedding(num_classes, embedding_dim)

    def forward(self, timestep: Tensor, class_labels: Tensor) -> Tensor:
        t_emb = self.time_linear_2(
            self.time_act(self.time_linear_1(self.time_freq(timestep)))
        )
        return add(t_emb, self.class_embed(class_labels))


class AdaLayerNormZero(nn.Module):
    """HF AdaLN-Zero via six ``Linear(H, H)`` (not fused+chunk)."""

    def __init__(self, hidden_size: int, num_embeds: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.emb = CombinedTimestepLabelEmbeddings(num_embeds, hidden_size)
        self.silu = SiLU()
        for name in _ADA_MOD_NAMES:
            setattr(self, name, NntileLinear(hidden_size, hidden_size))
        # Diffusers AdaLayerNormZero hardcodes eps=1e-6.
        self.norm = _identity_layer_norm(hidden_size, 1e-6)

    def _mods(self, cond: Tensor) -> tuple[Tensor, ...]:
        return tuple(getattr(self, name)(cond) for name in _ADA_MOD_NAMES)

    def forward(
        self,
        x: Tensor,
        timestep: Tensor,
        class_labels: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        cond = self.silu(self.emb(timestep, class_labels))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self._mods(cond)
        )
        x = _ada_modulate(self.norm(x), scale_msa, shift_msa)
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class DiTAttention(nn.Module):
    def __init__(self, config: DiTConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        heads = config.num_attention_heads
        head_dim = config.attention_head_dim
        bias = config.attention_bias
        self.query = NntileQKVProjection(hidden, head_dim, heads, bias=bias)
        self.key = NntileQKVProjection(hidden, head_dim, heads, bias=bias)
        self.value = NntileQKVProjection(hidden, head_dim, heads, bias=bias)
        self.out = NntileAttentionOutput(head_dim, heads, hidden, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(
            sdpa_kernel(
                self.query(x),
                self.key(x),
                self.value(x),
                mask=None,
                batch_ndim=2,
            )
        )


class DiTMlp(nn.Module):
    def __init__(self, config: DiTConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        inner = int(hidden * config.mlp_ratio)
        self.fc1 = NntileLinear(hidden, inner)
        approx = (
            "tanh" if config.activation_fn == "gelu-approximate" else "none"
        )
        self.act = GELU(approximate=approx)
        self.fc2 = NntileLinear(inner, hidden)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    def __init__(self, config: DiTConfig) -> None:
        super().__init__()
        self.norm1 = AdaLayerNormZero(
            config.hidden_size,
            config.num_embeds_ada_norm,
        )
        self.attn = DiTAttention(config)
        self.norm2 = _identity_layer_norm(config.hidden_size, config.norm_eps)
        self.mlp = DiTMlp(config)

    def forward(
        self,
        x: Tensor,
        timestep: Tensor,
        class_labels: Tensor,
    ) -> Tensor:
        h, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            x, timestep, class_labels
        )
        x = add(x, _ada_gate(self.attn(h), gate_msa))
        h = _ada_modulate(self.norm2(x), scale_mlp, shift_mlp)
        return add(x, _ada_gate(self.mlp(h), gate_mlp))


class DiT(nn.Module):
    """AdaLN-Zero DiT. Tokens are ``[B, T, C*p*p]`` (patchify on host)."""

    def __init__(self, config: DiTConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        hidden = config.hidden_size
        out_ch = (
            config.in_channels
            if config.out_channels is None
            else config.out_channels
        )
        self.out_channels = out_ch
        self.patch_embed = NntileLinear(config.patch_dim, hidden)
        self.pos_embed = nn.Parameter(
            sincos_2d_pos_embed(hidden, config.grid_size),
            requires_grad=False,
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(config) for _ in range(config.num_layers)]
        )
        self.norm_out = _identity_layer_norm(hidden, 1e-6)
        self.proj_out_shift = NntileLinear(hidden, hidden)
        self.proj_out_scale = NntileLinear(hidden, hidden)
        self.proj_out_silu = SiLU()
        self.proj_out_2 = NntileLinear(
            hidden, config.patch_size * config.patch_size * out_ch
        )

    def _add_pos(self, tokens: Tensor) -> Tensor:
        pos = self.pos_embed
        if pos.device != tokens.device:
            pos = pos.to(tokens.device)
        pos_b = scale_slice(pos, 0, int(tokens.size(0)))
        if int(pos_b.size(1)) != int(tokens.size(1)):
            raise ValueError("pos_embed T does not match tokens")
        return add(tokens, pos_b)

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        class_labels: Tensor,
    ) -> Tensor:
        """``hidden_states``: ``[B, T, C*p*p]`` (patchify NCHW on host)."""
        if hidden_states.dim() == 4:
            if hidden_states.device.type == "nntile":
                raise ValueError(
                    "DiT: patchify NCHW on host, then .to('nntile'); "
                    "no conv kernel"
                )
            hidden_states = patchify_nchw(
                hidden_states, self.config.patch_size
            )
        param_dev = next(self.parameters()).device
        if hidden_states.device != param_dev:
            raise ValueError(
                "DiT: tokens must be on the same device as the model "
                "(patchify on host, then .to('nntile'))"
            )
        tokens = self._add_pos(self.patch_embed(hidden_states))
        for block in self.blocks:
            tokens = block(tokens, timestep, class_labels)
        cond = self.blocks[0].norm1.emb(timestep, class_labels)
        cond = self.proj_out_silu(cond)
        shift = self.proj_out_shift(cond)
        scale = self.proj_out_scale(cond)
        tokens = _ada_modulate(self.norm_out(tokens), scale, shift)
        return self.proj_out_2(tokens)


__all__ = [
    "AdaLayerNormZero",
    "CombinedTimestepLabelEmbeddings",
    "DiT",
    "DiTBlock",
    "DiTConfig",
    "nchw_to_unpatchify_tokens",
    "patchify_nchw",
    "sincos_2d_pos_embed",
    "timestep_embedding_table",
    "unpatchify_nchw",
]
