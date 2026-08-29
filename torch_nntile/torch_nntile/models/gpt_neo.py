# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/models/gpt_neo.py
# GPT-Neo causal LM for device="nntile".

"""GPT-Neo stack mirroring ``nntile::model::gptneo``."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from torch_nntile.add_fiber import add_fiber
from torch_nntile.gemm import gemm
from torch_nntile.models.gpt2_minimal import make_causal_sdpa_mask
from torch_nntile.nn import Embedding, GELU, LayerNorm
from torch_nntile.nn.functional import add, mul_scalar
from torch_nntile.nn.linear import NntileLinear, prepare_sdpa_mask
from torch_nntile.nn.sdpa import nntile_model_transpose, sdpa_kernel


def make_local_causal_sdpa_mask(
    seq_len: int,
    window_size: int,
    device: torch.device | None = None,
) -> Tensor:
    """BOOL local-causal mask ``[seq, seq]`` for GPT-Neo local layers.

    Keep on CPU - nntile SDPA converts host bool/float masks; device-side
    ``where`` / float comparisons are not implemented yet.
    Allowed keys satisfy ``k <= q`` and ``q - k < window_size``.
    """
    q_idx = torch.arange(seq_len, dtype=torch.long, device="cpu")
    k_idx = torch.arange(seq_len, dtype=torch.long, device="cpu")
    mask = (
        (k_idx.unsqueeze(0) <= q_idx.unsqueeze(1))
        & ((q_idx.unsqueeze(1) - k_idx.unsqueeze(0)) < window_size)
    ).contiguous()
    if device is not None and device.type != "cpu":
        mask = mask.to(device)
    return mask


@dataclass
class GPTNeoConfig:
    vocab_size: int = 50257
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    max_position_embeddings: int = 2048
    head_dim: int = 128
    window_size: int = 256
    layer_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    attention_layers: list[str] = field(default_factory=list)
    eos_token_id: int = 50256
    bos_token_id: int = 50256
    name: str = "gpt-neo"

    def __post_init__(self) -> None:
        if (
            self.num_attention_heads > 0
            and self.hidden_size % self.num_attention_heads == 0
        ):
            self.head_dim = self.hidden_size // self.num_attention_heads
        if not self.attention_layers:
            # HF default: odd layers local, even global.
            self.attention_layers = [
                "local" if (i % 2 == 1) else "global"
                for i in range(self.num_hidden_layers)
            ]

    def validate(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "GPTNeoConfig: hidden_size must be divisible by "
                "num_attention_heads"
            )
        if len(self.attention_layers) != self.num_hidden_layers:
            raise ValueError(
                "GPTNeoConfig: attention_layers size must match "
                "num_hidden_layers"
            )

    def is_local_attention_layer(self, layer_id: int) -> bool:
        return self.attention_layers[layer_id] == "local"


class GPTNeoAttention(nn.Module):
    def __init__(self, config: GPTNeoConfig, *, local: bool = False) -> None:
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden = config.hidden_size
        self.local = local
        self.window_size = config.window_size
        self.q_weight = nn.Parameter(
            torch.empty(self.hidden, self.head_dim, self.n_heads)
        )
        self.k_weight = nn.Parameter(
            torch.empty(self.hidden, self.head_dim, self.n_heads)
        )
        self.v_weight = nn.Parameter(
            torch.empty(self.hidden, self.head_dim, self.n_heads)
        )
        self.o_weight = nn.Parameter(
            torch.empty(self.head_dim, self.n_heads, self.hidden)
        )
        self.o_bias = nn.Parameter(torch.zeros(self.hidden))
        # Host-built local masks (aux); keyed by seq_len.
        self._local_mask_cache: dict[int, Tensor] = {}
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for p in (
            self.q_weight,
            self.k_weight,
            self.v_weight,
            self.o_weight,
        ):
            nn.init.normal_(p, std=0.02)
        nn.init.zeros_(self.o_bias)

    def _project(self, x: Tensor, weight: Tensor) -> Tensor:
        proj = gemm(x, weight, ndim=1, batch_ndim=0)
        return nntile_model_transpose(proj, 1)

    def _output(self, attn_out: Tensor) -> Tensor:
        attn_t = nntile_model_transpose(attn_out, 3)
        out = gemm(attn_t, self.o_weight, ndim=2, batch_ndim=0)
        return add_fiber(self.o_bias, out, axis=out.dim() - 1, batch_ndim=0)

    def _cached_local_mask(self, seq_len: int, x: Tensor) -> Tensor:
        cached = self._local_mask_cache.get(seq_len)
        if cached is not None and cached.device == x.device:
            return cached
        mask = make_local_causal_sdpa_mask(
            seq_len,
            self.window_size,
            device=x.device,
        )
        self._local_mask_cache[seq_len] = mask
        return mask

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        *,
        is_causal: bool | None = None,
    ) -> Tensor:
        s = int(x.size(1))
        q = self._project(x, self.q_weight)
        k = self._project(x, self.k_weight)
        v = self._project(x, self.v_weight)
        # HF GPT-Neo scores are unscaled; cancel SDPA ``1/sqrt(d)``.
        # Use aten mul.Scalar - ``q * float`` may dispatch TensorxCPU-scalar.
        q = mul_scalar(q, float(self.head_dim) ** 0.5)
        # Local layers use a sliding causal window when the caller does not
        # supply an explicit mask (matches HF ``attention_layers`` / bias).
        if attn_mask is None and self.local:
            mask = self._cached_local_mask(s, x)
        elif is_causal is None:
            is_causal = attn_mask is None
            mask = prepare_sdpa_mask(
                attn_mask,
                x,
                q_len=s,
                is_causal=bool(is_causal),
            )
        else:
            mask = prepare_sdpa_mask(
                attn_mask,
                x,
                q_len=s,
                is_causal=bool(is_causal),
            )
        out = sdpa_kernel(
            q,
            k,
            v,
            mask=mask,
            batch_ndim=2,
        )
        return self._output(out)


class GPTNeoMLP(nn.Module):
    def __init__(self, config: GPTNeoConfig) -> None:
        super().__init__()
        self.c_fc = NntileLinear(
            config.hidden_size,
            config.intermediate_size,
        )
        self.c_proj = NntileLinear(
            config.intermediate_size,
            config.hidden_size,
        )
        self.act = GELU(approximate="tanh")

    def forward(self, x: Tensor) -> Tensor:
        return self.c_proj(self.act(self.c_fc(x)))


class GPTNeoBlock(nn.Module):
    def __init__(self, config: GPTNeoConfig, layer_id: int) -> None:
        super().__init__()
        local = config.is_local_attention_layer(layer_id)
        self.ln_1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = GPTNeoAttention(config, local=local)
        self.ln_2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = GPTNeoMLP(config)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        *,
        is_causal: bool | None = None,
    ) -> Tensor:
        residual = x
        x = self.ln_1(x)
        x = self.attn(x, attn_mask, is_causal=is_causal)
        x = add(residual, x)
        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        return add(residual, x)


class GPTNeoModel(nn.Module):
    def __init__(self, config: GPTNeoConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = Embedding(config.vocab_size, config.hidden_size)
        self.wpe = Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.h = nn.ModuleList(
            [
                GPTNeoBlock(config, i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self._position_ids_cache: dict[tuple[int, int], Tensor] = {}

    def _cached_position_ids(self, input_ids: Tensor) -> Tensor:
        batch, seq = int(input_ids.size(0)), int(input_ids.size(-1))
        key = (batch, seq)
        cached = self._position_ids_cache.get(key)
        if cached is not None and cached.device == input_ids.device:
            return cached
        position_ids = (
            torch.arange(seq, dtype=torch.long, device="cpu")
            .unsqueeze(0)
            .expand(batch, seq)
            .contiguous()
        )
        if input_ids.device.type != "cpu":
            position_ids = position_ids.to(input_ids.device)
        self._position_ids_cache[key] = position_ids
        return position_ids

    def clear_sequence_caches(self) -> None:
        self._position_ids_cache.clear()

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        if position_ids is None:
            position_ids = self._cached_position_ids(input_ids)
        x = add(self.wte(input_ids), self.wpe(position_ids))
        for block in self.h:
            x = block(x, attn_mask)
        return self.ln_f(x)


class GPTNeoCausal(nn.Module):
    """GPT-Neo causal LM (``nntile::model::gptneo::GptneoCausal``)."""

    def __init__(self, config: GPTNeoConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.transformer = GPTNeoModel(config)
        self.lm_head = NntileLinear(
            config.hidden_size, config.vocab_size, bias=False
        )
        # Weight tying intentionally unsupported (independent lm_head).

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        hidden = self.transformer(input_ids, position_ids, attn_mask)
        return self.lm_head(hidden)


__all__ = [
    "GPTNeoAttention",
    "GPTNeoBlock",
    "GPTNeoCausal",
    "GPTNeoConfig",
    "GPTNeoMLP",
    "GPTNeoModel",
    "make_causal_sdpa_mask",
    "make_local_causal_sdpa_mask",
]
