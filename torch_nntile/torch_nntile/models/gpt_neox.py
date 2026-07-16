# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/models/gpt_neox.py
# GPT-NeoX causal LM for device="nntile".

"""GPT-NeoX stack mirroring ``nntile::model::gptneox``.

Uses RoPE and parallel residual. Activations stay on ``device=nntile``;
position / RoPE tables are cached uploads (deleted NNGraph bind_data pattern).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from torch_nntile.gemm import gemm
from torch_nntile.models.gpt2_minimal import make_causal_sdpa_mask
from torch_nntile.nn.linear import NntileLinear, prepare_sdpa_mask
from torch_nntile.nn.sdpa import nntile_model_transpose, sdpa_kernel

try:
    from torch_nntile.rope import rope, rope_sin_cos_from_position_ids
except ImportError:  # pragma: no cover
    def rope(sin: Tensor, cos: Tensor, x: Tensor) -> Tensor:
        return x

    def rope_sin_cos_from_position_ids(
        position_ids: Tensor,
        head_dim: int,
        *,
        rope_theta: float = 10000.0,
    ) -> tuple[Tensor, Tensor]:
        del rope_theta
        b, s = position_ids.shape
        half = head_dim // 2
        z = torch.zeros(
            b, s, half, dtype=torch.float32, device=position_ids.device
        )
        return z, torch.ones_like(z)


@dataclass
class GPTNeoXConfig:
    vocab_size: int = 50280
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    max_position_embeddings: int = 2048
    head_dim: int = 64
    layer_norm_eps: float = 1e-5
    rotary_pct: float = 0.25
    rotary_emb_base: float = 10000.0
    use_parallel_residual: bool = True
    attention_bias: bool = True
    tie_word_embeddings: bool = False
    attention_layers: list[str] = field(default_factory=list)
    eos_token_id: int = 50256
    bos_token_id: int = 50256
    name: str = "gpt-neox"

    def __post_init__(self) -> None:
        if (
            self.num_attention_heads > 0
            and self.hidden_size % self.num_attention_heads == 0
        ):
            self.head_dim = self.hidden_size // self.num_attention_heads

    def validate(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "GPTNeoXConfig: hidden_size must be divisible by "
                "num_attention_heads"
            )

    @property
    def rotary_ndims(self) -> int:
        if self.rotary_pct <= 0:
            return 0
        dim = round(self.head_dim * self.rotary_pct)
        dim = max(2, dim)
        if dim % 2 != 0:
            dim -= 1
        if dim > self.head_dim:
            dim = self.head_dim
            if dim % 2 != 0:
                dim -= 1
        return dim


class GPTNeoXAttention(nn.Module):
    def __init__(self, config: GPTNeoXConfig) -> None:
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden = config.hidden_size
        self.rotary_ndims = config.rotary_ndims
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
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for p in (
            self.q_weight,
            self.k_weight,
            self.v_weight,
            self.o_weight,
        ):
            nn.init.normal_(p, std=0.02)

    def _project(self, x: Tensor, weight: Tensor) -> Tensor:
        proj = gemm(x, weight, ndim=1, batch_ndim=0)
        return nntile_model_transpose(proj, 1)

    def _output(self, attn_out: Tensor) -> Tensor:
        attn_t = nntile_model_transpose(attn_out, 3)
        return gemm(attn_t, self.o_weight, ndim=2, batch_ndim=0)

    def _apply_rope(self, x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
        # Partial RoPE on-device via narrow + rope + cat (nntile kernels).
        rot = self.rotary_ndims
        if rot <= 0:
            return x
        if sin.device != x.device:
            sin = sin.to(x.device)
            cos = cos.to(x.device)
        x_rot = torch.narrow(x, -1, 0, rot)
        x_pass = torch.narrow(x, -1, rot, self.head_dim - rot)
        x_rot = rope(sin, cos, x_rot)
        return torch.cat([x_rot, x_pass], dim=-1)

    def forward(
        self,
        x: Tensor,
        sin: Tensor | None = None,
        cos: Tensor | None = None,
        attn_mask: Tensor | None = None,
        *,
        is_causal: bool = True,
    ) -> Tensor:
        s = int(x.size(1))
        q = self._project(x, self.q_weight)
        k = self._project(x, self.k_weight)
        v = self._project(x, self.v_weight)
        if sin is not None and cos is not None:
            q = self._apply_rope(q, sin, cos)
            k = self._apply_rope(k, sin, cos)
        mask = prepare_sdpa_mask(
            attn_mask,
            x,
            q_len=s,
            is_causal=is_causal,
        )
        out = sdpa_kernel(
            q,
            k,
            v,
            mask=mask,
            batch_ndim=2,
        )
        return self._output(out)


class GPTNeoXMLP(nn.Module):
    def __init__(self, config: GPTNeoXConfig) -> None:
        super().__init__()
        self.dense_h_to_4h = NntileLinear(
            config.hidden_size, config.intermediate_size
        )
        self.dense_4h_to_h = NntileLinear(
            config.intermediate_size, config.hidden_size
        )
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.dense_4h_to_h(self.act(self.dense_h_to_4h(x)))


class GPTNeoXLayer(nn.Module):
    def __init__(self, config: GPTNeoXConfig) -> None:
        super().__init__()
        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.attention = GPTNeoXAttention(config)
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.mlp = GPTNeoXMLP(config)

    def forward(
        self,
        x: Tensor,
        sin: Tensor | None = None,
        cos: Tensor | None = None,
        attn_mask: Tensor | None = None,
        *,
        is_causal: bool = True,
    ) -> Tensor:
        if self.use_parallel_residual:
            normed = self.input_layernorm(x)
            attn_out = self.attention(
                normed, sin, cos, attn_mask, is_causal=is_causal
            )
            mlp_out = self.mlp(self.post_attention_layernorm(x))
            return x + attn_out + mlp_out
        residual = x
        x = self.input_layernorm(x)
        x = self.attention(x, sin, cos, attn_mask, is_causal=is_causal)
        x = residual + x
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual + x


class GPTNeoXModel(nn.Module):
    def __init__(self, config: GPTNeoXConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [GPTNeoXLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self._position_ids_cache: dict[tuple[int, int], Tensor] = {}
        self._rope_cache: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}

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

    def _cached_rope(
        self, position_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return sin/cos on ``position_ids.device`` (built once, then reused).

        Matches deleted NNGraph: RoPE tables are prepared on the host and
        bound once for training — never recomputed from activations.
        """
        batch, seq = int(position_ids.size(0)), int(position_ids.size(-1))
        key = (batch, seq)
        cached = self._rope_cache.get(key)
        if cached is not None and cached[0].device == position_ids.device:
            return cached
        pos_host = (
            torch.arange(seq, dtype=torch.long, device="cpu")
            .unsqueeze(0)
            .expand(batch, seq)
            .contiguous()
        )
        sin, cos = rope_sin_cos_from_position_ids(
            pos_host,
            self.config.rotary_ndims,
            rope_theta=self.config.rotary_emb_base,
        )
        if position_ids.device.type != "cpu":
            sin = sin.to(position_ids.device)
            cos = cos.to(position_ids.device)
        self._rope_cache[key] = (sin, cos)
        return sin, cos

    def clear_sequence_caches(self) -> None:
        self._position_ids_cache.clear()
        self._rope_cache.clear()

    def warm_sequence_caches(
        self,
        *,
        batch_sizes: list[int] | tuple[int, ...],
        seq_len: int,
        device: torch.device | str,
    ) -> None:
        """Prepare position_ids / RoPE tables once for training reuse."""
        device = torch.device(device)
        for batch in sorted({int(b) for b in batch_sizes}):
            if batch < 1:
                raise ValueError(f"batch size must be >= 1, got {batch}")
            probe = torch.empty(
                (batch, seq_len), dtype=torch.long, device=device
            )
            pos = self._cached_position_ids(probe)
            if self.config.rotary_ndims > 0:
                self._cached_rope(pos)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
        sin: Tensor | None = None,
        cos: Tensor | None = None,
        attn_mask: Tensor | None = None,
        *,
        is_causal: bool = True,
    ) -> Tensor:
        if position_ids is None:
            position_ids = self._cached_position_ids(input_ids)
        rotary_dim = self.config.rotary_ndims
        if (sin is None or cos is None) and rotary_dim > 0:
            sin, cos = self._cached_rope(position_ids)
        x = self.embed_in(input_ids)
        for layer in self.layers:
            x = layer(x, sin, cos, attn_mask, is_causal=is_causal)
        return self.final_layer_norm(x)


class GPTNeoXCausal(nn.Module):
    """GPT-NeoX causal LM (``nntile::model::gptneox::GptneoxCausal``)."""

    def __init__(self, config: GPTNeoXConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.gpt_neox = GPTNeoXModel(config)
        self.embed_out = NntileLinear(
            config.hidden_size, config.vocab_size, bias=False
        )
        # Weight tying intentionally unsupported (independent embed_out).

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
        sin: Tensor | None = None,
        cos: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        hidden = self.gpt_neox(
            input_ids,
            position_ids=position_ids,
            sin=sin,
            cos=cos,
            attn_mask=attn_mask,
            is_causal=True,
        )
        return self.embed_out(hidden)


__all__ = [
    "GPTNeoXAttention",
    "GPTNeoXCausal",
    "GPTNeoXConfig",
    "GPTNeoXLayer",
    "GPTNeoXMLP",
    "GPTNeoXModel",
    "make_causal_sdpa_mask",
]
