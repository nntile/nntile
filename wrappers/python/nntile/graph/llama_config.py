# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/graph/llama_config.py
# Python wrapper for Graph API LlamaConfig.
#
# @version 1.1.0

from __future__ import annotations

from ..nntile_graph import llama as _llama


def make_llama_config(
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    *,
    vocab_size: int = 32000,
    intermediate_size: int = 11008,
    num_hidden_layers: int = 32,
    max_position_embeddings: int = 2048,
    rms_norm_eps: float = 1e-6,
    rope_theta: float = 10000.0,
    attention_bias: bool = False,
    mlp_bias: bool = False,
) -> _llama.LlamaConfig:
    """Create a C++ LlamaConfig with validated head_dim."""
    cfg = _llama.LlamaConfig()
    cfg.hidden_size = hidden_size
    cfg.num_attention_heads = num_attention_heads
    cfg.num_key_value_heads = num_key_value_heads
    cfg.vocab_size = vocab_size
    cfg.intermediate_size = intermediate_size
    cfg.num_hidden_layers = num_hidden_layers
    cfg.max_position_embeddings = max_position_embeddings
    cfg.rms_norm_eps = rms_norm_eps
    cfg.rope_theta = rope_theta
    cfg.attention_bias = attention_bias
    cfg.mlp_bias = mlp_bias
    cfg.compute_head_dim()
    cfg.validate()
    return cfg


def from_torch_config(torch_config) -> _llama.LlamaConfig:
    """Create a C++ LlamaConfig from a HuggingFace LlamaConfig."""
    return make_llama_config(
        hidden_size=torch_config.hidden_size,
        num_attention_heads=torch_config.num_attention_heads,
        num_key_value_heads=torch_config.num_key_value_heads,
        vocab_size=torch_config.vocab_size,
        intermediate_size=torch_config.intermediate_size,
        num_hidden_layers=torch_config.num_hidden_layers,
        max_position_embeddings=torch_config.max_position_embeddings,
        rms_norm_eps=torch_config.rms_norm_eps,
        rope_theta=torch_config.rope_theta,
        attention_bias=getattr(torch_config, "attention_bias", False),
        mlp_bias=getattr(torch_config, "mlp_bias", False),
    )
