# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/llama_config.py
# LLaMa model config
#
# @version 1.1.0

from dataclasses import dataclass


@dataclass
class LlamaConfigNNTile:
    vocab_size: int
    vocab_embed_dim_tile: int
    hidden_size: int
    hidden_size_tile: int
    max_position_embeddings: int
    intermediate_size: int
    intermediate_size_tile: int
    n_attention_head: int
    n_head_tile: int
    num_key_value_heads: int
    activation_function: str = "silu"
    redux: bool = False
    dtype: str = "fp32"
    eos_token_id: int = 2
    bos_token_id: int = 1
    attention_bias: bool = False
    attention_dropout: float = 0.0
    rope_theta: float = 10000.
    rms_norm_eps: float = 1e-06
    num_hidden_layers: int = 1
    mlp_bias: bool = False
    flash_attention: bool = True
