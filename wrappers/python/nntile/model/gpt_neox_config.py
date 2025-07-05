# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/gpt_neox_config.py
# model config for NNTile version of GPT NeoX
#
# @version 1.1.0

from dataclasses import dataclass


@dataclass
class GPTNeoXConfig:
    vocab_size: int
    vocab_embed_dim_tile: int
    hidden_size: int
    hidden_size_tile: int
    intermediate_size: int
    intermediate_size_tile: int
    num_heads: int
    num_heads_tile: int
    activation_function: str = "gelu"
    dtype: str = "fp32"
    flash_attention: bool = False
    layer_norm_epsilon: float = 1e-5
    max_position_embeddings: int = 2048
    num_hidden_layers: int = 24
    redux: bool = False
    bos_token_id: int = 50256
    eos_token_id: int = 50256
    rotary_emb_base: int = 10000
    attention_bias: bool = False
    name: str = "gpt-neox"
