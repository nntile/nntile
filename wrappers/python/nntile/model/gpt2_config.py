# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/gpt2_config.py
# GPT2 model config
#
# @version 1.0.0

from dataclasses import dataclass


@dataclass
class GPT2ConfigNNTile:
    vocab_size: int
    vocab_embed_dim_tile: int
    embed_dim: int
    embed_dim_tile: int
    max_position_embeddings: int
    inner_dim: int
    inner_dim_tile: int
    layer_norm_epsilon: float
    num_hidden_layers: int
    n_head: int
    n_head_tile: int
    activation_function: str
    flashattention: bool = False
    use_redux: bool = False
    dtype: str = "fp32",
    eos_token_id: int = 50256
    