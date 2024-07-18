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
# @version 1.0.0

from typing import Dict


class LlamaConfigNNTile(Dict):
    def __init__(
        self,
        vocab_size: int = 32000,
        vocab_embed_dim_tile: int = 4096,
        hidden_size: int = 4096,
        hidden_size_tile: int = 4096,
        max_position_embeddings: int = 2048,
        intermediate_size: int = 11008,
        intermediate_size_tile: int = 11008,
        rms_norm_eps: float = 1e-06,
        num_hidden_layers: int = 32,
        n_attention_head: int = 32,
        n_head_tile: int = 32,
        num_key_value_heads: int = 32,
        num_key_value_head_tile: int = 32,
        activation_function: str = "silu",
        flashattention: bool = True,
        use_redux: bool = False,
        dtype: str = "fp32",
        eos_token_id: int = 2,
        bos_token_id: int = 1,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rope_theta: float = 10000.,
    ):
        self["vocab_size"] = vocab_size
        self["vocab_embed_dim_tile"] = vocab_embed_dim_tile
        self["hidden_size"] = hidden_size
        self["hidden_size_tile"] = hidden_size_tile
        self["max_position_embeddings"] = max_position_embeddings
        self["intermediate_size"] = intermediate_size
        self["intermediate_size_tile"] = intermediate_size_tile
        self["rms_norm_eps"] = rms_norm_eps
        self["num_hidden_layers"] = num_hidden_layers
        self["n_attention_heads"] = n_attention_head
        self["n_head_tile"] = n_head_tile
        self["num_key_value_heads"] = num_key_value_heads
        self["num_key_value_head_tile"] = num_key_value_head_tile
        self["activation_function"] = activation_function
        self["flashattention"] = flashattention
        self["redux"] = use_redux
        self["dtype"] = dtype
        self["eos_token_id"] = eos_token_id
        self["bos_token_id"] = bos_token_id
        self["attention_bias"] = attention_bias
        self["attention_dropout"] = attention_dropout
        self["rope_theta"] = rope_theta

    def __getattr__(self, attr):
        return self[attr]
