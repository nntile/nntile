# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/gpt_neox_block.py
# GPTNeoXBlock submodule of NNTile Python package
#
# @version 1.1.0

from typing import Optional

import numpy as np
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXConfig as ConfigTorch, GPTNeoXLayer as BlockTorch)

from nntile.layer.cache_utils import KVCache
from nntile.tensor import TensorMoments

from ..layer.add import Add
from ..layer.gpt_neox_attention import GPTNeoXAttention
from ..layer.layer_norm import LayerNorm
from .base_model import BaseModel
from .gpt_neox_config import GPTNeoXConfig
from .gpt_neox_mlp import GPTNeoXMLP


class GPTNeoXBlock(BaseModel):
    gpt_neox_mlp: GPTNeoXMLP
    input_norm: LayerNorm
    post_attn_norm: LayerNorm
    post_attn_add: Add
    attention_layer: GPTNeoXAttention
    post_mlp_add: Add
    config: GPTNeoXConfig

    # Construct model with all the provided data
    def __init__(self, x: TensorMoments, attention_layer: GPTNeoXAttention,
                 mlp_layer: GPTNeoXMLP, input_norm: LayerNorm,
                 post_attn_norm: LayerNorm,
                 post_attn_add: Add, post_mlp_add: Add,
                 config: GPTNeoXConfig,
                 ):
        # Init activations and list of layers
        layers = [input_norm, attention_layer, post_attn_add, post_attn_norm]
        layers = layers + mlp_layer.layers + [post_mlp_add]
        activations = [x] + input_norm.activations_output + \
                      attention_layer.activations_output + \
                      post_attn_add.activations_output + \
                      post_attn_norm.activations_output + \
                      mlp_layer.activations[1:] + \
                      post_mlp_add.activations_output
        # Fill Base Model with the generated data
        super().__init__(activations, layers, config)
        self.ln_1 = self.layers[0]
        self.attn = self.layers[1]
        self.ln_2 = self.layers[3]
        self.mlp = mlp_layer

    def forward_dynamic(
        self,
        x: TensorMoments,
        kv_cache: Optional[KVCache] = None
    ):
        use_parallel_residual = self.config.use_parallel_residual
        (ln_1, attention_layer, post_attn_add, ln_2) = self.layers[:4]
        post_mlp_add = self.layers[-1]

        x_normalized = ln_1.forward_dynamic(x)
        attn_outs, kv_cache = attention_layer.forward_dynamic(
            x_normalized, kv_cache=kv_cache
        )
        post_attn_outs = post_attn_add.forward_dynamic(attn_outs, x)

        if use_parallel_residual:
            ln_2_outs = ln_2.forward_dynamic(x)
        else:
            ln_2_outs = ln_2.forward_dynamic(post_attn_outs)

        mlp_outs = self.mlp.forward_dynamic(ln_2_outs)
        post_mlp_outs = post_mlp_add.forward_dynamic(mlp_outs, post_attn_outs)

        return post_mlp_outs, kv_cache

    @staticmethod
    def from_torch(
        torch_layer,
        x: TensorMoments,
        position_ids: np.ndarray,
        mask: np.ndarray,
        config: GPTNeoXConfig
    ):
        """
        torch_layer is HF module for GPT-NeoX Layer
        """
        use_parallel_residual = torch_layer.use_parallel_residual
        layer_norm_1 = LayerNorm.from_torch(
            torch_layer.input_layernorm,
            x
        )
        attention_layer = GPTNeoXAttention.from_torch(
                torch_layer.attention,
                layer_norm_1.activations_output[0],
                position_ids,
                mask,
                config
            )
        post_attn_add = Add.generate_simple(
                x, attention_layer.activations_output[0])
        if use_parallel_residual:
            layer_norm_2 = LayerNorm.from_torch(
                torch_layer.post_attention_layernorm,
                x
            )
        else:
            layer_norm_2 = LayerNorm.from_torch(
                torch_layer.post_attention_layernorm,
                post_attn_add.activations_output[0]
            )
        mlp_layer = GPTNeoXMLP.from_torch(
            torch_layer.mlp,
            layer_norm_2.activations_output[0],
            config)
        post_mlp_add = Add.generate_simple(
            mlp_layer.activations[-1],
            post_attn_add.activations_output[0])

        gpt_neox_block = GPTNeoXBlock(x, attention_layer,
                                            mlp_layer,
                                            layer_norm_1,
                                            layer_norm_2,
                                            post_attn_add,
                                            post_mlp_add,
                                            config)

        return gpt_neox_block

    def to_torch(self):
        torch_config = ConfigTorch(
            hidden_size=self.config.hidden_size,
            num_attention_heads=self.config.num_heads,
            intermediate_size=self.config.intermediate_size,
            rotary_pct=self.config.rotary_pct,
            rotary_emb_base=self.config.rotary_emb_base,
            attention_bias=self.config.attention_bias,
            use_cache=False,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            layer_norm_eps=self.config.layer_norm_epsilon,
            use_parallel_residual=self.config.use_parallel_residual,
        )
        layer_id = 0
        block_torch = BlockTorch(torch_config, layer_id)
        block_torch.input_layernorm = self.ln_1.to_torch()
        block_torch.attention = self.attn.to_torch()
        block_torch.post_attention_layernorm = self.ln_2.to_torch()
        block_torch.mlp = self.mlp.to_torch()
        return block_torch

    def to_torch_with_grads(self):
        torch_config = ConfigTorch(
            hidden_size=self.config.hidden_size,
            num_attention_heads=self.config.num_heads,
            intermediate_size=self.config.intermediate_size,
            rotary_pct=self.config.rotary_pct,
            rotary_emb_base=self.config.rotary_emb_base,
            attention_bias=self.config.attention_bias,
            use_cache=False,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            layer_norm_eps=self.config.layer_norm_epsilon,
            use_parallel_residual=self.config.use_parallel_residual,
        )
        layer_id = 0
        block_torch = BlockTorch(torch_config, layer_id)
        block_torch.input_layernorm = self.ln_1.to_torch_with_grads()
        block_torch.attention = self.attn.to_torch_with_grads()
        block_torch.post_attention_layernorm = self.ln_2.to_torch_with_grads()
        block_torch.mlp = self.mlp.to_torch_with_grads()
        return block_torch
