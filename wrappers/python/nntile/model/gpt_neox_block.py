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

import numpy as np
import torch
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXLayer as GPTNeoXBlockTorch, GPTNeoXConfig as GPTNeoXConfigTorch)

from nntile.tensor import TensorMoments

from ..layer.add import Add
from ..layer.gpt_neox_attention import GPTNeoXAttention
from ..layer.layer_norm import LayerNorm
from .base_model import BaseModel
from .gpt_neox_config import GPTNeoXConfig
from .gpt_neox_mlp import GPTNeoXMLP


class GPTNeoXBlock(BaseModel):
    next_tag: int
    gpt_neox_mlp: GPTNeoXMLP
    input_norm: LayerNorm
    post_attn_norm: LayerNorm
    post_attn_add: Add
    attention_layer: GPTNeoXAttention
    post_mlp_add: Add

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
        self.config = config
        # Fill Base Model with the generated data
        super().__init__(activations, layers, config)
        self.ln_1 = self.layers[0]
        self.attn = self.layers[1]
        self.ln_2 = self.layers[3]
        self.mlp = mlp_layer

    @staticmethod
    def from_torch(
        torch_layer,
        x: TensorMoments,
        position_ids: np.ndarray,
        mask: np.ndarray,
        config: GPTNeoXConfig,
        next_tag: int,
    ):
        """
        torch_layer is HF module for GPT-NeoX Layer
        """
        layer_norm_input_layer, next_tag = LayerNorm.from_torch(
            torch_layer.input_layernorm,
            x,
            next_tag
        )
        attention_layer, next_tag = GPTNeoXAttention.from_torch(
            torch_layer.attention,
            layer_norm_input_layer.activations_output[0],
            position_ids,
            mask,
            config,
            next_tag
        )
        post_attn_add, next_tag = Add.generate_simple(
            x, attention_layer.activations_output[0],
            next_tag)

        layer_norm_post_attn_layer, next_tag = LayerNorm.from_torch(
            torch_layer.post_attention_layernorm,
            post_attn_add.activations_output[0],
            next_tag
        )
        mlp_layer, next_tag = GPTNeoXMLP.from_torch(
            torch_layer.mlp,
            layer_norm_post_attn_layer.activations_output[0],
            config, next_tag)
        post_mlp_add, next_tag = Add.generate_simple(
            mlp_layer.activations[-1],
            post_attn_add.activations_output[0],
            next_tag)

        gpt_neox_block = GPTNeoXBlock(x, attention_layer,
                                            mlp_layer,
                                            layer_norm_input_layer,
                                            layer_norm_post_attn_layer,
                                            post_attn_add,
                                            post_mlp_add,
                                            config)

        return gpt_neox_block, next_tag

    def to_torch(self):
        torch_config = GPTNeoXConfigTorch(
            hidden_size=self.config.hidden_size,
            num_attention_heads=self.config.num_heads,
            intermediate_size=self.config.intermediate_size,
            rotary_pct=1.0,
            attention_bias=False,
            use_cache=False,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            layer_norm_eps=self.config.layer_norm_epsilon,
        )
        layer_id = 0
        block_torch = GPTNeoXBlockTorch(torch_config, layer_id)
        block_torch.input_layernorm = self.ln_1.to_torch()
        block_torch.attention = self.attn.to_torch()
        block_torch.post_attention_layernorm = self.ln_2.to_torch()
        block_torch.mlp = self.mlp.to_torch()
        return block_torch

    def to_torch_with_grads(self):
        torch_config = GPTNeoXConfigTorch(
            hidden_size=self.config.hidden_size,
            num_attention_heads=self.config.num_heads,
            intermediate_size=self.config.intermediate_size,
            rotary_pct=1.0,
            attention_bias=False,
            use_cache=False,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            layer_norm_eps=self.config.layer_norm_epsilon,
        )
        layer_id = 0
        block_torch = GPTNeoXBlockTorch(torch_config, layer_id)
        block_torch.input_layernorm = self.ln_1.to_torch_with_grads()
        block_torch.attention = self.attn.to_torch_with_grads()
        block_torch.post_attention_layernorm = self.ln_2.to_torch_with_grads()
        block_torch.mlp = self.mlp.to_torch_with_grads()
        return block_torch
