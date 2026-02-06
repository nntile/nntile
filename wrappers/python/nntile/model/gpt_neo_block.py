# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/gpt_neo_block.py
# GPTNeoBlock submodule of NNTile Python package
#
# @version 1.1.0

from typing import Optional

from transformers.models.gpt_neo.modeling_gpt_neo import (
    GPTNeoBlock as GPTNeoBlockTorch, GPTNeoConfig as GPTNeoConfigTorch)

from nntile.layer.cache_utils import KVCache
from nntile.tensor import TensorMoments

from ..layer.add import Add
from ..layer.gpt_neo_attention import GPTNeoAttention
from ..layer.layer_norm import LayerNorm
from .base_model import BaseModel
from .gpt_neo_config import GPTNeoConfig
from .gpt_neo_mlp import GPTNeoMLP


class GPTNeoBlock(BaseModel):
    gpt_neo_mlp: GPTNeoMLP
    input_norm: LayerNorm
    post_attn_norm: LayerNorm
    post_attn_add: Add
    attention_layer: GPTNeoAttention
    post_mlp_add: Add

    # Construct model with all the provided data
    def __init__(self, x: TensorMoments, attention_layer: GPTNeoAttention,
                 mlp_layer: GPTNeoMLP, input_norm: LayerNorm,
                 post_attn_norm: LayerNorm,
                 post_attn_add: Add, post_mlp_add: Add,
                 config: GPTNeoConfig,
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

    def forward_dynamic(
        self,
        x: TensorMoments,
        kv_cache: Optional[KVCache] = None
    ):
        (input_norm, attention_layer, post_attn_add, post_attn_norm) = self.layers[:4]  # noqa: E501
        post_mlp_add = self.layers[-1]

        x_normalized = input_norm.forward_dynamic(x)
        attn_outs, kv_cache = attention_layer.forward_dynamic(
            x_normalized, kv_cache=kv_cache
        )
        post_attn_outs = post_attn_add.forward_dynamic(attn_outs, x)
        post_attn_norm_outs = post_attn_norm.forward_dynamic(post_attn_outs)

        mlp_outs = self.mlp.forward_dynamic(post_attn_norm_outs)
        post_mlp_outs = post_mlp_add.forward_dynamic(mlp_outs, post_attn_outs)

        return post_mlp_outs, kv_cache

    @staticmethod
    def from_torch(
        torch_block, x: TensorMoments,
        config: GPTNeoConfig):
        """
        torch_gptneo_block is HF module for GPT-Neo Block
        """
        layer_norm_input_layer = LayerNorm.from_torch(
            torch_block.ln_1,
            x
        )
        attention_layer = GPTNeoAttention.from_torch(
            torch_block.attn,
            layer_norm_input_layer.activations_output[0],
            layer_norm_input_layer.activations_output[0],
            layer_norm_input_layer.activations_output[0],
            config)
        post_attn_add = Add.generate_simple(
            x, attention_layer.activations_output[0])

        layer_norm_post_attn_layer = LayerNorm.from_torch(
            torch_block.ln_2,
            post_attn_add.activations_output[0]
        )
        gpt2_mlp_module = GPTNeoMLP.from_torch(
            torch_block.mlp,
            layer_norm_post_attn_layer.activations_output[0],
            config)
        post_mlp_add = Add.generate_simple(
            gpt2_mlp_module.activations[-1],
            post_attn_add.activations_output[0])

        gpt_neo_block = GPTNeoBlock(x, attention_layer,
                                            gpt2_mlp_module,
                                            layer_norm_input_layer,
                                            layer_norm_post_attn_layer,
                                            post_attn_add,
                                            post_mlp_add,
                                            config)

        return gpt_neo_block

    def to_torch(self):
        config_torch = GPTNeoConfigTorch(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_heads,
            n_inner=self.config.intermediate_size,
            resid_dropout=0.0,
            embed_dropout=0.0,
            attention_dropout=0.0,
            layer_norm_epsilon=self.config.layer_norm_epsilon,
            use_cache=False,
        )
        gptneo_block_torch = GPTNeoBlockTorch(config_torch, self.attn.layer_id)
        gptneo_block_torch.ln_1 = self.ln_1.to_torch()
        gptneo_block_torch.attn = self.attn.to_torch()
        gptneo_block_torch.ln_2 = self.ln_2.to_torch()
        gptneo_block_torch.mlp = self.mlp.to_torch()
        return gptneo_block_torch

    def to_torch_with_grads(self):
        config_torch = GPTNeoConfigTorch(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_heads,
            n_inner=self.config.intermediate_size,
            resid_dropout=0.0,
            embed_dropout=0.0,
            attention_dropout=0.0,
            layer_norm_epsilon=self.config.layer_norm_epsilon,
            use_cache=False,
        )
        gptneo_block_torch = GPTNeoBlockTorch(config_torch, self.attn.layer_id)
        gptneo_block_torch.ln_1 = self.ln_1.to_torch_with_grads()
        gptneo_block_torch.attn = self.attn.to_torch_with_grads()
        gptneo_block_torch.ln_2 = self.ln_2.to_torch_with_grads()
        gptneo_block_torch.mlp = self.mlp.to_torch_with_grads()
        return gptneo_block_torch
