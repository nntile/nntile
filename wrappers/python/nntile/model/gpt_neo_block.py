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

from transformers.models.gpt_neo.modeling_gpt_neo import (
    GPTNeoBlock as GPTNeoBlockTorch, GPTNeoConfig as GPTNeoConfigTorch)

from nntile.tensor import TensorMoments

from ..layer.add import Add
from ..layer.gpt_neo_attention import GPTNeoAttention
from ..layer.layer_norm import LayerNorm
from .base_model import BaseModel
from .gptneo_config import GPTNeoConfig
from .gptneo_mlp import GPTNeoMLP


class GPTNeoBlock(BaseModel):
    next_tag: int
    gpt_neo_mlp: GPTNeoMLP
    input_norm: LayerNorm
    post_attn_norm: LayerNorm
    post_attn_add: Add
    attention_layer: GPTNeoAttention
    post_mlp_add: Add

    # Construct model with all the provided data
    def __init__(self, x: TensorMoments, attention_layer: GPTNeoAttention,
                 gpt2_mlp: GPTNeoMLP, input_norm: LayerNorm,
                 post_attn_norm: LayerNorm,
                 post_attn_add: Add, post_mlp_add: Add,
                 config: GPTNeoConfig,
                 ):
        # Init activations and list of layers
        layers = [input_norm, attention_layer, post_attn_add, post_attn_norm]
        layers = layers + gpt2_mlp.layers + [post_mlp_add]
        activations = [x] + input_norm.activations_output + \
                      attention_layer.activations_output + \
                      post_attn_add.activations_output + \
                      post_attn_norm.activations_output + \
                      gpt2_mlp.activations[1:] + \
                      post_mlp_add.activations_output
        self.config = config
        # Fill Base Model with the generated data
        super().__init__(activations, layers, config)
        self.ln_1 = self.layers[0]
        self.attn = self.layers[1]
        self.ln_2 = self.layers[3]
        self.mlp = gpt2_mlp

    @staticmethod
    def from_torch(
        torch_block, x: TensorMoments,
        config: GPTNeoConfig, next_tag: int):
        """
        torch_gptneo_block is HF module for GPT-Neo Block
        """
        layer_norm_input_layer, next_tag = LayerNorm.from_torch(
            torch_block.ln_1,
            x,
            next_tag
        )
        attention_layer, next_tag = GPTNeoAttention.from_torch(
            torch_block.attn,
            layer_norm_input_layer.activations_output[0],
            layer_norm_input_layer.activations_output[0],
            layer_norm_input_layer.activations_output[0],
            config, next_tag)
        post_attn_add, next_tag = Add.generate_simple(
            x, attention_layer.activations_output[0],
            next_tag)

        layer_norm_post_attn_layer, next_tag = LayerNorm.from_torch(
            torch_block.ln_2,
            post_attn_add.activations_output[0],
            next_tag
        )
        gpt2_mlp_module, next_tag = GPTNeoMLP.from_torch(
            torch_block.mlp,
            layer_norm_post_attn_layer.activations_output[0],
            config, next_tag)
        post_mlp_add, next_tag = Add.generate_simple(
            gpt2_mlp_module.activations[-1],
            post_attn_add.activations_output[0],
            next_tag)

        gpt_neo_block = GPTNeoBlock(x, attention_layer,
                                            gpt2_mlp_module,
                                            layer_norm_input_layer,
                                            layer_norm_post_attn_layer,
                                            post_attn_add,
                                            post_mlp_add,
                                            config)

        return gpt_neo_block, next_tag

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
