# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/gpt2_block.py
# GPT2Block submodule of NNTile Python package
#
# @version 1.1.0

from transformers import GPT2Config as GPT2ConfigTorch
from transformers.models.gpt2.modeling_gpt2 import GPT2Block as GPT2Block_torch

from nntile.tensor import TensorMoments

from ..layer.add import Add
from ..layer.gpt2_attention import GPT2Attention
from ..layer.layer_norm import LayerNorm
from .base_model import BaseModel
from .gpt2_config import GPT2ConfigNNTile
from .gpt2_mlp import GPT2MLP


class GPT2Block(BaseModel):
    next_tag: int
    gpt2_mlp: GPT2MLP
    input_norm: LayerNorm
    post_attn_norm: LayerNorm
    post_attn_add: Add
    attention_layer: GPT2Attention
    post_mlp_add: Add

    # Construct model with all the provided data
    def __init__(self, x: TensorMoments, attention_layer: GPT2Attention,
                 gpt2_mlp: GPT2MLP, input_norm: LayerNorm,
                 post_attn_norm: LayerNorm,
                 post_attn_add: Add, post_mlp_add: Add,
                 config: GPT2ConfigNNTile,
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
        super().__init__(activations, layers)
        self.ln_1 = self.layers[0]
        self.attn = self.layers[1]
        self.ln_2 = self.layers[3]
        self.mlp = gpt2_mlp

    @staticmethod
    def from_torch(
        torch_gpt2_block, x: TensorMoments,
        config: GPT2ConfigNNTile, next_tag: int):
        """
        torch_gpt2_block is HF module for GPT2 Block
        """
        layer_norm_input_layer, next_tag = LayerNorm.from_torch(
            torch_gpt2_block.ln_1,
            x,
            next_tag
        )
        attention_layer, next_tag = GPT2Attention.from_torch(
            torch_gpt2_block.attn,
            layer_norm_input_layer.activations_output[0],
            layer_norm_input_layer.activations_output[0],
            layer_norm_input_layer.activations_output[0],
            config, next_tag)
        post_attn_add, next_tag = Add.generate_simple(
            x, attention_layer.activations_output[0],
            next_tag)

        layer_norm_post_attn_layer, next_tag = LayerNorm.from_torch(
            torch_gpt2_block.ln_2,
            post_attn_add.activations_output[0],
            next_tag
        )
        gpt2_mlp_module, next_tag = GPT2MLP.from_torch(
            torch_gpt2_block.mlp,
            layer_norm_post_attn_layer.activations_output[0],
            config, next_tag)
        post_mlp_add, next_tag = Add.generate_simple(
            gpt2_mlp_module.activations[-1],
            post_attn_add.activations_output[0],
            next_tag)

        nntile_gpt2_decoder = GPT2Block(x, attention_layer,
                                            gpt2_mlp_module,
                                            layer_norm_input_layer,
                                            layer_norm_post_attn_layer,
                                            post_attn_add,
                                            post_mlp_add,
                                            config)

        return nntile_gpt2_decoder, next_tag

    def to_torch(self):
        config_torch = GPT2ConfigTorch(
            n_embd=self.config.hidden_size,
            n_layer=1,
            n_head=self.config.n_head,
            n_inner=self.config.intermediate_size,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            layer_norm_epsilon=self.config.layer_norm_epsilon,
            scale_attn_weights=True,
            use_cache=False,
            add_cross_attention=False,
        )
        gpt2_block_torch = GPT2Block_torch(config_torch)
        gpt2_block_torch.ln_1 = self.ln_1.to_torch()
        gpt2_block_torch.attn = self.attn.to_torch()
        gpt2_block_torch.ln_2 = self.ln_2.to_torch()
        gpt2_block_torch.mlp = self.mlp.to_torch()

        return gpt2_block_torch

    def to_torch_with_grads(self):
        config_torch = GPT2ConfigTorch(
            n_embd=self.config.hidden_size,
            n_layer=1,
            n_head=self.config.n_head,
            n_inner=self.config.intermediate_size,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            layer_norm_epsilon=self.config.layer_norm_epsilon,
            scale_attn_weights=True,
            use_cache=False,
            add_cross_attention=False,
        )
        gpt2_block_torch = GPT2Block_torch(config_torch)
        gpt2_block_torch.ln_1 = self.ln_1.to_torch_with_grads()
        gpt2_block_torch.attn = self.attn.to_torch_with_grads()
        gpt2_block_torch.ln_2 = self.ln_2.to_torch_with_grads()
        gpt2_block_torch.mlp = self.mlp.to_torch_with_grads()
        return gpt2_block_torch
