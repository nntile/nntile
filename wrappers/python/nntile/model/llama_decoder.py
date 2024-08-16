# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/llama_decoder.py
# LlamaDecoder submodule of NNTile Python package
#
# @version 1.1.0

import numpy as np
from transformers import LlamaConfig as LlamaConfig_torch
from transformers.models.llama.modeling_llama import (
    LlamaModel as LlamaModel_torch)

from nntile.tensor import TensorMoments

from ..layer.add import Add
from ..layer.llama_attention import LlamaAttention
from ..layer.rms_norm import RMSNorm
from .base_model import BaseModel
from .llama_config import LlamaConfigNNTile
from .llama_mlp import LlamaMLP as LlamaMLP_nntile


class LlamaDecoder(BaseModel):
    next_tag: int
    llama_mlp: LlamaMLP_nntile
    input_norm: RMSNorm
    post_attn_norm: RMSNorm
    post_attn_add: Add
    attention_layer: LlamaAttention
    post_mlp_add: Add

    # Construct model with all the provided data
    def __init__(self, x: TensorMoments, attention_layer: LlamaAttention,
                 llama_mlp: LlamaMLP_nntile, input_norm: RMSNorm,
                 post_attn_norm: RMSNorm,
                 post_attn_add: Add, post_mlp_add: Add,
                 config: LlamaConfigNNTile,
                 ):
        # Init activations and list of layers
        self.mlp = llama_mlp
        layers = [input_norm, attention_layer, post_attn_add, post_attn_norm]
        layers = layers + llama_mlp.layers + [post_mlp_add]
        activations = [x] + input_norm.activations_output + \
                      attention_layer.activations_output + \
                      post_attn_add.activations_output + \
                      post_attn_norm.activations_output + \
                      llama_mlp.activations[1:] + \
                      post_mlp_add.activations_output
        self.config = config
        # Fill Base Model with the generated data
        super().__init__(activations, layers)

    def forward_dynamic(self, x: TensorMoments):
        (input_norm, attention_layer, post_attn_add, post_attn_norm) = self.layers[:4]  # noqa: E501
        post_mlp_add = self.layers[-1]

        x_normalized = input_norm.forward_dynamic(x)
        attn_outs = attention_layer.forward_dynamic(x_normalized)
        post_attn_outs = post_attn_add.forward_dynamic(attn_outs, x)
        post_attn_norm_outs = post_attn_norm.forward_dynamic(post_attn_outs)

        mlp_outs = self.mlp.forward_dynamic(post_attn_norm_outs)
        post_mlp_outs = post_mlp_add.forward_dynamic(mlp_outs, post_attn_outs)

        return post_mlp_outs

    @staticmethod
    def from_torch(
        torch_llama_decoder, x: TensorMoments,
        position_ids: np.ndarray,
        mask: np.ndarray,
        config: LlamaConfigNNTile, next_tag: int):
        """
        torch_llama_decoder is HF module for LlamaDecoder block
        """
        rms_norm_input_layer, next_tag = RMSNorm.from_torch(
            torch_llama_decoder.input_layernorm, x,
            0, config.rms_norm_eps,
            next_tag, config.redux)
        attention_layer, next_tag = LlamaAttention.from_torch(
            torch_llama_decoder.self_attn,
            rms_norm_input_layer.activations_output[0],
            position_ids,
            mask,
            config,
            next_tag)
        post_attn_add, next_tag = Add.generate_simple(
            x, attention_layer.activations_output[0],
            next_tag)

        rms_norm_post_attn_layer, next_tag = RMSNorm.from_torch(
            torch_llama_decoder.post_attention_layernorm,
            post_attn_add.activations_output[0],
            0, config.rms_norm_eps,
            next_tag, config.redux)
        llama_mlp_module, next_tag = LlamaMLP_nntile.from_torch(
            torch_llama_decoder.mlp,
            rms_norm_post_attn_layer.activations_output[0],
            config, next_tag)
        post_mlp_add, next_tag = Add.generate_simple(
            llama_mlp_module.activations[-1],
            post_attn_add.activations_output[0],
            next_tag)

        nntile_llama_decoder = LlamaDecoder(x, attention_layer,
                                            llama_mlp_module,
                                            rms_norm_input_layer,
                                            rms_norm_post_attn_layer,
                                            post_attn_add,
                                            post_mlp_add,
                                            config)

        return nntile_llama_decoder, next_tag

    def to_torch(self):
        config_torch = LlamaConfig_torch(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            num_hidden_layers=1,
            vocab_size=self.config.vocab_size,
            max_position_embeddings=self.config.max_position_embeddings,
            rms_norm_eps=self.config.rms_norm_eps,
            n_attention_head=self.config.n_attention_head)

        llama_decoder_torch = LlamaModel_torch(config_torch).layers[0]
        llama_decoder_torch.input_layernorm = self.layers[0].to_torch()
        llama_decoder_torch.self_attn = self.layers[1].to_torch()
        p_attn_n = self.layers[3]
        llama_decoder_torch.post_attention_layernorm = p_attn_n.to_torch()
        llama_decoder_torch.mlp = self.mlp.to_torch()

        return llama_decoder_torch

    def to_torch_with_grads(self):
        config_torch = LlamaConfig_torch(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            num_hidden_layers=1,
            vocab_size=self.config.vocab_size,
            max_position_embeddings=self.config.max_position_embeddings,
            rms_norm_eps=self.config.rms_norm_eps,
            n_attention_head=self.config.n_attention_head)

        decoder_torch = LlamaModel_torch(config_torch).layers[0]
        decoder_torch.input_layernorm = self.layers[0].to_torch_with_grads()
        decoder_torch.self_attn = self.layers[1].to_torch_with_grads()
        p_attn_n = self.layers[3]
        decoder_torch.post_attention_layernorm = p_attn_n.to_torch_with_grads()
        decoder_torch.mlp = self.mlp.to_torch_with_grads()
        return decoder_torch
