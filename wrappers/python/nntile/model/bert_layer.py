# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/bert_layer.py
# Bert layer as a part of NNTile Python package
#
# @version 1.1.0

from transformers.models.bert.modeling_bert import (
    BertConfig as BertConfig_torch, BertLayer as BertLayer_torch)

from nntile.tensor import TensorMoments

from .base_model import BaseModel
from .bert_config import BertConfigNNTile
from .bert_modules import BertAttention, BertIntermediate, BertOutput


class BertLayer(BaseModel):
    next_tag: int

    def __init__(self, hidden_states: TensorMoments,
                  attention: BertAttention,
                  intermediate_block: BertIntermediate,
                  output_block: BertOutput,
                  config: BertConfigNNTile):

        self.dtype = config.dtype

        self.config = config

        self.bert_attention = attention
        self.intermediate_block = intermediate_block
        self.output_block = output_block

        activations = [hidden_states]
        activations.extend(attention.activations[1:])
        activations.extend(intermediate_block.activations[1:])
        activations.extend(output_block.activations[2:])

        layers = attention.layers + intermediate_block.layers + \
            output_block.layers

        # Fill Base Model with the generated data
        super().__init__(activations, layers)

    @staticmethod
    def from_torch(bert_layer_torch, hidden_states: TensorMoments,
                   config: BertConfigNNTile,
                   next_tag: int):

        if config.dtype not in ["fp32", "fp32_fast_tf32", "bf16",
                            "fp32_fast_fp16", "fp32_fast_bf16"]:
            raise TypeError("Only fp32, fp32_fast_tf32, bf16,"
            "fp32_fast_fp16, and fp32_fast_bf16 supported for weight type")

        bert_attention_nntile, next_tag = BertAttention.from_torch(
                                            bert_layer_torch.attention,
                                            hidden_states,
                                            config,
                                            next_tag)
        bert_interm_nntile, next_tag = BertIntermediate.from_torch(
                                            bert_layer_torch.intermediate,
                                            bert_attention_nntile.activations[-1],
                                            config.intermediate_size_tile,
                                            config, next_tag)

        bert_output_nntile, next_tag = BertOutput.from_torch(
                                            bert_layer_torch.output,
                                            bert_interm_nntile.activations[-1],
                                            bert_attention_nntile.activations[-1],
                                            config.hidden_size_tile,
                                            config,
                                            next_tag)
        bert_layer_nntile = BertLayer(hidden_states,
                                      bert_attention_nntile,
                                      bert_interm_nntile,
                                      bert_output_nntile,
                                      config)

        return bert_layer_nntile, next_tag

    def _make_default_torch_model(self):
        config_torch = BertConfig_torch()
        config_torch.vocab_size = self.config.vocab_size
        config_torch.hidden_size = self.config.hidden_size
        config_torch.max_position_embeddings = \
                        self.config.max_position_embeddings
        config_torch.layer_norm_eps = self.config.layer_norm_epsilon
        config_torch.type_vocab_size = self.config.type_vocab_size
        config_torch.hidden_dropout_prob = 0.
        config_torch.attention_probs_dropout_prob = 0.
        config_torch.intermediate_size = self.config.intermediate_size
        config_torch.num_attention_heads = self.config.num_attention_heads
        config_torch._attn_implementation = "eager"
        config_torch.add_cross_attention = False
        config_torch.chunk_size_feed_forward = 0
        config_torch.is_decoder = False

        bert_layer_torch = BertLayer_torch(config_torch)

        return bert_layer_torch

    def to_torch(self):

        bert_layer_torch = self._make_default_torch_model()
        bert_layer_torch.attention = self.bert_attention.to_torch()
        bert_layer_torch.intermediate = self.intermediate_block.to_torch()
        bert_layer_torch.output = self.output_block.to_torch()

        return bert_layer_torch

    def to_torch_with_grads(self):

        bert_layer_torch = self._make_default_torch_model()
        bert_layer_torch.attention = self.bert_attention.to_torch_with_grads()
        bert_layer_torch.intermediate = \
            self.intermediate_block.to_torch_with_grads()
        bert_layer_torch.output = self.output_block.to_torch_with_grads()

        return bert_layer_torch
