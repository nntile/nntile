# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/bert_encoder.py
# Bert encoder as a part of NNTile Python package
#
# @version 1.1.0

from typing import List

from transformers.models.bert.modeling_bert import (
    BertConfig as BertConfig_torch, BertEncoder as BertEncoder_torch)

from nntile.tensor import TensorMoments

from .base_model import BaseModel
from .bert_config import BertConfigNNTile
from .bert_layer import BertLayer


class BertEncoder(BaseModel):

    def __init__(self, hidden_states: TensorMoments,
                  list_bert_layers: List,
                  config: BertConfigNNTile):

        self.dtype = config.dtype

        self.config = config

        activations = [hidden_states]
        layers = []
        self.list_bert_layers = list_bert_layers
        for b_layer in list_bert_layers:
            activations.extend(b_layer.activations[1:])
            layers.extend(b_layer.layers)

        # Fill Base Model with the generated data
        super().__init__(activations, layers, config)

    @staticmethod
    def from_torch(bert_encoder_torch, hidden_states: TensorMoments,
                   config: BertConfigNNTile):

        if config.dtype not in ["fp32", "fp32_fast_tf32", "bf16",
                            "fp32_fast_fp16", "fp32_fast_bf16"]:
            raise TypeError("Only fp32, fp32_fast_tf32, bf16,"
            "fp32_fast_fp16, and fp32_fast_bf16 supported for weight type")

        list_bert_layers = []
        cur_hidden_states = hidden_states
        for bert_layers_torch in bert_encoder_torch.layer:
            current_layer = BertLayer.from_torch(bert_layers_torch,
                                                 cur_hidden_states,
                                                 config)
            cur_hidden_states = current_layer.activations[-1]
            list_bert_layers.append(current_layer)

        bert_layer_nntile = BertEncoder(hidden_states,
                                        list_bert_layers,
                                        config)

        return bert_layer_nntile

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
        config_torch.num_hidden_layers = self.config.num_hidden_layers

        bert_layer_torch = BertEncoder_torch(config_torch)

        return bert_layer_torch

    def to_torch(self):

        bert_encoder_torch = self._make_default_torch_model()

        for i, b_layer_nntile in enumerate(self.list_bert_layers):
            bert_encoder_torch.layer[i] = b_layer_nntile.to_torch()
        return bert_encoder_torch

    def to_torch_with_grads(self):

        bert_encoder_torch = self._make_default_torch_model()
        for i, b_layer_nntile in enumerate(self.list_bert_layers):
            bert_encoder_torch.layer[i] = b_layer_nntile.to_torch_with_grads()

        return bert_encoder_torch
