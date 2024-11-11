# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/roberta_modules.py
# Robert modules as a part of NNTile Python package.
# These modules are different in structure compared to modules in Bert
# model
#
# @version 1.1.0

import torch
from transformers.models.roberta.modeling_roberta import (
    RobertaConfig as RobertaConfig_torch, RobertaLMHead as RobertaLMHead_torch)

from nntile.tensor import TensorMoments, to_numpy

from ..layer.act import Act
from ..layer.layer_norm import LayerNorm
from ..layer.linear import Linear
from .base_model import BaseModel
from .bert_config import BertConfigNNTile


class RobertaLMHead(BaseModel):
    next_tag: int

    def __init__(self, X: TensorMoments,
                 lin1_layer: Linear,
                 act_fun: Act,
                 layer_norm: LayerNorm,
                 lin2_layer: Linear,
                 config: BertConfigNNTile):

        self.dtype = config.dtype

        self.config = config

        activations = [X]
        activations.extend(lin1_layer.activations_output)
        activations.extend(act_fun.activations_output)
        activations.extend(layer_norm.activations_output)
        activations.extend(lin2_layer.activations_output)

        layers = []
        layers.append(lin1_layer)
        layers.append(act_fun)
        layers.append(layer_norm)
        layers.append(lin2_layer)

        self.lin1_layer = lin1_layer
        self.layer_norm = layer_norm
        self.lin2_layer = lin2_layer

        # Fill Base Model with the generated data
        super().__init__(activations, layers, config)

    @staticmethod
    def from_torch(roberta_lm_head, X,
                   config: BertConfigNNTile, next_tag: int):

        if config.dtype not in ["fp32", "fp32_fast_tf32", "bf16",
                            "fp32_fast_fp16", "fp32_fast_bf16"]:
            raise TypeError("Only fp32, fp32_fast_tf32, bf16,"
            "fp32_fast_fp16, and fp32_fast_bf16 supported for weight type")

        lin1_layer, next_tag = Linear.from_torch(
                                    roberta_lm_head.dense,
                                    X,
                                    config.hidden_size_tile,
                                    config.redux, next_tag)
        activation_layer, next_tag = Act.generate_simple(
            lin1_layer.activations_output[0],
            "gelu", next_tag
        )

        layer_norm, next_tag = LayerNorm.from_torch(roberta_lm_head.layer_norm,
                                                   activation_layer.activations_output[0],
                                                   next_tag)

        lin2_layer, next_tag = Linear.from_torch(
                                    roberta_lm_head.decoder,
                                    layer_norm.activations_output[0],
                                    config.vocab_size,
                                    config.redux, next_tag)

        roberta_lmhead_nntile = RobertaLMHead(X, lin1_layer,
                                              activation_layer,
                                              layer_norm,
                                              lin2_layer,
                                              config)
        return roberta_lmhead_nntile, next_tag

    def to_torch(self):
        config_torch = RobertaConfig_torch()
        config_torch.hidden_size = self.config.hidden_size
        config_torch.intermediate_size = self.config.intermediate_size
        config_torch.layer_norm_eps = self.config.layer_norm_epsilon
        config_torch.vocab_size = self.config.vocab_size

        roberta_lm_head_torch = RobertaLMHead_torch(config_torch)
        roberta_lm_head_torch.dense = self.lin1_layer.to_torch()
        roberta_lm_head_torch.layer_norm = self.layer_norm.to_torch()
        roberta_lm_head_torch.decoder = self.lin2_layer.to_torch()
        roberta_lm_head_torch.bias = torch.nn.Parameter(
            torch.tensor(to_numpy(self.lin2_layer.b.value),
                        requires_grad=True))
        return roberta_lm_head_torch

    def to_torch_with_grads(self):
        config_torch = RobertaConfig_torch()
        config_torch.hidden_size = self.config.hidden_size
        config_torch.intermediate_size = self.config.intermediate_size
        config_torch.layer_norm_eps = self.config.layer_norm_epsilon
        config_torch.vocab_size = self.config.vocab_size
        config_torch.hidden_act = self.config.activation_function

        roberta_lm_head_torch = RobertaLMHead_torch(config_torch)
        roberta_lm_head_torch.dense = self.lin1_layer.to_torch_with_grads()
        roberta_lm_head_torch.layer_norm = \
            self.layer_norm.to_torch_with_grads()
        roberta_lm_head_torch.decoder = \
            self.lin2_layer.to_torch_with_grads()
        roberta_lm_head_torch.bias.grad = torch.tensor(
            to_numpy(self.lin2_layer.b.grad))
        return roberta_lm_head_torch
