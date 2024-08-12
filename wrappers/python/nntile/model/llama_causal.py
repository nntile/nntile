# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/llama_causal.py
# LlamaCausal model of NNTile Python package
#
# @version 1.1.0

import numpy as np
from transformers import LlamaConfig as LlamaConfig_torch
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM as LlamaCausalModel_torch)

from ..layer import Linear
from .base_model import BaseModel
from .llama import Llama as Llama_nntile
from .llama_config import LlamaConfigNNTile


class LlamaForCausalLM(BaseModel):
    next_tag: int
    llama_model_: Llama_nntile
    lin_: Linear

    def __init__(self,
                 llama_model_: Llama_nntile,
                 lin_head_: Linear,
                 config: LlamaConfigNNTile):
        self.dtype = config.dtype

        self.config = config

        if self.dtype not in ["fp32", "fp32_fast_tf32", "bf16"]:
            raise TypeError("Only fp32, fp32_fast_tf32 and bf16 are"
                            "supported for weight type")
        activations = []
        activations.extend(llama_model_.activations)
        layers = []
        layers.extend(llama_model_.layers)
        layers.append(lin_head_)

        self.llama_model_ = llama_model_
        self.lin_ = lin_head_

        activations.extend(lin_head_.activations_output)

        super().__init__(activations, layers)

    @staticmethod
    def from_torch(torch_llama_causal,
                   batch_size, batch_size_tile,
                   seq_len, seq_len_tile,
                   position_ids: np.ndarray,
                   mask: np.ndarray,
                   config: LlamaConfigNNTile,
                   next_tag: int):

        if config.dtype not in ["fp32", "fp32_fast_tf32", "bf16"]:
            raise TypeError("Only fp32, fp32_fast_tf32 and bf16 are"
                            "supported for weight type")

        llama_model, next_tag = Llama_nntile.from_torch(
                   torch_llama_causal.model,
                   batch_size, batch_size_tile,
                   seq_len, seq_len_tile,
                   position_ids,
                   mask,
                   config,
                   next_tag)
        lin_head, next_tag = Linear.from_torch(torch_llama_causal.lm_head,
                                               llama_model.activations[-1],
                                               config.vocab_size,
                                               config.redux, next_tag)

        causal_llama_nntile = LlamaForCausalLM(llama_model,
                                               lin_head,
                                               config)

        return causal_llama_nntile, next_tag

    def to_torch(self):
        config_torch = LlamaConfig_torch(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            num_hidden_layers=self.config.num_hidden_layers,
            vocab_size=self.config.vocab_size,
            max_position_embeddings=self.config.max_position_embeddings,
            rms_norm_eps=self.config.rms_norm_eps,
            n_attention_head=self.config.n_attention_head)

        llama_model_torch = LlamaCausalModel_torch(config_torch)
        llama_model_torch.model = self.llama_model_.to_torch()
        llama_model_torch.lm_head = self.lin_.to_torch()
        return llama_model_torch

    def to_torch_with_grads(self):
        config_torch = LlamaConfig_torch(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            num_hidden_layers=self.config.num_hidden_layers,
            vocab_size=self.config.vocab_size,
            max_position_embeddings=self.config.max_position_embeddings,
            rms_norm_eps=self.config.rms_norm_eps,
            n_attention_head=self.config.n_attention_head)

        llama_model_torch = LlamaCausalModel_torch(config_torch)
        llama_model_torch.model = self.llama_model_.to_torch_with_grads()
        llama_model_torch.lm_head = self.lin_.to_torch_with_grads()

        return llama_model_torch
