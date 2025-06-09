# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/gpt_neox_model.py
# GPTNeoX model of NNTile Python package
#
# @version 1.1.0

from typing import List

import numpy as np
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXConfig as ConfigTorch, GPTNeoXModel as ModelTorch)

from nntile.tensor import (
    Tensor_bf16, Tensor_fp32, Tensor_fp32_fast_bf16, Tensor_fp32_fast_fp16,
    Tensor_fp32_fast_tf32, Tensor_int64, TensorMoments, TensorTraits)

from ..layer import Embedding, LayerNorm
from .base_model import BaseModel
from .gpt_neox_block import GPTNeoXBlock
from .gpt_neox_config import GPTNeoXConfig


class GPTNeoXModel(BaseModel):
    emb_layer: Embedding
    final_lnorm: LayerNorm
    gpt_neox_blocks: List[GPTNeoXBlock]

    def __init__(self,
                 input_ids: TensorMoments,
                 emb_layer_: Embedding,
                 gpt_neox_blocks: List[GPTNeoXBlock],
                 lnorm_layer: LayerNorm,
                 config: GPTNeoXConfig
                 ):
        self.dtype = config.dtype

        self.config = config

        if self.dtype not in ["fp32", "tf32",
                              "bf16", "fp32_fast_fp16",
                              "fp32_fast_tf32",
                              "fp32_fast_bf16"]:
            raise TypeError("Only fp32, tf32, bf16, fp32_fast_tf32, "
                            "fp32_fast_fp16 and fp32_fast_bf16 are"
                            "supported for weight type")
        activations = [input_ids]
        activations += emb_layer_.activations_output
        layers = [emb_layer_]
        self.gpt_neox_blocks = gpt_neox_blocks
        self.emb_layer = layers[0]
        self.final_lnorm = lnorm_layer
        self.seq_len = input_ids.value.shape[0]

        for block_layer in gpt_neox_blocks:
            activations.extend(block_layer.activations[1:])
            layers.extend(block_layer.layers)

        layers.append(lnorm_layer)
        activations.extend(lnorm_layer.activations_output)

        super().__init__(activations, layers, config)

    @staticmethod
    def from_torch(model_torch: ModelTorch,
                   batch_size, batch_size_tile,
                   seq_len, seq_len_tile,
                   position_ids: np.ndarray,
                   mask: np.ndarray,
                   config: GPTNeoXConfig,
                   ):

        if config.dtype not in ["fp32", "tf32",
                              "bf16", "fp32_fast_fp16",
                              "fp32_fast_tf32",
                              "fp32_fast_bf16"]:
            raise TypeError("Only fp32, tf32, bf16, fp32_fast_tf32, "
                            "fp32_fast_fp16 and fp32_fast_bf16 are"
                            "supported for weight type")
        x_shape = [seq_len, batch_size]
        x_basetile = [seq_len_tile, batch_size_tile]
        x_traits = TensorTraits(x_shape, x_basetile)
        x_distr = [0] * x_traits.grid.nelems
        x_value = Tensor_int64(x_traits, x_distr)

        dtype2tensor_type = {"fp32": Tensor_fp32,
                             "tf32": Tensor_fp32_fast_tf32,
                             "fp32_fast_tf32": Tensor_fp32_fast_tf32,
                             "bf16": Tensor_bf16,
                             "fp32_fast_fp16": Tensor_fp32_fast_fp16,
                             "fp32_fast_bf16": Tensor_fp32_fast_bf16
                            }

        tensor_type = dtype2tensor_type[config.dtype]

        emb_layer = Embedding.generate_simple(
                                x_value,
                                tensor_type,
                                0,
                                config.vocab_size,
                                config.hidden_size,
                                config.hidden_size_tile,
                                config.hidden_size_tile,
                                )

        emb_layer.w.value.from_array(
            model_torch.embed_in.weight.cpu().detach().numpy().T
        )

        U = emb_layer.activations_output[0]
        gpt_neox_block_list = []

        for block_torch in model_torch.layers:
            block_nntile = GPTNeoXBlock.from_torch(
                block_torch, U, position_ids, mask, config
            )
            U = block_nntile.activations[-1]
            gpt_neox_block_list.append(block_nntile)

        lnorm_final = LayerNorm.from_torch(
                                    model_torch.final_layer_norm,
                                    U,
                                    config.redux)
        X = TensorMoments(x_value, None, False)
        nntile_model = GPTNeoXModel(X,
                            emb_layer,
                            gpt_neox_block_list,
                            lnorm_final,
                            config)

        return nntile_model

    def to_torch(self):
        config_torch = ConfigTorch(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_heads,
            intermediate_size=self.config.intermediate_size,
            rotary_pct=1.0,
            attention_bias=False,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            max_position_embeddings=self.config.max_position_embeddings,
            layer_norm_eps=self.config.layer_norm_epsilon,
            use_cache=False,
        )

        torch_model = ModelTorch(config_torch)
        torch_model.embed_in = self.emb_layer.to_torch()
        for i in range(self.config.num_hidden_layers):
            torch_model.layers[i] = self.gpt_neox_blocks[i].to_torch()

        torch_model.final_layer_norm = self.final_lnorm.to_torch()

        return torch_model

    def to_torch_with_grads(self):
        config_torch = ConfigTorch(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_heads,
            intermediate_size=self.config.intermediate_size,
            rotary_pct=1.0,
            attention_bias=False,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            max_position_embeddings=self.config.max_position_embeddings,
            layer_norm_eps=self.config.layer_norm_epsilon,
            use_cache=False,
        )

        torch_model = ModelTorch(config_torch)
        torch_model.embed_in = self.emb_layer.to_torch_with_grads()
        for i in range(self.config.num_hidden_layers):
            torch_model.layers[i] = (
                self.gpt_neox_blocks[i].to_torch_with_grads()
            )

        torch_model.final_layer_norm = (
            self.final_lnorm.to_torch_with_grads()
        )

        return torch_model
