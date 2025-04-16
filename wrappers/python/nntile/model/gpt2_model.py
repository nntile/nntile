# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/gpt2_model.py
# GPT2 model of NNTile Python package
#
# @version 1.1.0

from typing import List

import numpy as np
from transformers import GPT2Config as GPT2ConfigTorch
from transformers.models.gpt2.modeling_gpt2 import GPT2Model as GPT2Model_torch

from nntile.tensor import (
    Tensor_bf16, Tensor_fp32, Tensor_fp32_fast_bf16, Tensor_fp32_fast_fp16,
    Tensor_fp32_fast_tf32, Tensor_int64, TensorMoments, TensorTraits)

from ..layer import AddSlice, Embedding, LayerNorm
from .base_model import BaseModel
from .gpt2_block import GPT2Block
from .gpt2_config import GPT2ConfigNNTile


class GPT2Model(BaseModel):
    wte_layer: Embedding
    wpe_layer: Embedding
    add_slice_layer: AddSlice
    final_lnorm: LayerNorm
    gpt2_blocks: List[GPT2Block]

    def __init__(self,
                 input_ids: TensorMoments,
                 positional_ids: TensorMoments,
                 wpe_layer_: Embedding,
                 wte_layer_: Embedding,
                 add_slice_layer_: AddSlice,
                 gpt2_blocks: List[GPT2Block],
                 lnorm_layer: LayerNorm,
                 config: GPT2ConfigNNTile,
                 ):
        self.dtype = config.dtype

        self.config = config

        if self.dtype not in ["fp32", "tf32",
                              "bf16", "fp32_fast_fp16",
                              "fp32_fast_bf16"]:
            raise TypeError("Only fp32, tf32, bf16, fp32_fast_fp16,"
                            "fp32_fast_bf16 are"
                            "supported for weight type")
        activations = [input_ids, positional_ids]
        activations += wte_layer_.activations_output
        activations += wpe_layer_.activations_output
        activations += add_slice_layer_.activations_output
        layers = [wte_layer_, wpe_layer_, add_slice_layer_]
        self.gpt2_blocks = gpt2_blocks
        self.wte_layer = layers[0]
        self.wpe_layer = layers[1]
        self.add_slice_layer = layers[2]
        self.final_lnorm = lnorm_layer
        self.seq_len = input_ids.value.shape[0]

        for block_layer in gpt2_blocks:
            activations.extend(block_layer.activations[1:])
            layers.extend(block_layer.layers)

        layers.append(lnorm_layer)
        activations.extend(lnorm_layer.activations_output)

        super().__init__(activations, layers, config)

    @staticmethod
    def from_torch(torch_gpt2: GPT2Model_torch,
                   batch_size, batch_size_tile,
                   seq_len, seq_len_tile,
                   config: GPT2ConfigNNTile):

        if config.dtype not in ["fp32", "tf32",
                              "bf16", "fp32_fast_fp16",
                              "fp32_fast_bf16"]:
            raise TypeError("Only fp32, tf32, bf16, fp32_fast_fp16,"
                            "fp32_fast_bf16 are"
                            "supported for weight type")
        positional_ids_traits = TensorTraits([seq_len], [seq_len_tile])
        positional_ids_distr = [0] * positional_ids_traits.grid.nelems
        positional_ids_value = Tensor_int64(
            positional_ids_traits, positional_ids_distr
        )
        positional_ids_value.from_array(
            np.array(np.arange(seq_len), order="F", dtype=np.int64)
        )
        positional_ids = TensorMoments(positional_ids_value, None, False)

        x_shape = [seq_len, batch_size]
        x_basetile = [seq_len_tile, batch_size_tile]
        x_traits = TensorTraits(x_shape, x_basetile)
        x_distr = [0] * x_traits.grid.nelems
        x_value = Tensor_int64(x_traits, x_distr)

        dtype2tensor_type = {"fp32": Tensor_fp32,
                             "tf32": Tensor_fp32_fast_tf32,
                             "bf16": Tensor_bf16,
                             "fp32_fast_fp16": Tensor_fp32_fast_fp16,
                             "fp32_fast_bf16": Tensor_fp32_fast_bf16
                            }

        tensor_type = dtype2tensor_type[config.dtype]

        wte_layer = Embedding.generate_simple(
                                x_value,
                                tensor_type,
                                0,
                                config.vocab_size,
                                config.hidden_size,
                                config.hidden_size_tile,
                                config.hidden_size_tile)

        wpe_layer = Embedding.generate_simple(
                                positional_ids.value,
                                tensor_type,
                                0,
                                config.max_position_embeddings,
                                config.hidden_size,
                                config.hidden_size_tile,
                                config.hidden_size_tile)

        wte_layer.w.value.from_array(
            torch_gpt2.wte.weight.cpu().detach().numpy().T
        )
        wpe_layer.w.value.from_array(
            torch_gpt2.wpe.weight.cpu().detach().numpy().T
        )
        add_slice_layer = AddSlice.generate_simple(
            wte_layer.activations_output[0], wpe_layer.activations_output[0],
            2, redux=config.redux
        )

        U = add_slice_layer.activations_output[0]
        gpt2block_list = []

        for gpt2_block_torch in torch_gpt2.h:
            gpt2block_nntile = GPT2Block.from_torch(
                gpt2_block_torch, U, config
            )
            U = gpt2block_nntile.activations[-1]
            gpt2block_list.append(gpt2block_nntile)

        lnorm_final = LayerNorm.from_torch(
                                    torch_gpt2.ln_f,
                                    U,
                                    config.redux)
        X = TensorMoments(x_value, None, False)
        gpt2_nntile = GPT2Model(X,
                            positional_ids,
                            wpe_layer,
                            wte_layer,
                            add_slice_layer,
                            gpt2block_list,
                            lnorm_final,
                            config)

        return gpt2_nntile

    def to_torch(self):
        config_torch = GPT2ConfigTorch(
            vocab_size=self.config.vocab_size,
            n_positions=self.config.max_position_embeddings,
            n_embd=self.config.hidden_size,
            n_layer=self.config.num_hidden_layers,
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

        torch_gpt2 = GPT2Model_torch(config_torch)
        torch_gpt2.wte = self.wte_layer.to_torch()
        torch_gpt2.wpe = self.wpe_layer.to_torch()
        for i in range(self.config.num_hidden_layers):
            torch_gpt2.h[i] = self.gpt2_blocks[i].to_torch()

        torch_gpt2.ln_f = self.final_lnorm.to_torch()

        return torch_gpt2

    def to_torch_with_grads(self):
        config_torch = GPT2ConfigTorch(
            vocab_size=self.config.vocab_size,
            n_positions=self.config.max_position_embeddings,
            n_embd=self.config.hidden_size,
            n_layer=self.config.num_hidden_layers,
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

        torch_gpt2 = GPT2Model_torch(config_torch)
        torch_gpt2.wte = self.wte_layer.to_torch_with_grads()
        torch_gpt2.wpe = self.wpe_layer.to_torch_with_grads()
        for i in range(self.config.num_hidden_layers):
            torch_gpt2.h[i] = self.gpt2_blocks[i].to_torch_with_grads()

        torch_gpt2.ln_f = self.final_lnorm.to_torch_with_grads()

        return torch_gpt2
