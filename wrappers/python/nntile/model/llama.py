# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/llama.py
# Llama model of NNTile Python package
#
# @version 1.0.0

import numpy as np

from nntile.tensor import (
    Tensor_fp32, Tensor_int64, TensorMoments, TensorTraits)

from ..layer import Embedding, LlamaAttention, RMSNorm
from ..layer.add import Add
# from nntile.layer import Act, Linear, Prod
from .base_model import BaseModel
from .llama_config import LlamaConfigNNTile
from .llama_mlp import LlamaMLP as LlamaMLP_nntile


class Llama(BaseModel):
    next_tag: int

    def __init__(self,
                 input_ids: TensorMoments,
                 positional_ids: TensorMoments,
                 config: LlamaConfigNNTile,
                 next_tag: int,):
        self.dtype = config["dtype"]
        redux = config["redux"]

        if self.dtype not in ["fp32", "tf32", "bf16"]:
            raise TypeError("Only fp32, tf32 and bf16 are"
                            "supported for weight type")
        activations = [input_ids, positional_ids]
        layers = []

        self.num_hidden_layers = config["num_hidden_layers"]

        self.hidden_size = config["hidden_size"]
        self.hidden_size_tile = config["hidden_size_tile"]
        vocab_size = config["vocab_size"]
        vocab_embed_dim_tile = config["vocab_embed_dim_tile"]
        self.embed_dim = config["hidden_size"]
        embed_dim_tile = config["hidden_size_tile"]

        n_head = config["n_attention_heads"]
        n_head_tile = config["n_head_tile"]
        n_head_kv = config["num_key_value_heads"]

        if self.dtype == "fp32":
            embed_layer, next_tag = Embedding.generate_simple(
                input_ids.value,
                Tensor_fp32,
                0,
                vocab_size,
                self.embed_dim,
                embed_dim_tile,
                vocab_embed_dim_tile,
                next_tag,
            )
        layers.append(embed_layer)
        activations.extend(embed_layer.activations_output)

        for _ in range(self.num_hidden_layers):
            norm_layer, next_tag = RMSNorm.generate_simple(activations[-1], 1,
                                                       config["rms_norm_eps"],
                                                       next_tag,
                                                       redux)
            layers.append(norm_layer)
            activations.extend(norm_layer.activations_output)

            attn_layer, next_tag = LlamaAttention.generate_simple(
                                    activations[-1],
                                    n_head,
                                    n_head_tile,
                                    n_head_kv,
                                    next_tag,
                                    redux=redux)
            layers.append(attn_layer)
            activations.extend(attn_layer.activations_output)

            new_layer, next_tag = Add.generate_simple(
                activations[-3], activations[-1], next_tag
            )
            layers.append(new_layer)
            activations.extend(new_layer.activations_output)

            norm_layer, next_tag = RMSNorm.generate_simple(activations[-1], 1,
                                                       config["rms_norm_eps"],
                                                       next_tag,
                                                       redux)
            layers.append(norm_layer)
            activations.extend(norm_layer.activations_output)

            mlp_subnetwork = LlamaMLP_nntile(activations[-1], config, next_tag)
            next_tag = mlp_subnetwork.next_tag
            activations.extend(mlp_subnetwork.activations[1:])
            layers.extend(mlp_subnetwork.layers)

            new_layer, next_tag = Add.generate_simple(
                norm_layer.activations_output, activations[-1], next_tag
            )
            layers.append(new_layer)
            activations.extend(new_layer.activations_output)

        norm_layer, next_tag = RMSNorm.generate_simple(activations[-1], 1,
                                                       config["rms_norm_eps"],
                                                       next_tag,
                                                       redux)
        layers.append(norm_layer)
        activations.extend(norm_layer.activations_output)

        self.next_tag = next_tag
        super().__init__(activations, layers)

    @staticmethod
    def from_torch(torch_llama,
                   batch_size: int,
                   batch_size_tile: int,
                   seq_len: int,
                   seq_len_tile: int,
                   config: LlamaConfigNNTile,
                   next_tag: int):
        positional_ids_traits = TensorTraits([seq_len], [seq_len_tile])
        positional_ids_distr = [0] * positional_ids_traits.grid.nelems
        positional_ids_value = Tensor_int64(
            positional_ids_traits, positional_ids_distr, next_tag
        )
        next_tag = positional_ids_value.next_tag
        positional_ids_value.from_array(
            np.array(np.zeros(seq_len), order="F", dtype=np.int64)
        )
        positional_ids = TensorMoments(positional_ids_value, None, False)

        x_traits = TensorTraits(
            [seq_len, batch_size], [seq_len_tile, batch_size_tile]
        )
        x_distr = [0] * x_traits.grid.nelems
        x = Tensor_int64(x_traits, x_distr, next_tag)
        next_tag = x.next_tag
        x_grad = None
        x_grad_required = False
        x_moments = TensorMoments(x, x_grad, x_grad_required)

        llama_nntile = Llama(x_moments, positional_ids, config, next_tag)
        for p_nntile, (name, p_torch) in zip(llama_nntile.parameters,
                                             torch_llama.named_parameters()):
            print(p_nntile.value.shape, p_torch.shape, name)
            p_nntile.value.from_array(p_torch.cpu().detach().numpy().T)

        return llama_nntile, llama_nntile.next_tag
