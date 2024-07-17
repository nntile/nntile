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
    Tensor_bool, Tensor_fp32, Tensor_int64, TensorMoments, TensorTraits)

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
        activations = [input_ids]
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

        seq_len = input_ids.value.shape[0]
        seq_len_tile = input_ids.value.basetile_shape[0]
        mask_traits = TensorTraits(
            (seq_len, seq_len), (seq_len_tile, seq_len_tile)
        )
        mask_distr = [0] * mask_traits.grid.nelems
        self.mask = Tensor_bool(mask_traits, mask_distr, next_tag)
        next_tag = self.mask.next_tag
        mask_np = np.array(
            np.triu(np.ones((seq_len, seq_len))), dtype=bool, order="F"
        )
        self.mask.from_array(mask_np)

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
            input_act = activations[-1]
            norm_layer, next_tag = RMSNorm.generate_simple(activations[-1], 0,
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
                                    mask=self.mask,
                                    redux=redux)
            layers.append(attn_layer)
            activations.extend(attn_layer.activations_output)

            new_layer, next_tag = Add.generate_simple(
                input_act, activations[-1], next_tag
            )
            layers.append(new_layer)
            activations.extend(new_layer.activations_output)

            norm_layer, next_tag = RMSNorm.generate_simple(activations[-1], 0,
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
                norm_layer.activations_output[0], activations[-1], next_tag
            )
            layers.append(new_layer)
            activations.extend(new_layer.activations_output)

        norm_layer, next_tag = RMSNorm.generate_simple(activations[-1], 0,
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
        nntile_params = llama_nntile.parameters
        torch_named_parameters = list(torch_llama.named_parameters())
        # Copy embedding parameters
        print(torch_named_parameters[0][0])
        nntile_params[0].value.from_array(
            torch_named_parameters[0][1].cpu().detach().numpy().T)
        # Copy the last RMSNorm layer parameters
        print(torch_named_parameters[-1][0])
        nntile_params[-1].value.from_array(
            torch_named_parameters[-1][1].cpu().detach().numpy())
        for layer_idx in range(config["num_hidden_layers"]):
            # First 4 parameters are for Attention (q, k, v, o)
            tmp_q_shape = nntile_params[layer_idx * 9 + 2].value.shape.copy()
            tmp_q_shape[:2] = tmp_q_shape[1::-1]
            print(torch_named_parameters[1 + 9 * layer_idx][0])
            nntile_params[layer_idx * 9 + 2].value.from_array(
                np.moveaxis(
                    torch_named_parameters[1 + 9 * layer_idx][1].detach()
                    .cpu()
                    .numpy()
                    .reshape(*tmp_q_shape),
                    0,
                    1,
                )
            )
            print(torch_named_parameters[2 + 9 * layer_idx][0])
            w_k_shape = nntile_params[layer_idx * 9 + 3].value.shape
            nntile_params[layer_idx * 9 + 3].value.from_array(
                torch_named_parameters[2 + 9 * layer_idx][1].detach()
                .cpu()
                .numpy()
                .reshape(*w_k_shape)
            )
            w_v_shape = nntile_params[layer_idx * 9 + 4].value.shape
            print(torch_named_parameters[3 + 9 * layer_idx][0])
            nntile_params[layer_idx * 9 + 4].value.from_array(
                torch_named_parameters[3 + 9 * layer_idx][1].detach()
                .cpu()
                .numpy()
                .reshape(*w_v_shape)
            )
            print(torch_named_parameters[4 + 9 * layer_idx][0])
            tmp_w_shape = nntile_params[layer_idx * 9 + 5].value.shape.copy()
            tmp_w_shape[1:3] = tmp_w_shape[2:0:-1]
            nntile_params[layer_idx * 9 + 5].value.from_array(
                np.moveaxis(
                    torch_named_parameters[4 + 9 * layer_idx][1].detach()
                    .cpu()
                    .numpy()
                    .reshape(*tmp_w_shape),
                    1,
                    2,
                )
            )
            # Next 3 parameters are for LlamaMLP
            print(torch_named_parameters[1 + layer_idx * 9 + 4][0])
            nntile_params[7 + layer_idx * 9].value.from_array(
                torch_named_parameters[1 + layer_idx * 9 + 4][1].
                cpu().detach().numpy())
            print(torch_named_parameters[1 + layer_idx * 9 + 5][0])
            nntile_params[8 + layer_idx * 9].value.from_array(
                torch_named_parameters[1 + layer_idx * 9 + 5][1].
                cpu().detach().numpy())
            print(torch_named_parameters[1 + layer_idx * 9 + 6][0])
            nntile_params[9 + layer_idx * 9].value.from_array(
                torch_named_parameters[1 + layer_idx * 9 + 6][1].
                cpu().detach().numpy())
            # Next is input RMSNorm
            print(torch_named_parameters[1 + layer_idx * 9 + 7][0])
            nntile_params[1 + layer_idx * 9].value.from_array(
                torch_named_parameters[1 + layer_idx * 9 + 7][1].
                cpu().detach().numpy())
            # Next is RMSNorm after skip connection after attention
            print(torch_named_parameters[1 + layer_idx * 9 + 8][0])
            nntile_params[6 + layer_idx * 9].value.from_array(
                torch_named_parameters[1 + layer_idx * 9 + 8][1].
                cpu().detach().numpy())

        return llama_nntile, llama_nntile.next_tag

    def unregister(self):
        super().unregister()
        if self.mask:
            self.mask.unregister()
