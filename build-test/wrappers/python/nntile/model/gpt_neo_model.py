# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/gpt_neo_model.py
# GPTNeo model of NNTile Python package
#
# @version 1.1.0

from typing import List, Optional

import numpy as np
from transformers.models.gpt_neo.modeling_gpt_neo import (
    GPTNeoConfig as GPTNeoConfigTorch, GPTNeoModel as GPTNeoModelTorch)

import nntile.utils.constructors as nntc
from nntile.layer.cache_utils import KVCacheStorage
from nntile.tensor import (
    Tensor_bf16, Tensor_fp32, Tensor_fp32_fast_bf16, Tensor_fp32_fast_fp16,
    Tensor_fp32_fast_tf32, Tensor_int64, TensorMoments, TensorTraits)

from ..layer import AddSlice, Embedding, LayerNorm
from .base_model import BaseModel
from .gpt_neo_block import GPTNeoBlock
from .gpt_neo_config import GPTNeoConfig


class GPTNeoModel(BaseModel):
    wte_layer: Embedding
    wpe_layer: Embedding
    add_slice_layer: AddSlice
    final_lnorm: LayerNorm
    gpt_neo_blocks: List[GPTNeoBlock]

    def __init__(self,
                 input_ids: TensorMoments,
                 positional_ids: TensorMoments,
                 wpe_layer_: Embedding,
                 wte_layer_: Embedding,
                 add_slice_layer_: AddSlice,
                 gpt_neo_blocks: List[GPTNeoBlock],
                 lnorm_layer: LayerNorm,
                 config: GPTNeoConfig,
                 ):
        self.dtype = config.dtype

        self.config = config

        if self.dtype not in ["fp32", "tf32",
                              "bf16", "fp32_fast_fp16",
                              "fp32_fast_tf32",
                              "fp32_fast_bf16"]:
            raise TypeError("Only fp32, tf32, bf16, fp32_fast_tf32, "
                            "fp32_fast_fp16 and fp32_fast_bf16 are "
                            "supported for weight type")
        activations = [input_ids, positional_ids]
        activations += wte_layer_.activations_output
        activations += wpe_layer_.activations_output
        activations += add_slice_layer_.activations_output
        layers = [wte_layer_, wpe_layer_, add_slice_layer_]
        self.gpt_neo_blocks = gpt_neo_blocks
        self.wte_layer = layers[0]
        self.wpe_layer = layers[1]
        self.add_slice_layer = layers[2]
        self.final_lnorm = lnorm_layer
        self.seq_len = input_ids.value.shape[0]

        for block_layer in gpt_neo_blocks:
            activations.extend(block_layer.activations[1:])
            layers.extend(block_layer.layers)

        layers.append(lnorm_layer)
        activations.extend(lnorm_layer.activations_output)

        super().__init__(activations, layers, config)

    def forward_dynamic(
            self,
            x: TensorMoments,
            use_cache: bool = False,
            kv_caches: Optional[KVCacheStorage] = None
        ):
        seq_size = x.value.shape[0]
        cache_list = None
        if kv_caches is not None:
            if not kv_caches.is_initialized():
                kv_caches.init(len(self.gpt_neo_blocks), self.seq_len, 1)
            cache_list = kv_caches.get_cache()

        kvcache_size = len(cache_list[0]) if kv_caches else 0
        pos_ids_np = np.asfortranarray(
            np.arange(
                kvcache_size, kvcache_size + seq_size, dtype=np.int64
            )
        )
        pos_ids_nnt_tm = TensorMoments(
            nntc.from_array(
                pos_ids_np, basetile_shape=(x.value.basetile_shape[0],)
            ),
            None,
            False,
        )

        outs_inp = self.wte_layer.forward_dynamic(x)
        outs_pos = self.wpe_layer.forward_dynamic(pos_ids_nnt_tm)
        embedded_input = self.add_slice_layer.forward_dynamic(
            outs_inp, outs_pos)

        block_out = embedded_input
        for lid, block_layer in enumerate(self.gpt_neo_blocks):
            block_out, updated_cache = block_layer.forward_dynamic(
                block_out,
                kv_cache=cache_list[lid] if cache_list else None
            )
            if cache_list:
                cache_list[lid] = updated_cache
        normalized_outs = self.final_lnorm.forward_dynamic(block_out)
        return normalized_outs, kv_caches

    @staticmethod
    def from_torch(torch_gpt_neo: GPTNeoModelTorch,
                   batch_size, batch_size_tile,
                   seq_len, seq_len_tile,
                   config: GPTNeoConfig):

        if config.dtype not in ["fp32", "tf32",
                              "bf16", "fp32_fast_fp16",
                              "fp32_fast_tf32",
                              "fp32_fast_bf16"]:
            raise TypeError("Only fp32, tf32, bf16, fp32_fast_tf32, "
                            "fp32_fast_fp16 and fp32_fast_bf16 are"
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
                             "fp32_fast_tf32": Tensor_fp32_fast_tf32,
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
            torch_gpt_neo.wte.weight.cpu().detach().numpy().T
        )
        wpe_layer.w.value.from_array(
            torch_gpt_neo.wpe.weight.cpu().detach().numpy().T
        )
        add_slice_layer = AddSlice.generate_simple(
            wte_layer.activations_output[0], wpe_layer.activations_output[0],
            2, redux=config.redux
        )

        U = add_slice_layer.activations_output[0]
        gpt_neo_block_list = []

        for block_torch in torch_gpt_neo.h:
            block_nntile = GPTNeoBlock.from_torch(
                block_torch, U, config
            )
            U = block_nntile.activations[-1]
            gpt_neo_block_list.append(block_nntile)

        lnorm_final = LayerNorm.from_torch(
                                    torch_gpt_neo.ln_f,
                                    U,
                                    config.redux)
        X = TensorMoments(x_value, None, False)
        config.attention_types = torch_gpt_neo.config.attention_types
        nntile_gpt_neo = GPTNeoModel(X,
                            positional_ids,
                            wpe_layer,
                            wte_layer,
                            add_slice_layer,
                            gpt_neo_block_list,
                            lnorm_final,
                            config)

        return nntile_gpt_neo

    def to_torch(self):
        config_torch = GPTNeoConfigTorch(
            vocab_size=self.config.vocab_size,
            max_position_embeddings=self.config.max_position_embeddings,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_hidden_layers,
            num_heads=self.config.num_heads,
            intermediate_size=self.config.intermediate_size,
            resid_dropout=0.0,
            embed_dropout=0.0,
            attention_dropout=0.0,
            layer_norm_epsilon=self.config.layer_norm_epsilon,
            use_cache=False,
            attention_types=self.config.attention_types
            )

        torch_gpt_neo = GPTNeoModelTorch(config_torch)
        torch_gpt_neo.wte = self.wte_layer.to_torch()
        torch_gpt_neo.wpe = self.wpe_layer.to_torch()
        for i in range(self.config.num_hidden_layers):
            torch_gpt_neo.h[i] = self.gpt_neo_blocks[i].to_torch()

        torch_gpt_neo.ln_f = self.final_lnorm.to_torch()

        return torch_gpt_neo

    def to_torch_with_grads(self):
        config_torch = GPTNeoConfigTorch(
            vocab_size=self.config.vocab_size,
            max_position_embeddings=self.config.max_position_embeddings,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_hidden_layers,
            num_heads=self.config.num_heads,
            intermediate_size=self.config.intermediate_size,
            resid_dropout=0.0,
            embed_dropout=0.0,
            attention_dropout=0.0,
            layer_norm_epsilon=self.config.layer_norm_epsilon,
            use_cache=False,
            attention_types=self.config.attention_types
            )

        torch_gpt_neo = GPTNeoModelTorch(config_torch)
        torch_gpt_neo.wte = self.wte_layer.to_torch_with_grads()
        torch_gpt_neo.wpe = self.wpe_layer.to_torch_with_grads()
        for i in range(self.config.num_hidden_layers):
            torch_gpt_neo.h[i] = self.gpt_neo_blocks[i].to_torch_with_grads()

        torch_gpt_neo.ln_f = self.final_lnorm.to_torch_with_grads()

        return torch_gpt_neo
