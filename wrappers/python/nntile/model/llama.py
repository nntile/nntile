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
# @version 1.1.0

from typing import List, Optional

import numpy as np
from transformers import LlamaConfig as LlamaConfig_torch
from transformers.models.llama.modeling_llama import (
    LlamaModel as LlamaModel_torch)

from nntile.layer.cache_utils import KVCacheStorage
from nntile.tensor import (
    Tensor_bf16, Tensor_fp32, Tensor_fp32_fast_tf32, Tensor_int64,
    TensorMoments, TensorTraits)

from ..layer import Embedding, RMSNorm
from .base_model import BaseModel
from .llama_config import LlamaConfigNNTile
from .llama_decoder import LlamaDecoder


class Llama(BaseModel):
    embd_layer: Embedding
    final_rmsnorm: RMSNorm
    list_decoder: List[LlamaDecoder]

    def __init__(self,
                 input_ids: TensorMoments,
                 emb_layer_: Embedding,
                 decoders: List[LlamaDecoder],
                 rms_norm_layer: RMSNorm,
                 config: LlamaConfigNNTile):
        self.dtype = config.dtype

        self.config = config

        if self.dtype not in ["fp32", "fp32_fast_tf32", "bf16"]:
            raise TypeError("Only fp32, fp32_fast_tf32 and bf16 are"
                            "supported for weight type")
        activations = [input_ids] + emb_layer_.activations_output
        layers = [emb_layer_]
        self.decoders = decoders
        self.embd_layer = emb_layer_
        self.final_rmsnorm = rms_norm_layer
        self.seq_len = input_ids.value.shape[0]

        for dec_layer in decoders:
            activations.extend(dec_layer.activations[1:])
            layers.extend(dec_layer.layers)

        activations.extend(rms_norm_layer.activations_output)
        layers.append(rms_norm_layer)

        super().__init__(activations, layers, config)

    def forward_dynamic(
            self,
            x: TensorMoments,
            use_cache: bool = False,
            kv_caches: Optional[KVCacheStorage] = None
        ):
        cache_list = None
        if kv_caches is not None:
            if not kv_caches.is_initialized():
                kv_caches.init(len(self.decoders), self.seq_len, 1)
            cache_list = kv_caches.get_cache()

        x_emb = self.embd_layer.forward_dynamic(x)

        dec_out = x_emb
        for lid, dec_layer in enumerate(self.decoders):
            dec_out, updated_cache = dec_layer.forward_dynamic(
                dec_out,
                kv_cache=cache_list[lid] if cache_list else None
            )
            if cache_list:
                cache_list[lid] = updated_cache
        normalized_outs = self.final_rmsnorm.forward_dynamic(dec_out)
        return normalized_outs, kv_caches

    @staticmethod
    def from_torch(torch_llama,
                   batch_size, batch_size_tile,
                   seq_len, seq_len_tile,
                   position_ids: np.ndarray,
                   mask: np.ndarray,
                   config: LlamaConfigNNTile):

        if config.dtype not in ["fp32", "fp32_fast_tf32", "bf16"]:
            raise TypeError("Only fp32, fp32_fast_tf32 and bf16 are"
                            "supported for weight type")

        x_shape = [seq_len, batch_size]
        x_basetile = [seq_len_tile, batch_size_tile]
        x_traits = TensorTraits(x_shape, x_basetile)
        x_distr = [0] * x_traits.grid.nelems
        x_value = Tensor_int64(x_traits, x_distr)

        dtype2tensor_type = {"fp32": Tensor_fp32,
                             "bf16": Tensor_bf16,
                             "fp32_fast_tf32": Tensor_fp32_fast_tf32
                            }

        tensor_type = dtype2tensor_type[config.dtype]

        embed_layer = Embedding.generate_simple(
                                    x_value, tensor_type, 0,
                                    config.vocab_size,
                                    config.hidden_size,
                                    config.hidden_size_tile,
                                    config.hidden_size_tile)

        embed_layer.w.value.from_array(torch_llama.embed_tokens.weight.cpu().detach().numpy().T)
        U = embed_layer.activations_output[0]
        decoders_list = []

        for decoder_llama_torch in torch_llama.layers:
            decoder_nntile_layer = LlamaDecoder.from_torch(
                decoder_llama_torch, U, position_ids, mask, config)
            U = decoder_nntile_layer.activations[-1]
            decoders_list.append(decoder_nntile_layer)

        rms_norm_final = RMSNorm.from_torch(
                                    torch_llama.norm,
                                    decoders_list[-1].activations[-1],
                                    0,
                                    config.rms_norm_eps,
                                    config.redux)
        X = TensorMoments(x_value, None, False)
        llama_nntile = Llama(X,
                             embed_layer,
                             decoders_list,
                             rms_norm_final,
                             config)

        return llama_nntile

    def to_torch(self):
        config_torch = LlamaConfig_torch(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            num_hidden_layers=self.config.num_hidden_layers,
            vocab_size=self.config.vocab_size,
            max_position_embeddings=self.config.max_position_embeddings,
            rms_norm_eps=self.config.rms_norm_eps,
            n_attention_head=self.config.n_attention_head)

        llama_model_torch = LlamaModel_torch(config_torch)
        llama_model_torch.embed_tokens = self.layers[0].to_torch()
        for i in range(self.config.num_hidden_layers):
            llama_model_torch.layers[i] = self.decoders[i].to_torch()

        llama_model_torch.norm = self.final_rmsnorm.to_torch()

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

        llama_model_torch = LlamaModel_torch(config_torch)
        llama_model_torch.embed_tokens = self.embd_layer.to_torch_with_grads()
        for i in range(self.config.num_hidden_layers):
            llama_model_torch.layers[i] = \
                self.decoders[i].to_torch_with_grads()

        llama_model_torch.norm = self.final_rmsnorm.to_torch_with_grads()

        return llama_model_torch
