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

from typing import Optional

import numpy as np
from transformers import LlamaConfig as LlamaConfig_torch
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM as LlamaCausalModel_torch)

import nntile
from nntile.layer.cache_utils import KVCacheStorage
from nntile.model.generation.llm import LLMGenerationMixin
from nntile.types import TensorMoments

from ..layer import Linear
from .base_model import BaseModel
from .llama import Llama as Llama_nntile
from .llama_config import LlamaConfigNNTile


class LlamaForCausalLM(BaseModel, LLMGenerationMixin):
    llama_model_: Llama_nntile
    lin_: Linear

    def __init__(self,
                 llama_model_: Llama_nntile,
                 lin_head_: Linear,
                 config: LlamaConfigNNTile):
        self.dtype = config.dtype

        self.config = config
        self.eos_token_id = self.config.eos_token_id

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

        super().__init__(activations, layers, config)

    def set_input(self, x: nntile.tensor.Tensor):
        expected_shape = self.activations[0].value.shape
        if not compare_shapes(x.shape, expected_shape):
            raise Exception(
                "Mismatch shapes. Got: ", x.shape,
                " Expected: ", expected_shape
            )

        nntile.functions.copy_async(x, self.activations[0].value)

    def get_output(self) -> nntile.tensor.Tensor:
        return self.activations[-1].value

    def forward(self, x: nntile.tensor.Tensor) -> nntile.tensor.Tensor:
        self.set_input(x)
        self.forward_async()
        return self.get_output()

    def forward_dynamic(
            self, x: TensorMoments,
            use_cache: bool = False,
            kv_caches: Optional[KVCacheStorage] = None
        ):
        llama_logits, kv_caches = self.llama_model_.forward_dynamic(
            x, use_cache=use_cache, kv_caches=kv_caches
        )
        out_logits = self.lin_.forward_dynamic(llama_logits)
        return out_logits, kv_caches

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        seq_len: int,
        batch_size: int = 1,
        batch_size_tile: int | None = None,
        seq_len_tile: int | None = None,
        hidden_size_tile: int | None = None,
        intermediate_size_tile: int | None = None,
        n_head_tile: int | None = None,
        dtype: str = 'fp32',
        flash_attention: bool = False,
        cache_dir: str | None = None,
    ):
        # TODO: where should be global repo with all this logic.
        # We need to design it.
        # For now, manual code for usability
        return create_llama_model_from_torch_pretrained(
            model_name=model_name,
            batch_size=batch_size,
            batch_size_tile=batch_size_tile,
            seq_len=seq_len,
            seq_len_tile=seq_len_tile,
            hidden_size_tile=hidden_size_tile,
            intermediate_size_tile=intermediate_size_tile,
            n_head_tile=n_head_tile,
            dtype=dtype,
            flash_attention=flash_attention,
            cache_dir=cache_dir,
        )

    @staticmethod
    def from_torch(torch_llama_causal,
                   batch_size, batch_size_tile,
                   seq_len, seq_len_tile,
                   position_ids: np.ndarray,
                   mask: np.ndarray,
                   config: LlamaConfigNNTile):

        if config.dtype not in ["fp32", "fp32_fast_tf32", "bf16"]:
            raise TypeError("Only fp32, fp32_fast_tf32 and bf16 are"
                            "supported for weight type")

        llama_model = Llama_nntile.from_torch(
                   torch_llama_causal.model,
                   batch_size, batch_size_tile,
                   seq_len, seq_len_tile,
                   position_ids,
                   mask,
                   config)
        lin_head = Linear.from_torch(torch_llama_causal.lm_head,
                                               llama_model.activations[-1],
                                               config.vocab_size,
                                               config.redux)

        causal_llama_nntile = LlamaForCausalLM(llama_model,
                                               lin_head,
                                               config)

        return causal_llama_nntile

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


def create_llama_model_from_torch_pretrained(
    model_name: str,
    batch_size: int,
    batch_size_tile: int | None,
    seq_len: int,
    seq_len_tile: int | None,
    hidden_size_tile: int | None,
    intermediate_size_tile: int | None,
    n_head_tile: int | None,
    dtype: str,
    flash_attention: bool,
    cache_dir: str | None = None
):
    model_torch = LlamaCausalModel_torch.from_pretrained(
        model_name, cache_dir=cache_dir
    )
    model_torch.eval()

    llama_config_nntile = LlamaConfigNNTile(
        vocab_size=model_torch.vocab_size,
        vocab_embed_dim_tile=model_torch.config.hidden_size,
        hidden_size=model_torch.config.hidden_size,
        hidden_size_tile=hidden_size_tile or model_torch.config.hidden_size,
        max_position_embeddings=model_torch.config.max_position_embeddings,
        num_hidden_layers=model_torch.config.num_hidden_layers,
        rms_norm_eps=model_torch.config.rms_norm_eps,
        n_attention_head=model_torch.config.num_attention_heads,
        num_key_value_heads=model_torch.config.num_key_value_heads,
        intermediate_size=model_torch.config.intermediate_size,
        intermediate_size_tile=intermediate_size_tile or model_torch.config.intermediate_size,  # noqa: E501
        n_head_tile=n_head_tile or model_torch.config.num_attention_heads,
        dtype=dtype,
        flash_attention=flash_attention
    )

    single_batch_pos_ids = np.arange(seq_len).reshape(1, seq_len)
    pos_ids = np.repeat(single_batch_pos_ids, batch_size, axis=0)

    mask = np.array(np.triu(np.ones((seq_len, seq_len))),
                        dtype=bool, order="F")

    llama_causal_nntile = LlamaForCausalLM.from_torch(
        model_torch,
        batch_size,
        batch_size_tile or batch_size,
        seq_len,
        seq_len_tile or seq_len,
        pos_ids,
        mask,
        llama_config_nntile
    )

    return llama_causal_nntile


def compare_shapes(iterable1, iterable2):
    return all(x == y for x, y in zip(iterable1, iterable2))
