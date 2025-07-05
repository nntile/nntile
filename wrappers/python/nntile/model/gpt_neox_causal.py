# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/gpt_neox_causal.py
# GPTNeoXForCausalLM model of NNTile Python package
#
# @version 1.1.0

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import GPTNeoXConfig as ConfigTorch
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXForCausalLM as ModelTorch)

import nntile
from nntile.layer.cache_utils import KVCacheStorage
from nntile.model.generation.llm import LLMGenerationMixin
from nntile.types import TensorMoments

from ..layer import Linear
from .base_model import BaseModel
from .gpt_neox_config import GPTNeoXConfig
from .gpt_neox_model import GPTNeoXModel


class GPTNeoXForCausalLM(BaseModel, LLMGenerationMixin):
    gpt_neox_model_: GPTNeoXModel
    lin_: Linear

    def __init__(self,
                 gpt_neox_model_: GPTNeoXModel,
                 lin_head_: Linear,
                 config: GPTNeoXConfig):
        self.dtype = config.dtype

        self.config = config
        self.eos_token_id = self.config.eos_token_id

        if self.dtype not in ["fp32", "tf32",
                              "bf16", "fp32_fast_fp16",
                              "fp32_fast_bf16"]:
            raise TypeError("Only fp32, tf32, bf16, fp32_fast_fp16,"
                            "fp32_fast_bf16 are"
                            "supported for weight type")
        activations = []
        activations.extend(gpt_neox_model_.activations)
        layers = []
        layers.extend(gpt_neox_model_.layers)
        layers.append(lin_head_)

        self.gpt_neox_model_ = gpt_neox_model_
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
        gpt_neox_logits, kv_caches = self.gpt_neox_model_.forward_dynamic(
            x, use_cache=use_cache, kv_caches=kv_caches
        )
        out_logits = self.lin_.forward_dynamic(gpt_neox_logits)
        return out_logits, kv_caches

    @staticmethod
    def from_torch(torch_causal,
                   batch_size, batch_size_tile,
                   seq_len, seq_len_tile,
                   position_ids: np.ndarray,
                   mask: np.ndarray,
                   config: GPTNeoXConfig,
                   ):

        if config.dtype not in ["fp32", "tf32",
                              "bf16", "fp32_fast_fp16",
                              "fp32_fast_bf16"]:
            raise TypeError("Only fp32, tf32, bf16, fp32_fast_fp16,"
                            "fp32_fast_bf16 are"
                            "supported for weight type")

        neox_model = GPTNeoXModel.from_torch(
                   torch_causal.gpt_neox,
                   batch_size, batch_size_tile,
                   seq_len, seq_len_tile,
                   position_ids,
                   mask,
                   config)
        lin_head = Linear.from_torch(torch_causal.embed_out,
                                               neox_model.activations[-1],
                                               config.vocab_size,
                                               config.redux)

        nntile_causal = GPTNeoXForCausalLM(neox_model,
                                          lin_head,
                                          config)

        return nntile_causal

    def to_torch(self):
        config_torch = ConfigTorch(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_heads,
            intermediate_size=self.config.intermediate_size,
            rotary_pct=self.config.rotary_pct,
            rotary_emb_base=self.config.rotary_emb_base,
            attention_bias=self.config.attention_bias,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            max_position_embeddings=self.config.max_position_embeddings,
            layer_norm_eps=self.config.layer_norm_epsilon,
            use_cache=False,
        )

        causal_torch = ModelTorch(config_torch)
        causal_torch.gpt_neox = self.gpt_neox_model_.to_torch()
        causal_torch.embed_out = self.lin_.to_torch()
        return causal_torch

    def to_torch_with_grads(self):
        config_torch = ConfigTorch(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_heads,
            intermediate_size=self.config.intermediate_size,
            rotary_pct=self.config.rotary_pct,
            rotary_emb_base=self.config.rotary_emb_base,
            attention_bias=self.config.attention_bias,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            max_position_embeddings=self.config.max_position_embeddings,
            layer_norm_eps=self.config.layer_norm_epsilon,
            use_cache=False,
        )

        causal_torch = ModelTorch(config_torch)
        causal_torch.gpt_neox = self.gpt_neox_model_.to_torch_with_grads()
        causal_torch.embed_out = self.lin_.to_torch_with_grads()
        return causal_torch

    @classmethod
    def from_pretrained(
        cls,
        batch_size: int,
        batch_size_tile: int,
        seq_len: int,
        cache_dir: str | None = None,
        remote_model_name: str = "EleutherAI/gpt-neox-20b",
        dtype: str = "fp32"
    ):
        return create_gpt_neox_model_from_torch_pretrained(
            batch_size,
            batch_size_tile,
            seq_len,
            cache_dir=cache_dir,
            remote_model_name=remote_model_name,
            dtype=dtype
        )


def compare_shapes(iterable1, iterable2):
    return all(x == y for x, y in zip(iterable1, iterable2))


def create_gpt_neox_model_from_torch_pretrained(
    batch_size: int,
    batch_size_tile: int,
    seq_len: int,
    cache_dir: str | None = None,
    remote_model_name: str = "EleutherAI/gpt-neox-20b",
    dtype: str = "fp32"
):
    model_torch = ModelTorch.from_pretrained(
        remote_model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16
    )
    model_torch.embed_out.weight = nn.Parameter(
        model_torch.embed_out.weight.detach().clone()
    )
    model_torch.eval()

    config_nntile = GPTNeoXConfig(
        vocab_size=model_torch.config.vocab_size,
        vocab_embed_dim_tile=model_torch.config.hidden_size,
        hidden_size=model_torch.config.hidden_size,
        hidden_size_tile=model_torch.config.hidden_size,
        intermediate_size=model_torch.config.intermediate_size,
        intermediate_size_tile=model_torch.config.intermediate_size,
        num_heads=model_torch.config.num_attention_heads,
        num_heads_tile=model_torch.config.num_attention_heads,
        dtype=dtype,
        layer_norm_epsilon=model_torch.config.layer_norm_eps,
        max_position_embeddings=model_torch.config.max_position_embeddings,
        num_hidden_layers=model_torch.config.num_hidden_layers,
        redux=False,
        bos_token_id=model_torch.config.bos_token_id,
        eos_token_id=model_torch.config.eos_token_id,
        rotary_pct=model_torch.config.rotary_pct,
        rotary_emb_base=model_torch.config.rotary_emb_base,
        attention_bias=model_torch.config.attention_bias,
    )

    single_batch_pos_ids = np.arange(seq_len).reshape(1, seq_len)
    pos_ids = np.repeat(single_batch_pos_ids, batch_size, axis=0)

    mask = np.array(np.triu(np.ones((seq_len, seq_len))),
                        dtype=bool, order="F")

    causal_nntile = GPTNeoXForCausalLM.from_torch(
        model_torch,
        batch_size,
        batch_size_tile or batch_size,
        seq_len,
        seq_len,
        pos_ids,
        mask,
        config_nntile
    )

    return causal_nntile
