# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/gpt_neo_causal.py
# GPTNeoForCausalLM model of NNTile Python package
#
# @version 1.1.0

from typing import Optional

import torch.nn as nn
from transformers import GPTNeoConfig as ConfigTorch
from transformers.models.gpt_neo.modeling_gpt_neo import (
    GPTNeoForCausalLM as ModelTorch)

import nntile
from nntile.layer.cache_utils import KVCacheStorage
from nntile.model.generation.llm import LLMGenerationMixin
from nntile.types import TensorMoments

from ..layer import Linear
from .base_model import BaseModel
from .gpt_neo_config import GPTNeoConfig
from .gpt_neo_model import GPTNeoModel


class GPTNeoForCausalLM(BaseModel, LLMGenerationMixin):
    gpt_neo_model_: GPTNeoModel
    lin_: Linear

    def __init__(self,
                 gpt_neo_model_: GPTNeoModel,
                 lin_head_: Linear,
                 config: GPTNeoConfig):
        self.dtype = config.dtype

        self.config = config
        self.eos_token_id = self.config.eos_token_id

        if self.dtype not in ["fp32", "tf32",
                              "bf16", "fp32_fast_fp16",
                              "fp32_fast_tf32",
                              "fp32_fast_bf16"]:
            raise TypeError("Only fp32, tf32, bf16, fp32_fast_tf32, "
                            "fp32_fast_fp16 and fp32_fast_bf16 are "
                            "supported for weight type")
        activations = []
        activations.extend(gpt_neo_model_.activations)
        layers = []
        layers.extend(gpt_neo_model_.layers)
        layers.append(lin_head_)

        self.gpt_neo_model_ = gpt_neo_model_
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
        gpt_neo_logits, kv_caches = self.gpt_neo_model_.forward_dynamic(
            x, use_cache=use_cache, kv_caches=kv_caches
        )
        out_logits = self.lin_.forward_dynamic(gpt_neo_logits)
        return out_logits, kv_caches

    @staticmethod
    def from_torch(torch_gpt_neo_causal,
                   batch_size, batch_size_tile,
                   seq_len, seq_len_tile,
                   config: GPTNeoConfig,
                   ):

        if config.dtype not in ["fp32", "tf32",
                              "bf16", "fp32_fast_fp16",
                              "fp32_fast_tf32",
                              "fp32_fast_bf16"]:
            raise TypeError("Only fp32, tf32, bf16, fp32_fast_tf32, "
                            "fp32_fast_fp16 and fp32_fast_bf16 are "
                            "supported for weight type")

        nntile_gpt_neo = GPTNeoModel.from_torch(
                   torch_gpt_neo_causal.transformer,
                   batch_size, batch_size_tile,
                   seq_len, seq_len_tile,
                   config
                   )
        lin_head = Linear.from_torch(torch_gpt_neo_causal.lm_head,
                                               nntile_gpt_neo.activations[-1],
                                               config.vocab_size,
                                               config.redux)

        nntile_gpt_neo_causal = GPTNeoForCausalLM(nntile_gpt_neo,
                                               lin_head,
                                               config)

        return nntile_gpt_neo_causal

    def to_torch(self):
        config_torch = ConfigTorch(
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

        model_torch = ModelTorch(config_torch)
        model_torch.transformer = self.gpt_neo_model_.to_torch()
        model_torch.lm_head = self.lin_.to_torch()
        return model_torch

    def to_torch_with_grads(self):
        config_torch = ConfigTorch(
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

        model_torch = ModelTorch(config_torch)
        model_torch.transformer = self.gpt_neo_model_.to_torch_with_grads()
        model_torch.lm_head = self.lin_.to_torch_with_grads()
        return model_torch

    @classmethod
    def from_pretrained(
        cls,
        batch_size: int,
        batch_size_tile: int,
        seq_len: int,
        cache_dir: str | None = None,
        remote_model_name: str = "EleutherAI/gpt-neo-1.3B",
        dtype: str = "fp32"
    ):
        return create_gpt_neo_model_from_torch_pretrained(
            batch_size,
            batch_size_tile,
            seq_len,
            cache_dir=cache_dir,
            remote_model_name=remote_model_name,
            dtype=dtype
        )


def create_gpt_neo_model_from_torch_pretrained(
    batch_size: int,
    batch_size_tile: int,
    seq_len: int,
    cache_dir: str | None = None,
    remote_model_name: str = "EleutherAI/gpt-neo-1.3B",
    dtype: str = "fp32"
):
    model_torch = ModelTorch.from_pretrained(
        remote_model_name, cache_dir=cache_dir
    )
    config = model_torch.config
    config.attention_dropout = 0.0
    config.resid_dropout = 0.0
    config.embed_dropout = 0.0
    # Current version splits lm_head and wte parameters,
    # shared parameters will be supported soon
    model_torch.lm_head.weight = nn.Parameter(
        model_torch.lm_head.weight.detach().clone()
    )
    model_torch.config.intermediate_size = (
        model_torch.config.intermediate_size or
        4 * model_torch.config.hidden_size
    )
    model_torch.eval()

    config_nntile = GPTNeoConfig(
        vocab_size=model_torch.config.vocab_size,
        vocab_embed_dim_tile=model_torch.config.hidden_size,
        hidden_size=model_torch.config.hidden_size,
        hidden_size_tile=model_torch.config.hidden_size,
        intermediate_size=model_torch.config.intermediate_size,
        intermediate_size_tile=model_torch.config.intermediate_size,
        num_heads=model_torch.config.num_heads,
        num_heads_tile=model_torch.config.num_heads,
        attention_types=model_torch.config.attention_types,
        dtype=dtype,
        layer_norm_epsilon=model_torch.config.layer_norm_epsilon,
        max_position_embeddings=model_torch.config.max_position_embeddings,
        num_hidden_layers=model_torch.config.num_layers,
        bos_token_id=model_torch.config.bos_token_id,
        eos_token_id=model_torch.config.eos_token_id,
        window_size=model_torch.config.window_size
    )

    causal_nntile = GPTNeoForCausalLM.from_torch(
        model_torch,
        batch_size,
        batch_size_tile or batch_size,
        seq_len,
        seq_len,
        config_nntile
    )
    return causal_nntile


def compare_shapes(iterable1, iterable2):
    return all(x == y for x, y in zip(iterable1, iterable2))
