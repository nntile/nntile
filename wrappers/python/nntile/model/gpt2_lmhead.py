# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/gpt2_lmhead.py
# GPT2LMHead model of NNTile Python package
#
# @version 1.1.0

from typing import Optional

from transformers import GPT2Config as GPT2ConfigTorch
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2LMHeadModel as GPT2Model_torch)

import nntile
from nntile.layer.cache_utils import KVCacheStorage
from nntile.model.generation.llm import LLMGenerationMixin
from nntile.tensor import TensorMoments

from ..layer import Linear
from .base_model import BaseModel
from .gpt2_config import GPT2ConfigNNTile
from .gpt2_model import GPT2Model as GPT2_nntile


class GPT2LMHead(BaseModel, LLMGenerationMixin):
    gpt2_model_: GPT2_nntile
    lin_: Linear

    def __init__(self,
                 gpt2_model_: GPT2_nntile,
                 lin_head_: Linear,
                 config: GPT2ConfigNNTile):
        self.dtype = config.dtype

        self.config = config
        self.eos_token_id = self.config.eos_token_id

        if self.dtype not in ["fp32", "fp16", "tf32",
                              "bf16", "fp32_fast_fp16",
                              "fp32_fast_tf32",
                              "fp32_fast_bf16"]:
            raise TypeError("Only fp32, fp16, tf32, bf16, fp32_fast_tf32, "
                            "fp32_fast_fp16 and fp32_fast_bf16 are "
                            "supported for weight type")
        activations = []
        activations.extend(gpt2_model_.activations)
        layers = []
        layers.extend(gpt2_model_.layers)
        layers.append(lin_head_)

        self.gpt2_model_ = gpt2_model_
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
        print("GPT2LMHead.forward_dynamic called")
        if kv_caches is None and use_cache:
            kv_caches = KVCacheStorage()

        gpt2_logits, kv_caches = self.gpt2_model_.forward_dynamic(
            x, use_cache=use_cache, kv_caches=kv_caches
        )
        out_logits = self.lin_.forward_dynamic(gpt2_logits)
        return out_logits, kv_caches

    @staticmethod
    def from_torch(torch_gpt2_lmhead,
                   batch_size, batch_size_tile,
                   seq_len, seq_len_tile,
                   config: GPT2ConfigNNTile):
        if config.dtype not in ["fp32", "fp16", "tf32",
                              "bf16", "fp32_fast_fp16",
                              "fp32_fast_tf32",
                              "fp32_fast_bf16"]:
            raise TypeError("Only fp32, fp16, tf32, bf16, fp32_fast_tf32, "
                            "fp32_fast_fp16 and fp32_fast_bf16 are "
                            "supported for weight type")

        nntile_gpt2 = GPT2_nntile.from_torch(
                   torch_gpt2_lmhead.transformer,
                   batch_size, batch_size_tile,
                   seq_len, seq_len_tile,
                   config)
        lin_head = Linear.from_torch(torch_gpt2_lmhead.lm_head,
                                               nntile_gpt2.activations[-1],
                                               config.vocab_size,
                                               config.redux)

        nntile_gpt2_lmhead = GPT2LMHead(nntile_gpt2,
                                               lin_head,
                                               config)

        return nntile_gpt2_lmhead

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

        gpt2_model_torch = GPT2Model_torch(config_torch)
        gpt2_model_torch.transformer = self.gpt2_model_.to_torch()
        gpt2_model_torch.lm_head = self.lin_.to_torch()
        return gpt2_model_torch

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

        gpt2_model_torch = GPT2Model_torch(config_torch)
        gpt2_model_torch.transformer = self.gpt2_model_.to_torch_with_grads()
        gpt2_model_torch.lm_head = self.lin_.to_torch_with_grads()
        return gpt2_model_torch

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        batch_size: int,
        batch_size_tile: int,
        seq_len_tile: int,
        cache_dir: str | None = None,
    ):
        # TODO: where should be global repo with all this logic.
        # We need to design it.
        # For now, manual code for usability
        return create_gpt2_model_from_torch_pretrained(
            model_name,
            batch_size,
            batch_size_tile,
            seq_len_tile,
            cache_dir=cache_dir,
        )


def compare_shapes(iterable1, iterable2):
    return all(x == y for x, y in zip(iterable1, iterable2))


PretrainedGpt2Configs = {
    "gpt2": GPT2ConfigTorch(
        vocab_size=50257,
        vocab_embed_dim_tile=384,
        embed_dim=768,
        embed_dim_tile=384,
        max_position_embeddings=1024,
        inner_dim=3072,
        inner_dim_tile=1536,
        layer_norm_epsilon=1e-05,
        num_hidden_layers=12,
        n_head=12,
        n_head_tile=12,
        activation_function="gelutanh",
        flashattention=False,
        use_redux=False,
        dtype="fp32",
    )
}


def create_gpt2_model_from_torch_pretrained(
    model_name: str,
    batch_size: int,
    batch_size_tile: int,
    seq_len_tile: int,
    cache_dir: str | None = None,
):
    import torch.nn as nn
    if model_name not in PretrainedGpt2Configs:
        raise Exception(
            f"Unsupported pretrained model: {model_name}."
            "Try create manually with GPT2Model_nntile.from_torch."
            "Currently supported: {list(PretrainedGpt2Configs.keys())}"
        )

    nntile_model_config = PretrainedGpt2Configs[model_name]

    model_torch = GPT2Model_torch.from_pretrained(
        model_name, cache_dir=cache_dir
    )

    config = model_torch.config
    config.attn_pdrop = 0
    config.embd_pdrop = 0
    config.resid_pdrop = 0
    # Current version splits lm_head and wte parameters,
    # shared parameters will be supported soon
    model_torch.lm_head.weight = nn.Parameter(
        model_torch.lm_head.weight.detach().clone()
    )

    inner_dim = (
        config.n_inner if config.n_inner is not None
        else 4 * config.hidden_size
    )
    config.n_inner = inner_dim

    nntile_model = GPT2LMHead.from_torch(
        model_torch,
        batch_size,
        batch_size_tile,
        config.n_positions,
        seq_len_tile,
        nntile_model_config,
    )

    return nntile_model
