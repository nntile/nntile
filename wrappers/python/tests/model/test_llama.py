# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/model/test_llama.py
# Test for nntile.model.Llama
# Each test is generated in float precision by Torch, then it is downcasted
# into NNTile type. So, implementation of double precision is NOT checked.
#
# @version 1.0.0

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from transformers.models.llama import (
    LlamaConfig as LlamaConfig_torch, LlamaModel as LlamaModel_torch)

import nntile
from nntile.model.llama import Llama as LlamaModel
from nntile.model.llama_config import LlamaConfigNNTile
from nntile.tensor import to_numpy

# NNTile dtype via corresponding Tensor type
dtype2nntile = {
        'fp32': nntile.tensor.Tensor_fp32,
        'fp32_fast_tf32': nntile.tensor.Tensor_fp32_fast_tf32,
        'bf16': nntile.tensor.Tensor_bf16,
}

dtype2tol = {
        'fp32': {'rtol': 1e-6},
        'fp32_fast_tf32': {'rtol': 1e-4},
        'bf16': {'rtol': 1.6e-2},
}


def assert_close_by_frobnorm(a: np.ndarray, b: np.ndarray, rtol: float):
    np.testing.assert_array_less(
            np.linalg.norm(a - b),
            rtol * np.linalg.norm(a)
    )


@dataclass
class LlamaTestParams:
    vocab_size: int
    vocab_embed_dim_tile: int
    hidden_size: int
    hidden_size_tile: int
    max_position_embeddings: int
    intermediate_size: int
    intermediate_size_tile: int
    rms_norm_eps: float
    num_hidden_layers: int
    num_attention_heads: int
    num_attention_heads_tile: int
    num_key_value_heads: int
    activation_function: str = "silu"
    flashattention: bool = True
    attention_bias: bool = False
    attention_dropout: float = 0.0
    rope_theta: float = 10000.
    seq_len: int = 1
    seq_len_tile: int = 1
    batch_size: int = 1
    batch_size_tile: int = 1
    dtype: str = "fp32"
    redux: bool = False


TEST_PARAMS = [
    pytest.param(
        LlamaTestParams(
            vocab_size=32000,
            vocab_embed_dim_tile=32,
            hidden_size=128,
            hidden_size_tile=32,
            max_position_embeddings=1024,
            intermediate_size=384,
            intermediate_size_tile=96,
            rms_norm_eps=1e-6,
            num_hidden_layers=1,
            num_attention_heads=16,
            num_attention_heads_tile=8,
            num_key_value_heads=4,
            activation_function="silu",
            flashattention=False,
            attention_bias=False,
            attention_dropout=0.0,
            rope_theta=2.,
            seq_len=64,
            seq_len_tile=16,
            batch_size=4,
            batch_size_tile=1,
            dtype='fp32',
            redux=False
        )
    ),
]


def generate_inputs(params: LlamaTestParams):
    torch_config = LlamaConfig_torch(
            vocab_size=params.vocab_size,
            hidden_size=params.hidden_size,
            max_position_embeddings=params.max_position_embeddings,
            intermediate_size=params.intermediate_size,
            num_attention_heads=params.num_attention_heads,
            num_key_value_heads=params.num_key_value_heads,
            attention_bias=params.attention_bias,
            use_cache=False,
            attention_dropout=0.0,
            num_hidden_layers=params.num_hidden_layers,
            rms_norm_eps=params.rms_norm_eps
    )

    torch_model = LlamaModel_torch(
            torch_config,
    )
    nntile_config = LlamaConfigNNTile(
            vocab_size=params.vocab_size,
            vocab_embed_dim_tile=params.hidden_size_tile,
            hidden_size=params.hidden_size,
            hidden_size_tile=params.hidden_size_tile,
            intermediate_size=params.intermediate_size,
            intermediate_size_tile=params.intermediate_size_tile,
            num_hidden_layers=params.num_hidden_layers,
            rms_norm_eps=params.rms_norm_eps,
            max_position_embeddings=torch_config.max_position_embeddings,
            n_attention_head=torch_config.num_attention_heads,
            n_head_tile=torch_config.num_attention_heads,
            num_key_value_heads=torch_config.num_key_value_heads,
    )
    gen = np.random.default_rng()
    pos_ids = gen.integers(
            params.seq_len,
            size=(params.batch_size, params.seq_len),
            dtype=np.int64
    )
    mask = np.array(np.triu(np.ones((params.seq_len, params.seq_len))),
                    dtype=bool, order="F")
    nntile_model, _ = LlamaModel.from_torch(
            torch_model, params.batch_size, params.batch_size_tile,
            params.seq_len, params.seq_len_tile, pos_ids,
            mask, nntile_config, 0)
    nntile_model.clear_gradients()
    gen = np.random.default_rng()
    x_random = gen.integers(0,
                            params.seq_len,
                            nntile_model.activations[0].value.shape)

    x_nntile = np.array(x_random, np.int64, order='F')
    nntile_model.activations[0].value.from_array(x_nntile)
    x_torch = torch.tensor(x_nntile.T)
    y_grad_random = gen.standard_normal((params.hidden_size,
                                         params.seq_len,
                                         params.batch_size))
    y_grad_nntile = np.array(y_grad_random, dtype=np.float32, order='F')
    nntile_model.activations[-1].grad.from_array(y_grad_nntile)
    y_grad_torch = torch.Tensor(y_grad_nntile.T)
    return torch_model, nntile_model, x_torch, pos_ids, y_grad_torch


@pytest.mark.parametrize("params", TEST_PARAMS)
class TestLlama:
    def test_coercion(
        self, starpu_simple, torch_rng, params: LlamaTestParams
    ):
        torch_model, nntile_model, _, _, _ = generate_inputs(params)
        torch_model_other = nntile_model.to_torch()
        nntile_model.unregister()

        assert_close_by_frobnorm(
            torch_model.embed_tokens.weight.detach().numpy(),
            torch_model_other.embed_tokens.weight.detach().numpy(),
            **dtype2tol[params.dtype]
        )
        for i in range(params.num_hidden_layers):
            assert_close_by_frobnorm(
                torch_model.layers[i].mlp.down_proj.weight.detach().numpy(),
                torch_model_other.layers[i].mlp.down_proj.weight.detach().numpy(),
                **dtype2tol[params.dtype]
            )
            assert_close_by_frobnorm(
                torch_model.layers[i].mlp.up_proj.weight.detach().numpy(),
                torch_model_other.layers[i].mlp.up_proj.weight.detach().numpy(),
                **dtype2tol[params.dtype]
            )
            assert_close_by_frobnorm(
                torch_model.layers[i].mlp.gate_proj.weight.detach().numpy(),
                torch_model_other.layers[i].mlp.gate_proj.weight.detach().numpy(),
                **dtype2tol[params.dtype]
            )

            assert_close_by_frobnorm(
                torch_model.layers[i].post_attention_layernorm.weight.detach().numpy(),
                torch_model_other.layers[i].post_attention_layernorm.weight.detach().numpy(),
                **dtype2tol[params.dtype]
            )
            assert_close_by_frobnorm(
                torch_model.layers[i].input_layernorm.weight.detach().numpy(),
                torch_model_other.layers[i].input_layernorm.weight.detach().numpy(),
                **dtype2tol[params.dtype]
            )

            assert_close_by_frobnorm(
                torch_model.layers[i].self_attn.q_proj.weight.detach().numpy(),
                torch_model_other.layers[i].self_attn.q_proj.weight.detach().numpy(),
                **dtype2tol[params.dtype]
            )
            assert_close_by_frobnorm(
                torch_model.layers[i].self_attn.v_proj.weight.detach().numpy(),
                torch_model_other.layers[i].self_attn.v_proj.weight.detach().numpy(),
                **dtype2tol[params.dtype]
            )
            assert_close_by_frobnorm(
                torch_model.layers[i].self_attn.o_proj.weight.detach().numpy(),
                torch_model_other.layers[i].self_attn.o_proj.weight.detach().numpy(),
                **dtype2tol[params.dtype]
            )
            assert_close_by_frobnorm(
                torch_model.layers[i].self_attn.k_proj.weight.detach().numpy(),
                torch_model_other.layers[i].self_attn.k_proj.weight.detach().numpy(),
                **dtype2tol[params.dtype]
            )

    def test_forward(
        self, starpu_simple, torch_rng, params: LlamaTestParams
    ):
        torch_model, nntile_model, x, pos_ids, _ = generate_inputs(params)
        y = torch_model(x, position_ids=torch.tensor(pos_ids),
                        return_dict=True)
        nntile_model.forward_async()
        y_nntile_np = to_numpy(nntile_model.activations[-1].value)
        nntile_model.unregister()
        assert_close_by_frobnorm(
                y.last_hidden_state.detach().numpy(),
                y_nntile_np.T,
                **dtype2tol[params.dtype]
        )

    def test_forward_backward(
        self, starpu_simple, torch_rng, params: LlamaTestParams
    ):
        torch_model, nntile_model, x, pos_ids, y_grad = generate_inputs(params)
        y = torch_model(x, position_ids=torch.tensor(pos_ids))
        nntile_model.forward_async()
        y_nntile_np = to_numpy(nntile_model.activations[-1].value).T
        res = (y.last_hidden_state * y_grad).sum()
        res.backward()
        nntile_model.backward_async()
        torch_model_other = nntile_model.to_torch_with_grads()
        nntile_model.unregister()
        assert_close_by_frobnorm(
                y.last_hidden_state.detach().numpy(),
                y_nntile_np,
                **dtype2tol[params.dtype]
        )
        # Embedding grad
        assert_close_by_frobnorm(
            torch_model.embed_tokens.weight.grad.detach().numpy(),
            torch_model_other.embed_tokens.weight.grad.detach().numpy(),
            **dtype2tol[params.dtype]
        )
        for i in range(params.num_hidden_layers):

            # MLP gradients
            assert_close_by_frobnorm(
                torch_model.layers[i].mlp.up_proj.weight.grad.detach().numpy(),
                torch_model_other.layers[i].mlp.up_proj.weight.grad.detach().numpy(),
                **dtype2tol[params.dtype]
            )
            assert_close_by_frobnorm(
                torch_model.layers[i].mlp.gate_proj.weight.grad.detach().numpy(),
                torch_model_other.layers[i].mlp.gate_proj.weight.grad.detach().numpy(),
                **dtype2tol[params.dtype]
            )
            assert_close_by_frobnorm(
                torch_model.layers[i].mlp.down_proj.weight.grad.detach().numpy(),
                torch_model_other.layers[i].mlp.down_proj.weight.grad.detach().numpy(),
                **dtype2tol[params.dtype]
            )
            # Normalizations gradients
            assert_close_by_frobnorm(
                torch_model.layers[i].post_attention_layernorm.weight.grad.detach().numpy(),
                torch_model_other.layers[i].post_attention_layernorm.weight.grad.detach().numpy(),
                **dtype2tol[params.dtype]
            )
            assert_close_by_frobnorm(
                torch_model.layers[i].input_layernorm.weight.grad.detach().numpy(),
                torch_model_other.layers[i].input_layernorm.weight.grad.detach().numpy(),
                **dtype2tol[params.dtype]
            )
            # Attention gradients
            assert_close_by_frobnorm(
                torch_model.layers[i].self_attn.q_proj.weight.grad.detach().numpy(),
                torch_model_other.layers[i].self_attn.q_proj.weight.grad.detach().numpy(),
                **dtype2tol[params.dtype]
            )
            assert_close_by_frobnorm(
                torch_model.layers[i].self_attn.v_proj.weight.grad.detach().numpy(),
                torch_model_other.layers[i].self_attn.v_proj.weight.grad.detach().numpy(),
                **dtype2tol[params.dtype]
            )
            assert_close_by_frobnorm(
                torch_model.layers[i].self_attn.o_proj.weight.grad.detach().numpy(),
                torch_model_other.layers[i].self_attn.o_proj.weight.grad.detach().numpy(),
                **dtype2tol[params.dtype]
            )
            assert_close_by_frobnorm(
                torch_model.layers[i].self_attn.k_proj.weight.grad.detach().numpy(),
                torch_model_other.layers[i].self_attn.k_proj.weight.grad.detach().numpy(),
                **dtype2tol[params.dtype]
            )
        assert_close_by_frobnorm(
                torch_model.norm.weight.grad.detach().numpy(),
                torch_model_other.norm.weight.grad.detach().numpy(),
                **dtype2tol[params.dtype]
            )
