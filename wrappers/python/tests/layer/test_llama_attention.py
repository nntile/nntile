# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/layer/test_llama_attention.py
# Test for nntile.layer.LlamaAttention
#
# @version 1.0.0

from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import ( \
        LlamaConfig as LlamaConfig_torch, \
        LlamaAttention as LlamaAttention_torch)

import nntile
import nntile.utils
from nntile.tensor import TensorTraits, TensorMoments
from nntile.layer import LlamaAttention

@dataclass
class TestParamsLlamaAttention:
    n_emb: int
    n_emb_tile: int
    n_seq: int
    n_seq_tile: int
    n_batch: int
    n_batch_tile: int
    n_head: int
    n_head_tile: int
    n_head_kv: int
    dtype: np.dtype
    bias: bool
    layer_idx: int=0

TEST_PARAMS = \
[ \
    TestParamsLlamaAttention(n_emb=128, n_emb_tile=128, n_seq=64, \
        n_seq_tile=64, n_batch=3, n_batch_tile=3, n_head=8, n_head_tile=4, \
        n_head_kv=4, dtype=np.dtypes.Float32DType(), bias=True), \
]

def generate_inputs(params: TestParamsLlamaAttention):
    torch_layer_config = LlamaConfig_torch( \
            hidden_size=params.n_emb, \
            num_attention_heads=params.n_head, \
            num_key_value_heads=params.n_head_kv, \
            attention_bias=params.bias, \
            use_cache=False, \
            attention_dropout=0.0 \
            )
    torch_layer = LlamaAttention_torch(
            torch_layer_config, \
            layer_idx=params.layer_idx \
            )
    x_shape = [params.n_emb, params.n_seq, params.n_batch]
    x_basetile = [params.n_emb_tile, params.n_seq_tile, params.n_batch_tile]
    x_traits = TensorTraits(x_shape, x_basetile)
    x_distr = [0] * x_traits.grid.nelems
    x_type = nntile.utils.constructors.np2nnt_type_mapping[type(params.dtype)]
    x_value = x_type(x_traits, x_distr, 0)
    x_grad = x_type(x_traits, x_distr, 0)
    X = TensorMoments(x_value, x_grad, grad_required=True)
    nntile_layer = nntile.layer.LlamaAttention.from_torch(torch_layer, X, \
            params.n_head_tile)
    return torch_layer, nntile_layer

@pytest.mark.parametrize("params", TEST_PARAMS)
class TestLlamaAttention:
    def test_from_to_torch(self, starpu_simple, torch_rng, \
            params: TestParamsLlamaAttention):
        torch_layer, nntile_layer = generate_inputs(params)
        torch_layer_other = nntile_layer.to_torch()
        nntile_layer.unregister()
        nntile_layer.x.unregister()
        nntile_layer.y.unregister()
        assert torch.equal(torch_layer.q_proj.weight, \
                torch_layer_other.q_proj.weight)
        assert torch.equal(torch_layer.k_proj.weight, \
                torch_layer_other.k_proj.weight)
        assert torch.equal(torch_layer.v_proj.weight, \
                torch_layer_other.v_proj.weight)
        assert torch.equal(torch_layer.o_proj.weight, \
                torch_layer_other.o_proj.weight)
        if params.bias:
            assert torch.equal(torch_layer.q_proj.bias, \
                    torch_layer_other.q_proj.bias)
            assert torch.equal(torch_layer.k_proj.bias, \
                    torch_layer_other.k_proj.bias)
            assert torch.equal(torch_layer.v_proj.bias, \
                    torch_layer_other.v_proj.bias)
            assert torch.equal(torch_layer.o_proj.bias, \
                    torch_layer_other.o_proj.bias)

    def test_forward(self, starpu_simple, torch_rng, \
            params: TestParamsLlamaAttention):

## Helper function returns bool value true if test passes
#def helper(dtype: np.dtype):
#    n_emb = 128
#    n_seq = 64
#    n_batch = 3
#    n_head = 8
#    n_head_tile = 4
#    n_head_kv = 4
#    head_size = n_emb // n_head
#    np.random.seed(1)
#    # Describe single-tile tensor, located at node 0
#    X_shape = [n_emb, n_seq, n_batch]
#    X_traits = nntile.tensor.TensorTraits(X_shape, X_shape)
#    mpi_distr = [0]
#    next_tag = 0
#    # Tensor objects
#    X_value = Tensor[dtype](X_traits, mpi_distr, next_tag)
#    next_tag = X_value.next_tag
#    X_grad = Tensor[dtype](X_traits, mpi_distr, next_tag)
#    next_tag = X_grad.next_tag
#    # Set initial value for input
#    rand_X = np.random.randn(*X_shape)
#    np_X = np.array(rand_X, dtype=dtype, order='F')
#    X_value.from_array(np_X)
#    nntile.tensor.clear_async(X_grad)
#    X = nntile.tensor.TensorMoments(X_value, X_grad, True)
#    # Define attention layer
#    layer, next_tag = LlamaAttention_nntile.generate_simple(X, \
#            n_head, n_head_tile, n_head_kv, next_tag, bias=True, mask=None, \
#            redux=False)
#    # Define numpy arrays and nntile tensors
#    rand_W_Q = np.random.randn(*layer.w_q.value.shape)
#    #rand_W_Q = np.ones(layer.w_q.value.shape)
#    np_W_Q = np.array(rand_W_Q, dtype=dtype, order='F')
#    layer.w_q.value.from_array(np_W_Q)
#    nntile.tensor.clear_async(layer.w_q.grad)
#
#    rand_bias_Q = np.random.randn(*layer.in_proj_bias_q.value.shape)
#    #rand_bias_Q = np.zeros(layer.in_proj_bias_q.value.shape)
#    np_inproj_bias_Q = np.array(rand_bias_Q, dtype=dtype, order='F')
#    layer.in_proj_bias_q.value.from_array(np_inproj_bias_Q)
#    nntile.tensor.clear_async(layer.in_proj_bias_q.grad)
#
#    rand_W_K = np.random.randn(*layer.w_k.value.shape)
#    #rand_W_K = np.ones(layer.w_k.value.shape)
#    np_W_K = np.array(rand_W_K, dtype=dtype, order='F')
#    layer.w_k.value.from_array(np_W_K)
#    nntile.tensor.clear_async(layer.w_k.grad)
#
#    rand_bias_K = np.random.randn(*layer.in_proj_bias_k.value.shape)
#    #rand_bias_K = np.zeros(layer.in_proj_bias_k.value.shape)
#    np_inproj_bias_K = np.array(rand_bias_K, dtype=dtype, order='F')
#    layer.in_proj_bias_k.value.from_array(np_inproj_bias_K)
#    nntile.tensor.clear_async(layer.in_proj_bias_k.grad)
#
#    rand_W_V = np.random.randn(*layer.w_v.value.shape)
#    #rand_W_V = np.ones(layer.w_v.value.shape)
#    np_W_V = np.array(rand_W_V, dtype=dtype, order='F')
#    layer.w_v.value.from_array(np_W_V)
#    nntile.tensor.clear_async(layer.w_v.grad)
#
#    rand_bias_V = np.random.randn(*layer.in_proj_bias_v.value.shape)
#    #rand_bias_V = np.zeros(layer.in_proj_bias_v.value.shape)
#    np_inproj_bias_V = np.array(rand_bias_V, dtype=dtype, order='F')
#    layer.in_proj_bias_v.value.from_array(np_inproj_bias_V)
#    nntile.tensor.clear_async(layer.in_proj_bias_v.grad)
#
#    rand_W = np.random.randn(*layer.w.value.shape)
#    #rand_W = np.ones(layer.w.value.shape)
#    np_W = np.array(rand_W, dtype=dtype, order='F')
#    layer.w.value.from_array(np_W)
#    nntile.tensor.clear_async(layer.w.grad)
#
#    np_out_proj_bias = np.array(np.random.randn(n_emb), dtype=dtype, order='F')
#    #np_out_proj_bias = np.array(np.zeros(n_emb), dtype=dtype, order='F')
#    layer.out_proj_bias.value.from_array(np_out_proj_bias)
#    nntile.tensor.clear_async(layer.out_proj_bias.grad)
#
#    rand_Y_grad = np.random.randn(*X_shape)
#    np_Y_grad = np.array(rand_Y_grad, dtype=dtype, order='F')
#    layer.y.grad.from_array(np_Y_grad)
#    # Check result of forward pass layer.y.value
#    layer.forward_async()
#    np_Y_nntile = np.zeros(layer.y.value.shape, dtype=dtype, order='F')
#    layer.y.value.to_array(np_Y_nntile)
#    # Define Torch tensors and layer
#    X_tensor = torch.tensor(np_X.T, requires_grad=True)
#    torch_layer = layer.to_torch()
#
#    attn_output = torch_layer(X_tensor, \
#            position_ids=torch.zeros((n_batch, n_seq), dtype=torch.long))
#    np_Y_torch = attn_output[0].data.numpy().T
#    # Compare
#    norm = np.linalg.norm(np_Y_torch)
#    diff = np.linalg.norm(np_Y_torch - np_Y_nntile)
#    # print("Forward diff = {}".format(diff/norm))
#    if diff > norm*1e-4:
#        import ipdb; ipdb.set_trace()
#        return False
#    # Check backward
#    layer.backward_async()
#    torch_layer_grads = layer.to_torch_with_grads()
#    layer.x.grad.to_array(np_X)
#
#    attn_grad = torch.tensor(np_Y_grad.T)
#    res = (attn_output[0]*attn_grad).sum()
#    res.backward()
#    np_X_torch = np.array(X_tensor.grad).T
#
#    norm = torch.norm(torch_layer.o_proj.bias.grad)
#    diff = torch.norm(torch_layer.o_proj.bias.grad - torch_layer_grads.o_proj.bias.grad)
#    # print("Error in grad for outproj bias = {}".format(diff / norm))
#    if diff > norm*1e-4:
#        import ipdb; ipdb.set_trace()
#        return False
#
#    norm1 = torch.norm(torch_layer.q_proj.bias.grad)
#    diff1 = torch.norm(torch_layer.q_proj.bias.grad - torch_layer_grads.q_proj.bias.grad)
#
#    norm2 = torch.norm(torch_layer.k_proj.bias.grad)
#    diff2 = torch.norm(torch_layer.k_proj.bias.grad - torch_layer_grads.k_proj.bias.grad)
#
#    norm3 = torch.norm(torch_layer.v_proj.bias.grad)
#    diff3 = torch.norm(torch_layer.v_proj.bias.grad - torch_layer_grads.v_proj.bias.grad)
#
#    if diff1*diff1+diff2*diff2+diff3*diff3 > (norm1*norm1+norm2*norm2+norm3*norm3)*1e-8:
#        import ipdb; ipdb.set_trace()
#        return False
#
#    norm = np.linalg.norm(np_X_torch)
#    diff = np.linalg.norm(np_X_torch - np_X)
#    if diff > norm*1e-4:
#        import ipdb; ipdb.set_trace()
#        return False
#    norm = torch.norm(torch_layer.q_proj.weight.grad)
#    diff = torch.norm(torch_layer.q_proj.weight.grad - torch_layer_grads.q_proj.weight.grad)
#    if diff > norm*1e-4:
#        import ipdb; ipdb.set_trace()
#        return False
#    norm = torch.norm(torch_layer.k_proj.weight.grad)
#    diff = torch.norm(torch_layer.k_proj.weight.grad - torch_layer_grads.k_proj.weight.grad)
#    if diff > norm*1e-4:
#        import ipdb; ipdb.set_trace()
#        return False
#    norm = torch.norm(torch_layer.v_proj.weight.grad)
#    diff = torch.norm(torch_layer.v_proj.weight.grad - torch_layer_grads.v_proj.weight.grad)
#    if diff > norm*1e-4:
#        import ipdb; ipdb.set_trace()
#        return False
#    norm = torch.norm(torch_layer.o_proj.weight.grad)
#    diff = torch.norm(torch_layer.o_proj.weight.grad - torch_layer_grads.o_proj.weight.grad)
#    if diff > norm*1e-4:
#        import ipdb; ipdb.set_trace()
#        return False
#    # Unregister
#    X.unregister()
#    layer.unregister()
#    return True
#
## Test runner for different precisions
#def test():
#    for dtype in dtypes:
#        assert helper(dtype)
#
## Repeat tests
#def test_repeat():
#    for dtype in dtypes:
#        assert helper(dtype)
#
#if __name__ == "__main__":
#    test()
#    test_repeat()
