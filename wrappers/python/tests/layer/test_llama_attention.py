# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/layer/test_llama_attention.py
# Test for nntile.layer.llama_attention
#
# @version 1.0.0

# All necesary imports
import nntile
import numpy as np
# Set up StarPU configuration and init it
config = nntile.starpu.Config(1, 0, 0)
# Init all NNTile-StarPU codelets
nntile.starpu.init()
# Define list of tested types
dtypes = [np.float32, np.float64]
# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
        np.float64: nntile.tensor.Tensor_fp64}
# Get attention layer
LlamaAttention_nntile = nntile.layer.LlamaAttention
# Get attention from LLaMa
import torch
import transformers
from transformers.models.llama.modeling_llama import (LlamaAttention, \
        LlamaConfig, LlamaSdpaAttention)

# Helper function returns bool value true if test passes
def helper(dtype: np.dtype):
    n_emb = 4
    n_seq = 2
    n_batch = 1
    n_head = 2
    n_head_tile = 2
    n_head_kv = 2
    head_size = n_emb // n_head
    np.random.seed(1)
    # Describe single-tile tensor, located at node 0
    X_shape = [n_emb, n_seq, n_batch]
    X_traits = nntile.tensor.TensorTraits(X_shape, X_shape)
    mpi_distr = [0]
    next_tag = 0
    # Tensor objects
    X_value = Tensor[dtype](X_traits, mpi_distr, next_tag)
    next_tag = X_value.next_tag
    X_grad = Tensor[dtype](X_traits, mpi_distr, next_tag)
    next_tag = X_grad.next_tag
    # Set initial value for input
    rand_X = np.random.randn(*X_shape)
    np_X = np.array(rand_X, dtype=dtype, order='F')
    X_value.from_array(np_X)
    nntile.tensor.clear_async(X_grad)
    X = nntile.tensor.TensorMoments(X_value, X_grad, True)
    # Define attention layer
    layer, next_tag = LlamaAttention_nntile.generate_simple(X, X, X, \
            n_head, n_head_tile, n_head_kv, next_tag, bias=True, mask=None, \
            redux=False)
    # Define numpy arrays and nntile tensors
    #rand_W_Q = np.random.randn(*layer.w_q.value.shape)
    rand_W_Q = np.ones(layer.w_q.value.shape)
    np_W_Q = np.array(rand_W_Q, dtype=dtype, order='F')
    layer.w_q.value.from_array(np_W_Q)
    nntile.tensor.clear_async(layer.w_q.grad)

    #rand_bias_Q = np.random.randn(*layer.in_proj_bias_q.value.shape)
    rand_bias_Q = np.zeros(layer.in_proj_bias_q.value.shape)
    np_inproj_bias_Q = np.array(rand_bias_Q, dtype=dtype, order='F')
    layer.in_proj_bias_q.value.from_array(np_inproj_bias_Q)
    nntile.tensor.clear_async(layer.in_proj_bias_q.grad)

    #rand_W_K = np.random.randn(*layer.w_k.value.shape)
    rand_W_K = np.ones(layer.w_k.value.shape)
    np_W_K = np.array(rand_W_K, dtype=dtype, order='F')
    layer.w_k.value.from_array(np_W_K)
    nntile.tensor.clear_async(layer.w_k.grad)

    #rand_bias_K = np.random.randn(*layer.in_proj_bias_k.value.shape)
    rand_bias_K = np.zeros(layer.in_proj_bias_k.value.shape)
    np_inproj_bias_K = np.array(rand_bias_K, dtype=dtype, order='F')
    layer.in_proj_bias_k.value.from_array(np_inproj_bias_K)
    nntile.tensor.clear_async(layer.in_proj_bias_k.grad)

    #rand_W_V = np.random.randn(*layer.w_v.value.shape)
    rand_W_V = np.ones(layer.w_v.value.shape)
    np_W_V = np.array(rand_W_V, dtype=dtype, order='F')
    layer.w_v.value.from_array(np_W_V)
    nntile.tensor.clear_async(layer.w_v.grad)

    #rand_bias_V = np.random.randn(*layer.in_proj_bias_v.value.shape)
    rand_bias_V = np.zeros(layer.in_proj_bias_v.value.shape)
    np_inproj_bias_V = np.array(rand_bias_V, dtype=dtype, order='F')
    layer.in_proj_bias_v.value.from_array(np_inproj_bias_V)
    nntile.tensor.clear_async(layer.in_proj_bias_v.grad)

    #rand_W = np.random.randn(*layer.w.value.shape)
    rand_W = np.ones(layer.w.value.shape)
    np_W = np.array(rand_W, dtype=dtype, order='F')
    layer.w.value.from_array(np_W)
    nntile.tensor.clear_async(layer.w.grad)

    #np_out_proj_bias = np.array(np.random.randn(n_emb), dtype=dtype, order='F')
    np_out_proj_bias = np.array(np.zeros(n_emb), dtype=dtype, order='F')
    layer.out_proj_bias.value.from_array(np_out_proj_bias)
    nntile.tensor.clear_async(layer.out_proj_bias.grad)

    rand_Y_grad = np.random.randn(*X_shape)
    np_Y_grad = np.array(rand_Y_grad, dtype=dtype, order='F')
    layer.y.grad.from_array(np_Y_grad)
    # Check result of forward pass layer.y.value
    layer.forward_async()
    np_Y_nntile = np.zeros(layer.y.value.shape, dtype=dtype, order='F')
    layer.y.value.to_array(np_Y_nntile)
    # Define Torch tensors and layer
    X_tensor = torch.tensor(np_X.T, requires_grad=True)
    torch_layer_config = LlamaConfig(hidden_size=n_emb, \
            num_attention_heads=n_head, num_key_value_heads=n_head_kv, \
            attention_bias=True, use_cache=False, attention_dropout=0.0)
    torch_layer = LlamaAttention(torch_layer_config, layer_idx=1)
    W_Q_tensor = torch.tensor(np_W_Q.reshape(n_emb, n_emb), requires_grad=True)
    W_K_tensor = torch.tensor(np_W_K.reshape(n_head_kv*head_size, n_emb) \
            , requires_grad=True)
    W_V_tensor = torch.tensor(np_W_V.reshape(n_head_kv*head_size, n_emb) \
            , requires_grad=True)
    torch_layer.q_proj.weight.data = W_Q_tensor
    torch_layer.k_proj.weight.data = W_K_tensor
    torch_layer.v_proj.weight.data = W_V_tensor
    W_out_tensor = torch.tensor(np_W.reshape(n_emb, n_emb), requires_grad=True)
    torch_layer.o_proj.weight.data = W_out_tensor
    out_proj_bias = torch.tensor(np_out_proj_bias.reshape(-1), requires_grad=True)
    torch_layer.o_proj.bias.data = out_proj_bias
    # print(torch.norm(torch_layer.in_proj_bias).item())
    torch_layer.q_proj.bias.data = \
            torch.tensor(np_inproj_bias_Q.transpose().reshape(-1))
    torch_layer.k_proj.bias.data = \
            torch.tensor(np_inproj_bias_K.transpose().reshape(-1))
    torch_layer.v_proj.bias.data = \
            torch.tensor(np_inproj_bias_V.transpose().reshape(-1))

    attn_output = torch_layer(X_tensor, \
            position_ids=torch.zeros((n_batch, n_seq), dtype=torch.long))
    np_Y_torch = attn_output[0].data.numpy().T
    # Compare
    norm = np.linalg.norm(np_Y_torch)
    diff = np.linalg.norm(np_Y_torch - np_Y_nntile)
    # print("Forward diff = {}".format(diff/norm))
    if diff > norm*1e-4:
        import ipdb; ipdb.set_trace()
        return False
    # Check backward
    layer.backward_async()
    layer.x_q.grad.to_array(np_X)
    layer.out_proj_bias.grad.to_array(np_out_proj_bias)

    layer.w_q.grad.to_array(np_W_Q)
    layer.w_k.grad.to_array(np_W_K)
    layer.w_v.grad.to_array(np_W_V)
    layer.w.grad.to_array(np_W)
    layer.in_proj_bias_q.grad.to_array(np_inproj_bias_Q)
    layer.in_proj_bias_k.grad.to_array(np_inproj_bias_K)
    layer.in_proj_bias_v.grad.to_array(np_inproj_bias_V)

    np_W_Q_nntile = np_W_Q
    np_W_K_nntile = np_W_K
    np_W_V_nntile = np_W_V
    np_W_nntile = np_W

    np_inproj_bias_Q_nntile = np_inproj_bias_Q
    np_inproj_bias_K_nntile = np_inproj_bias_K
    np_inproj_bias_V_nntile = np_inproj_bias_V

    attn_grad = torch.tensor(np_Y_grad.T)
    res = (attn_output[0]*attn_grad).sum()
    res.backward()
    np_X_torch = np.array(X_tensor.grad).T
    np_W_Q_torch = np.array(torch_layer.q_proj.weight.grad)
    np_W_K_torch = np.array(torch_layer.k_proj.weight.grad)
    np_W_V_torch = np.array(torch_layer.v_proj.weight.grad)
    np_W_torch = np.array(torch_layer.o_proj.weight.grad)

    np_out_proj_bias_torch = np.array(torch_layer.o_proj.bias.grad)
    np_in_proj_bias_q_torch = np.array(torch_layer.q_proj.bias.grad)
    np_in_proj_bias_k_torch = np.array(torch_layer.k_proj.bias.grad)
    np_in_proj_bias_v_torch = np.array(torch_layer.v_proj.bias.grad)
    norm = np.linalg.norm(np_out_proj_bias_torch)
    diff = np.linalg.norm(np_out_proj_bias_torch - np_out_proj_bias)
    # print("Error in grad for outproj bias = {}".format(diff / norm))
    if diff > norm*1e-4:
        import ipdb; ipdb.set_trace()
        return False

    norm1 = np.linalg.norm(np_in_proj_bias_q_torch)
    diff1 = np.linalg.norm(np_in_proj_bias_q_torch - \
            np_inproj_bias_Q_nntile.reshape(-1))

    norm2 = np.linalg.norm(np_in_proj_bias_k_torch)
    diff2 = np.linalg.norm(np_in_proj_bias_k_torch - \
            np_inproj_bias_K_nntile.reshape(-1))

    norm3 = np.linalg.norm(np_in_proj_bias_v_torch)
    diff3 = np.linalg.norm(np_in_proj_bias_v_torch - \
            np_inproj_bias_V_nntile.reshape(-1))

    if diff1*diff1+diff2*diff2+diff3*diff3 > (norm1*norm1+norm2*norm2+norm3*norm3)*1e-8:
        import ipdb; ipdb.set_trace()
        return False

    norm = np.linalg.norm(np_X_torch)
    diff = np.linalg.norm(np_X_torch - np_X)
    if diff > norm*1e-4:
        import ipdb; ipdb.set_trace()
        return False
    norm = np.linalg.norm(np_W_Q_torch)
    diff = np.linalg.norm(np_W_Q_torch - np_W_Q_nntile.reshape(n_emb, n_emb))
    if diff > norm*1e-4:
        import ipdb; ipdb.set_trace()
        return False
    norm = np.linalg.norm(np_W_K_torch)
    diff = np.linalg.norm(np_W_K_torch - np_W_K_nntile.reshape(n_head_kv*head_size, n_emb))
    if diff > norm*1e-4:
        import ipdb; ipdb.set_trace()
        return False
    norm = np.linalg.norm(np_W_V_torch)
    diff = np.linalg.norm(np_W_V_torch - np_W_V_nntile.reshape(n_head_kv*head_size, n_emb))
    if diff > norm*1e-4:
        import ipdb; ipdb.set_trace()
        return False
    norm = np.linalg.norm(np_W_torch)
    diff = np.linalg.norm(np_W_torch - np_W_nntile.reshape(n_emb, n_emb))
    if diff > norm*1e-4:
        import ipdb; ipdb.set_trace()
        return False
    # Unregister
    X.unregister()
    layer.unregister()
    return True

# Test runner for different precisions
def test():
    for dtype in dtypes:
        assert helper(dtype)

# Repeat tests
def test_repeat():
    for dtype in dtypes:
        assert helper(dtype)

if __name__ == "__main__":
    test()
    test_repeat()
