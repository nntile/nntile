# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/layer/test_attention.py
# Test for nntile.layer.attention
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-07-22

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
Attention = nntile.layer.Attention
# Get attention from PyTorch
import torch
from torch.nn import MultiheadAttention

# Helper function returns bool value true if test passes
def helper(dtype: np.dtype):
    n_emb = 128
    n_emb_k = 112
    n_emb_v = 96
    n_seq = 256
    n_batch = 48
    n_head = 8
    # Describe single-tile tensor, located at node 0
    X_Q_shape = [n_emb, n_seq, n_batch]
    X_K_shape = [n_emb_k, n_seq, n_batch]
    X_V_shape = [n_emb_v, n_seq, n_batch]
    X_Q_traits = nntile.tensor.TensorTraits(X_Q_shape, X_Q_shape)
    X_K_traits = nntile.tensor.TensorTraits(X_K_shape, X_K_shape)
    X_V_traits = nntile.tensor.TensorTraits(X_V_shape, X_V_shape)
    mpi_distr = [0]
    next_tag = 0
    # Tensor objects
    X_Q_value = Tensor[dtype](X_Q_traits, mpi_distr, next_tag)
    next_tag = X_Q_value.next_tag
    X_Q_grad = Tensor[dtype](X_Q_traits, mpi_distr, next_tag)
    next_tag = X_Q_grad.next_tag
    X_K_value = Tensor[dtype](X_K_traits, mpi_distr, next_tag)
    next_tag = X_K_value.next_tag
    X_K_grad = Tensor[dtype](X_K_traits, mpi_distr, next_tag)
    next_tag = X_K_grad.next_tag
    X_V_value = Tensor[dtype](X_V_traits, mpi_distr, next_tag)
    next_tag = X_V_value.next_tag
    X_V_grad = Tensor[dtype](X_V_traits, mpi_distr, next_tag)
    next_tag = X_V_grad.next_tag
    # Set initial value for input
    rand_X_Q = np.random.randn(*X_Q_shape)
    np_X_Q = np.array(rand_X_Q, dtype=dtype, order='F')
    X_Q_value.from_array(np_X_Q)
    nntile.tensor.clear_async(X_Q_grad)
    X_Q = nntile.tensor.TensorMoments(X_Q_value, X_Q_grad, True)
    rand_X_K = np.random.randn(*X_K_shape)
    np_X_K = np.array(rand_X_K, dtype=dtype, order='F')
    X_K_value.from_array(np_X_K)
    nntile.tensor.clear_async(X_K_grad)
    X_K = nntile.tensor.TensorMoments(X_K_value, X_K_grad, True)
    rand_X_V = np.random.randn(*X_V_shape)
    np_X_V = np.array(rand_X_V, dtype=dtype, order='F')
    X_V_value.from_array(np_X_V)
    nntile.tensor.clear_async(X_V_grad)
    X_V = nntile.tensor.TensorMoments(X_V_value, X_V_grad, True)
    # Define attention layer
    layer, next_tag = Attention.generate_simple(X_Q, X_K, X_V, \
            n_head, next_tag, True)
    # Define numpy arrays and nntile tensors
    rand_W_Q = np.random.randn(*layer.w_q.value.shape)
    np_W_Q = np.array(rand_W_Q, dtype=dtype, order='F')
    layer.w_q.value.from_array(np_W_Q)
    nntile.tensor.clear_async(layer.w_q.grad)

    rand_bias_Q = np.random.randn(*layer.in_proj_bias_q.value.shape)
    np_inproj_bias_Q = np.array(rand_bias_Q, dtype=dtype, order='F')
    layer.in_proj_bias_q.value.from_array(np_inproj_bias_Q)
    nntile.tensor.clear_async(layer.in_proj_bias_q.grad)

    rand_W_K = np.random.randn(*layer.w_k.value.shape)
    np_W_K = np.array(rand_W_K, dtype=dtype, order='F')
    layer.w_k.value.from_array(np_W_K)
    nntile.tensor.clear_async(layer.w_k.grad)

    rand_bias_K = np.random.randn(*layer.in_proj_bias_k.value.shape)
    np_inproj_bias_K = np.array(rand_bias_K, dtype=dtype, order='F')
    layer.in_proj_bias_k.value.from_array(np_inproj_bias_K)
    nntile.tensor.clear_async(layer.in_proj_bias_k.grad)
        
    rand_W_V = np.random.randn(*layer.w_v.value.shape)
    np_W_V = np.array(rand_W_V, dtype=dtype, order='F')
    layer.w_v.value.from_array(np_W_V)
    nntile.tensor.clear_async(layer.w_v.grad)

    rand_bias_V = np.random.randn(*layer.in_proj_bias_v.value.shape)
    np_inproj_bias_V = np.array(rand_bias_V, dtype=dtype, order='F')
    layer.in_proj_bias_v.value.from_array(np_inproj_bias_V)
    nntile.tensor.clear_async(layer.in_proj_bias_v.grad)
        
    rand_W = np.random.randn(*layer.w.value.shape)
    np_W = np.array(rand_W, dtype=dtype, order='F')
    layer.w.value.from_array(np_W)
    nntile.tensor.clear_async(layer.w.grad)

    np_out_proj_bias = np.array(np.random.randn(n_emb), dtype=dtype, order='F')
    layer.out_proj_bias.value.from_array(np_out_proj_bias)
    nntile.tensor.clear_async(layer.out_proj_bias.grad)

    rand_Y_grad = np.random.randn(*X_Q_shape)
    np_Y_grad = np.array(rand_Y_grad, dtype=dtype, order='F')
    layer.y.grad.from_array(np_Y_grad)
    # Check result of forward pass layer.y.value
    layer.forward_async()
    np_Y_nntile = np.zeros(layer.y.value.shape, dtype=dtype, order='F')
    layer.y.value.to_array(np_Y_nntile)
    # Define Torch tensors and layer
    X_Q_tensor = torch.tensor(np_X_Q.T, requires_grad=True)
    X_K_tensor = torch.tensor(np_X_K.T, requires_grad=True)
    X_V_tensor = torch.tensor(np_X_V.T, requires_grad=True)
    torch_layer = MultiheadAttention(n_emb, n_head, kdim=n_emb_k, \
            vdim=n_emb_v, batch_first=True, bias=True)
    W_Q_tensor = torch.tensor(np_W_Q.reshape(n_emb, n_emb), requires_grad=True)
    W_K_tensor = torch.tensor(np_W_K.reshape(n_emb, n_emb_k), requires_grad=True)
    W_V_tensor = torch.tensor(np_W_V.reshape(n_emb, n_emb_v), requires_grad=True)
    torch_layer.q_proj_weight.data = W_Q_tensor
    torch_layer.k_proj_weight.data = W_K_tensor
    torch_layer.v_proj_weight.data = W_V_tensor
    W_out_tensor = torch.tensor(np_W.reshape(n_emb, n_emb), requires_grad=True)
    torch_layer.out_proj.weight.data = W_out_tensor
    out_proj_bias = torch.tensor(np_out_proj_bias.reshape(-1), requires_grad=True)
    torch_layer.out_proj.bias.data = out_proj_bias
    # print(torch.norm(torch_layer.in_proj_bias).item())
    in_proj_bias = torch.tensor(np.hstack([ \
            np_inproj_bias_Q.transpose().reshape(-1), \
            np_inproj_bias_K.transpose().reshape(-1), \
            np_inproj_bias_V.transpose().reshape(-1)]), \
            requires_grad=True)
    torch_layer.in_proj_bias.data = in_proj_bias

    attn_output = torch_layer(X_Q_tensor, X_K_tensor, X_V_tensor, \
            need_weights=False)
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
    layer.x_q.grad.to_array(np_X_Q)
    layer.x_k.grad.to_array(np_X_K)
    layer.x_v.grad.to_array(np_X_V)
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

    np_inproj_nntile = np.hstack([np_inproj_bias_Q_nntile, \
            np_inproj_bias_K_nntile, np_inproj_bias_V_nntile]).transpose()\
            .reshape(-1)

    attn_grad = torch.tensor(np_Y_grad.T)
    res = (attn_output[0]*attn_grad).sum()
    res.backward()
    np_X_Q_torch = np.array(X_Q_tensor.grad).T
    np_X_K_torch = np.array(X_K_tensor.grad).T
    np_X_V_torch = np.array(X_V_tensor.grad).T
    np_W_Q_torch = np.array(torch_layer.q_proj_weight.grad)
    np_W_K_torch = np.array(torch_layer.k_proj_weight.grad)
    np_W_V_torch = np.array(torch_layer.v_proj_weight.grad)
    np_W_torch = np.array(torch_layer.out_proj.weight.grad)
    
    np_out_proj_bias_torch = np.array(torch_layer.out_proj.bias.grad)
    norm = np.linalg.norm(np_out_proj_bias_torch)
    diff = np.linalg.norm(np_out_proj_bias_torch - np_out_proj_bias)
    # print("Error in grad for outproj bias = {}".format(diff / norm))
    if diff > norm*1e-4:
        import ipdb; ipdb.set_trace()
        return False
    
    np_inproj_bias_torch = np.array(torch_layer.in_proj_bias.grad)
    norm = np.linalg.norm(np_inproj_bias_torch)
    diff = np.linalg.norm(np_inproj_bias_torch - np_inproj_nntile)
    # print("Error in grad for inproj bias = {}".format(diff / norm))
    if diff > norm*1e-4:
        import ipdb; ipdb.set_trace()
        return False
    
    norm = np.linalg.norm(np_X_Q_torch)
    diff = np.linalg.norm(np_X_Q_torch - np_X_Q)
    if diff > norm*1e-4:
        import ipdb; ipdb.set_trace()
        return False
    norm = np.linalg.norm(np_X_K_torch)
    diff = np.linalg.norm(np_X_K_torch - np_X_K)
    if diff > norm*1e-4:
        import ipdb; ipdb.set_trace()
        return False
    norm = np.linalg.norm(np_X_V_torch)
    diff = np.linalg.norm(np_X_V_torch - np_X_V)
    if diff > norm*1e-4:
        import ipdb; ipdb.set_trace()
        return False
    norm = np.linalg.norm(np_W_Q_torch)
    diff = np.linalg.norm(np_W_Q_torch - np_W_Q_nntile.reshape(n_emb, n_emb))
    if diff > norm*1e-4:
        import ipdb; ipdb.set_trace()
        return False
    norm = np.linalg.norm(np_W_K_torch)
    diff = np.linalg.norm(np_W_K_torch - np_W_K_nntile.reshape(n_emb, n_emb_k))
    if diff > norm*1e-4:
        import ipdb; ipdb.set_trace()
        return False
    norm = np.linalg.norm(np_W_V_torch)
    diff = np.linalg.norm(np_W_V_torch - np_W_V_nntile.reshape(n_emb, n_emb_v))
    if diff > norm*1e-4:
        import ipdb; ipdb.set_trace()
        return False
    norm = np.linalg.norm(np_W_torch)
    diff = np.linalg.norm(np_W_torch - np_W_nntile.reshape(n_emb, n_emb))
    if diff > norm*1e-4:
        import ipdb; ipdb.set_trace()
        return False
    # Unregister
    X_Q.unregister()
    X_K.unregister()
    X_V.unregister()
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

