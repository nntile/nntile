# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/layer/test_attention.py
# Test for nntile.layer.attention
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-03-24

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
torch_dtype = {np.float32: torch.float32,
               np.float64: torch.float64}

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
    X_Q = nntile.tensor.TensorMoments(X_Q_value, X_Q_grad, True)
    rand_X_K = np.random.randn(*X_K_shape)
    np_X_K = np.array(rand_X_K, dtype=dtype, order='F')
    X_K_value.from_array(np_X_K)
    X_K = nntile.tensor.TensorMoments(X_K_value, X_K_grad, True)
    rand_X_V = np.random.randn(*X_V_shape)
    np_X_V = np.array(rand_X_V, dtype=dtype, order='F')
    X_V_value.from_array(np_X_V)
    X_V = nntile.tensor.TensorMoments(X_V_value, X_V_grad, True)
    # Define attention layer
    layer, next_tag = Attention.generate_simple_mpiroot(X_Q, X_K, X_V, \
            n_head, next_tag)
    # Define numpy arrays and nntile tensors
    np_W_Q = []
    np_W_K = []
    np_W_V = []
    np_W = []
    for i in range(n_head):
        rand_W_Q = np.random.randn(*layer.w_q[i].value.shape)
        np_W_Q.append(np.array(rand_W_Q, dtype=dtype, order='F'))
        layer.w_q[i].value.from_array(np_W_Q[i])
        rand_W_K = np.random.randn(*layer.w_k[i].value.shape)
        np_W_K.append(np.array(rand_W_K, dtype=dtype, order='F'))
        layer.w_k[i].value.from_array(np_W_K[i])
        rand_W_V = np.random.randn(*layer.w_v[i].value.shape)
        np_W_V.append(np.array(rand_W_V, dtype=dtype, order='F'))
        layer.w_v[i].value.from_array(np_W_V[i])
        rand_W = np.random.randn(*layer.w[i].value.shape)
        np_W.append(np.array(rand_W, dtype=dtype, order='F'))
        layer.w[i].value.from_array(np_W[i])
    # Check result of forward pass layer.y.value
    layer.forward_async()
    np_Y2 = np.zeros(layer.y.value.shape, dtype=dtype, order='F')
    layer.y.value.to_array(np_Y2)
    # Define Torch tensors and layer
    X_Q_tensor = torch.tensor(np_X_Q.T, requires_grad=True)
    X_K_tensor = torch.tensor(np_X_K.T, requires_grad=True)
    X_V_tensor = torch.tensor(np_X_V.T, requires_grad=True)
    torch_layer = MultiheadAttention(n_emb, n_head, kdim=n_emb_k, \
            vdim=n_emb_v, batch_first=True, bias=False)
    W_Q = np.vstack(np_W_Q)
    W_K = np.vstack(np_W_K)
    W_V = np.vstack(np_W_V)
    torch_layer.q_proj_weight.data = torch.tensor(W_Q, requires_grad=True)
    torch_layer.k_proj_weight.data = torch.tensor(W_K, requires_grad=True)
    torch_layer.v_proj_weight.data = torch.tensor(W_V, requires_grad=True)
    W_out = np.hstack(np_W)
    W_out_tensor = torch.tensor(W_out, requires_grad=True)
    torch_layer.out_proj.weight.data = W_out_tensor
    attn_output = torch_layer(X_Q_tensor, X_K_tensor, X_V_tensor, \
            need_weights=False)

    res = attn_output[0].sum()
    np_Y = attn_output[0].data.numpy().T
    X_Q.unregister()
    X_K.unregister()
    X_V.unregister()
    layer.unregister()
    # Compare
    norm = np.linalg.norm(np_Y)
    diff = np.linalg.norm(np_Y-np_Y2)
    return diff < norm * 1e-4

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

