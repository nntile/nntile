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
# @date 2023-03-23

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
    n_seq = 256
    n_batch = 48
    n_head = 8
    # Describe single-tile tensor, located at node 0
    A_shape = [n_emb, n_seq, n_batch]
    A_traits = nntile.tensor.TensorTraits(A_shape, A_shape)
    mpi_distr = [0]
    next_tag = 0
    # Tensor objects
    A = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    A_grad = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A_grad.next_tag
    # Set initial value for input
    rand_A = np.random.randn(*A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A.from_array(np_A)
    A_moments = nntile.tensor.TensorMoments(A, A_grad, True)
    # Define attention layer
    layer, next_tag = Attention.generate_simple_mpiroot(A_moments, n_head, \
            next_tag)
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
    A_tensor = torch.tensor(np_A.T)
    torch_layer = MultiheadAttention(n_emb, n_head, batch_first=True, \
            bias=False)
    W_Q = np.vstack(np_W_Q)
    W_K = np.vstack(np_W_K)
    W_V = np.vstack(np_W_V)
    W = np.vstack([W_Q, W_K, W_V])
    W_tensor = torch.tensor(W)
    torch_layer.in_proj_weight.data = W_tensor
    W_out = np.hstack(np_W)
    W_out_tensor = torch.tensor(W_out)
    torch_layer.out_proj.weight.data = W_out_tensor
    attn_output, attn_weights = torch_layer(A_tensor, A_tensor, A_tensor, \
            average_attn_weights=False)
    np_Y = attn_output.detach().numpy().T
    A_moments.unregister()
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

