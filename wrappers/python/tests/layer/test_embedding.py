# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/layer/test_embedding.py
# Test for nntile.layer.embedding
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
# Get attention from PyTorch
import torch
from torch.nn import Embedding

# Helper function returns bool value true if test passes
def helper(dtype: np.dtype):
    # Describe single-tile tensor, located at node 0
    index_shape = [4, 5, 6]
    vocab_size = 1000
    emb_size = 100
    emb_size_tile = 60
    ndim = len(index_shape)
    axis = ndim
    index_traits = nntile.tensor.TensorTraits(index_shape, index_shape)
    mpi_distr = [0]
    next_tag = 0
    # Tensor objects
    nntile_index = nntile.tensor.Tensor_int64(index_traits, mpi_distr, next_tag)
    next_tag = nntile_index.next_tag
    # Set initial values of tensors
    rand_index = np.random.randint(0, vocab_size, size=index_shape)
    np_index = np.array(rand_index, dtype=np.int64, order='F')
    nntile_index.from_array(np_index)
    torch_index = torch.tensor(np_index)
    rand_vocab = np.random.randn(emb_size, vocab_size)
    np_vocab = np.array(rand_vocab, dtype=dtype, order='F')
    rand_embed_grad = np.random.randn(*index_shape, emb_size)
    np_embed_grad = np.array(rand_embed_grad, dtype=dtype, order='F')
    # Define NNTile embedding layer
    nntile_layer, next_tag = nntile.layer.Embedding.generate_simple( \
            nntile_index, Tensor[dtype], axis, vocab_size, emb_size, \
            emb_size_tile, emb_size_tile, next_tag)
    nntile_layer.w.value.from_array(np_vocab)
    # Define PyTorch embedding layer
    torch_layer = Embedding(vocab_size, emb_size)
    torch_layer.weight.data = torch.tensor(np_vocab.T)
    # NNTile forward
    nntile_layer.forward_async()
    nntile_embed = np.zeros(nntile_layer.y.value.shape, dtype=dtype, order='F')
    nntile_layer.y.value.to_array(nntile_embed)
    # PyTorch forward
    torch_embed = torch_layer(torch_index)
    # Check forward
    assert (torch_embed.data.numpy() == nntile_embed).all()
    # NNTile backward
    nntile_layer.y.grad.from_array(np_embed_grad)
    nntile.tensor.clear_async(nntile_layer.w.grad)
    nntile_layer.backward_async()
    nntile_layer.w.grad.to_array(np_vocab)
    # PyTorch backward
    res = (torch_embed*torch.tensor(np_embed_grad)).sum()
    res.backward()
    np_vocab_torch = torch_layer.weight.grad.numpy().T
    assert (np.linalg.norm(np_vocab_torch-np_vocab) \
            / np.linalg.norm(np_vocab_torch) < 1e-6)
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
