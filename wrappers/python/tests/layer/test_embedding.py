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
# @version 1.1.0

import numpy as np
import pytest
import torch
from numpy.testing import assert_equal

import nntile
import nntile.utils.constructors as nntc
from nntile.layer import Embedding

# Define mapping between numpy and nntile types
Tensor = {
    np.float32: nntile.tensor.Tensor_fp32,
    np.float64: nntile.tensor.Tensor_fp64,
}

dtype2nntile = {
    'fp16': nntile.tensor.Tensor_fp16,
    'bf16': nntile.tensor.Tensor_bf16,
    'fp32': nntile.tensor.Tensor_fp32,
}

dtype2np = {
    'fp16': np.float16,
    'bf16': np.float16,
    'fp32': np.float32,
}

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_embedding(context, dtype: np.dtype):
    # Describe single-tile tensor, located at node 0
    index_shape = [4, 5, 6]
    vocab_size = 1000
    emb_size = 100
    emb_size_tile = 60
    ndim = len(index_shape)
    axis = ndim
    index_traits = nntile.tensor.TensorTraits(index_shape, index_shape)
    mpi_distr = [0]
    # Tensor objects
    nntile_index = nntile.tensor.Tensor_int64(
        index_traits, mpi_distr
    )
    # Set initial values of tensors
    rng = np.random.default_rng(42)
    rand_index = rng.integers(0, vocab_size, index_shape)
    np_index = np.array(rand_index, dtype=np.int64, order="F")
    nntile_index.from_array(np_index)
    torch_index = torch.tensor(np_index)
    rand_vocab = rng.standard_normal((emb_size, vocab_size))
    np_vocab = np.array(rand_vocab, dtype=dtype, order="F")
    rand_embed_grad = rng.standard_normal(index_shape + [emb_size])
    np_embed_grad = np.array(rand_embed_grad, dtype=dtype, order="F")
    # Define NNTile embedding layer
    nntile_layer = Embedding.generate_simple(
        nntile_index,
        Tensor[dtype],
        axis,
        vocab_size,
        emb_size,
        emb_size_tile,
        emb_size_tile,
    )
    nntile_layer.w.value.from_array(np_vocab)
    # Define PyTorch embedding layer
    torch_layer = torch.nn.Embedding(vocab_size, emb_size)
    torch_layer.weight.data = torch.tensor(np_vocab.T)
    # NNTile forward
    nntile_layer.forward_async()
    nntile_embed = np.zeros(nntile_layer.y.value.shape, dtype=dtype, order="F")
    nntile_layer.y.value.to_array(nntile_embed)
    # PyTorch forward
    torch_embed = torch_layer(torch_index)
    # Check forward
    assert_equal(torch_embed.data.numpy(), nntile_embed)
    # NNTile backward
    nntile_layer.y.grad.from_array(np_embed_grad)
    nntile.tensor.clear_async(nntile_layer.w.grad)
    nntile_layer.backward_async()
    nntile_layer.w.grad.to_array(np_vocab)
    # PyTorch backward
    res = (torch_embed * torch.tensor(np_embed_grad)).sum()
    res.backward()
    np_vocab_torch = torch_layer.weight.grad.numpy().T
    abs_error = np.linalg.norm(np_vocab_torch - np_vocab)
    rel_error = abs_error / np.linalg.norm(np_vocab_torch)
    assert rel_error < 1e-6


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_embedding_dynamic(context, numpy_rng, dtype):
    # Describe single-tile tensor, located at node 0
    index_shape = [4, 5, 6]
    vocab_size = 1000
    emb_size = 100
    emb_size_tile = 60
    ndim = len(index_shape)
    axis = ndim

    # # Set initial values of tensors
    np_index = np.asfortranarray(
        numpy_rng.integers(0, vocab_size, index_shape, dtype=np.int64)
    )
    # nntile_index.from_array(np_index)
    nntile_index = nntc.from_array(np_index)
    torch_index = torch.tensor(np_index)
    np_vocab = np.asfortranarray(
        numpy_rng.standard_normal((emb_size, vocab_size), dtype=dtype)
    )

    # Define NNTile embedding layer
    nntile_layer = Embedding.generate_simple(
        nntile_index,
        Tensor[dtype],
        axis,
        vocab_size,
        emb_size,
        emb_size_tile,
        emb_size_tile,
    )
    nntile_layer.w.value.from_array(np_vocab)
    # Define PyTorch embedding layer
    torch_layer = torch.nn.Embedding(vocab_size, emb_size)
    torch_layer.weight.data = torch.tensor(np_vocab.T)
    # NNTile forward
    y = nntile_layer.forward_dynamic(
        nntile.tensor.TensorMoments(nntile_index, None, False)
    )
    nntile_embed = nntc.to_numpy(y.value)
    # PyTorch forward
    torch_embed = torch_layer(torch_index)
    # Check forward
    assert_equal(torch_embed.data.numpy(), nntile_embed)

    index_dyn_shape = [7, 8, 9]
    np_dyn_index = np.asfortranarray(
        numpy_rng.integers(0, vocab_size, index_dyn_shape, dtype=np.int64)
    )
    torch_dyn_index = torch.tensor(np_dyn_index)
    y_dyn = nntile_layer.forward_dynamic(
        nntile.tensor.TensorMoments(nntc.from_array(np_dyn_index), None, False)
    )
    nntile_dyn_embed = nntc.to_numpy(y_dyn.value)

    torch_dyn_embed = torch_layer(torch_dyn_index)

    assert_equal(torch_dyn_embed.data.numpy(), nntile_dyn_embed)


@pytest.mark.benchmark
@pytest.mark.parametrize("dtype", ['fp32'])
def test_bench_embedding_forward_async(context_cuda, benchmark_operation, dtype: str):
    index_shape = [64, 32, 16]
    vocab_size = 2048
    emb_size = 128
    emb_size_tile = emb_size
    axis = len(index_shape)

    index_traits = nntile.tensor.TensorTraits(index_shape, index_shape)
    mpi_distr = [0]
    nntile_index = nntile.tensor.Tensor_int64(index_traits, mpi_distr)

    rng = np.random.default_rng(42)
    np_index = np.array(rng.integers(0, vocab_size, index_shape), dtype=np.int64, order="F")
    nntile_index.from_array(np_index)

    layer = Embedding.generate_simple(
        nntile_index,
        dtype2nntile[dtype],
        axis,
        vocab_size,
        emb_size,
        emb_size_tile,
        emb_size_tile,
    )

    np_vocab = np.array(rng.standard_normal((emb_size, vocab_size)), dtype=dtype2np[dtype], order="F")
    layer.w.value.from_array(np_vocab)

    out_np = np.zeros(layer.y.value.shape, dtype=dtype2np[dtype], order="F")

    def bench_fn():
        layer.forward_async()
        nntile.starpu.wait_for_all()

    nntile.starpu.wait_for_all()
    benchmark_operation(bench_fn)

@pytest.mark.benchmark
@pytest.mark.parametrize("dtype", ['fp32'])
def test_bench_embedding_forward_backward_async(context_cuda, benchmark_operation, dtype: str):
    index_shape = [64, 32, 16]
    vocab_size = 2048
    emb_size = 128
    emb_size_tile = emb_size
    axis = len(index_shape)

    index_traits = nntile.tensor.TensorTraits(index_shape, index_shape)
    mpi_distr = [0]
    nntile_index = nntile.tensor.Tensor_int64(index_traits, mpi_distr)

    rng = np.random.default_rng(42)
    np_index = np.array(rng.integers(0, vocab_size, index_shape), dtype=np.int64, order="F")
    nntile_index.from_array(np_index)

    layer = Embedding.generate_simple(
        nntile_index,
        dtype2nntile[dtype],
        axis,
        vocab_size,
        emb_size,
        emb_size_tile,
        emb_size_tile,
    )

    np_vocab = np.array(rng.standard_normal((emb_size, vocab_size)), dtype=dtype2np[dtype], order="F")
    layer.w.value.from_array(np_vocab)

    # forward once and set grad
    layer.forward_async()
    grad_np = np.array(rng.standard_normal(layer.y.value.shape), dtype=dtype2np[dtype], order="F")
    layer.y.grad.from_array(grad_np)
    nntile.tensor.clear_async(layer.w.grad)

    def bench_fn():
        layer.backward_async()
        nntile.starpu.wait_for_all()

    nntile.starpu.wait_for_all()
    benchmark_operation(bench_fn)
