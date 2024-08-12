# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/embedding.py
# Embedding layer of NNTile Python package
#
# @version 1.1.0

import torch
from torch.nn import Embedding as Embedding_torch

import nntile.utils.constructors as nntc
from nntile.layer.base_layer import BaseLayer
from nntile.tensor import (
    Tensor_int64, TensorMoments, TensorTraits, clear_async, embedding_async,
    embedding_backward_async, to_numpy)


class Embedding(BaseLayer):
    x: Tensor_int64
    y: TensorMoments
    w: TensorMoments

    # Construct linear layer with all the provided data
    def __init__(
        self, x: Tensor_int64, y: TensorMoments, w: TensorMoments, axis: int
    ):
        # Redirect to BaseClass initialization
        super().__init__([x], [y], [w], [])
        # Named storage
        self.x = x
        self.y = y
        self.w = w
        self.w.grad.set_reduction_add()
        self.axis = axis

    # Simple generator for the embedding layer
    @staticmethod
    def generate_simple(
        x: Tensor_int64,
        TensorType,
        axis: int,
        vocab_size: int,
        emb_size: int,
        y_emb_tile: int,
        w_emb_tile: int,
        next_tag: int,
    ):
        # Check embedding tile sizes
        if y_emb_tile % w_emb_tile != 0:
            raise ValueError("y_emb_tile % w_emb_tile != 0")
        # Embeddings vocabulary
        w_shape = [emb_size, vocab_size]
        w_basetile = [w_emb_tile, vocab_size]
        w_traits = TensorTraits(w_shape, w_basetile)
        w_distr = [0] * w_traits.grid.nelems
        w_value = TensorType(w_traits, w_distr, next_tag)
        next_tag = w_value.next_tag
        w_grad = TensorType(w_traits, w_distr, next_tag)
        next_tag = w_grad.next_tag
        w = TensorMoments(w_value, w_grad, True)
        # Output embeddings
        y_shape = x.shape.copy()
        y_shape.insert(axis, emb_size)
        y_basetile = x.basetile_shape.copy()
        y_basetile.insert(axis, y_emb_tile)
        y_traits = TensorTraits(y_shape, y_basetile)
        y_distr = [0] * y_traits.grid.nelems
        y_value = TensorType(y_traits, y_distr, next_tag)
        next_tag = y_value.next_tag
        y_grid = TensorType(y_traits, y_distr, next_tag)
        next_tag = y_grid.next_tag
        y = TensorMoments(y_value, y_grid, True)
        # Create embedding layer with all the provided data
        layer = Embedding(x, y, w, axis)
        # Return layer and next tag to be used
        return (layer, next_tag)

    # Forward propagation of the embedding layer
    def forward_async(self):
        clear_async(self.y.value)
        embedding_async(self.x, self.w.value, self.y.value, self.axis)
        self.x.wont_use()
        self.w.value.wont_use()
        self.y.value.wont_use()

    def forward_dynamic(self, x: TensorMoments):
        y_shape = x.value.shape.copy()
        y_shape.insert(self.axis, self.w.value.shape[0])

        y_basetile = x.value.basetile_shape.copy()
        y_basetile.insert(self.axis, self.y.value.basetile_shape[self.axis])

        y = nntc.empty(
            y_shape, basetile_shape=y_basetile, dtype=type(self.w.value)
        )
        embedding_async(x.value, self.w.value, y, self.axis)
        return TensorMoments(y, None, False)

    # Backward propagation of the embedding layer
    def backward_async(self):
        # redux=1 leads to performance loss, as each embedding_backward is a
        # sparse operation, but reduction plays with a full dense vocabulary
        embedding_backward_async(
            self.x, self.y.grad, self.w.grad, self.axis, redux=0
        )
        self.x.wont_use()
        self.y.grad.wont_use()
        self.w.grad.wont_use()

    def to_torch(self):
        torch_emb = Embedding_torch(
            self.w.value.shape[1], self.w.value.shape[0]
        )
        torch_emb.weight.data = torch.tensor(
            to_numpy(self.w.value).T, requires_grad=True
        )
        return torch_emb

    def to_torch_with_grads(self):
        torch_emb = Embedding_torch(
            self.w.value.shape[1], self.w.value.shape[0]
        )
        torch_emb.weight.data = torch.tensor(
            to_numpy(self.w.value).T, requires_grad=True
        )
        torch_emb.weight.grad = torch.tensor(to_numpy(self.w.grad).T)
        return torch_emb
