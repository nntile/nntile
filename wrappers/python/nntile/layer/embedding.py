# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/embedding.py
# Embedding layer of NNTile Python package
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-04-21

from nntile.tensor import TensorTraits, Tensor, TensorOrNone, TensorMoments, \
        Tensor_int64, clear_async, embedding_async, embedding_backward_async
from nntile.layer.base_layer import BaseLayer
import numpy as np
from typing import List

class Embedding(BaseLayer):
    x: Tensor_int64
    y: TensorMoments
    w: TensorMoments

    # Construct linear layer with all the provided data
    def __init__(self, x: Tensor_int64, y: TensorMoments, w: TensorMoments, \
            axis: int):
        # Redirect to BaseClass initialization
        super().__init__([x], [y], [w], [])
        # Named storage
        self.x = x
        self.y = y
        self.w = w
        self.axis = axis

    # Simple generator for the embedding layer
    @staticmethod
    def generate_simple(x: Tensor_int64, TensorType, axis: int, \
            vocab_size: int, emb_size: int, y_emb_tile: int, w_emb_tile: int, \
            next_tag: int):
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

    # Backward propagation of the linear layer
    def backward_async(self):
        embedding_backward_async(self.x, self.y.grad, self.w.grad, self.axis)

