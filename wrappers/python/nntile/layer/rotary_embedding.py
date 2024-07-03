# @copyright (c) 2022-2024 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/rotary_embedding.py
# Rotary Positional Embedding layer of NNTile Python package
#
# @version 1.0.0
# @author Gleb Karpov
# @date 2024-06-29

from nntile.tensor import TensorTraits, Tensor, TensorOrNone, TensorMoments, clear_async, rope_async
from nntile.layer.base_layer import BaseLayer
import numpy as np
from typing import List


class RotaryEmbedding(BaseLayer):
    x: TensorMoments
    y: TensorMoments
    sin: Tensor
    cos: Tensor

    # Construct rotary embedding layer with all the provided data
    def __init__(self, x: TensorMoments, y: TensorMoments, sin: Tensor, cos: Tensor):
        # Redirect to BaseClass initialization
        super().__init__([x], [y], [], [sin, cos])
        # Named storage
        self.x = x
        self.y = y
        self.sin = sin
        self.cos = cos

    # Simple generator for the embedding layer
    @staticmethod
    def generate_simple(x: TensorMoments, next_tag: int):
        head_size, n_seq, _, n_head = x.value.shape
        head_size_tile, n_seq_tile, _, n_head_tile = x.value.basetile_shape

        cos_shape = [int(head_size / 2) , n_seq, n_head]
        sin_shape = [int(head_size / 2) , n_seq, n_head]

        cos_basetile = [int(head_size_tile / 2), n_seq_tile, n_head_tile]
        sin_basetile = [int(head_size_tile / 2), n_seq_tile, n_head_tile]

        cos_traits = TensorTraits(cos_shape, cos_basetile)
        sin_traits = TensorTraits(sin_shape, sin_basetile)

        cos_distr = [0] * cos_traits.grid.nelems
        sin_distr = [0] * sin_traits.grid.nelems

        cos = type(x.value)(cos_traits, cos_distr, next_tag)
        next_tag = cos.next_tag

        sin = type(x.value)(sin_traits, sin_distr, next_tag)
        next_tag = sin.next_tag

        y_shape = x.value.shape
        y_basetile_shape = x.value.basetile_shape
        y_traits = TensorTraits(y_shape, y_basetile_shape)
        y_distr = [0] * y_traits.grid.nelems
        y_value = type(x.value)(y_traits, y_distr, next_tag)
        next_tag = y_value.next_tag

        # Create gradient of Y with the same traits and distribution as Y
        y_grad = type(x.value)(y_traits, y_distr, next_tag)
        next_tag = y_grad.next_tag
        # Define Y as TensorMoments
        y = TensorMoments(y_value, y_grad, True)

        # Create rotary embedding layer with all the provided data
        layer = RotaryEmbedding(x, y, sin, cos) 
        # Return layer and next tag to be used
        return (layer, next_tag)

    # Forward propagation of the embedding layer
    def forward_async(self):
        clear_async(self.y.value)
        rope_async(self.sin, self.cos, self.x.value, self.y.value, 2)

    # # Backward propagation of the embedding layer
    # def backward_async(self):
    #     embedding_backward_async(self.x, self.y.grad, self.w.grad, self.axis, \
    #             redux=0)
