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
    def generate_simple(x: TensorMoments, y: TensorMoments, next_tag: int):
        head_size, n_seq, n_batch, n_head = x.value.shape
        head_size_tile, n_seq_tile, n_batch_tile, n_head_tile = x.value.basetile_shape

        n_emb = n_head * head_size
        n_emb_tile = n_head_tile * head_size_tile

        cos_shape = [n_emb, n_seq]
        sin_shape = [n_emb, n_seq]

        cos_basetile = [n_emb_tile, n_seq_tile]
        sin_basetile = [n_emb_tile, n_seq_tile]

        cos_traits = TensorTraits(cos_shape, cos_basetile)
        sin_traits = TensorTraits(sin_shape, sin_basetile)

        cos_distr = [0] * cos_traits.grid.nelems
        sin_distr = [0] * sin_traits.grid.nelems

        cos = type(x.value)(cos_traits, cos_distr, next_tag)
        next_tag = cos.next_tag

        sin = type(x.value)(sin_traits, sin_distr, next_tag)
        next_tag = sin.next_tag

        inv_freq = 1.0 / (10000 ** (np.arange(0, n_emb, 2, dtype=np.int64).float() / n_emb))
        t = np.arange(n_seq, dtype=np.int64).type_as(inv_freq)
        freqs = np.outer(inv_freq, t)
        
        cos.from_array(np.cos(freqs))
        sin.from_array(np.sin(freqs))

        # Create rotary embedding layer with all the provided data
        layer = RotaryEmbedding(x, y, sin, cos) 
        # Return layer and next tag to be used
        return (layer, next_tag)

    # Forward propagation of the embedding layer
    def forward_async(self):
        clear_async(self.y.value)
        rope_async(self.sin, self.cos, self.x, self.y.value, 2)

    # # Backward propagation of the embedding layer
    # def backward_async(self):
    #     embedding_backward_async(self.x, self.y.grad, self.w.grad, self.axis, \
    #             redux=0)
