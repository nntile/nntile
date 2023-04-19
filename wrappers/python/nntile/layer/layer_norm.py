# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/layer_norm.py
# LayerNorm of NNTile Python package
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-04-19

from nntile.tensor import TensorTraits, Tensor_fp32, Tensor_fp64, Tensor, \
        TensorOrNone, TensorMoments, \
        bias_async, copy_async, sum_async, norm_async, set_async, pow_async, \
        biasprod_async, scalprod_async, axpy_async
from nntile.layer.base_layer import BaseLayer
import numpy as np
from typing import List

class LayerNorm(BaseLayer):
    x: TensorMoments
    y: TensorMoments
    mean: Tensor
    inv_stddev: Tensor
    scalprod: Tensor
    axis: int
    eps: float

    # Construct normalization layer with all the provided data
    def __init__(self, x: TensorMoments, y: TensorMoments, mean: Tensor,
            inv_stddev: Tensor, scalprod: Tensor, axis: int, eps: float):
        # Redirect to BaseLayer initialization
        super().__init__([x], [y], [], [mean, inv_stddev, scalprod])
        self.x = x
        self.y = y
        self.mean = mean
        self.inv_stddev = inv_stddev
        self.scalprod = scalprod
        self.axis = axis
        self.l = self.x.value.shape[axis]
        self.eps = eps ** 0.5 # This value is used to init deviation

    # Simple generator for the normalization layer
    @staticmethod
    def generate_simple(x: TensorMoments, axis: int, eps: float,
            next_tag: int):
        # Get traits of X
        x_traits = TensorTraits(x.value.shape, x.value.basetile_shape)
        # Create Y with the same traits and distribution as X
        y_value = type(x.value)(x_traits, x.value.distribution, next_tag)
        next_tag = y_value.next_tag
        # Create grad Y with the same traits and distribution as X
        y_grad = type(x.value)(x_traits, x.value.distribution, next_tag)
        next_tag = y_grad.next_tag
        # Wrap Y
        y = TensorMoments(y_value, y_grad, True)
        # Define auxiliary tensors to hold mean, inverse of stddev and scalar
        # products along given axis
        mean_shape = x.value.shape[:axis] + x.value.shape[axis+1:]
        mean_basetile = x.value.basetile_shape[:axis] \
                + x.value.basetile_shape[axis+1:]
        mean_traits = TensorTraits(mean_shape, mean_basetile)
        mean_distr = []
        x_distr = x.value.distribution
        # Set distribution of mean tensor as X tensor with 0 index in provided
        # axis
        for i in range(mean_traits.grid.nelems):
            mean_tile_index = mean_traits.grid.linear_to_index(i)
            x_tile_index = mean_tile_index[0:axis] + [0] \
                    + mean_tile_index[axis:]
            x_tile_offset = x.value.grid.index_to_linear(x_tile_index)
            mean_distr.append(x_distr[x_tile_offset])
        mean = type(x.value)(mean_traits, mean_distr, next_tag)
        next_tag = mean.next_tag
        inv_stddev = type(x.value)(mean_traits, mean_distr, next_tag)
        next_tag = inv_stddev.next_tag
        scalprod = type(x.value)(mean_traits, mean_distr, next_tag)
        next_tag = scalprod.next_tag
        # Create LayerNorm object with all the provided tensors
        layer = LayerNorm(x, y, mean, inv_stddev, scalprod, axis, eps)
        # Return layer and next tag to be used
        return (layer, next_tag)

    # Forward propagation of the normalization layer
    def forward_async(self):
        # Get means over given axis
        sum_async(1.0/self.l, self.x.value, 0.0, self.mean, self.axis)
        # Init Y as a copy of X
        copy_async(self.x.value, self.y.value)
        # Subtract mean from Y
        bias_async(-1.0, self.mean, self.y.value, self.axis)
        # Compute standard deviation of self.y.value
        set_async(self.eps, self.inv_stddev)
        norm_async(1.0/self.l**0.5, self.y.value, 1.0, self.inv_stddev, \
                self.axis)
        # Invert stddev (to multiply by it instead of dividing)
        pow_async(1.0, -1.0, self.inv_stddev)
        # Finally, normalize output
        biasprod_async(self.inv_stddev, self.y.value, self.axis)

    def backward_async(self):
        # Copy Y into dX
        copy_async(self.y.value, self.x.grad)
        # Get mean of product of dY and Y over the given axis
        scalprod_async(-1.0/self.l, self.y.grad, self.y.value, 0.0, \
                self.mean, self.axis)
        # Multiply dX by the mean
        biasprod_async(self.mean, self.x.grad, self.axis)
        # Add dY to dX
        axpy_async(1.0, self.y.grad, self.x.grad)
        # Get mean value of dY over the given axis
        sum_async(1.0/self.l, self.y.grad, 0.0, self.mean, self.axis)
        # Subtract mean from dX
        bias_async(-1.0, self.mean, self.x.grad, self.axis)
        # Multiply dX by the inverse stddev
        biasprod_async(self.inv_stddev, self.x.grad, self.axis)

