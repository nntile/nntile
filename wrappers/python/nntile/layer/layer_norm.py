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
# @date 2023-04-26

from nntile.tensor import TensorTraits, Tensor_fp32, Tensor_fp64, Tensor, \
        TensorOrNone, TensorMoments, \
        bias_slice_async, copy_async, sum_slice_async, norm_async, \
        fill_async, pow_async, \
        biasprod_async, scalprod_async, axpy_async, biasprod_outer_async, \
        bias_outer_async, sum_fiber_async, scalprod_outer_async, \
        clear_async
from nntile.layer.base_layer import BaseLayer
import numpy as np
from typing import List

class LayerNorm(BaseLayer):
    x: TensorMoments
    y: TensorMoments
    gamma: TensorMoments
    beta: TensorMoments
    tmp_y_value: Tensor
    tmp_y_grad: Tensor
    mean: Tensor
    inv_stddev: Tensor
    axis: int
    eps: float

    # Construct normalization layer with all the provided data
    def __init__(self, x: TensorMoments, y: TensorMoments, \
            gamma: TensorMoments, beta: TensorMoments, tmp_y_value: Tensor, \
            tmp_y_grad: Tensor, mean: Tensor, inv_stddev: Tensor, axis: int, \
            eps: float):
        # Redirect to BaseLayer initialization
        super().__init__([x], [y], [gamma, beta], [tmp_y_value, tmp_y_grad, \
                mean, inv_stddev])
        self.x = x
        self.y = y
        self.gamma = gamma
        self.beta = beta
        self.tmp_y_value = tmp_y_value
        self.tmp_y_grad = tmp_y_grad
        self.mean = mean
        self.inv_stddev = inv_stddev
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
        x_distr = x.value.distribution
        y_value = type(x.value)(x_traits, x_distr, next_tag)
        next_tag = y_value.next_tag
        # Create grad Y with the same traits and distribution as X
        y_grad = type(x.value)(x_traits, x_distr, next_tag)
        next_tag = y_grad.next_tag
        # Wrap Y
        y = TensorMoments(y_value, y_grad, True)
        # Gamma parameter
        gamma_shape = [x.value.shape[axis]]
        gamma_basetile = [x.value.basetile_shape[axis]]
        gamma_traits = TensorTraits(gamma_shape, gamma_basetile)
        gamma_distr = []
        for i in range(x.value.grid.shape[axis]):
            gamma_distr.append(x_distr[x.value.grid.stride[axis]*i])
        gamma_value = type(x.value)(gamma_traits, gamma_distr, next_tag)
        next_tag = gamma_value.next_tag
        gamma_grad = type(x.value)(gamma_traits, gamma_distr, next_tag)
        next_tag = gamma_grad.next_tag
        gamma = TensorMoments(gamma_value, gamma_grad, True)
        # Beta parameter
        beta_value = type(x.value)(gamma_traits, gamma_distr, next_tag)
        next_tag = beta_value.next_tag
        beta_grad = type(x.value)(gamma_traits, gamma_distr, next_tag)
        next_tag = beta_grad.next_tag
        beta = TensorMoments(beta_value, beta_grad, True)
        # Temporary tensor for normalized input
        tmp_y_value = type(x.value)(x_traits, x_distr, next_tag)
        next_tag = tmp_y_value.next_tag
        # Temporary tensor for gradient of normalized input
        tmp_y_grad = type(x.value)(x_traits, x_distr, next_tag)
        next_tag = tmp_y_grad.next_tag
        # Define auxiliary tensors to hold mean, inverse of stddev and scalar
        # products along given axis
        mean_shape = x.value.shape[:axis] + x.value.shape[axis+1:]
        mean_basetile = x.value.basetile_shape[:axis] \
                + x.value.basetile_shape[axis+1:]
        mean_traits = TensorTraits(mean_shape, mean_basetile)
        mean_distr = []
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
        # Create LayerNorm object with all the provided tensors
        layer = LayerNorm(x, y, gamma, beta, tmp_y_value, tmp_y_grad, mean, \
                inv_stddev, axis, eps)
        # Init gamma and beta
        clear_async(beta.value)
        fill_async(1.0, gamma.value)
        # Return layer and next tag to be used
        return (layer, next_tag)

    # Forward propagation of the normalization layer
    def forward_async(self):
        # Get means over given axis
        sum_slice_async(1.0/self.l, self.x.value, 0.0, self.mean, self.axis)
        # Init Y as a copy of X
        copy_async(self.x.value, self.y.value)
        # Subtract mean from Y
        bias_slice_async(-1.0, self.mean, 1.0, self.y.value, self.axis)
        # Compute standard deviation of self.y.value
        fill_async(self.eps, self.inv_stddev)
        norm_async(1.0/self.l**0.5, self.y.value, 1.0, self.inv_stddev, \
                self.axis)
        # Invert stddev (to multiply by it instead of dividing)
        pow_async(1.0, -1.0, self.inv_stddev)
        # Finally, normalize input
        biasprod_async(self.inv_stddev, self.y.value, self.axis)
        # Copy normalized input for the backward phase
        copy_async(self.y.value, self.tmp_y_value)
        # Scale output
        biasprod_outer_async(self.gamma.value, self.y.value, self.axis)
        # Shift output
        bias_outer_async(1.0, self.beta.value, self.y.value, self.axis)

    def backward_async(self):
        # Accumulate gradient over beta
        sum_fiber_async(1.0, self.y.grad, 1.0, self.beta.grad, self.axis)
        # Accumulate gradient over gamma
        scalprod_outer_async(1.0, self.y.grad, self.tmp_y_value, 1.0, \
                self.gamma.grad, self.axis)
        # Define gradient over normalized input
        copy_async(self.y.grad, self.tmp_y_grad)
        biasprod_outer_async(self.gamma.value, self.tmp_y_grad, self.axis)
        # Get mean of product of tmp_Y_grad and tmp_Y_value over the given axis
        scalprod_async(-1.0/self.l, self.tmp_y_grad, self.tmp_y_value, 0.0, \
                self.mean, self.axis)
        # Multiply tmp_Y_value by the mean
        biasprod_async(self.mean, self.tmp_y_value, self.axis)
        # Add tmp_Y_grad to tmp_Y_value
        axpy_async(1.0, self.tmp_y_grad, self.tmp_y_value)
        # Get mean value of tmp_Y_grad over the given axis
        sum_slice_async(1.0/self.l, self.tmp_y_grad, 0.0, self.mean, self.axis)
        # Subtract mean from tmp_Y_value
        bias_slice_async(-1.0, self.mean, 1.0, self.tmp_y_value, self.axis)
        # Multiply tmp_Y_value by the inverse stddev
        biasprod_async(self.inv_stddev, self.tmp_y_value, self.axis)
        # Accumulate gradient from tmp_Y_value
        axpy_async(1.0, self.tmp_y_value, self.x.grad)

