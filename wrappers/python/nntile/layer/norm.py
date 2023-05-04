# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/norm.py
# Normalization layer of NNTile Python package
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-02-15

from nntile.tensor import TensorTraits, Tensor_fp32, Tensor_fp64, Tensor, \
        TensorOrNone, TensorMoments, \
        clear_async, copy_async, sumnorm_async, normalize_async
from nntile.layer.base_layer import BaseLayer
import numpy as np
from typing import List

class Norm(BaseLayer):
    x: TensorMoments
    y: TensorMoments
    gb: TensorMoments
    sumnorm: Tensor
    y_last: Tensor
    axis: int
    l: int
    eps: float

    # Construct normalization layer with all the provided data
    def __init__(self, x: TensorMoments, y: TensorMoments, gb: TensorMoments,
            sumnorm: Tensor, y_last: Tensor, axis: int, eps: float):
        # Redirect to BaseLayer initialization
        super().__init__([x], [y], [gb], [sumnorm, y_last])
        self.x = x
        self.y = y
        self.gb = gb
        self.sumnorm = sumnorm
        self.y_last = y_last
        self.axis = axis
        self.l = self.x.value.shape[axis]
        self.eps = eps

    # Simple generator for the normalization layer
    @staticmethod
    def generate_simple(x: TensorMoments, axis: int, eps: float,
            next_tag: int):
        # Get traits of X
        x_traits = TensorTraits(x.value.shape, x.value.basetile_shape)
        # Create Y with the same traits and distribution as X
        y = type(x.value)(x_traits, x.value.distribution, next_tag)
        next_tag = y.next_tag
        # Create grad Y with the same traits and distribution as X
        y_grad = type(x.value)(x_traits, x.value.distribution, next_tag)
        next_tag = y_grad.next_tag
        # Wrap Y
        y_moments = TensorMoments(y, y_grad, True)
        # Gamma and beta simply contain 2 numbers
        gb_traits = TensorTraits([2], [2])
        # Home node for gamma-and-beta tensor is 0
        gb = type(x.value)(gb_traits, [0], next_tag)
        next_tag = gb.next_tag
        # Init gamma=1 and beta=0
        if type(x.value) is Tensor_fp32:
            np_gb = np.array([1.0, 0.0], dtype=np.float32, order='F')
        if type(x.value) is Tensor_fp64:
            np_gb = np.array([1.0, 0.0], dtype=np.float64, order='F')
        # TODO: fix for MPI case (only 0-node shall launch the following line)
        gb.from_array(np_gb)
        # derivative over gamma and beta
        gb_grad = type(x.value)(gb_traits, [0], next_tag)
        next_tag = gb_grad.next_tag
        # Wrap Gamma and beta
        gb_moments = TensorMoments(gb, gb_grad, True)
        # Define auxiliary tensor to hold sums and norms along axis
        sumnorm_shape = [2] + x.value.shape[:axis] + x.value.shape[axis+1:]
        sumnorm_basetile = [2] + x.value.basetile_shape[:axis] \
                + x.value.basetile_shape[axis+1:]
        sumnorm_traits = TensorTraits(sumnorm_shape, sumnorm_basetile)
        sumnorm_distr = []
        x_distr = x.value.distribution
        for i in range(sumnorm_traits.grid.nelems):
            sumnorm_tile_index = sumnorm_traits.grid.linear_to_index(i)
            x_tile_index = sumnorm_tile_index[1:axis+1] + [0] \
                    + sumnorm_tile_index[axis+1:]
            x_tile_offset = x.value.grid.index_to_linear(x_tile_index)
            sumnorm_distr.append(x_distr[x_tile_offset])
        sumnorm = type(x.value)(sumnorm_traits, sumnorm_distr, next_tag)
        next_tag = sumnorm.next_tag
        # Store Y as implementation of backward propagation is based on it
        y_last = type(x.value)(x_traits, x.value.distribution, next_tag)
        next_tag = y_last.next_tag
        # No need to store X, as backward propagation does not need it
        # Create normalization layer with all the provided tensors
        layer = Norm(x, y_moments, gb_moments, sumnorm, y_last, axis, eps)
        # Return layer and next tag to be used
        return (layer, next_tag)

    # Create a new layer that works with a different input size, but relies on
    # the same layer parameters
    def rebatch(self, new_x: TensorMoments, batch_axis: int, next_tag: int):
        # Shortcuts for new and old shapes and basetiles
        new_shape = new_x.value.shape
        new_basetile = new_x.value.basetile_shape
        shape = self.x.value.shape
        basetile = self.x.value.basetile_shape
        # Check if new and old shapes differ only in 1 axis
        check_new = new_shape[:batch_axis] + new_shape[batch_axis+1:]
        check_old = shape[:batch_axis] + shape[batch_axis+1:]
        if check_new != check_old:
            raise ValueError
        # Check if new and old basetiles differ only in 1 axis
        check_new = new_basetile[:batch_axis] + new_basetile[batch_axis+1:]
        check_old = basetile[:batch_axis] + basetile[batch_axis+1:]
        if check_new != check_old:
            raise ValueError
        # Define new Y and dY with the same shape and basetile as new X
        new_x_traits = TensorTraits(new_x.value.shape,
                new_x.value.basetile_shape)
        new_y = type(new_x.value)(new_x_traits, new_x.value.distribution,
                next_tag)
        next_tag = new_y.next_tag
        new_y_grad = type(new_x.value)(new_x_traits, new_x.value.distribution,
                next_tag)
        next_tag = new_y_grad.next_tag
        # Wrap new_Y
        new_y_moments = TensorMoments(new_y, new_y_grad, True)
        # If normalization axis is the same as batch_axis then we can use the
        # same sumnorm as before
        if batch_axis == self.axis:
            new_sumnorm = self.sumnorm
        else:
            new_sumnorm_shape = [2] + new_shape[:self.axis] \
                    + new_shape[self.axis+1:]
            new_sumnorm_basetile = [2] + new_basetile[:self.axis] \
                    + new_basetile[self.axis+1:]
            new_sumnorm_traits = TensorTraits(new_sumnorm_shape,
                    new_sumnorm_basetile)
            new_sumnorm_distr = []
            new_x_distr = new_x.value.distribution
            for i in range(new_sumnorm_traits.grid.nelems):
                new_sumnorm_tile_index = \
                        new_sumnorm_traits.grid.linear_to_index(i)
                new_x_tile_index = new_sumnorm_tile_index[1:self.axis+1] \
                        + [0] + new_sumnorm_tile_index[self.axis+1:]
                new_x_tile_offset = new_x.value.grid.index_to_linear( \
                        new_x_tile_index)
                new_sumnorm_distr.append(new_x_distr[new_x_tile_offset])
            new_sumnorm = type(new_x.value)(new_sumnorm_traits, \
                    new_sumnorm_distr, next_tag)
            next_tag = new_sumnorm.next_tag
        new_y_last = type(new_x.value)(new_x_traits,
                new_x.value.distribution, next_tag)
        next_tag = new_y_last.next_tag
        new_layer = Norm(new_x, new_y_moments, self.gb, new_sumnorm, new_y_last,
                self.axis, self.eps)
        return (new_layer, next_tag)

    # Forward propagation of the normalization layer
    def forward_async(self):
        # Clear auxiliary tensor for sums and norms of slices along given axis
        clear_async(self.sumnorm)
        # Accumulate sums and norms of slices along given axis
        sumnorm_async(self.x.value, self.sumnorm, self.axis)
        # Init Y as a copy of X
        copy_async(self.x.value, self.y.value)
        # Normalize Y inplace
        normalize_async(self.gb.value, self.sumnorm, self.y.value, self.l,
                self.eps, self.axis)
        # Copy Y to utilize it during backward propagation
        copy_async(self.y.value, self.y_last)
        # Hint for StarPU that gamma-and-beta, sumnorm and y_last tensors will
        # not be used soon and it is advised to offload data from GPU 
        #self.gb.value.wont_use()
        #self.sumnorm.wont_use()
        #self.y_last.wont_use()

    def backward_async(self):
        raise NotImplementedError

