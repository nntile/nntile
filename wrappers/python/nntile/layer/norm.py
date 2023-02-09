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
# @date 2023-02-09

import nntile.tensor as tensor
import numpy as np
from typing import List

class Norm:
    x: tensor.Tensor
    dx: tensor.Tensor
    y: tensor.Tensor
    dy: tensor.Tensor
    gb: tensor.Tensor
    dgb: tensor.TensorOrNone
    sumnorm: tensor.Tensor
    y_last: tensor.Tensor
    axis: int
    l: int
    eps: float
    params: List[tensor.Tensor]
    grads: List[tensor.Tensor]

    # Construct normalization layer with all the provided data
    def __init__(self, x, dx, y, dy, gb, dgb, sumnorm, y_last, axis, eps):
        self.x = x
        self.dx = dx
        self.y = y
        self.dy = dy
        self.gb = gb
        self.dgb = dgb
        self.sumnorm = sumnorm
        self.y_last = y_last
        self.axis = axis
        self.l = self.x.shape[axis]
        self.eps = eps
        self.params = [self.gb]
        self.grads = [self.dgb]

    # Simple generator for the normalization layer
    @staticmethod
    def generate_block_cyclic(x, dx, axis, eps, next_tag):
        # Check if X and dX correspond to each other
        if x.shape != dx.shape or x.basetile_shape != dx.basetile_shape:
            raise ValueError
        # Get traits of X
        x_traits = tensor.TensorTraits(x.shape, x.basetile_shape)
        # Create Y with the same traits and distribution as X
        y = type(x)(x_traits, x.distribution, next_tag)
        next_tag = y.next_tag
        # Create dY with the same traits and distribution as X
        dy = type(x)(x_traits, x.distribution, next_tag)
        next_tag = dy.next_tag
        # Gamma and beta simply contain 2 numbers
        gb_traits = tensor.TensorTraits([2], [2])
        # Home node for gamma-and-beta tensor is 0
        gb = type(x)(gb_traits, [0], next_tag)
        next_tag = gb.next_tag
        # Init gamma=1 and beta=0
        np_gb = np.array([1.0, 0.0], dtype=np.float32, order='F')
        # TODO: fix for MPI case (only 0-node shall launch the following line)
        gb.from_array(np_gb)
        # derivative over gamma and beta
        dgb = type(x)(gb_traits, [0], next_tag)
        next_tag = dgb.next_tag
        # Define auxiliary tensor to hold sums and norms along axis
        sumnorm_shape = [2] + x.shape[:axis] + x.shape[axis+1:]
        sumnorm_basetile = [2] + x.basetile_shape[:axis] \
                + x.basetile_shape[axis+1:]
        sumnorm_traits = tensor.TensorTraits(sumnorm_shape, sumnorm_basetile)
        sumnorm_distr = []
        x_distr = x.distribution
        for i in range(sumnorm_traits.grid.nelems):
            sumnorm_tile_index = sumnorm_traits.grid.linear_to_index(i)
            x_tile_index = sumnorm_tile_index[1:axis+1] + [0] \
                    + sumnorm_tile_index[axis+1:]
            x_tile_offset = x.grid.index_to_linear(x_tile_index)
            sumnorm_distr.append(x_distr[x_tile_offset])
        sumnorm = type(x)(sumnorm_traits, sumnorm_distr, next_tag)
        next_tag = sumnorm.next_tag
        # Store Y as implementation of backward propagation is based on it
        y_last = type(x)(x_traits, x.distribution, next_tag)
        next_tag = y_last.next_tag
        # No need to store X, as backward propagation does not need it
        # Create normalization layer with all the provided tensors
        layer = Norm(x, dx, y, dy, gb, dgb, sumnorm, y_last, axis, eps)
        # Return layer and next tag to be used
        return (layer, next_tag)

    # Create a new layer that works with a different input size, but relies on
    # the same layer parameters
    def rebatch(self, new_x, new_dx, batch_axis, next_tag):
        # Shortcuts for new and old shapes and basetiles
        new_shape = new_x.shape
        new_basetile = new_x.basetile_shape
        shape = self.x.shape
        basetile = self.x.basetile_shape
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
        # Check if new X and dX correspond to each other
        if new_x.shape != new_dx.shape \
                or new_x.basetile_shape != new_dx.basetile_shape:
            raise ValueError
        # Define new Y and dY with the same shape and basetile as new X
        new_x_traits = tensor.TensorTraits(new_x.shape, new_x.basetile_shape)
        new_y = type(new_x)(new_x_traits, new_x.distribution, next_tag)
        next_tag = new_y.next_tag
        new_dy = type(new_x)(new_x_traits, new_x.distribution, next_tag)
        next_tag = new_dy.next_tag
        # If normalization axis is the same as batch_axis then we can use the
        # same sumnorm as before
        if batch_axis == self.axis:
            new_sumnorm = self.sumnorm
        else:
            new_sumnorm_shape = [2] + new_shape[:self.axis] \
                    + new_shape[self.axis+1:]
            new_sumnorm_basetile = [2] + new_basetile[:self.axis] \
                    + new_basetile[self.axis+1:]
            new_sumnorm_traits = tensor.TensorTraits(new_sumnorm_shape, \
                    new_sumnorm_basetile)
            new_sumnorm_distr = []
            new_x_distr = new_x.distribution
            for i in range(new_sumnorm_traits.grid.nelems):
                new_sumnorm_tile_index = \
                        new_sumnorm_traits.grid.linear_to_index(i)
                new_x_tile_index = new_sumnorm_tile_index[1:self.axis+1] \
                        + [0] + new_sumnorm_tile_index[self.axis+1:]
                new_x_tile_offset = new_x.grid.index_to_linear( \
                        new_x_tile_index)
                new_sumnorm_distr.append(new_x_distr[new_x_tile_offset])
            new_sumnorm = type(new_x)(new_sumnorm_traits, \
                    new_sumnorm_distr, next_tag)
            next_tag = new_sumnorm.next_tag
        new_y_last = type(new_x)(new_x_traits, new_x.distribution, \
                next_tag)
        next_tag = new_y_last.next_tag
        new_layer = Norm(new_x, new_dx, new_y, new_dy, self.gb, \
                self.dgb, new_sumnorm, new_y_last, self.axis, self.eps)
        return (new_layer, next_tag)

    # Forward propagation of the normalization layer
    def forward_async(self):
        # Clear auxiliary tensor for sums and norms of slices along given axis
        tensor.clear_async(self.sumnorm)
        # Accumulate sums and norms of slices along given axis
        tensor.sumnorm_async(self.x, self.sumnorm, self.axis)
        # Init Y as a copy of X
        tensor.copy_async(self.x, self.y)
        # Normalize Y inplace
        tensor.normalize_async(self.gb, self.sumnorm, self.y, self.l,
                self.eps, self.axis)
        # Copy Y to utilize it during backward propagation
        tensor.copy_async(self.y, self.y_last)
        # Destroy values stored in tensor X
        self.x.invalidate_submit()
        # Hint for StarPU that gamma-and-beta, sumnorm and y_last tensors will
        # not be used soon and it is advised to offload data from GPU 
        self.gb.wont_use()
        self.sumnorm.wont_use()
        self.y_last.wont_use()

    def backward_async(self):
        raise NotImplementedError

