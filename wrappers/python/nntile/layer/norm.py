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
# @date 2023-02-07

import nntile.nntile_core.tensor as tensor
import numpy as np

class Norm_fp32:
    x: tensor.Tensor_fp32
    dx: tensor.Tensor_fp32
    y: tensor.Tensor_fp32
    dy: tensor.Tensor_fp32
    gb: tensor.Tensor_fp32
    sumnorm: tensor.Tensor_fp32
    y_last: tensor.Tensor_fp32
    axis: int
    l: int
    eps: float
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

    @staticmethod
    def generate_block_cyclic(x, dx, axis, eps, next_tag):
        if x.shape != dx.shape or x.basetile_shape != dx.basetile_shape:
            raise ValueError
        x_traits = tensor.TensorTraits(x.shape, x.basetile_shape)
        y = tensor.Tensor_fp32(x_traits, x.distribution, next_tag)
        next_tag = y.next_tag
        dy = tensor.Tensor_fp32(x_traits, x.distribution, next_tag)
        next_tag = dy.next_tag
        gb_traits = tensor.TensorTraits([2], [2])
        gb = tensor.Tensor_fp32(gb_traits, [0], next_tag)
        next_tag = gb.next_tag
        np_gb = np.array([1.0, 0.0], dtype=np.float32, order='F')
        gb.from_array(np_gb)
        dgb = tensor.Tensor_fp32(gb_traits, [0], next_tag)
        next_tag = dgb.next_tag
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
        sumnorm = tensor.Tensor_fp32(sumnorm_traits, sumnorm_distr, next_tag)
        next_tag = sumnorm.next_tag
        y_last = tensor.Tensor_fp32(x_traits, x.distribution, next_tag)
        next_tag = y_last.next_tag
        layer = Norm_fp32(x, dx, y, dy, gb, dgb, sumnorm, y_last, axis, eps)
        return (layer, next_tag)

    def rebatch(self, new_x, new_dx, batch_axis, next_tag):
        # Shortcuts for new and old shapes and basetiles
        new_shape = new_x.shape
        new_basetile =new_x.basetile_shape
        shape = self.x.shape
        basetile = self.x.basetile_shape
        # In case nothing is changed do nothing
        if new_shape == shape and new_basetile == basetile:
            return self
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
        new_y = tensor.Tensor_fp32(new_x_traits, new_x.distribution, next_tag)
        next_tag = new_y.next_tag
        new_dy = tensor.Tensor_fp32(new_x_traits, new_x.distribution, next_tag)
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
            new_sumnorm = tensor.Tensor_fp32(new_sumnorm_traits, \
                    new_sumnorm_distr, next_tag)
            next_tag = new_sumnorm.next_tag
        new_y_last = tensor.Tensor_fp32(new_x_traits, new_x.distribution, \
                next_tag)
        next_tag = new_y_last.next_tag
        new_layer = Norm_fp32(new_x, new_dx, new_y, new_dy, self.gb, \
                self.dgb, new_sumnorm, new_y_last, self.axis, self.eps)
        return (new_layer, next_tag)

    def forward_async(self):
        tensor.clear_async_fp32(self.sumnorm)
        tensor.sumnorm_async_fp32(self.x, self.sumnorm, self.axis)
        tensor.copy_async_fp32(self.x, self.y)
        tensor.normalize_async_fp32(self.gb, self.sumnorm, self.y, self.l,
                self.eps, self.axis)
        tensor.copy_async_fp32(self.y, self.y_last)
        self.x.invalidate_submit()
        self.gb.wont_use()
        self.sumnorm.wont_use()
        self.y_last.wont_use()

    def backward_async(self):
        raise NotImplementedError

