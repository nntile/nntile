# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/linear.py
# Linear layer of NNTile Python package
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-02-11

import nntile.tensor as tensor
import numpy as np
from typing import List

class Linear:
    x: tensor.Tensor
    dx: tensor.Tensor
    y: tensor.Tensor
    dy: tensor.Tensor
    w: tensor.Tensor
    dw: tensor.Tensor
    b: tensor.TensorOrNone
    db: tensor.TensorOrNone
    side: str
    trans_x: tensor.TransOp
    trans_w: tensor.TransOp
    ndim: int
    params: List[tensor.Tensor]
    grads: List[tensor.Tensor]

    # Construct linear layer with all the provided data
    def __init__(self, x, dx, y, dy, w, dw, b, db, side, trans_x, trans_w,
            ndim):
        self.x = x
        self.dx = dx
        self.y = y
        self.dy = dy
        self.w = w
        self.dw = dw
        self.b = b
        self.db = db
        self.side = side
        self.trans_x = trans_x
        self.trans_w = trans_w
        self.ndim = ndim
        self.params = [w, b]
        self.grads = [dw, db]

    # Simple generator for the linear layer
    @staticmethod
    def generate_block_cyclic(x, dx, side, new_shape, new_basetile_shape,
            next_tag):
        # Define shapes
        if side == 'L':
            w_shape = [x.shape[-1]] + [new_shape]
            w_basetile = [x.basetile_shape[-1]] + [new_basetile_shape]
            y_shape = x.shape[:-1] + [new_shape]
            y_basetile = x.basetile_shape[:-1] + [new_basetile_shape]
        else:
            w_shape = [new_shape] + [x.shape[0]]
            w_basetile = [new_basetile_shape] + [x.basetile_shape[0]]
            y_shape = [new_shape] + x.shape[1:]
            y_basetile = [new_basetile_shape] + x.basetile_shape[1:]
        # Define W
        w_traits = tensor.TensorTraits(w_shape, w_basetile)
        # TODO change distribution
        w_distr = [0] * w_traits.grid.nelems
        w = type(x)(w_traits, w_distr, next_tag)
        next_tag = w.next_tag
        # Define Y
        y_traits = tensor.TensorTraits(y_shape, y_basetile)
        # TODO change distribution
        y_distr = [0] * y_traits.grid.nelems
        y = type(x)(y_traits, y_distr, next_tag)
        next_tag = y.next_tag
        # Create dW with the same traits and distribution as W
        dw = type(x)(w_traits, w.distribution, next_tag)
        next_tag = dw.next_tag
        # Create dY with the same traits and distribution as Y
        dy = type(x)(y_traits, y.distribution, next_tag)
        next_tag = dy.next_tag
        # Create activation layer with all the provided tensors
        layer = Linear(x, dx, y, dy, w, dw, None, None, side, tensor.notrans,
                tensor.notrans, 1)
        # Return layer and next tag to be used
        return (layer, next_tag)

    # Forward propagation of the linear layer
    def forward_async(self):
        # Perform actual gemm
        if self.side == 'L':
            tensor.gemm_async(1.0, self.trans_x, self.x, self.trans_w, self.w,
                    0.0, self.y, self.ndim)
        else:
            tensor.gemm_async(1.0, self.trans_w, self.w, self.trans_x, self.x,
                    0.0, self.y, self.ndim)
        # Destroy values stored in tensor X
        self.x.invalidate_submit()
        # Hint for StarPU that W tensor will
        # not be used soon and it is advised to offload data from GPU 
        self.w.wont_use()

