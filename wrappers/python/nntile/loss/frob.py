# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/loss/frob.py
# Frobenius norm loss of NNTile Python package
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-02-09

import nntile.tensor as tensor
import numpy as np

class Frob:
    x: tensor.Tensor
    dx: tensor.Tensor
    y: tensor.Tensor
    val: tensor.Tensor
    tmp: tensor.Tensor

    # Constructor of loss with all the provided data
    def __init__(self, x: tensor.Tensor, dx: tensor.Tensor, y: tensor.Tensor,
            val: tensor.Tensor, tmp: tensor.Tensor):
        self.x = x
        self.dx = dx
        self.y = y
        self.val = val
        self.tmp = tmp

    # Simple generator for the normalization layer
    @staticmethod
    def generate_block_cyclic(x: tensor.Tensor, dx: tensor.Tensor,
            y: tensor.Tensor, val: tensor.Tensor, next_tag) -> tuple:
        ndim = len(x.grid.shape)
        tmp_traits = tensor.TensorTraits(x.grid.shape, [1]*ndim)
        tmp = type(x)(tmp_traits, x.distribution, next_tag)
        next_tag = tmp.next_tag
        loss = Frob(x, dx, y, val, tmp)
        return loss, next_tag

    # Get both value and gradient
    def value_grad_async(self):
        # Get gradient into dX
        self.grad_async()
        # Get value ||dX||
        tensor.nrm2_async(self.dx, self.val, self.tmp)
        # Ignore temporary values
        self.tmp.invalidate_submit()
        # Compute loss as 0.5*||dX||^2
        tensor.prod_async(self.val, self.val)
        tensor.axpy_async(-0.5, self.val, self.val)

    # Get value only
    def value_async(self):
        # Value requires gradient in any case
        self.value_grad_async()
        # Gradient is unnecessary to store
        self.dx.invalidate_submit()

    # Get gradient only
    def grad_async(self):
        # Put X into gradient dX
        tensor.copy_async(self.x, self.dx)
        # Define gradient dX as X-Y
        tensor.axpy_async(-1, self.y, self.dx)
        # Values X and Y are not needed anymore
        self.x.invalidate_submit()
        self.y.invalidate_submit()

