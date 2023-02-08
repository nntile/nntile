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
# @date 2023-02-08

import nntile.nntile_core.tensor as tensor
import numpy as np

class Frob_fp32:
    x: tensor.Tensor_fp32
    y: tensor.Tensor_fp32
    dx: tensor.Tensor_fp32
    tmp: tensor.Tensor_fp32
    val: tensor.Tensor_fp32
    val_tmp: tensor.Tensor_fp32

    # Constructor of loss with all the provided data
    def __init__(self, x, dx, y, val, tmp):
        self.x = x
        self.dx = dx
        self.y = y
        self.val = val
        self.tmp = tmp

    # Simple generator for the normalization layer
    @staticmethod
    def generate_block_cyclic(x, dx, y, val, next_tag):
        ndim = len(x.grid.shape)
        tmp_traits = tensor.TensorTraits(x.grid.shape, [1]*ndim)
        tmp = tensor.Tensor_fp32(tmp_traits, x.distribution, next_tag)
        next_tag = tmp.next_tag
        layer = Frob_fp32(x, dx, y, val, tmp)
        return layer, next_tag

    # Get only value
    def value(self):
        tensor.copy_async_fp32(self.x, self.dx)
        tensor.axpy2_async_fp32(-1, self.y, self.dx)
        tensor.nrm2_async_fp32(self.dx, self.val, self.tmp)
        tensor.prod_async_fp32(self.val, self.val)
        tensor.axpy2_async_fp32(-0.5, self.val, self.val)

    # Get both value and gradient
    def grad(self):
        self.value()

