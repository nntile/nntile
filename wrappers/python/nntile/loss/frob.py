# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/loss/frob.py
# Frobenius norm loss of NNTile Python package
#
# @version 1.1.0

from nntile.tensor import (
    Tensor, TensorMoments, TensorTraits, axpy_async, copy_async, nrm2_async,
    prod_inplace_async, scal_inplace_async)


class Frob:
    x: TensorMoments
    y: Tensor
    val: Tensor
    tmp: Tensor

    # Constructor of loss with all the provided data
    def __init__(
        self,
        x: TensorMoments,
        y: Tensor,
        val: Tensor,
        val_sqrt: Tensor,
        tmp: Tensor,
    ):
        self.x = x
        self.y = y
        self.val_sqrt = val_sqrt
        self.val = val
        self.tmp = tmp

    def unregister(self):
        self.y.unregister()
        self.val_sqrt.unregister()
        self.val.unregister()
        self.tmp.unregister()

    # Simple geenrator
    @staticmethod
    def generate_simple(x: TensorMoments, next_tag: int) -> tuple:
        ndim = len(x.value.grid.shape)
        x_traits = TensorTraits(x.value.shape, x.value.basetile_shape)
        y = type(x.value)(x_traits, x.value.distribution, next_tag)
        next_tag = y.next_tag
        val_traits = TensorTraits([], [])
        val = type(x.value)(val_traits, [0], next_tag)
        next_tag = val.next_tag
        val2 = type(x.value)(val_traits, [0], next_tag)
        next_tag = val2.next_tag
        tmp_traits = TensorTraits(x.value.grid.shape, [1] * ndim)
        tmp = type(x.value)(tmp_traits, x.value.distribution, next_tag)
        next_tag = tmp.next_tag
        loss = Frob(x, y, val, val2, tmp)
        return loss, next_tag

    # Get value and gradient if needed
    def calc_async(self):
        # Put X into gradient grad X
        copy_async(self.x.value, self.x.grad)
        # Define gradient dX as X-Y
        axpy_async(-1, self.y, self.x.grad)
        # Values Y are not needed anymore
        # self.y.invalidate_submit()
        # Get value ||grad X||
        nrm2_async(1.0, self.x.grad, 0.0, self.val_sqrt, self.tmp)
        # Ignore temporary values
        # self.tmp.invalidate_submit()
        # Invalidate gradient if it is unnecessary
        # if self.x.grad_required is False:
        #    self.x.grad.invalidate_submit()
        # Compute loss as 0.5*||dX||^2
        copy_async(self.val_sqrt, self.val)
        prod_inplace_async(self.val_sqrt, self.val)
        scal_inplace_async(0.5, self.val)
