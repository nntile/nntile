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
# @date 2023-02-13

from nntile.tensor import TensorTraits, Tensor, TensorMoments, TransOp, \
        trans, notrans, copy_async, gemm_async
from nntile.layer.base_layer import BaseLayer
import numpy as np
from typing import List

class Linear(BaseLayer):
    side: str
    trans_x: TransOp
    ndim: int
    x: TensorMoments
    y: TensorMoments
    w: TensorMoments
    b: TensorMoments

    # Construct linear layer with all the provided data
    def __init__(self, side: str, trans_x: TransOp, x: TensorMoments,
            y: TensorMoments, w: TensorMoments, b: TensorMoments, ndim: int):
        # Bias is not yet supported
        if b.value is not None:
            raise NotImplementedError
        # Check parameter side
        if side != 'L' and side != 'R':
            raise ValueError("side must be either 'L' or 'R'")
        # Check parameter ndim
        if ndim <= 0:
            raise ValueError("ndim must be positive integer")
        # Redirect to BaseClass initialization
        super().__init__([x], [y], [w, b])
        # Set up local named parameters
        self.side = side
        self.trans_x = trans_x
        self.ndim = ndim
        self.x = x
        self.y = y
        self.w = w
        self.b = b

    # Simple generator for the linear layer
    @staticmethod
    def generate_simple_mpiroot(x: TensorMoments, side: str, trans_x: TransOp,
            ndim: int, add_shape: List[int], add_basetile_shape: List[int],
            next_tag: int):
        # Define shapes
        if side == 'L':
            if trans_x == notrans:
                w_shape = x.value.shape[-ndim:] + add_shape
                w_tile = x.value.basetile_shape[-ndim:] + add_basetile_shape
                y_shape = x.value.shape[:-ndim] + add_shape
                y_tile = x.value.basetile_shape[:-ndim] + add_basetile_shape
            else:
                w_shape = x.value.shape[:ndim] + add_shape
                w_tile = x.value.basetile_shape[:ndim] + add_basetile_shape
                y_shape = x.value.shape[ndim:] + add_shape
                y_tile = x.value.basetile_shape[ndim:] + add_basetile_shape
        else:
            if trans_x == notrans:
                w_shape = add_shape + x.value.shape[:ndim]
                w_tile = add_basetile_shape + x.value.basetile_shape[:ndim]
                y_shape = add_shape + x.value.shape[ndim:]
                y_tile = add_basetile_shape + x.value.basetile_shape[ndim:]
            else:
                w_shape = add_shape + x.value.shape[-ndim:]
                w_tile = add_basetile_shape + x.value.basetile_shape[-ndim:]
                y_shape = add_shape + x.value.shape[:-ndim]
                y_tile = add_basetile_shape + x.value.basetile_shape[:-ndim]
        # Define W
        w_traits = TensorTraits(w_shape, w_tile)
        # TODO change distribution
        w_distr = [0] * w_traits.grid.nelems
        w_value = type(x.value)(w_traits, w_distr, next_tag)
        next_tag = w_value.next_tag
        # Create gradient of W with the same traits and distribution as W
        w_grad = type(x.value)(w_traits, w_distr, next_tag)
        next_tag = w_grad.next_tag
        # Define W as TensorMoments
        w = TensorMoments(w_value, w_grad, True)
        # Define Y
        y_traits = TensorTraits(y_shape, y_tile)
        # TODO change distribution
        y_distr = [0] * y_traits.grid.nelems
        y_value = type(x.value)(y_traits, y_distr, next_tag)
        next_tag = y_value.next_tag
        # Create gradient of Y with the same traits and distribution as Y
        y_grad = type(x.value)(y_traits, y_distr, next_tag)
        next_tag = y_grad.next_tag
        # Define Y as TensorMoments
        y = TensorMoments(y_value, y_grad, True)
        # Bias is ignored for now
        b = TensorMoments(None, None, False)
        # Create linear layer with all the provided data
        layer = Linear(side, trans_x, x, y, w, b, ndim)
        # Return layer and next tag to be used
        return (layer, next_tag)

    # Forward propagation of the linear layer
    def forward_async(self):
        # Perform actual gemm
        if self.side == 'L':
            gemm_async(1.0, self.trans_x, self.x.value, notrans, self.w.value,
                    0.0, self.y.value, self.ndim)
        else:
            gemm_async(1.0, notrans, self.w.value, self.trans_x, self.x.value,
                    0.0, self.y.value, self.ndim)
        # Hint for StarPU that W tensor will
        # not be used soon and it is advised to offload data from GPU
        self.w.value.wont_use()

    # Backward propagation of the linear layer
    def backward_async(self):
        # Obtain 
        # Perform actual gemms
        if self.side == 'L':
            if self.trans_x == notrans:
                gemm_async(1.0, trans, self.x.grad, notrans, self.y.grad,
                        0.0, self.w.grad, len(self.x.value.shape)-self.ndim)
                gemm_async(1.0, notrans, self.y.grad, trans, self.w.value, 0.0,
                        self.x.grad, len(self.w.value.shape)-self.ndim)
            else:
                gemm_async(1.0, notrans, self.x.grad, notrans, self.y.grad,
                        0.0, self.w.grad, len(self.x.value.shape)-self.ndim)
                gemm_async(1.0, notrans, self.w.value, trans, self.y.grad, 0.0,
                        self.x.grad, len(self.w.value.shape)-self.ndim)
        else:
            if self.trans_x == notrans:
                gemm_async(1.0, notrans, self.y.grad, trans, self.x.grad,
                        0.0, self.w.grad, len(self.x.value.shape)-self.ndim)
                gemm_async(1.0, trans, self.w.value, notrans, self.y.grad, 0.0,
                        self.x.grad, len(self.w.value.shape)-self.ndim)
            else:
                gemm_async(1.0, notrans, self.y.grad, notrans, self.x.grad,
                        0.0, self.w.grad, len(self.x.value.shape)-self.ndim)
                gemm_async(1.0, trans, self.y.grad, notrans, self.w.value, 0.0,
                        self.x.grad, len(self.w.value.shape)-self.ndim)
        # Hint StarPU to offload certain buffers
        self.w.value.wont_use()
        self.w.grad.wont_use()
        # Hint StarPU to delete data from certain buffers
        self.y.grad.invalidate_submit()

