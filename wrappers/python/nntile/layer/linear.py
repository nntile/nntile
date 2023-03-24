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
# @date 2023-02-15

from nntile.tensor import TensorTraits, Tensor, TensorOrNone, TensorMoments, \
        TransOp, trans, notrans, copy_async, gemm_async, randn_async
from nntile.layer.base_layer import BaseLayer
import numpy as np
from typing import List

class Linear(BaseLayer):
    side: str
    trans_x: TransOp
    x: TensorMoments
    y: TensorMoments
    w: TensorMoments
    ndim: int
    #b: TensorMoments
    #b_axis: int
    x_copy: TensorOrNone

    # Construct linear layer with all the provided data
    def __init__(self, side: str, trans_x: TransOp, x: TensorMoments,
            y: TensorMoments, w: TensorMoments, ndim: int,
            #b: TensorMoments, b_axis: int, # No bias as of now
            x_copy: TensorOrNone):
        # Check parameter side
        if side != 'L' and side != 'R':
            raise ValueError("side must be either 'L' or 'R'")
        # Check parameter ndim
        if ndim <= 0:
            raise ValueError("ndim must be positive integer")
        # Redirect to BaseClass initialization
        super().__init__([x], [y], [w])
        # Set up local named parameters
        self.side = side
        self.trans_x = trans_x
        self.ndim = ndim
        self.x = x
        self.y = y
        self.w = w
        #self.b = b
        #self.b_axis = b_axis
        self.x_copy = x_copy

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
        #b = TensorMoments(None, None, False)
        # Copy of input for backward
        x_traits = TensorTraits(x.value.shape, x.value.basetile_shape)
        x_copy = type(x.value)(x_traits, x.value.distribution, next_tag)
        next_tag = x_copy.next_tag
        # Create linear layer with all the provided data
        layer = Linear(side, trans_x, x, y, w, ndim, x_copy)
        # Return layer and next tag to be used
        return (layer, next_tag)

    # Random initialization of weights
    def init_randn_async(self):
        seed = 100
        randn_async(self.w.value, [0]*len(self.w.value.shape),
                self.w.value.shape, seed, 0.0, 1.)

    # Forward propagation of the linear layer
    def forward_async(self):
        # Copy input X for backward if needed
        if self.w.grad_required:
            copy_async(self.x.value, self.x_copy)
            # Hint for StarPU that X_copy tensor will
            # not be used soon and it is advised to offload data from GPU
            self.x_copy.wont_use()
        # Perform actual gemm
        if self.side == 'L':
            # y = op(x) * w
            gemm_async(1.0, self.trans_x, self.x.value, notrans, self.w.value,
                    0.0, self.y.value, self.ndim)
        else:
            # y = w * op(x)
            gemm_async(1.0, notrans, self.w.value, self.trans_x, self.x.value,
                    0.0, self.y.value, self.ndim)
        # Hint for StarPU that W tensor will
        # not be used soon and it is advised to offload data from GPU
        self.w.value.wont_use()

    # Backward propagation of the linear layer
    def backward_async(self):
        # Gradient over W (weights)
        if self.w.grad_required:
            gemm_ndim = len(self.x.value.shape) - self.ndim
            if self.side == 'L':
                if self.trans_x == notrans:
                    gemm_async(1.0, trans, self.x_copy, notrans, self.y.grad,
                            0.0, self.w.grad, gemm_ndim)
                else:
                    gemm_async(1.0, notrans, self.x_copy, notrans, self.y.grad,
                            0.0, self.w.grad, gemm_ndim)
            else:
                if self.trans_x == notrans:
                    gemm_async(1.0, notrans, self.y.grad, trans, self.x_copy,
                            0.0, self.w.grad, gemm_ndim)
                else:
                    gemm_async(1.0, notrans, self.y.grad, notrans, self.x_copy,
                            0.0, self.w.grad, gemm_ndim)
            # Hint StarPU to delete x_copy buffer
            self.x_copy.invalidate_submit()
            # Hint StarPU to offload gradient over W if needed
            self.w.grad.wont_use()
        # Gradient over X (input)
        if self.x.grad_required:
            if self.side == 'L':
                gemm_ndim = len(self.w.value.shape) - self.ndim
                if self.trans_x == notrans:
                    gemm_async(1.0, notrans, self.y.grad, trans, self.w.value,
                            0.0, self.x.grad, gemm_ndim)
                else:
                    gemm_async(1.0, notrans, self.w.value, trans, self.y.grad,
                            0.0, self.x.grad, gemm_ndim)
            else:
                if self.trans_x == notrans:
                    gemm_async(1.0, trans, self.w.value, notrans, self.y.grad,
                            0.0, self.x.grad, gemm_ndim)
                else:
                    gemm_async(1.0, trans, self.y.grad, notrans, self.w.value,
                            0.0, self.x.grad, gemm_ndim)
            # Hint StarPU to offload certain buffers
            self.w.value.wont_use()

    # Unregister all internal tensors
    def unregister(self):
        self.w.unregister()
        if self.x_copy is not None:
            self.x_copy.unregister()

