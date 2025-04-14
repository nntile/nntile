# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/add_slice.py
# Add slice layer of NNTile Python package
#
# @version 1.1.0

import nntile.utils.constructors as nntc
from nntile.tensor import (
    TensorMoments, TensorTraits, add_inplace_async, add_slice_async,
    sum_slice_async)

from .base_layer import BaseLayer


class AddSlice(BaseLayer):
    def __init__(
        self,
        x: TensorMoments,
        y: TensorMoments,
        u: TensorMoments,
        axis: int,
        redux: bool = False,
    ):
        super().__init__([x, y], [u], [], [])
        # Set up local named parameters
        self.x = x
        self.y = y
        self.y.grad.set_reduction_add()
        self.u = u
        self.axis = axis
        if redux:
            self.redux = 1
        else:
            self.redux = 0

    # Forward propagation of the add_slice layer
    def forward_async(self):
        # Add slice operation
        add_slice_async(
            1.0, self.y.value, 1.0, self.x.value, self.u.value, self.axis
        )

    def forward_dynamic(self, x: TensorMoments, slice_tensor: TensorMoments):
        y = nntc.empty_like(x.value)
        add_slice_async(1.0, slice_tensor.value, 1.0, x.value, y, self.axis)
        return TensorMoments(y, None, False)

    def backward_async(self):
        add_inplace_async(1.0, self.u.grad, 1.0, self.x.grad)
        sum_slice_async(
            1.0, self.u.grad, 1.0, self.y.grad, self.axis, redux=self.redux
        )

    # Simple generator for the add_slice layer
    @staticmethod
    def generate_simple(
        x: TensorMoments,
        y: TensorMoments,
        axis: int,
        next_tag: int,
        redux: bool = False,
    ):
        # Get traits of X
        u_traits = TensorTraits(x.value.shape, x.value.basetile_shape)
        # Create Y with the same traits and distribution as X
        u_value = type(x.value)(u_traits, x.value.distribution, next_tag)
        next_tag = u_value.next_tag
        u_grad = type(x.value)(u_traits, x.value.distribution, next_tag)
        next_tag = u_grad.next_tag
        u = TensorMoments(u_value, u_grad, True)
        # Create activation layer with all the provided tensors
        layer = AddSlice(x, y, u, axis, redux)
        # Return layer and next tag to be used
        return (layer, next_tag)
