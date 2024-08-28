# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/act.py
# Activation layer of NNTile Python package
#
# @version 1.1.0

from typing import Callable

import nntile.utils.constructors as nntc
from nntile.layer.base_layer import BaseLayer
from nntile.tensor import (
    Tensor, TensorMoments, TensorTraits, copy_async, gelu_async,
    gelu_backward_async, gelutanh_async, gelutanh_backward_async,
    gelutanh_inplace_async, relu_backward_async, relu_forward_async,
    silu_backward_async, silu_forward_async)


class Act(BaseLayer):
    x: TensorMoments
    y: TensorMoments
    activations = {
        "relu": (relu_forward_async, relu_backward_async),
        "gelu": (gelu_async, gelu_backward_async),
        "gelutanh": (gelutanh_inplace_async, gelutanh_backward_async),
        "silu": (silu_forward_async, silu_backward_async),
    }
    funcname: str
    func: Callable[[Tensor], None]
    dfunc: Callable[[Tensor], None]

    # Construct activation layer with all the provided data
    def __init__(self, x: TensorMoments, y: TensorMoments, funcname: str):
        # Check if activation is actually implemented
        if funcname not in Act.activations:
            raise ValueError
        # Redirect to BaseLayer initialization
        super().__init__([x], [y], [], [])
        # Set up local named parameters
        self.x = x
        self.y = y
        self.funcname = funcname
        self.func, self.dfunc = Act.activations[funcname]

    # Simple generator for the normalization layer
    @staticmethod
    def generate_simple(x: TensorMoments, funcname: str, next_tag: int):
        # Get traits of X
        x_traits = TensorTraits(x.value.shape, x.value.basetile_shape)
        # Create Y with the same traits and distribution as X
        y_value = type(x.value)(x_traits, x.value.distribution, next_tag)
        next_tag = y_value.next_tag
        y_grad = type(x.value)(x_traits, x.value.distribution, next_tag)
        next_tag = y_grad.next_tag
        y = TensorMoments(y_value, y_grad, True)
        # Create activation layer with all the provided tensors
        layer = Act(x, y, funcname)
        # Return layer and next tag to be used
        return (layer, next_tag)

    # Forward propagation of the activation layer
    def forward_async(self):
        if self.funcname == "relu":
            relu_forward_async(self.x.value, self.y.value)
        if self.funcname == "silu":
            silu_forward_async(self.x.value, self.y.value)
        if self.funcname == "gelutanh":
            gelutanh_async(self.x.value, self.y.value)
        if self.funcname == "gelu":
            copy_async(self.x.value, self.y.value)
            gelu_async(self.y.value)
        self.x.value.wont_use()
        self.y.value.wont_use()

    def forward_dynamic(self, x: TensorMoments):
        y = nntc.zeros(
            x.value.shape,
            dtype=type(x.value),
            basetile_shape=x.value.basetile_shape,
        )
        if self.funcname == "relu":
            relu_forward_async(x.value, y)
        if self.funcname == "silu":
            silu_forward_async(x.value, y)
        if self.funcname == "gelutanh":
            gelutanh_async(x.value, y)
        if self.funcname == "gelu":
            copy_async(x.value, y)
            gelu_async(y)

        return TensorMoments(y, None, False)

    # Backward propagation of the activation layer
    def backward_async(self):
        # Gradient over X (input)
        if self.x.grad_required:
            self.dfunc(self.x.value, self.y.grad, self.x.grad)
            self.x.value.wont_use()
            self.x.grad.wont_use()
            self.y.grad.wont_use()
