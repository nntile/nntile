# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/act.py
# Activation layer of NNTile Python package
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-02-15

from nntile.tensor import TensorTraits, Tensor, TensorOrNone, TensorMoments, \
        copy_async, prod_async, relu_async, drelu_async
from nntile.layer.base_layer import BaseLayer
import numpy as np
from typing import List, Callable

class Act(BaseLayer):
    x: TensorMoments
    y: TensorMoments
    activations = {'relu': (relu_async, drelu_async),
            }
    func: Callable[[Tensor], None]
    dfunc: Callable[[Tensor], None]
    x_copy: TensorOrNone

    # Construct activation layer with all the provided data
    def __init__(self, x: TensorMoments, y: TensorMoments, funcname: str,
            x_copy: TensorOrNone):
        # Check if activation is actually implemented
        if funcname not in Act.activations:
            raise ValueError
        # Redirect to BaseLayer initialization
        super().__init__([x], [y], [])
        # Set up local named parameters
        self.x = x
        self.y = y
        self.func, self.dfunc = Act.activations[funcname]
        self.x_copy = x_copy

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
        # Copy of input for backward
        x_copy = type(x.value)(x_traits, x.value.distribution, next_tag)
        next_tag = x_copy.next_tag
        # Create activation layer with all the provided tensors
        layer = Act(x, y, funcname, x_copy)
        # Return layer and next tag to be used
        return (layer, next_tag)

    # Forward propagation of the activation layer
    def forward_async(self):
        # Copy input X for backward if needed
        if self.x.grad_required:
            copy_async(self.x.value, self.x_copy)
            # Hint for StarPU that X_copy tensor will
            # not be used soon and it is advised to offload data from GPU
            self.x_copy.wont_use()
        # Init Y as a copy of X
        copy_async(self.x.value, self.y.value)
        # Non-linear activation of Y inplace
        self.func(self.y.value)

    # Backward propagation of the activation layer
    def backward_async(self):
        # Gradient over X (input)
        if self.x.grad_required:
            # Copy X_copy into gradient of X
            copy_async(self.x_copy, self.x.grad)
            # Get derivative of activation functions at X inplace
            self.dfunc(self.x.grad)
            # Per-element product of gradient of Y and f'(X)
            prod_async(self.y.grad, self.x.grad)

    # Unregister all internal tensors
    def unregister(self):
        if self.x_copy is not None:
            self.x_copy.unregister()

