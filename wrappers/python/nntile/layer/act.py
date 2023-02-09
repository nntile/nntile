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
# @date 2023-02-09

import nntile.tensor as tensor
import numpy as np
from typing import List, Callable

class Act:
    x: tensor.Tensor
    dx: tensor.TensorOrNone
    y: tensor.Tensor
    dy: tensor.Tensor
    activations = {'relu': (tensor.relu_async, tensor.drelu_async),
            }
    func: Callable[[tensor.Tensor], None]
    dfunc: Callable[[tensor.Tensor], None]
    params: List[tensor.Tensor]
    grads: List[tensor.TensorOrNone]

    # Construct activation layer with all the provided data
    def __init__(self, x: tensor.Tensor, dx: tensor.TensorOrNone,
            y: tensor.Tensor, dy: tensor.Tensor, funcname: str):
        self.x = x
        self.dx = dx
        self.y = y
        self.dy = dy
        if funcname not in Act.activations:
            raise ValueError
        self.func, self.dfunc = Act.activations[funcname]
        self.params = []
        self.grads = []

    # Simple generator for the normalization layer
    @staticmethod
    def generate_block_cyclic(x: tensor.Tensor, dx, funcname, next_tag):
        # Check if X and dX correspond to each other
        if x.shape != dx.shape or x.basetile_shape != dx.basetile_shape:
            raise ValueError
        # Get traits of X
        x_traits = tensor.TensorTraits(x.shape, x.basetile_shape)
        # Create Y with the same traits and distribution as X
        y = type(x)(x_traits, x.distribution, next_tag)
        next_tag = y.next_tag
        # Create dY with the same traits and distribution as X
        dy = type(x)(x_traits, x.distribution, next_tag)
        next_tag = dy.next_tag
        # Create activation layer with all the provided tensors
        layer = Act(x, dx, y, dy, funcname)
        # Return layer and next tag to be used
        return (layer, next_tag)

    # Forward propagation of the activation layer
    def forward_async(self):
        # Init Y as a copy of X
        tensor.copy_async(self.x, self.y)
        # Non-linear activation of Y inplace
        self.func(self.y)
        # If dX is actually provided
        if self.dx is not None:
            # Copy X into dX to utilize it during backward propagation
            tensor.copy_async(self.x, self.dx)
            # Hint for StarPU that dX tensor will
            # not be used soon and it is advised to offload data from GPU 
            self.dx.wont_use()
        # Destroy values stored in tensor X
        self.x.invalidate_submit()

    # Backward propagation of the activation layer
    def backward_async(self):
        # Do nothing if dX is None
        if self.dx is None:
            return
        # Get derivative of activation functions at X inplace of dX
        self.dfunc(self.dx)
        # Per-element product of dY and f'(x)
        tensor.prod_async(self.dy, self.dx)
        # Destroy values stored in tensor dY
        self.dy.invalidate_submit()

