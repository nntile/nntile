# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/model.py
# Base model API of NNTile Python package
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-02-09

import nntile.nntile_core.tensor as tensor
import numpy as np
from typing import List

class Model_fp32:
    x: tensor.Tensor_fp32
    activation_x: List[tensor.Tensor_fp32]
    activation_dx: List[tensor.Tensor_fp32]
    layers: List
    params: List[tensor.Tensor_fp32]
    grads: List[tensor.Tensor_fp32]

    # Construct model with all the provided data
    def __init__(self, x, activation_x, activation_dx, layers):
        self.x = x
        self.activation_x = activation_x
        self.activation_dx = activation_dx
        self.layers = layers
        self.params = []
        self.grads = []
        for l in layers:
            self.params.extend(l.params)
            self.grads.extend(l.grads)

    # Add a new layer with corresponding new activations
    def append(self, activation_y, activation_dy, layer):
        self.activation_x.append(self.activation_y)
        self.activation_dx.append(self.activation_dy)
        self.layers.append(layer)
        self.params.append(layer.params)
        self.grads.append(layer.grads)

    # Forward propagation
    def forward_async(self):
        for l in self.layers:
            l.forward_async()

    # Backward propagation
    def backward_async(self):
        for l in self.layers:
            l.backward_async()

