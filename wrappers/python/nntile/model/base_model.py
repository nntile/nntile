# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/base_model.py
# Base model API of NNTile Python package
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-05-11

from nntile.tensor import TensorTraits, Tensor, TensorOrNone, TensorMoments, \
        clear_async
from nntile.layer.base_layer import BaseLayer
import numpy as np
from typing import List

class BaseModel:
    activations: List[TensorMoments]
    parameters: List[TensorMoments]
    layers: List[BaseLayer]

    # Construct model with all the provided data
    def __init__(self, activations: List[TensorMoments],
            layers: List[BaseLayer]):
        self.activations = activations
        self.layers = layers
        self.parameters = []
        for l in layers:
            self.parameters.extend(l.parameters)

    # Add a new layer with corresponding new activations
    def append(self, layer: BaseLayer):
        self.activations.append(layer.activations_output)
        self.layers.append(layer)
        self.parameters.append(layer.parameters)

    # Forward propagation
    def forward_async(self):
        for l in self.layers:
            l.forward_async()

    # Backward propagation
    def backward_async(self):
        for l in reversed(self.layers):
            l.backward_async()

    # Clear gradients of activations and parameters
    def clear_gradients(self):
        for t in self.activations:
            if t.grad is not None and t.grad_required:
                clear_async(t.grad)
        for t in self.parameters:
            if t.grad is not None and t.grad_required:
                clear_async(t.grad)

    # Unregister all tensors related to this model
    def unregister(self):
        for l in self.layers:
            l.unregister()
        for x in self.activations:
            x.unregister()

    def get_parameters(self):
        return self.parameters

