# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/base_model.py
# Base model API of NNTile Python package
#
# @version 1.1.0

from typing import List

from nntile.layer.base_layer import BaseLayer
from nntile.tensor import TensorMoments, clear_async


class BaseModel:
    activations: List[TensorMoments]
    parameters: List[TensorMoments]
    layers: List[BaseLayer]

    # Construct model with all the provided data
    def __init__(
        self, activations: List[TensorMoments], layers: List[BaseLayer]
    ):
        self.activations = activations
        self.layers = layers
        self.parameters = []
        for layer in layers:
            self.parameters.extend(layer.parameters)

    # Add a new layer with corresponding new activations
    def append(self, layer: BaseLayer):
        self.activations.append(layer.activations_output)
        self.layers.append(layer)
        self.parameters.append(layer.parameters)

    # Forward propagation
    def forward_async(self):
        for layer in self.layers:
            layer.forward_async()

    # Forward propagation with dynamic shapes
    def forward_dynamic(self, x: TensorMoments):
        out = x
        for layer in self.layers:
            out = layer.forward_dynamic(out)
        return out

    # Backward propagation
    def backward_async(self):
        for layer in reversed(self.layers):
            layer.backward_async()

    # Clear all gradients (parameters and inter-layer activations)
    def clear_gradients(self):
        self.clear_parameters_grads()
        self.clear_activations_grads()

    # Clear gradients of parameters
    def clear_parameters_grads(self):
        for tensor in self.parameters:
            if tensor.grad is not None and tensor.grad_required:
                clear_async(tensor.grad)

    # Clear gradients of inter-layer activations
    def clear_activations_grads(self):
        for tensor in self.activations:
            if tensor.grad is not None and tensor.grad_required:
                clear_async(tensor.grad)

    # Unregister all tensors related to this model
    def unregister(self):
        for layer in self.layers:
            layer.unregister()
        for x in self.activations:
            x.unregister()

    def get_parameters(self):
        return self.parameters

    def get_flops_forward(self):
        flops = 0
        for layer in self.layers:
            flops += layer.get_forward_flops()
        return flops

    def get_flops_backward(self):
        flops = 0
        for layer in self.layers:
            flops += layer.get_backward_flops()
        return flops
