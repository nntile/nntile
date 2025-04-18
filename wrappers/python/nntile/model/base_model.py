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

from typing import List, Union

from nntile.layer.base_layer import BaseLayer
from nntile.tensor import Tensor, TensorMoments, clear_async


class BaseModel:
    activations: List[TensorMoments]
    parameters: List[TensorMoments]
    layers: List[BaseLayer]
    temporaries: List[Union[Tensor, TensorMoments]]

    # Construct model with all the provided data
    def __init__(
        self, activations: List[TensorMoments], layers: List[BaseLayer],
        config=None
    ):
        self.config = config
        self.activations = activations
        self.layers = layers
        self.parameters = []
        for layer in layers:
            self.parameters.extend(layer.parameters)
        self.temporaries = []
        for layer in layers:
            self.temporaries.extend(layer.temporaries)

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

    def ooc_force_parameters(self, portion: float = 0.0):
        """Choose the first `portion` of parameters force them to be OOC
            in advance"""
        enable_count = int(len(self.parameters) * portion)
        disabled_count = len(self.parameters) - enable_count
        for i in range(enable_count):
            self.parameters[i].value.ooc_enable()
        for i in range(disabled_count):
            self.parameters[enable_count + i].value.ooc_disable()

    def ooc_force_gradients(self, portion: float = 0.0):
        """Choose the first `portion` of gradients force them to be OOC
            in advance"""
        enable_count = int(len(self.parameters) * portion)
        disabled_count = len(self.parameters) - enable_count
        for i in range(enable_count):
            if self.parameters[i].grad is not None:
                self.parameters[i].grad.ooc_enable()
        for i in range(disabled_count):
            if self.parameters[enable_count + i].grad is not None:
                self.parameters[enable_count + i].grad.ooc_disable()

    def ooc_force_activations(self, portion: float = 0.0):
        """Choose the first `portion` of activations force them to be OOC
            in advance"""
        enable_count = int(len(self.activations) * portion)
        disabled_count = len(self.activations) - enable_count
        for i in range(enable_count):
            self.activations[i].value.ooc_enable()
            if self.activations[i].grad is not None:
                self.activations[i].grad.ooc_enable()
        for i in range(disabled_count):
            self.activations[enable_count + i].value.ooc_disable()
            if self.activations[enable_count + i].grad is not None:
                self.activations[enable_count + i].grad.ooc_disable()

    def ooc_force_temporaries(self, portion: float = 0.0):
        """Choose the first `portion` of temporaries force them to be OOC
        in advance"""
        enable_count = int(len(self.temporaries) * portion)
        disabled_count = len(self.temporaries) - enable_count
        for i in range(enable_count):
            if isinstance(self.temporaries[i], TensorMoments):
                self.temporaries[i].value.ooc_enable()
                if self.temporaries[i].grad is not None:
                    self.temporaries[i].grad.ooc_enable()
            else:
                self.temporaries[i].ooc_enable()
        for i in range(disabled_count):
            if isinstance(self.temporaries[enable_count + i], TensorMoments):
                self.temporaries[enable_count + i].value.ooc_disable()
                if self.temporaries[enable_count + i].grad is not None:
                    self.temporaries[enable_count + i].grad.ooc_disable()
            else:
                self.temporaries[enable_count + i].ooc_disable()
