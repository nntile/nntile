# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/base_layer.py
# Base layer API of NNTile Python package
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-02-12

from nntile.tensor import TensorMoments
import numpy as np
from typing import List

class BaseLayer(object):
    # Input activations with moments
    activations_input: List[TensorMoments]
    # Output activations with moments
    activations_output: List[TensorMoments]
    # Layer parameters with moments
    parameters: List[TensorMoments]

    def __init__(self, activations_input: List[TensorMoments],
            activations_output: List[TensorMoments],
            parameters: List[TensorMoments]):
        self.activations_input = activations_input
        self.activations_output = activations_output
        self.parameters = parameters

    def forward_async(self):
        raise NotImplementedError

    def forward(self):
        self.forward_async()
        starpu.wait_for_all()

    def backward_async(self):
        raise NotImplementedError

    def backward(self):
        self.forward_async()
        starpu.wait_for_all()

