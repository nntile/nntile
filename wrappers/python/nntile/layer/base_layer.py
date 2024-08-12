# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/base_layer.py
# Base layer API of NNTile Python package
#
# @version 1.1.0

from typing import List, Union

import numpy as np

import nntile
from nntile.tensor import Tensor, TensorMoments, randn_async


class BaseLayer(object):
    # Input activations with moments
    activations_input: List[TensorMoments]
    # Output activations with moments
    activations_output: List[TensorMoments]
    # Layer parameters with moments
    parameters: List[TensorMoments]
    # Auxiliary tensors or tensors with moments
    temporaries: List[Union[Tensor, TensorMoments]]

    def __init__(self, activations_input: List[TensorMoments],
            activations_output: List[TensorMoments],
            parameters: List[TensorMoments],
            temporaries: List[Union[Tensor, TensorMoments]]):
        self.activations_input = activations_input
        self.activations_output = activations_output
        self.parameters = parameters
        self.temporaries = temporaries

    # Random initialization of parameters
    def init_randn_async(self):
        seed = 100
        for p in self.parameters:
            mean = 0.0
            stddev = 1.0 / np.sqrt(p.value.nelems)
            randn_async(p.value, [0] * len(p.value.shape), p.value.shape,
                    seed, mean, stddev)

    def forward_async(self):
        raise NotImplementedError

    def forward(self):
        self.forward_async()
        nntile.starpu.wait_for_all()

    def backward_async(self):
        raise NotImplementedError

    def backward(self):
        self.forward_async()
        nntile.starpu.wait_for_all()

    # Unregister layer weights and temporary tensors
    def unregister(self):
        for p in self.parameters:
            p.unregister()
        for t in self.temporaries:
            if t is not None:
                t.unregister()

    def get_forward_flops(self):
        return 0

    def get_backward_flops(self):
        return 0
