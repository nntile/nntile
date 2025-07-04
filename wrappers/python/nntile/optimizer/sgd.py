# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/optimizer/sgd.py
# Implementation of SGD with momentum option within nntile package
#
# @version 1.1.0

import numpy as np

import nntile
from nntile.tensor import TensorTraits


class SGD:
    def __init__(
        self,
        params,
        lr,
        momentum=0.0,
        nesterov=False,
        weight_decay=0.0,
        damping=0.0,
        dtype=np.float32,
    ):
        self.params = params
        self.nesterov = nesterov
        self.num_iter = 0
        self.dtype = dtype
        if dtype == np.float32:
            self.lr = np.float32(lr)
            self.momentum = np.float32(momentum)
            self.weight_decay = np.float32(weight_decay)
            self.damping = np.float32(damping)
        elif dtype == np.float64:
            self.lr = np.float64(lr)
            self.momentum = np.float64(momentum)
            self.weight_decay = np.float64(weight_decay)
            self.damping = np.float64(damping)
        self.states = []
        if momentum > 0:
            for p in self.params:
                p_traits = TensorTraits(p.value.shape, p.value.basetile_shape)
                self.states.append(
                    type(p.value)(
                        p_traits, p.value.distribution
                    )
                )

    def unregister(self):
        if self.momentum > 0:
            for s in self.states:
                s.unregister()

    def step(self):
        for i, p in enumerate(self.params):
            if self.weight_decay != 0.0:
                nntile.tensor.add_inplace_async(
                    self.weight_decay, p.value, 1.0, p.grad
                )

            if self.momentum > 0:
                if self.num_iter == 0:
                    nntile.tensor.copy_async(p.grad, self.states[i])
                else:
                    nntile.tensor.add_inplace_async(
                        1 - self.damping, p.grad, self.momentum, self.states[i]
                    )
                if self.nesterov:
                    nntile.tensor.add_inplace_async(
                        self.momentum, self.states[i], 1.0, p.grad
                    )
                else:
                    nntile.tensor.copy_async(self.states[i], p.grad)
            nntile.tensor.add_inplace_async(-self.lr, p.grad, 1.0, p.value)
        self.num_iter += 1

    def get_nbytes(self):
        nbytes = 0
        for state in self.states:
            nbytes += state.get_nbytes()
        return nbytes

    def force_offload_disk(self, portion: float = 0.0):
        """Choose the first `portion` of parameters, whose optimizer
        states are forced to be offloaded to disk in advance"""
        enable_count = int(len(self.states) * portion)
        disabled_count = len(self.states) - enable_count
        for i in range(enable_count):
            self.states[i].force_offload_disk_enable()
        for i in range(disabled_count):
            self.states[enable_count + i].force_offload_disk_disable()

    def force_offload_ram(self, portion: float = 0.0):
        """Choose the first `portion` of parameters, whose optimizer
        states are forced to be offloaded to RAM in advance"""
        enable_count = int(len(self.states) * portion)
        disabled_count = len(self.states) - enable_count
        for i in range(enable_count):
            self.states[i].force_offload_ram_enable()
        for i in range(disabled_count):
            self.states[enable_count + i].force_offload_ram_disable()
