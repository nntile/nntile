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
# @version 1.0.0

import nntile
import numpy as np
from nntile.tensor import TensorTraits

class SGD:
    def __init__(self, params, lr, next_tag,
                 momentum=0., nesterov=False,
                 weight_decay=0., damping=0., dtype=np.float32):
        self.params = params
        self.nesterov = nesterov
        self.num_iter = 0
        self.dtype=dtype
        self.next_tag = next_tag
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
        if momentum > 0:
            self.states = []
            for p in self.params:
                p_traits = TensorTraits(p.value.shape, p.value.basetile_shape)
                self.states.append(type(p.value)(p_traits, p.value.distribution, self.next_tag))
                self.next_tag = self.states[-1].next_tag

    def get_next_tag(self):
        return self.next_tag
    
    def unregister(self):
        if self.momentum > 0:
            for s in self.states:
                s.unregister()
            
    def step(self):
        for i, p in enumerate(self.params):
            if self.weight_decay != 0.:
                nntile.tensor.axpy_async(self.weight_decay, p.value, p.grad)

            if self.momentum > 0:
                if self.num_iter == 0:
                    nntile.tensor.copy_async(p.grad, self.states[i])
                else:
                    nntile.tensor.scal_inplace_async(self.momentum, self.states[i])
                    nntile.tensor.axpy_async(1 - self.damping, p.grad, self.states[i])
                if self.nesterov:
                    nntile.tensor.axpy_async(self.momentum, self.states[i], p.grad)
                else:
                    nntile.tensor.copy_async(self.states[i], p.grad)
            nntile.tensor.axpy_async(-self.lr, p.grad, p.value)
        self.num_iter += 1
