# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/optimizer/adam.py
# Implementation of Adam with amsgrad option within nntile package
#
# @version 1.0.0
# @author Aleksandr Katrutsa
# @date 2023-02-17

import nntile
import numpy as np
from nntile.tensor import TensorTraits

class Adam:
    def __init__(self, params, lr, next_tag,
                 beta1=0.9, beta2=0.999, amsgrad=False,
                 weight_decay=0., eps=1e-10, dtype=np.float32):
        self.params = params
        self.next_tag = next_tag
        self.amsgrad = amsgrad
        self.num_iter = 1
        self.dtype=dtype

        self.first_moments = []
        self.second_moments = []
        self.max_second_moments = []
        self.denoms = []
        for p in self.params:
            p_traits = TensorTraits(p.value.shape, p.value.basetile_shape)
            self.first_moments.append(type(p.value)(p_traits, p.value.distribution, self.next_tag))
            self.next_tag = self.first_moments[-1].next_tag
            self.second_moments.append(type(p.value)(p_traits, p.value.distribution, self.next_tag))
            self.next_tag = self.second_moments[-1].next_tag
            self.denoms.append(type(p.value)(p_traits, p.value.distribution, self.next_tag))
            self.next_tag = self.denoms[-1].next_tag
            if self.amsgrad:
                self.max_second_moments.append(type(p.value)(p_traits, p.value.distribution, self.next_tag))
                self.next_tag = self.max_second_moments[-1].next_tag
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.eps = eps

    def get_next_tag(self):
        return self.next_tag
    
    def unregister(self):
        for i in range(len(self.first_moments)):
            self.first_moments[i].unregister()
            self.second_moments[i].unregister()
            self.denoms[i].unregister()
            if self.amsgrad:
                self.max_second_moments[i].unregister()

    def step(self):
        for i, p in enumerate(self.params):
            if self.weight_decay != 0.:
                nntile.tensor.axpy_async(self.weight_decay, p.value, p.grad)
            

            # Update first moments
            if self.num_iter == 1:
                nntile.tensor.copy_async(p.grad, self.first_moments[i])
                nntile.tensor.axpy_async(-self.beta1, self.first_moments[i], self.first_moments[i])
            else:
                nntile.tensor.axpy_async(self.beta1 - 1, self.first_moments[i], self.first_moments[i])
                nntile.tensor.axpy_async(1 - self.beta1, p.grad, self.first_moments[i])

            # Update second moments
            nntile.tensor.prod_async(p.grad, p.grad)
            if self.num_iter == 1:
                nntile.tensor.copy_async(p.grad, self.second_moments[i])
                nntile.tensor.axpy_async(-self.beta2, self.second_moments[i], self.second_moments[i])
            else:
                nntile.tensor.axpy_async(self.beta2 - 1, self.second_moments[i], self.second_moments[i])
                nntile.tensor.axpy_async(1 - self.beta2, p.grad, self.second_moments[i])
            
            # Mult tensor by scalar
            if self.dtype == np.float32:
                step_size = np.float32(-self.lr / (1 - np.power(self.beta1, self.num_iter)))
            elif self.dtype == np.float64:
                step_size = np.float64(-self.lr / (1 - np.power(self.beta1, self.num_iter)))

            if self.amsgrad:
                nntile.tensor.maximum_async(self.second_moments[i], self.max_second_moments[i])
                nntile.tensor.copy_async(self.max_second_moments[i], self.denoms[i])
            else:
                nntile.tensor.copy_async(self.second_moments[i], self.denoms[i])
            
            nntile.tensor.sqrt_async(self.denoms[i])
            if self.dtype == np.float32:
                scale_factor = 1. / (1 - np.power(self.beta2, self.num_iter))
                scale_factor = np.sqrt(scale_factor, dtype=np.float32)
            elif self.dtype == np.float64:
                scale_factor = np.float64(1. / np.sqrt(1 - np.power(self.beta2, self.num_iter),dtype=np.float64))

            nntile.tensor.axpy_async(scale_factor, self.denoms[i], self.denoms[i])

            nntile.tensor.addcdiv_async(step_size, self.eps,
                                        self.first_moments[i], self.denoms[i], p.value)
        self.num_iter += 1