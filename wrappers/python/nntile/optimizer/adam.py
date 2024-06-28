# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/optimizer/adam.py
# Implementation of Adam with amsgrad option within nntile package
#
# @version 1.0.0

import nntile
import numpy as np
from nntile.tensor import TensorTraits
import pickle
import torch

class Adam:
    def __init__(self, params, lr, next_tag, beta1=0.9, beta2=0.999, \
            amsgrad=False, weight_decay=0., eps=1e-8, dtype=np.float32):
        self.params = params
        self.next_tag = next_tag
        self.amsgrad = amsgrad
        self.num_iter = 1
        self.dtype=dtype
        self.first_moments = []
        self.second_moments = []
        self.max_second_moments = []
        for p in self.params:
            p_traits = TensorTraits(p.value.shape, p.value.basetile_shape)
            self.first_moments.append(type(p.value)(p_traits, \
                    p.value.distribution, self.next_tag))
            self.next_tag = self.first_moments[-1].next_tag
            self.second_moments.append(type(p.value)(p_traits, \
                    p.value.distribution, self.next_tag))
            self.next_tag = self.second_moments[-1].next_tag
            if self.amsgrad:
                self.max_second_moments.append(type(p.value)(p_traits, \
                        p.value.distribution, self.next_tag))
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
            if self.amsgrad:
                self.max_second_moments[i].unregister()

    def step(self):
        for i, p in enumerate(self.params):
            if self.weight_decay != 0.:
                nntile.tensor.axpy_async(self.weight_decay, p.value, p.grad)
            # Update first moments
            if self.num_iter == 1:
                nntile.tensor.clear_async(self.first_moments[i])
                nntile.tensor.add_async(1-self.beta1, p.grad, 0.0, \
                        self.first_moments[i])
            else:
                nntile.tensor.add_async(1-self.beta1, p.grad, self.beta1, \
                        self.first_moments[i])
            # Update second moments
            if self.num_iter == 1:
                nntile.tensor.clear_async(self.second_moments[i])
                if self.amsgrad:
                    nntile.tensor.clear_async(self.max_second_moments[i])
            nntile.tensor.hypot_async(np.sqrt(1-self.beta2), p.grad, \
                    np.sqrt(self.beta2), self.second_moments[i])
            # Mult tensor by scalar
            step_size = -self.lr / (1 - np.power(self.beta1, \
                    self.num_iter))
            scale_factor = 1 - np.power(self.beta2, self.num_iter)
            scale_factor = np.sqrt(scale_factor)
            if self.amsgrad:
                nntile.tensor.maximum_async(self.second_moments[i], \
                        self.max_second_moments[i])
                nntile.tensor.addcdiv_async(step_size/scale_factor, \
                        self.eps*scale_factor, self.first_moments[i], \
                        self.max_second_moments[i], p.value)
                self.max_second_moments[i].wont_use()
            else:
                nntile.tensor.addcdiv_async(step_size*scale_factor, \
                        self.eps*scale_factor, self.first_moments[i], \
                        self.second_moments[i], p.value)
            p.value.wont_use()
            p.grad.wont_use()
            self.first_moments[i].wont_use()
            self.second_moments[i].wont_use()
        self.num_iter += 1

    def save_state(self, path):
        first_moments = []
        second_moments = []
        max_second_moments = []
        for i in range(len(self.first_moments)):
            f_m = np.array(np.zeros(self.first_moments[i].shape, dtype=self.dtype), order="F")
            self.first_moments[i].to_array(f_m)
            first_moments.append(f_m.copy())

            s_m = np.array(np.zeros(self.second_moments[i].shape, dtype=self.dtype), order="F")
            self.second_moments[i].to_array(s_m)
            second_moments.append(s_m.copy())

        for m_s_m_nntile in self.max_second_moments:
            m_s_m = np.array(np.zeros(self.max_second_moments[i].shape, dtype=self.dtype), order="F")
            self.max_second_moments[i].to_array(m_s_m)
            max_second_moments.append(m_s_m.copy())

        stored_data = {
            "first_moments": first_moments,
            "second_moments": second_moments,
            "max_second_moments": max_second_moments,
            "num_iter": self.num_iter,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "lr": self.lr,
            "eps": self.eps,
            "weight_decay": self.weight_decay
        }
        with open(path, 'wb') as fp:
            pickle.dump(stored_data, fp)

    def load_state(self, path):
        with open(path, 'rb') as fp:
            stored_states = pickle.load(fp)

        self.lr = stored_states["lr"]
        self.beta1 = stored_states["beta1"]
        self.beta2 = stored_states["beta2"]
        self.eps = stored_states["eps"]
        self.num_iter = stored_states["num_iter"]
        self.weight_decay = stored_states["weight_decay"]

        first_moments = stored_states["first_moments"]
        second_moments = stored_states["second_moments"]
        max_second_moments = stored_states["max_second_moments"]
        for i in range(len(first_moments)):
            self.first_moments[i].from_array(first_moments[i])
            self.second_moments[i].from_array(second_moments[i])

        for i in range(len(max_second_moments)):
            self.max_second_moments[i].from_array(max_second_moments[i])



class FusedAdam:
    def __init__(self, params, lr, next_tag, beta1=0.9, beta2=0.999, \
            weight_decay=0., eps=1e-8, dtype=np.float32, start_lr=None, \
            full_lr_iter=None):
        self.params = params
        self.next_tag = next_tag
        self.num_iter = 1
        self.dtype=dtype
        self.first_moments = []
        self.second_moments = []
        for p in self.params:
            p_traits = TensorTraits(p.value.shape, p.value.basetile_shape)
            self.first_moments.append(type(p.value)(p_traits, \
                    p.value.distribution, self.next_tag))
            self.next_tag = self.first_moments[-1].next_tag
            self.second_moments.append(type(p.value)(p_traits, \
                    p.value.distribution, self.next_tag))
            self.next_tag = self.second_moments[-1].next_tag
        self.lr = lr
        self.start_lr = start_lr
        self.full_lr_iter = full_lr_iter
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

    def step(self):
        cur_lr = self.lr
        if self.start_lr is not None and self.full_lr_iter is not None:
            if self.num_iter < self.full_lr_iter and self.full_lr_iter > 1:
                cur_lr = (self.lr-self.start_lr) / (self.full_lr_iter-1)
                cur_lr = cur_lr*(self.num_iter-1) + self.start_lr
        for i, p in enumerate(self.params):
            nntile.tensor.fused_adam_step(p.value, p.grad, \
                    self.first_moments[i], self.second_moments[i], cur_lr, \
                    self.eps, self.beta1, self.beta2, self.weight_decay, \
                    self.num_iter)
            p.value.wont_use()
            # dP can be deleted
            #p.grad.wont_use()
            p.grad.invalidate_submit()
            self.first_moments[i].wont_use()
            self.second_moments[i].wont_use()
        self.num_iter += 1

    def save_state(self, path, dtype="fp32"):
        first_moments = []
        second_moments = []
        for i in range(len(self.first_moments)):
            f_m = np.array(np.zeros(self.first_moments[i].shape, dtype=self.dtype), order="F")
            self.first_moments[i].to_array(f_m)
            s_m = np.array(np.zeros(self.second_moments[i].shape, dtype=self.dtype), order="F")
            self.second_moments[i].to_array(s_m)
            if dtype == "fp32":
                first_moments.append(f_m.copy())
                second_moments.append(s_m.copy())
            elif dtype == "fp16":
                first_moments.append(torch.tensor(f_m, dtype=torch.float16))
                second_moments.append(torch.tensor(s_m, dtype=torch.float16))
            elif dtype == "bf16":
                first_moments.append(torch.tensor(f_m, dtype=torch.bfloat16))
                second_moments.append(torch.tensor(s_m, dtype=torch.bfloat16))

        stored_data = {
            "first_moments": first_moments,
            "second_moments": second_moments,
            "num_iter": self.num_iter,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "lr": self.lr,
            "start_lr": self.start_lr,
            "full_lr_iter": self.full_lr_iter,
            "eps": self.eps,
            "weight_decay": self.weight_decay
        }
        with open(path, 'wb') as fp:
            pickle.dump(stored_data, fp)

    def load_state(self, path):
        with open(path, 'rb') as fp:
            stored_states = pickle.load(fp)

        self.lr = stored_states["lr"]
        self.start_lr = stored_states["start_lr"]
        self.full_lr_iter = stored_states["full_lr_iter"]
        self.beta1 = stored_states["beta1"]
        self.beta2 = stored_states["beta2"]
        self.eps = stored_states["eps"]
        self.num_iter = stored_states["num_iter"]
        self.weight_decay = stored_states["weight_decay"]

        first_moments = stored_states["first_moments"]
        second_moments = stored_states["second_moments"]
        for i in range(len(first_moments)):
            self.first_moments[i].from_array(first_moments[i].to(torch.float32))
            self.second_moments[i].from_array(second_moments[i].to(torch.float32))
