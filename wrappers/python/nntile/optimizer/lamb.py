# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/optimizer/lamb.py
# Implementation of LAMB optimizer within nntile package
#
# @version 1.1.0

import pickle

import numpy as np
import torch

import nntile
from nntile.tensor import TensorTraits


class Lamb:
    def __init__(
        self,
        params,
        lr,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,
        eps=1e-8,
        min_trust=0.0,
        max_trust=10.0,
        dtype=np.float32,
        start_lr=None,
        full_lr_iter=None,
    ):
        self.params = params
        self.num_iter = 1
        self.dtype = dtype
        self.first_moments = []
        self.second_moments = []
        for p in self.params:
            p_traits = TensorTraits(p.value.shape, p.value.basetile_shape)
            self.first_moments.append(
                type(p.value)(p_traits, p.value.distribution)
            )
            self.second_moments.append(
                type(p.value)(p_traits, p.value.distribution)
            )
        self.lr = lr
        self.start_lr = start_lr
        self.full_lr_iter = full_lr_iter
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.eps = eps
        self.min_trust = min_trust
        self.max_trust = max_trust

    def unregister(self):
        for p in self.params:
            p.value.unregister()
        for m in self.first_moments:
            m.unregister()
        for m in self.second_moments:
            m.unregister()

    def step(self):
        cur_lr = self.lr
        if self.start_lr is not None and self.full_lr_iter is not None:
            if self.num_iter < self.full_lr_iter and self.full_lr_iter > 1:
                cur_lr = (self.lr - self.start_lr) / (self.full_lr_iter - 1)
                cur_lr = cur_lr * (self.num_iter - 1) + self.start_lr
        for i, p in enumerate(self.params):
            nntile.tensor.fused_lamb_step(
                p.value,
                p.grad,
                self.first_moments[i],
                self.second_moments[i],
                cur_lr,
                self.eps,
                self.beta1,
                self.beta2,
                self.weight_decay,
                self.min_trust,
                self.max_trust,
                self.num_iter,
            )
        self.num_iter += 1

    def get_state(self):
        state = {
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "weight_decay": self.weight_decay,
            "eps": self.eps,
            "min_trust": self.min_trust,
            "max_trust": self.max_trust,
            "dtype": self.dtype,
            "num_iter": self.num_iter,
            "start_lr": self.start_lr,
            "full_lr_iter": self.full_lr_iter,
        }
        return state

    def set_state(self, state):
        self.lr = state["lr"]
        self.beta1 = state["beta1"]
        self.beta2 = state["beta2"]
        self.weight_decay = state["weight_decay"]
        self.eps = state["eps"]
        self.min_trust = state["min_trust"]
        self.max_trust = state["max_trust"]
        self.dtype = state["dtype"]
        self.num_iter = state["num_iter"]
        self.start_lr = state["start_lr"]
        self.full_lr_iter = state["full_lr_iter"]

    def __getstate__(self):
        state = self.get_state()
        return state

    def __setstate__(self, state):
        self.set_state(state)
