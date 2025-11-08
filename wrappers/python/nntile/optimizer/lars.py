# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                 2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/optimizer/lars.py
# Implementation of LARS optimizer within nntile package
#
# @version 1.1.0

import numpy as np
import torch

import nntile


class Lars:
    def __init__(
        self,
        params,
        lr,
        trust_ratio=0.02,
        weight_decay=0.0,
        dtype=np.float32,
    ):
        self.params = params
        self.lr = lr
        self.trust_ratio = trust_ratio
        self.weight_decay = weight_decay
        self.dtype = dtype

    def unregister(self):
        pass  # No additional tensors to unregister for LARS

    def step(self):
        """
        Perform LARS step.
        """
        for p in self.params:
            # Create scalar tensors for norms
            grad_norm_traits = nntile.tensor.TensorTraits([], [])
            p_norm_traits = nntile.tensor.TensorTraits([], [])
            norm_distr = [0] * grad_norm_traits.grid.nelems
            grad_norm = nntile.tensor.Tensor_fp32(grad_norm_traits, norm_distr)
            p_norm = nntile.tensor.Tensor_fp32(p_norm_traits, norm_distr)
            
            nntile.functions.fused_lars_step(
                p.value,
                p.grad,
                self.lr,
                self.trust_ratio,
                grad_norm,
                p_norm,
                self.weight_decay,
            )
            p.value.wont_use()
            # dP can be deleted
            # p.grad.wont_use()
            p.grad.invalidate_submit()
            
            # Unregister norm tensors
            grad_norm.unregister()
            p_norm.unregister()

    def save_state(self, path):
        stored_data = {
            "lr": self.lr,
            "trust_ratio": self.trust_ratio,
            "weight_decay": self.weight_decay,
        }
        with open(path, "wb") as fp:
            torch.save(stored_data, fp)

    def load_state(self, path):
        stored_states = torch.load(path)
        self.lr = stored_states["lr"]
        self.trust_ratio = stored_states["trust_ratio"]
        self.weight_decay = stored_states["weight_decay"]

    def get_nbytes(self):
        return 0  # LARS doesn't store additional state tensors
