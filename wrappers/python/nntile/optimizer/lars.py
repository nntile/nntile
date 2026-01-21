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
        self.params = list(params)
        self.trust_ratio = dtype(trust_ratio)
        self.weight_decay = dtype(weight_decay)
        self.lr = dtype(lr)
        self.dtype = dtype

        # Pre-allocate scalar tensors that hold norms for each parameter
        self._grad_norm_tensors = []
        self._p_norm_tensors = []
        for param in self.params:
            traits = nntile.tensor.TensorTraits([], [])
            distr = [0] * traits.grid.nelems
            norm_tensor_cls = self._norm_tensor_class(param.value)
            grad_norm = norm_tensor_cls(traits, distr)
            p_norm = norm_tensor_cls(traits, distr)
            self._grad_norm_tensors.append(grad_norm)
            self._p_norm_tensors.append(p_norm)

    def unregister(self):
        for tensor in self._grad_norm_tensors:
            tensor.unregister()
        for tensor in self._p_norm_tensors:
            tensor.unregister()

    def _norm_tensor_class(self, tensor):
        if isinstance(tensor, nntile.tensor.Tensor_fp64):
            return nntile.tensor.Tensor_fp64
        return nntile.tensor.Tensor_fp32

    def _tensor_norm(self, tensor):
        """Return L2 norm of a tensor by syncing data to host."""
        np_dtype = (
            np.float64 if isinstance(tensor, nntile.tensor.Tensor_fp64) else np.float32
        )
        if tensor.shape == []:
            buffer = np.zeros((1,), dtype=np_dtype)
        else:
            buffer = np.zeros(tuple(tensor.shape), dtype=np_dtype, order="F")
        tensor.to_array(buffer)
        return float(np.linalg.norm(buffer))

    def _prepare_norms(self, values, name):
        """Convert user-provided norms to a validated float list."""
        if values is None:
            return None
        if np.isscalar(values):
            values = [values]
        else:
            values = list(values)
        if len(values) != len(self.params):
            raise ValueError(
                f"Expected {len(self.params)} entries for {name}, "
                f"got {len(values)}"
            )
        return [float(v) for v in values]

    def _set_scalar_tensor(self, tensor, value):
        np_dtype = (
            np.float64 if isinstance(tensor, nntile.tensor.Tensor_fp64) else np.float32
        )
        tensor.from_array(np.array([value], dtype=np_dtype))

    def step(self, *args, weight_norms=None, grad_norms=None):
        """
        Perform one LARS step.
        Compatible with both of the following call styles:
            step()                       # norms computed internally
            step(weight_norms, grad_norms)
            step(weight_norms=..., grad_norms=...)
        """
        if args:
            if len(args) > 2:
                raise TypeError(
                    "Lars.step accepts at most two positional arguments "
                    "(weight_norms, grad_norms)"
                )
            if weight_norms is not None or grad_norms is not None:
                raise TypeError(
                    "Provide norms either positionally or via keywords, not both"
                )
            weight_norms = args[0]
            if len(args) == 2:
                grad_norms = args[1]

        weight_norms = self._prepare_norms(weight_norms, "weight_norms")
        grad_norms = self._prepare_norms(grad_norms, "grad_norms")

        if weight_norms is None:
            weight_norms = [self._tensor_norm(p.value) for p in self.params]
        if grad_norms is None:
            grad_norms = [self._tensor_norm(p.grad) for p in self.params]

        if len(weight_norms) != len(self.params) or len(grad_norms) != len(
            self.params
        ):
            raise ValueError("Incorrect number of norms provided for LARS step")

        for idx, p in enumerate(self.params):
            self._set_scalar_tensor(self._grad_norm_tensors[idx], grad_norms[idx])
            self._set_scalar_tensor(self._p_norm_tensors[idx], weight_norms[idx])

            nntile.functions.fused_lars_step(
                p.value,
                p.grad,
                self.lr,
                self.trust_ratio,
                self._grad_norm_tensors[idx],
                self._p_norm_tensors[idx],
                self.weight_decay,
            )
            p.value.wont_use()
            p.grad.invalidate_submit()

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
        nbytes = 0
        for tensor in self._grad_norm_tensors:
            nbytes += tensor.get_nbytes()
        for tensor in self._p_norm_tensors:
            nbytes += tensor.get_nbytes()
        return nbytes
