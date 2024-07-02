# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/tensor.py
# Multiprecision tensor with operations
#
# @version 1.0.0

from .nntile_core import tensor as core_tensor

# Multiprecision tensor as a union type for all precisions
Tensor = core_tensor.Tensor_fp32 | core_tensor.Tensor_fp64 \
        | core_tensor.Tensor_fp32_fast_tf32 | core_tensor.Tensor_bf16
# Optional tensor argument
TensorOrNone = Tensor | None
# Union of multiprecision tensor and float
TensorOrFloat = Tensor | float
TensorFloatOrInt = Tensor | core_tensor.Tensor_int64

# Struct meant for tensor, its gradient and a flag if gradient is required
class TensorMoments(object):
    value: TensorOrNone
    grad: TensorOrNone
    grad_required: bool

    def __init__(self, value: TensorOrNone, grad: TensorOrNone,
            grad_required: bool):
        self.value = value
        self.grad = grad
        self.grad_required = grad_required

    def __del__(self):
        self.unregister()

    def unregister(self):
        if self.value is not None:
            self.value.unregister()
            self.value = None
        if self.grad is not None:
            self.grad.unregister()
            self.grad = None
