from .nntile_core import tensor as core_tensor
from .nntile_core.tensor import TensorTraits, Tensor_fp32, Tensor_fp64, \
        Tensor_int64, Tensor_fp16, Tensor_bool
from .nntile_core import TransOp, notrans, trans
from typing import Union, List

# Multiprecision tensor as a union type for all precisions
Tensor = Union[core_tensor.Tensor_fp32, core_tensor.Tensor_fp64]
# Optional tensor argument
TensorOrNone = Union[Tensor, None]
# Union of multiprecision tensor and float
TensorOrFloat = Union[Tensor, float]
TensorFloatOrInt = Union[Tensor, core_tensor.Tensor_int64]

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
        if self.grad is not None:
            self.grad.unregister()
