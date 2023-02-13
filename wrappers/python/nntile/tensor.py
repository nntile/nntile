# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/tensor.py
# Multiprecision tensor with operations
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-02-13

from .nntile_core import tensor as core_tensor
from .nntile_core.tensor import TensorTraits, Tensor_fp32, Tensor_fp64
from .nntile_core import TransOp, notrans, trans
from typing import Union, List

# Multiprecision tensor as a union type for all precisions
Tensor = Union[core_tensor.Tensor_fp32, core_tensor.Tensor_fp64]
# Optional tensor argument
TensorOrNone = Union[Tensor, None]
# Union of multiprecision tensor and float
TensorOrFloat = Union[Tensor, float]

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

# Wrapper for multiprecision gemm
def gemm_async(alpha: float, trans_A: TransOp, A: Tensor, trans_B: TransOp,
        B: Tensor, beta: float, C: Tensor, ndim: int) -> None:
    if type(A) is not type(B) or type(A) is not type(C):
        raise TypeError
    if type(A) is core_tensor.Tensor_fp32:
        core_tensor.gemm_async_fp32(alpha, trans_A, A, trans_B, B, beta, C,
                ndim)
    else:
        core_tensor.gemm_async_fp64(alpha, trans_A, A, trans_B, B, beta, C,
                ndim)

# Wrapper for multiprecision ReLU
def relu_async(x: Tensor) -> None:
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.relu_async_fp32(x)
    else:
        core_tensor.relu_async_fp64(x)

# Wrapper for multiprecision derivative of ReLU
def drelu_async(x: Tensor) -> None:
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.drelu_async_fp32(x)
    else:
        core_tensor.drelu_async_fp64(x)

# Wrapper for multiprecision sumnorm
def sumnorm_async(x: Tensor, sumnorm: Tensor, axis: int) -> None:
    if type(x) is not type(sumnorm):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.sumnorm_async_fp32(x, sumnorm, axis)
    else:
        core_tensor.sumnorm_async_fp64(x, sumnorm, axis)

# Wrapper for multiprecision softmax
def softmax_async(maxsumexp: Tensor, x: Tensor, axis: int) -> None:
    if type(maxsumexp) is not type(x):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.softmax_async_fp32(maxsumexp, x, axis)
    else:
        core_tensor.softmax_async_fp64(maxsumexp, x, axis)

# Wrapper for multiprecision scatter
def scatter_async(x: Tensor, y: Tensor) -> None:
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.scatter_async_fp32(x, y)
    else:
        core_tensor.scatter_async_fp64(x, y)

# Wrapper for multiprecision randn
def randn_async(x: Tensor, start: List[int], shape: List[int], seed: int,
        mean: float, dev: float) -> None:
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.randn_async_fp32(x, start, shape, seed, mean, dev)
    else:
        core_tensor.randn_async_fp64(x, start, shape, seed, mean, dev)

# Wrapper for multiprecision prod
def prod_async(x: Tensor, y: Tensor) -> None:
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.prod_async_fp32(x, y)
    else:
        core_tensor.prod_async_fp64(x, y)

# Wrapper for multiprecision nrm2
def nrm2_async(x: Tensor, y: Tensor, tmp: Tensor) -> None:
    if type(x) is not type(y) or type(x) is not type(tmp):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.nrm2_async_fp32(x, y, tmp)
    else:
        core_tensor.nrm2_async_fp64(x, y, tmp)

# Wrapper for multiprecision normalize
def normalize_async(gb: Tensor, x: Tensor, y: Tensor, l: int, eps: float,
        axis: int) -> None:
    if type(x) is not type(y) or type(x) is not type(gb):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.normalize_async_fp32(gb, x, y, l, eps, axis)
    else:
        core_tensor.normalize_async_fp64(gb, x, y, l, eps, axis)

# Wrapper for multiprecision maxsumexp
def maxsumexp_async(x: Tensor, maxsumexp: Tensor, axis: int) -> None:
    if type(x) is not type(maxsumexp):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.maxsumexp_async_fp32(x, maxsumexp, axis)
    else:
        core_tensor.maxsumexp_async_fp64(x, maxsumexp, axis)

# Wrapper for multiprecision bias
def bias_async(bias: Tensor, x: Tensor, axis: int) -> None:
    if type(bias) is not type(x):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.bias_async_fp32(bias, x, axis)
    else:
        core_tensor.bias_async_fp64(bias, x, axis)

# Wrapper for multiprecision gather
def gather_async(x: Tensor, y: Tensor) -> None:
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.gather_async_fp32(x, y)
    else:
        core_tensor.gather_async_fp64(x, y)

# Wrapper for multiprecision copy_intersection
def copy_intersection_async(x: Tensor, x_offset: List[int], y: Tensor,
        y_offset: List[int]) -> None:
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.copy_intersection_async_fp32(x, x_offset, y, y_offset)
    else:
        core_tensor.copy_intersection_async_fp64(x, x_offset, y, y_offset)

# Wrapper for multiprecision copy
def copy_async(x: Tensor, y: Tensor) -> None:
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.copy_async_fp32(x, y)
    else:
        core_tensor.copy_async_fp64(x, y)

# Wrapper for multiprecision clear
def clear_async(x: Tensor) -> None:
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.clear_async_fp32(x)
    else:
        core_tensor.clear_async_fp64(x)

# Wrapper for multiprecision axpy
def axpy_async(alpha: TensorOrFloat, x: Tensor, y: Tensor) -> None:
    if type(x) is not type(y):
        raise TypeError
    if type(alpha) is Tensor:
        if type(alpha) is not type(x):
            raise TypeError
        if type(x) is core_tensor.Tensor_fp32:
            core_tensor.axpy_async_fp32(alpha, x, y)
        else:
            core_tensor.axpy_async_fp64(alpha, x, y)
    else:
        if type(x) is core_tensor.Tensor_fp32:
            core_tensor.axpy2_async_fp32(alpha, x, y)
        else:
            core_tensor.axpy2_async_fp64(alpha, x, y)

