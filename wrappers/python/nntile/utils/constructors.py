# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/utils/constructors.py
# Certain auxiliary constructors for NNTile-Numpy interoperability
#
# @version 1.0.0

from typing import Sequence

import numpy as np

from nntile.functions import clear_async, fill_async
from nntile.nntile_core.tensor import (
    Tensor_bf16, Tensor_bool, Tensor_fp32, Tensor_fp32_fast_tf32, Tensor_fp64,
    Tensor_int64, TensorTraits)
from nntile.types import Tensor

nnt2np_type_mapping = {
    Tensor_fp32: np.float32,
    Tensor_fp32_fast_tf32: np.float32,
    Tensor_bf16: np.float32,
    Tensor_fp64: np.float64,
    Tensor_int64: np.int64,
    Tensor_bool: bool,
}

np2nnt_type_mapping = {
    np.dtypes.Float32DType: Tensor_fp32,
    np.dtypes.Float64DType: Tensor_fp64,
    np.dtypes.IntDType: Tensor_int64,
    np.dtypes.Int32DType: Tensor_int64,
    np.dtypes.Int64DType: Tensor_int64,
    np.dtypes.BoolDType: Tensor_bool,
}


def empty(
    shape: Sequence[int],
    basetile_shape: Sequence[int] | None = None,
    dtype: Tensor = Tensor_fp32,
    mpi_distr: Sequence[int] | None = None,
    next_tag: int = 0
):
    A_traits = TensorTraits(shape, basetile_shape or shape)
    if mpi_distr is None:
        mpi_distr = [0] * A_traits.grid.nelems
    A_value = dtype(A_traits, mpi_distr, next_tag)
    return A_value


def empty_like(
    A: np.ndarray,
    basetile_shape: Sequence[int] | None = None,
    mpi_distr: Sequence[int] | None = None,
    next_tag: int = 0
):
    dtype = np2nnt_type_mapping[type(A.dtype)]
    return empty(A.shape, basetile_shape, dtype, mpi_distr, next_tag)


def from_array(
    A: np.ndarray,
    basetile_shape: Sequence[int] | None = None,
    mpi_distr: Sequence[int] | None = None,
    next_tag: int = 0,
):
    A_value = empty_like(A, basetile_shape, mpi_distr, next_tag)
    A_value.from_array(A)
    return A_value


def to_numpy(tensor_nnt):
    np_res = np.zeros(
        tensor_nnt.shape,
        order="F",
        dtype=nnt2np_type_mapping[type(tensor_nnt)]
    )
    tensor_nnt.to_array(np_res)
    return np_res


def zeros(
    shape: Sequence[int],
    basetile_shape: Sequence[int] | None = None,
    dtype: Tensor = Tensor_fp32,
    mpi_distr: Sequence[int] | None = None,
    next_tag: int = 0,
):
    A_value = empty(shape, basetile_shape, dtype, mpi_distr, next_tag)
    clear_async(A_value)
    return A_value


def zeros_like(
    A: np.ndarray,
    basetile_shape: Sequence[int] | None = None,
    mpi_distr: Sequence[int] | None = None,
    next_tag: int = 0
):
    dtype = np2nnt_type_mapping[type(A.dtype)]
    return zeros(A.shape, basetile_shape, dtype, mpi_distr, next_tag)


def full(
    shape: Sequence[int],
    basetile_shape: Sequence[int] | None = None,
    dtype: Tensor = Tensor_fp32,
    fill_value: float = 0.0,
    mpi_distr: Sequence[int] | None = None,
    next_tag: int = 0,
):
    A_value = empty(shape, basetile_shape, dtype, mpi_distr, next_tag)
    fill_async(fill_value, A_value)
    return A_value


def full_like(
    A: np.ndarray,
    basetile_shape: Sequence[int] | None = None,
    fill_value: float = 0.0,
    mpi_distr: Sequence[int] | None = None,
    next_tag: int = 0
):
    dtype = np2nnt_type_mapping[type(A.dtype)]
    return full(A.shape, basetile_shape, dtype, fill_value, mpi_distr,
            next_tag)


def ones(
    shape: Sequence[int],
    basetile_shape: Sequence[int] | None = None,
    dtype: Tensor = Tensor_fp32,
    mpi_distr: Sequence[int] | None = None,
    next_tag: int = 0,
):
    return full(shape, basetile_shape, dtype, 1, mpi_distr, next_tag)


def ones_like(
    A: np.ndarray,
    basetile_shape: Sequence[int] | None = None,
    mpi_distr: Sequence[int] | None = None,
    next_tag: int = 0
):
    dtype = np2nnt_type_mapping[type(A.dtype)]
    return ones(A.shape, basetile_shape, dtype, mpi_distr, next_tag)
