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

from nntile.functions import fill_async
from nntile.nntile_core.tensor import (
    Tensor_bf16, Tensor_bool, Tensor_fp32, Tensor_fp32_fast_tf32, Tensor_fp64,
    Tensor_int64, TensorTraits)

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


def from_array(
    A: np.ndarray,
    basetile_shape: Sequence[int] | None = None,
    mpi_distr=[0],
    next_tag=0,
):
    A_traits = TensorTraits(A.shape, basetile_shape or A.shape)

    A_value = np2nnt_type_mapping[type(A.dtype)](A_traits, mpi_distr, next_tag)

    A_value.from_array(A)
    return A_value


def to_numpy(tensor_nnt):
    np_res = np.zeros(
        tensor_nnt.shape, order="F", dtype=nnt2np_type_mapping[type(tensor_nnt)]
    )
    tensor_nnt.to_array(np_res)
    return np_res


def zeros(shape: Sequence[int], dtype=Tensor_fp32):
    np_dtype = nnt2np_type_mapping[dtype]
    return from_array(np.zeros(shape, dtype=np_dtype))


def full(shape: Sequence[int], fill_value, dtype=Tensor_fp32):
    nnt_tensor = zeros(shape, dtype=dtype)
    fill_async(fill_value, nnt_tensor)
    return nnt_tensor


def ones(shape: Sequence[int], dtype=Tensor_fp32):
    return full(shape, 1, dtype=dtype)
