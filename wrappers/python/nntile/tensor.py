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

from .nntile_core.tensor import TensorTraits, Tensor_fp32, Tensor_fp64, \
        Tensor_int64, Tensor_fp16, Tensor_bool
from .nntile_core import TransOp, notrans, trans

from .utils.constructors import astensor, asarray, zeros, ones

from .types import *
from .functions import *
