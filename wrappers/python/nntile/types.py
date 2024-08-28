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
# @version 1.1.0

import sys
from typing import TYPE_CHECKING, Protocol, Sequence

if sys.version_info >= (3, 12):
    from collections.abc import Buffer
else:
    from typing_extensions import Buffer

from .nntile_core.tensor import (
    Tensor_bf16, Tensor_fp32, Tensor_fp32_fast_tf32, Tensor_fp64, Tensor_int64,
    TensorTraits)
from .nntile_core.tile import TileTraits

if TYPE_CHECKING:
    from .nntile_core.tensor import Tensor as TensorProtocol
    from .nntile_core.tile import Tile as TileProtocol
else:
    class TileProtocol(Protocol):
        """Tile type specification for duck typing."""

        def __init__(self, traits: TileTraits): ...
        def from_array(self, array: Buffer): ...
        def to_array(self, array: Buffer): ...
        def unregister(self) -> None: ...

    class TensorProtocol(TileProtocol, Protocol):
        """Tensor type specification for duck typing."""

        def __init__(self, traits: TensorTraits, distribution: Sequence[int],
                     last_tag: int):            ...

        @property
        def distribution(self) -> list[int]: ...
        @property
        def next_tag(self) -> int: ...

        def from_array(self, array: Buffer) -> None: ...
        def to_array(self, array: Buffer) -> None: ...

        def get_tile(self, linear_offset: int) -> TileProtocol: ...
        def print_scalar_async(self) -> None: ...

        def set_reduction_add(self) -> None: ...
        def set_reduction_hypot(self) -> None: ...
        def set_reduction_maxsumexp(self) -> None: ...

        def invalidate_submit(self) -> None: ...
        def unregister(self) -> None: ...
        def wont_use(self) -> None: ...


# Multiprecision tensor as a union type for all precisions
Tensor = Tensor_fp32 | Tensor_fp64 | Tensor_fp32_fast_tf32 | Tensor_bf16
# Optional tensor argument
TensorOrNone = Tensor | None
# Union of multiprecision tensor and float
TensorOrFloat = Tensor | float
TensorFloatOrInt = Tensor | Tensor_int64


class TensorMoments(object):
    """Tensor, its gradient, and a flag if gradient is required."""

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

    def get_nbytes(self):
        if self.grad is None:
            return self.value.get_nbytes()
        else:
            return self.value.get_nbytes() + self.grad.get_nbytes()
