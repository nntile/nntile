# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/gap.py
# Global average pooling layer of NNTile Python package
#
# @version 1.1.0

from nntile.layer.base_layer import BaseLayer
from nntile.tensor import (
    TensorMoments, TensorTraits, add_slice_inplace_async, sum_slice_async,
    transpose_async,
)


class GAP(BaseLayer):
    x: TensorMoments
    y: TensorMoments

    def __init__(self, x: TensorMoments, y: TensorMoments, yT: TensorMoments):
        self.x = x
        self.y = y
        self.yT = yT
        super().__init__([x], [y], [], [yT])

    @staticmethod
    def generate_simple(x: TensorMoments):
        yT_shape = x.value.shape[1:]
        yT_basetile_shape = x.value.basetile_shape[1:]
        yT_traits = TensorTraits(yT_shape, yT_basetile_shape)
        yT_distr = [0] * yT_traits.grid.nelems
        yT_value = type(x.value)(yT_traits, yT_distr)
        yT_grad = type(x.value)(yT_traits, yT_distr)
        yT = TensorMoments(yT_value, yT_grad, True)

        y_shape = yT_shape[::-1]
        y_basetile_shape = yT_basetile_shape[::-1]
        y_traits = TensorTraits(y_shape, y_basetile_shape)
        y_distr = [0] * y_traits.grid.nelems
        y_value = type(x.value)(y_traits, y_distr)
        y_grad = type(x.value)(y_traits, y_distr)
        y = TensorMoments(y_value, y_grad, True)

        return GAP(x, y, yT)

    def forward_async(self):
        alpha = 1 / self.x.value.shape[0]
        sum_slice_async(alpha, self.x.value, 0.0, self.yT.value, 0)
        transpose_async(1.0, self.yT.value, self.y.value, 1)

    def backward_async(self):
        alpha = 1 / self.x.value.shape[0]
        transpose_async(1.0, self.y.grad, self.yT.grad, 1)
        add_slice_inplace_async(alpha, self.yT.grad, 0.0, self.x.grad, 0)
