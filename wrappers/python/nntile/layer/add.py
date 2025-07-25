# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/add.py
# Add layer of NNTile Python package.
# It is used in skip-connection operation
#
# @version 1.1.0

import nntile.utils.constructors as nntc
from nntile.layer.base_layer import BaseLayer
from nntile.tensor import (
    TensorMoments, TensorTraits, add_async, add_inplace_async)


class Add(BaseLayer):
    def __init__(self, x: TensorMoments, y: TensorMoments, res: TensorMoments):
        self.x = x
        self.y = y
        self.res = res
        super().__init__([x, y], [res], [], [])

    @staticmethod
    def generate_simple(x: TensorMoments, y: TensorMoments):
        res_traits = TensorTraits(y.value.shape, y.value.basetile_shape)
        # Create result tensor with the same distribution as y
        res_value = type(y.value)(res_traits, y.value.distribution)
        res_grad = type(y.value)(res_traits, y.value.distribution)
        res = TensorMoments(res_value, res_grad, True)
        return Add(x, y, res)

    def forward_async(self):
        add_async(1.0, self.x.value, 1.0, self.y.value, self.res.value)
        self.x.value.wont_use()
        self.y.value.wont_use()
        self.res.value.wont_use()

    def forward_dynamic(self, x1: TensorMoments, x2: TensorMoments):
        y = nntc.empty_like(x1.value)
        add_async(1.0, x1.value, 1.0, x2.value, y)
        return TensorMoments(y, None, False)

    def backward_async(self):
        add_inplace_async(1.0, self.res.grad, 1.0, self.x.grad)
        add_inplace_async(1.0, self.res.grad, 1.0, self.y.grad)
        self.x.grad.wont_use()
        self.y.grad.wont_use()
        self.res.grad.wont_use()
