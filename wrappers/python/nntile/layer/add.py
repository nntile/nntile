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
from nntile.tensor import TensorMoments, TensorTraits, add_async, copy_async


class Add(BaseLayer):
    def __init__(self, x: TensorMoments, y: TensorMoments, res: TensorMoments):
        self.x = x
        self.y = y
        self.res = res
        super().__init__([x, y], [res], [], [])

    @staticmethod
    def generate_simple(x: TensorMoments, y: TensorMoments, next_tag: int):
        res_traits = TensorTraits(y.value.shape, y.value.basetile_shape)
        res_distr = [0] * res_traits.grid.nelems
        res_value = type(y.value)(res_traits, res_distr, next_tag)
        next_tag = res_value.next_tag
        res_grad = type(y.value)(res_traits, res_distr, next_tag)
        next_tag = res_grad.next_tag
        res = TensorMoments(res_value, res_grad, True)
        return Add(x, y, res), next_tag

    def forward_async(self):
        copy_async(self.x.value, self.res.value)
        add_async(1, self.y.value, 1, self.res.value)
        self.x.value.wont_use()
        self.y.value.wont_use()
        self.res.value.wont_use()

    def forward_dynamic(self, x1: TensorMoments, x2: TensorMoments):
        y = nntc.clone(x1.value)
        add_async(1.0, x2.value, 1.0, y)
        return TensorMoments(y, None, False)

    def backward_async(self):
        add_async(1, self.res.grad, 1, self.x.grad)
        add_async(1, self.res.grad, 1, self.y.grad)
        self.x.grad.wont_use()
        self.y.grad.wont_use()
        self.res.grad.wont_use()
