# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/conv2d.py
# Conv2d layer of NNTile Python package
#
# @version 1.1.0

from typing import Sequence

import torch
import torch.nn as nn

from nntile.layer.base_layer import BaseLayer
from nntile.tensor import (
    TensorMoments, TensorTraits, conv2d_bwd_input_inplace_async,
    conv2d_bwd_weight_inplace_async, conv2d_inplace_async, to_numpy)


class Conv2d(BaseLayer):
    x: TensorMoments
    y: TensorMoments
    w: TensorMoments
    padding: Sequence[int]
    redux: bool

    # Construct linear layer with all the provided data
    def __init__(self, x: TensorMoments, y: TensorMoments, w: TensorMoments,
            padding: Sequence[int] = [0, 0], stride: Sequence[int] = [1, 1],
            dilation: Sequence[int] = [1, 1], redux: bool = False):
        # Set up Base layer
        super().__init__([x], [y], [w], [])
        # Set up local named parameters
        self.x = x
        if self.x.grad is not None:
            self.x.grad.set_reduction_add()
        self.y = y
        self.y.value.set_reduction_add()
        self.w = w
        self.w.grad.set_reduction_add()
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        if redux:
            self.redux = 1
        else:
            self.redux = 0

    # Simple generator for the Conv2d layer
    @staticmethod
    def generate_simple(x: TensorMoments, kernel_shape: Sequence[int],
            out_channels: int, padding: Sequence[int] = [0, 0],
            stride: Sequence[int] = [1, 1], dilation: Sequence[int] = [1, 1],
            next_tag: int = 0, redux: bool = False):
        # Define shapes
        w_shape = kernel_shape + [x.value.shape[2], out_channels]
        y_shape_w = x.value.shape[0] + 2 * padding[0] - 1 \
                - dilation[0] * (kernel_shape[0] - 1)
        y_shape_w = y_shape_w // stride[0] + 1
        y_shape_h = x.value.shape[1] + 2 * padding[1] - 1 \
                - dilation[1] * (kernel_shape[1] - 1)
        y_shape_h = y_shape_h // stride[1] + 1
        y_shape = [y_shape_w, y_shape_h, out_channels, x.value.shape[3]]
        y_tile = x.value.basetile_shape[:2] + [out_channels] + \
                x.value.basetile_shape[3:]
        # Define W
        w_traits = TensorTraits(w_shape, w_shape)
        w_distr = [0]
        w_value = type(x.value)(w_traits, w_distr, next_tag)
        next_tag = w_value.next_tag
        # Create gradient of W with the same traits and distribution as W
        w_grad = type(x.value)(w_traits, w_distr, next_tag)
        next_tag = w_grad.next_tag
        # Define W as TensorMoments
        w = TensorMoments(w_value, w_grad, True)
        # Define Y
        y_traits = TensorTraits(y_shape, y_tile)
        # TODO change distribution
        y_distr = [0] * y_traits.grid.nelems
        y_value = type(x.value)(y_traits, y_distr, next_tag)
        next_tag = y_value.next_tag
        # Create gradient of Y with the same traits and distribution as Y
        y_grad = type(x.value)(y_traits, y_distr, next_tag)
        next_tag = y_grad.next_tag
        # Define Y as TensorMoments
        y = TensorMoments(y_value, y_grad, True)
        # Create linear layer with all the provided data
        layer = Conv2d(x, y, w, padding=padding, stride=stride,
                dilation=dilation, redux=redux)
        # Return layer and next tag to be used
        return (layer, next_tag)

    # Forward propagation of the conv2d layer
    def forward_async(self):
        conv2d_inplace_async(1.0, self.x.value, self.w.value, 0.0,
                self.y.value, self.padding, self.stride, self.dilation)

    # Backward propagation of the conv2d layer
    def backward_async(self):
        if self.x.grad_required:
            conv2d_bwd_input_inplace_async(1.0, self.y.grad, self.w.value, 1.0,
                    self.x.grad, self.padding, self.stride, self.dilation)
        if self.w.grad_required:
            conv2d_bwd_weight_inplace_async(1.0, self.x.value, self.y.grad,
                    0.0, self.w.grad, self.padding, self.stride, self.dilation)

    def to_torch(self):
        torch_layer = nn.Conv2d(
                self.w.value.shape[2],
                self.w.value.shape[3],
                kernel_size=self.w.value.shape[1::-1],
                padding=self.padding[::-1],
                stride=self.stride[::-1],
                dilation=self.dilation[::-1],
                bias=False
        )
        torch_layer.weight.data = torch.tensor(to_numpy(self.w.value).T,
                requires_grad=True)
        return torch_layer

    def to_torch_with_grads(self):
        torch_layer = self.to_torch()
        torch_layer.weight.grad = torch.tensor(to_numpy(self.w.grad).T)
        return torch_layer

    @classmethod
    def from_torch(cls, torch_layer, x, next_tag):
        redux = 0
        nntile_layer, next_tag = cls.generate_simple(x,
                list(torch_layer.kernel_size[::-1]),
                torch_layer.out_channels,
                padding=torch_layer.padding[::-1],
                stride=torch_layer.stride[::-1],
                dilation=torch_layer.dilation[::-1],
                next_tag=next_tag,
                redux=redux
        )
        nntile_layer.w.value.from_array(
                torch_layer.weight.detach().cpu().numpy().T
        )
        return nntile_layer, next_tag
