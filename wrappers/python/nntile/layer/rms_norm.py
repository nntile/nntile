# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/rms_norm.py
# RMSNorm of NNTile Python package
#
# @version 1.1.0

import torch
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm as RMSNorm_torch)

import nntile.utils.constructors as nntc
from nntile.layer.base_layer import BaseLayer
from nntile.tensor import (
    Tensor, TensorMoments, TensorTraits, add_async, copy_async, fill_async,
    hypot_scalar_inverse_async, norm_slice_async, prod_fiber3_async,
    prod_slice_async, sumprod_fiber_async, sumprod_slice_async, to_numpy)


class RMSNorm(BaseLayer):
    x: TensorMoments
    y: TensorMoments
    gamma: TensorMoments
    tmp_y_value: Tensor
    tmp_y_grad: Tensor
    mean: Tensor
    inv_stddev: Tensor
    axis: int
    eps: float

    # Construct normalization layer with all the provided data
    def __init__(self, x: TensorMoments, y: TensorMoments,
            gamma: TensorMoments, tmp_y_value: Tensor,
            tmp_y_grad: Tensor, mean: Tensor, inv_stddev: Tensor, axis: int,
            eps: float, redux: bool = False):
        # Redirect to BaseLayer initialization
        super().__init__([x], [y], [gamma], [tmp_y_value, tmp_y_grad,
                inv_stddev, mean])
        self.x = x
        self.y = y
        self.gamma = gamma
        self.gamma.grad.set_reduction_add()
        self.tmp_y_value = tmp_y_value
        self.tmp_y_grad = tmp_y_grad
        self.inv_stddev = inv_stddev
        self.inv_stddev.set_reduction_hypot()
        self.mean = mean
        self.mean.set_reduction_add()
        self.axis = axis
        self.l = self.x.value.shape[axis]
        self.eps = eps ** 0.5  # This value is used to init deviation
        if redux:
            self.redux = 1
        else:
            self.redux = 0

    # Simple generator for the normalization layer
    @staticmethod
    def generate_simple(x: TensorMoments, axis: int, eps: float,
            next_tag: int, redux: bool = False):
        # Get traits of X
        x_traits = TensorTraits(x.value.shape, x.value.basetile_shape)
        # Create Y with the same traits and distribution as X
        x_distr = x.value.distribution
        y_value = type(x.value)(x_traits, x_distr, next_tag)
        next_tag = y_value.next_tag
        # Create grad Y with the same traits and distribution as X
        y_grad = type(x.value)(x_traits, x_distr, next_tag)
        next_tag = y_grad.next_tag
        # Wrap Y
        y = TensorMoments(y_value, y_grad, True)
        # Gamma parameter
        gamma_shape = [x.value.shape[axis]]
        gamma_basetile = [x.value.basetile_shape[axis]]
        gamma_traits = TensorTraits(gamma_shape, gamma_basetile)
        gamma_distr = []
        for i in range(x.value.grid.shape[axis]):
            gamma_distr.append(x_distr[x.value.grid.stride[axis] * i])
        gamma_value = type(x.value)(gamma_traits, gamma_distr, next_tag)
        next_tag = gamma_value.next_tag
        gamma_grad = type(x.value)(gamma_traits, gamma_distr, next_tag)
        next_tag = gamma_grad.next_tag
        gamma = TensorMoments(gamma_value, gamma_grad, True)
        # Temporary tensor for normalized input
        tmp_y_value = type(x.value)(x_traits, x_distr, next_tag)
        next_tag = tmp_y_value.next_tag
        # Temporary tensor for gradient of normalized input
        tmp_y_grad = type(x.value)(x_traits, x_distr, next_tag)
        next_tag = tmp_y_grad.next_tag
        inv_stddev_shape = x.value.shape[:axis] + x.value.shape[axis + 1:]
        inv_stddev_basetile = x.value.basetile_shape[:axis] \
                + x.value.basetile_shape[axis + 1:]
        inv_stddev_traits = TensorTraits(inv_stddev_shape, inv_stddev_basetile)
        inv_stddev_distr = []
        # Set distribution of mean tensor as X tensor with 0 index in provided
        # axis
        for i in range(inv_stddev_traits.grid.nelems):
            inv_stddev_tile_index = inv_stddev_traits.grid.linear_to_index(i)
            x_tile_index = inv_stddev_tile_index[0:axis] + [0] \
                    + inv_stddev_tile_index[axis:]
            x_tile_offset = x.value.grid.index_to_linear(x_tile_index)
            inv_stddev_distr.append(x_distr[x_tile_offset])
        inv_stddev = type(x.value)(inv_stddev_traits, inv_stddev_distr,
                                   next_tag)
        next_tag = inv_stddev.next_tag
        mean = type(x.value)(inv_stddev_traits, inv_stddev_distr, next_tag)
        next_tag = mean.next_tag

        # Create RMSNorm object with all the provided tensors
        layer = RMSNorm(x, y, gamma, tmp_y_value, tmp_y_grad, mean,
                inv_stddev, axis, eps, redux=redux)
        # Init gamma and beta
        fill_async(1.0, gamma.value)
        # Return layer and next tag to be used
        return (layer, next_tag)

    # Forward propagation of the normalization layer
    def forward_async(self):
        # Compute standard deviation of self.y.value
        norm_slice_async(1.0 / self.l**0.5, self.x.value, 0.0,
                self.inv_stddev, self.axis, redux=self.redux)
        hypot_scalar_inverse_async(self.eps, 1.0, self.inv_stddev)
        # Finally, normalize input
        copy_async(self.x.value, self.tmp_y_value)
        prod_slice_async(self.inv_stddev, 1.0, self.tmp_y_value, self.axis)
        # inv_stddev can be offloaded from GPU
        self.inv_stddev.wont_use()
        # Scale normalized input for the backward phase
        prod_fiber3_async(self.gamma.value, 1.0, self.tmp_y_value,
                self.y.value, self.axis)
        # tmp_Y_value can be offloaded from GPU
        self.tmp_y_value.wont_use()
        # gamma can be offloaded from GPU
        self.gamma.value.wont_use()
        # Y can be offloaded from GPU
        self.y.value.wont_use()

    # Dynamic forward propagation of the normalization layer
    def forward_dynamic(self, x: TensorMoments):
        inv_stddev = nntc.empty(
            x.value.shape[: self.axis] + x.value.shape[self.axis + 1 :],
            basetile_shape=x.value.basetile_shape[: self.axis]
            + x.value.basetile_shape[self.axis + 1 :],
            dtype=type(x.value),
        )
        tmp_y_value = nntc.empty_like(x.value)
        y_value = nntc.empty_like(x.value)

        # Finally, normalize input
        norm_slice_async(
            1.0 / x.value.shape[self.axis] ** 0.5,
            x.value,
            0.0,
            inv_stddev,
            self.axis,
            redux=self.redux,
        )
        hypot_scalar_inverse_async(self.eps, 1.0, inv_stddev)

        # Finally, normalize input
        copy_async(x.value, tmp_y_value)
        prod_slice_async(inv_stddev, 1.0, tmp_y_value, self.axis)

        # Scale normalized input for the backward phase
        prod_fiber3_async(
            self.gamma.value, 1.0, tmp_y_value, y_value, self.axis
        )

        return TensorMoments(y_value, None, False)

    # Backward propagation of the normalization layer
    def backward_async(self):
        # Accumulate gradient over gamma
        sumprod_fiber_async(1.0, self.y.grad, self.tmp_y_value, 1.0,
                self.gamma.grad, self.axis, redux=self.redux)
        # d_gamma can be offloaded from GPU
        self.gamma.grad.wont_use()
        # Define gradient over normalized input
        prod_fiber3_async(self.gamma.value, 1.0, self.y.grad,
                self.tmp_y_grad, self.axis)
        # dY can be offloaded from GPU
        self.y.grad.wont_use()
        # gamma can be offloaded from GPU
        self.gamma.value.wont_use()
        # Get mean of product of tmp_Y_grad and tmp_Y_value over the given axis
        sumprod_slice_async(-1.0 / self.l, self.tmp_y_grad, self.tmp_y_value,
                0.0, self.mean, self.axis, redux=self.redux)
        # Multiply tmp_Y_value by the mean
        prod_slice_async(self.mean, 1.0, self.tmp_y_value, self.axis)
        # Add tmp_Y_grad to tmp_Y_value
        add_async(1., self.tmp_y_grad, 1., self.tmp_y_value)
        # tmp_Y_grad can be deleted
        self.tmp_y_grad.invalidate_submit()
        # mean can be deleted
        self.mean.invalidate_submit()
        # Multiply tmp_Y_value by the inverse stddev
        prod_slice_async(self.inv_stddev, 1.0, self.tmp_y_value, self.axis)
        # inv_stddev can be deleted
        self.inv_stddev.invalidate_submit()
        # Accumulate gradient from tmp_Y_value
        add_async(1., self.tmp_y_value, 1., self.x.grad)
        # tmp_Y_value can be deleted
        self.tmp_y_value.invalidate_submit()
        # dX can offloade from GPU
        self.x.grad.wont_use()

    @staticmethod
    def from_torch(torch_rmsnorm, x: TensorMoments,
                   axis: int, eps: float,
                   next_tag: int, redux: bool = False):
        rmsnorm_layer, next_tag = RMSNorm.generate_simple(x, axis,
                                                          eps, next_tag,
                                                          redux)
        rmsnorm_layer.parameters[0].value.from_array(
            torch_rmsnorm.weight.data.cpu().detach().numpy())

        return rmsnorm_layer, next_tag

    def to_torch(self):
        target_shape = self.activations_input[0].value.shape
        torch_layer = RMSNorm_torch(target_shape[self.axis],
                                    self.eps**2)
        torch_layer.weight.data = torch.tensor(
                                to_numpy(self.parameters[0].value),
                                requires_grad=True)
        return torch_layer

    def to_torch_with_grads(self):
        torch_layer = self.to_torch()
        torch_layer.weight.grad = torch.tensor(
                                to_numpy(self.parameters[0].grad))

        return torch_layer
