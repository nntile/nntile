# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/layer_norm.py
# LayerNorm of NNTile Python package
#
# @version 1.1.0

import nntile.utils.constructors as nntc
from nntile.layer.base_layer import BaseLayer
from nntile.tensor import (
    Tensor, TensorMoments, TensorTraits, add_async, add_fiber_async,
    add_slice3_async, add_slice_async, clear_async, fill_async,
    hypot_scalar_inverse_async, norm_slice_async, prod_fiber3_async,
    prod_slice_async, sum_fiber_async, sum_slice_async, sumprod_fiber_async,
    sumprod_slice_async)


class LayerNorm(BaseLayer):
    x: TensorMoments
    y: TensorMoments
    gamma: TensorMoments
    beta: TensorMoments
    tmp_y_value: Tensor
    tmp_y_grad: Tensor
    mean: Tensor
    inv_stddev: Tensor
    axis: int
    eps: float

    # Construct normalization layer with all the provided data
    def __init__(
        self,
        x: TensorMoments,
        y: TensorMoments,
        gamma: TensorMoments,
        beta: TensorMoments,
        tmp_y_value: Tensor,
        tmp_y_grad: Tensor,
        mean: Tensor,
        inv_stddev: Tensor,
        axis: int,
        eps: float,
        redux: bool = False,
    ):
        # Redirect to BaseLayer initialization
        super().__init__(
            [x],
            [y],
            [gamma, beta],
            [tmp_y_value, tmp_y_grad, mean, inv_stddev],
        )
        self.x = x
        self.y = y
        self.gamma = gamma
        self.gamma.grad.set_reduction_add()
        self.beta = beta
        self.beta.grad.set_reduction_add()
        self.tmp_y_value = tmp_y_value
        self.tmp_y_grad = tmp_y_grad
        self.mean = mean
        self.mean.set_reduction_add()
        self.inv_stddev = inv_stddev
        self.inv_stddev.set_reduction_hypot()
        self.axis = axis
        self.l = self.x.value.shape[axis]
        self.eps = eps**0.5  # This value is used to init deviation
        if redux:
            self.redux = 1
        else:
            self.redux = 0

    # Simple generator for the normalization layer
    @staticmethod
    def generate_simple(
        x: TensorMoments,
        axis: int,
        eps: float,
        next_tag: int,
        redux: bool = False,
    ):
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
        # Beta parameter
        beta_value = type(x.value)(gamma_traits, gamma_distr, next_tag)
        next_tag = beta_value.next_tag
        beta_grad = type(x.value)(gamma_traits, gamma_distr, next_tag)
        next_tag = beta_grad.next_tag
        beta = TensorMoments(beta_value, beta_grad, True)
        # Temporary tensor for normalized input
        tmp_y_value = type(x.value)(x_traits, x_distr, next_tag)
        next_tag = tmp_y_value.next_tag
        # Temporary tensor for gradient of normalized input
        tmp_y_grad = type(x.value)(x_traits, x_distr, next_tag)
        next_tag = tmp_y_grad.next_tag
        # Define auxiliary tensors to hold mean, inverse of stddev and scalar
        # products along given axis
        mean_shape = x.value.shape[:axis] + x.value.shape[axis + 1 :]
        mean_basetile = (
            x.value.basetile_shape[:axis] + x.value.basetile_shape[axis + 1 :]
        )
        mean_traits = TensorTraits(mean_shape, mean_basetile)
        mean_distr = []
        # Set distribution of mean tensor as X tensor with 0 index in provided
        # axis
        for i in range(mean_traits.grid.nelems):
            mean_tile_index = mean_traits.grid.linear_to_index(i)
            x_tile_index = (
                mean_tile_index[0:axis] + [0] + mean_tile_index[axis:]
            )
            x_tile_offset = x.value.grid.index_to_linear(x_tile_index)
            mean_distr.append(x_distr[x_tile_offset])
        mean = type(x.value)(mean_traits, mean_distr, next_tag)
        next_tag = mean.next_tag
        inv_stddev = type(x.value)(mean_traits, mean_distr, next_tag)
        next_tag = inv_stddev.next_tag
        # Create LayerNorm object with all the provided tensors
        layer = LayerNorm(
            x,
            y,
            gamma,
            beta,
            tmp_y_value,
            tmp_y_grad,
            mean,
            inv_stddev,
            axis,
            eps,
            redux=redux,
        )
        # Init gamma and beta
        clear_async(beta.value)
        fill_async(1.0, gamma.value)
        # Return layer and next tag to be used
        return (layer, next_tag)

    # Forward propagation of the normalization layer
    def forward_async(self):
        # Get means over given axis
        sum_slice_async(
            1.0 / self.l,
            self.x.value,
            0.0,
            self.mean,
            self.axis,
            redux=self.redux,
        )
        # Y = X - mean
        add_slice3_async(
            -1.0, self.mean, 1.0, self.x.value, self.tmp_y_value, self.axis
        )
        # mean can be offloaded from GPU
        self.mean.wont_use()
        # X can be offloaded from GPU
        self.x.value.wont_use()
        # Compute standard deviation of self.y.value
        # fill_async(self.eps, self.inv_stddev)
        norm_slice_async(
            1.0 / self.l**0.5,
            self.tmp_y_value,
            0.0,
            self.inv_stddev,
            self.axis,
            redux=self.redux,
        )
        hypot_scalar_inverse_async(self.eps, 1.0, self.inv_stddev)
        # Invert stddev (to multiply by it instead of dividing)
        # pow_async(1.0, -1.0, self.inv_stddev)
        # Finally, normalize input
        prod_slice_async(self.inv_stddev, 1.0, self.tmp_y_value, self.axis)
        # inv_stddev can be offloaded from GPU
        self.inv_stddev.wont_use()
        # Scale normalized input for the backward phase
        prod_fiber3_async(
            self.gamma.value, 1.0, self.tmp_y_value, self.y.value, self.axis
        )
        # tmp_Y_value can be offloaded from GPU
        self.tmp_y_value.wont_use()
        # gamma can be offloaded from GPU
        self.gamma.value.wont_use()
        # Shift output
        add_fiber_async(1.0, self.beta.value, 1.0, self.y.value, self.axis, 0)
        # beta can be offloaded from GPU
        self.beta.value.wont_use()
        # Y can be offloaded from GPU
        self.y.value.wont_use()

    # Forward propagation of the normalization layer
    def forward_dynamic(self, x: TensorMoments):
        mean_shape = (
            x.value.shape[: self.axis] + x.value.shape[self.axis + 1 :]
        )
        mean_basetile = (
            x.value.basetile_shape[: self.axis]
            + x.value.basetile_shape[self.axis + 1 :]
        )

        tmp_y_value = nntc.empty_like(x.value)
        y = TensorMoments(nntc.empty_like(x.value), None, False)
        mean = nntc.empty(mean_shape, mean_basetile, dtype=type(x.value))
        inv_stddev = nntc.empty(mean_shape, mean_basetile, dtype=type(x.value))

        num_layers = x.value.shape[self.axis]

        # Get means over given axis
        sum_slice_async(
            1.0 / num_layers, x.value, 0.0, mean, self.axis, redux=self.redux
        )
        # Y = X - mean
        add_slice3_async(-1.0, mean, 1.0, x.value, tmp_y_value, self.axis)
        # mean can be offloaded from GPU
        mean.wont_use()

        # Compute standard deviation of self.y.value
        norm_slice_async(
            1.0 / num_layers**0.5,
            tmp_y_value,
            0.0,
            inv_stddev,
            self.axis,
            redux=self.redux,
        )
        hypot_scalar_inverse_async(self.eps, 1.0, inv_stddev)
        # Finally, normalize input
        prod_slice_async(inv_stddev, 1.0, tmp_y_value, self.axis)
        # inv_stddev can be offloaded from GPU
        inv_stddev.wont_use()
        # Scale normalized input for the backward phase
        prod_fiber3_async(
            self.gamma.value, 1.0, tmp_y_value, y.value, self.axis
        )
        # tmp_Y_value can be offloaded from GPU
        tmp_y_value.wont_use()
        # gamma can be offloaded from GPU
        self.gamma.value.wont_use()
        # Shift output
        add_fiber_async(1.0, self.beta.value, 1.0, y.value, self.axis, 0)
        # beta can be offloaded from GPU
        self.beta.value.wont_use()
        return y

    # Backward propagation of the normalization layer
    def backward_async(self):
        # Accumulate gradient over beta
        sum_fiber_async(
            1.0,
            self.y.grad,
            1.0,
            self.beta.grad,
            self.axis,
            0,
            redux=self.redux,
        )
        # d_beta can be offloaded from GPU
        self.beta.grad.wont_use()
        # Accumulate gradient over gamma
        sumprod_fiber_async(
            1.0,
            self.y.grad,
            self.tmp_y_value,
            1.0,
            self.gamma.grad,
            self.axis,
            redux=self.redux,
        )
        # d_gamma can be offloaded from GPU
        self.gamma.grad.wont_use()
        # Define gradient over normalized input
        prod_fiber3_async(
            self.gamma.value, 1.0, self.y.grad, self.tmp_y_grad, self.axis
        )
        # dY can be offloaded from GPU
        self.y.grad.wont_use()
        # gamma can be offloaded from GPU
        self.gamma.value.wont_use()
        # Get mean of product of tmp_Y_grad and tmp_Y_value over the given axis
        sumprod_slice_async(
            -1.0 / self.l,
            self.tmp_y_grad,
            self.tmp_y_value,
            0.0,
            self.mean,
            self.axis,
            redux=self.redux,
        )
        # Multiply tmp_Y_value by the mean
        prod_slice_async(self.mean, 1.0, self.tmp_y_value, self.axis)
        # Add tmp_Y_grad to tmp_Y_value
        add_async(1.0, self.tmp_y_grad, 1.0, self.tmp_y_value)
        # Get mean value of tmp_Y_grad over the given axis
        sum_slice_async(
            1.0 / self.l,
            self.tmp_y_grad,
            0.0,
            self.mean,
            self.axis,
            redux=self.redux,
        )
        # tmp_Y_grad can be deleted
        self.tmp_y_grad.invalidate_submit()
        # Subtract mean from tmp_Y_value
        add_slice_async(-1.0, self.mean, 1.0, self.tmp_y_value, self.axis)
        # mean can be deleted
        self.mean.invalidate_submit()
        # Multiply tmp_Y_value by the inverse stddev
        prod_slice_async(self.inv_stddev, 1.0, self.tmp_y_value, self.axis)
        # inv_stddev can be deleted
        self.inv_stddev.invalidate_submit()
        # Accumulate gradient from tmp_Y_value
        # axpy_async(1.0, self.tmp_y_value, self.x.grad)
        add_async(1.0, self.tmp_y_value, 1.0, self.x.grad)
        # tmp_Y_value can be deleted
        self.tmp_y_value.invalidate_submit()
        # dX can offloade from GPU
        self.x.grad.wont_use()
