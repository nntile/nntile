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
# @version 1.0.0

import math

from nntile.layer.base_layer import BaseLayer
from nntile.tensor import (Tensor, TensorMoments, add_async, add_fiber_async,
                           copy_async, hypot_scalar_inverse_async,
                           norm_slice_async, ones, pow_async, prod_async,
                           prod_fiber_async, sum_fiber_async,
                           sumprod_fiber_async, zeros)


class BatchNorm2d(BaseLayer):
    def __init__(
        self,
        x: TensorMoments,
        y: TensorMoments,
        weight: TensorMoments,
        bias: TensorMoments,
        eps: float = 1e-05,
        redux: bool = False,
    ):
        self.x = x
        self.eps = eps**0.5  # we will square it in forward (implementation constraints)

        self.weight = weight
        self.bias = bias

        self.dtype = type(x.value)

        self.x_normalized = zeros(self.x.value.shape, dtype=self.dtype)

        self.y = y

        self.x_unbiased_copy = zeros(self.y.value.shape, dtype=self.dtype)
        self.inv_std = zeros(self.x.value.shape[1:2], dtype=self.dtype)

        self.std_tmp_3dim = zeros(self.x.value.shape[1:], dtype=self.dtype)
        self.std_tmp_2dim = zeros(
            self.x.value.shape[1:2] + self.x.value.shape[3:], dtype=self.dtype
        )

        self.n_channels = self.x.value.shape[1]
        self.numel_in_channel = math.prod(self.x.value.shape) / self.n_channels

        self.redux = redux

        self.x_grad_tmp = zeros(self.x.value.shape, dtype=self.dtype)

        self.tmp_buff_full = zeros(self.x.value.shape, dtype=self.dtype)
        self.tmp_buff_channels = zeros(self.x.value.shape[1:2], dtype=self.dtype)

    @classmethod
    def generate_simple(cls, x, eps: float = 1e-05, redux=False):
        y = TensorMoments(
            zeros(x.value.shape, dtype=type(x.value)),
            zeros(x.value.shape, dtype=type(x.value)),
            grad_required=True,
        )
        n_channels = x.value.shape[1]

        weight = TensorMoments(
            ones([n_channels], dtype=type(x.value)),
            ones([n_channels], dtype=type(x.value)),
            grad_required=True,
        )
        bias = TensorMoments(
            ones([n_channels], dtype=type(x.value)),
            ones([n_channels], dtype=type(x.value)),
            grad_required=True,
        )

        return cls(x, y, weight, bias, eps=eps, redux=redux)

    def forward_async(self):
        self._normalize_forward()
        self._learnable_transform_forward()

        self.y.value.wont_use()

    def backward_async(self):
        self._learnable_transform_backward()
        self._normalize_backward()

        # could be earlier. But this is straightforward we delete grad
        self.x.grad.wont_use()
        self.y.grad.wont_use()

    def _normalize_forward(self):
        # TODO: add add_fiber3_async and remove copy
        copy_async(self.x.value, self.x_normalized)

        # mean
        x_mean = self.tmp_buff_channels
        sum_fiber_async(1.0 / self.numel_in_channel, self.x.value, 0.0, x_mean, 1, 0)
        self.x.value.wont_use()

        # X_res = X - mean
        add_fiber_async(-1.0, x_mean, 1.0, self.x_normalized, 1, 0)
        self.tmp_buff_channels.wont_use()
        # copy for backward
        copy_async(self.x_normalized, self.x_unbiased_copy)
        self.x_unbiased_copy.wont_use()

        # inverse std
        norm_slice_async(
            1.0, self.x_normalized, 0.0, self.std_tmp_3dim, 0, redux=self.redux
        )
        norm_slice_async(
            1.0, self.std_tmp_3dim, 0.0, self.std_tmp_2dim, 2, redux=self.redux
        )
        self.std_tmp_3dim.invalidate_submit()
        norm_slice_async(
            1.0 / self.numel_in_channel**0.5,
            self.std_tmp_2dim,
            0.0,
            self.inv_std,
            1,
            redux=self.redux,
        )
        self.std_tmp_2dim.invalidate_submit()

        hypot_scalar_inverse_async(self.eps, 1.0, self.inv_std)

        # X_res = X_res/std
        prod_fiber_async(self.inv_std, 1.0, self.x_normalized, 1)
        self.inv_std.wont_use()

    def _learnable_transform_forward(self):
        # y = weight * y + bias
        copy_async(self.x_normalized, self.y.value)
        self.x_normalized.wont_use()
        prod_fiber_async(self.weight.value, 1.0, self.y.value, 1)
        self.weight.value.wont_use()
        add_fiber_async(1.0, self.bias.value, 1.0, self.y.value, 1, 0)
        self.bias.value.wont_use()

    def _compute_grad_normalized_input_over_x(self, grad_nnt):
        """
        compute gradient for nominator inplace
        """
        grad_mean = self.tmp_buff_channels  # zeros([self.n_channels])
        sum_fiber_async(1.0 / self.numel_in_channel, grad_nnt, 0.0, grad_mean, 1, 0)

        # grad computing over mean
        add_fiber_async(-1.0, grad_mean, 1.0, grad_nnt, 1, 0)
        self.tmp_buff_channels.invalidate_submit()

    def _compute_grad_inv_std_over_x(self, grad_nnt):
        """
        compute gradient for denominator inplace
        """
        # Compute grad d(out)/d(variance)
        inv_denom_grad = self.tmp_buff_channels  # zeros([self.n_channels])
        sum_fiber_async(1.0, grad_nnt, 0.0, inv_denom_grad, 1, 0)

        # x_var_eps = self.inv_std**-2
        # xvar_grad = denom_grad[None, :, None, None] * -0.5*(xvar_eps)**(-1.5)
        x_var_eps_ref = self.inv_std
        xvar_grad = x_var_eps_ref  # inplace for reuse. Invalidates self.inv_std

        pow_async(-0.5, -2.0 * -1.5, xvar_grad)  # fuse two powers from above
        prod_async(inv_denom_grad, xvar_grad)

        # Compute grad d(variance)/d(x)
        x_normalized_grad = grad_nnt  # zeros(self.x.value.shape)
        mean_grad = self.tmp_buff_channels  #  zeros([self.n_channels])

        # x_normalized_grad = 1.0/self.numel_in_channel*2*(A-xmean[None, :, None, None])
        # x_grad = (-1*x_normalized_grad.sum([0,2,3])[None, :, None,None]/self.numel_in_channel + x_normalized_grad)
        add_async(
            1.0 / self.numel_in_channel * 2,
            self.x_unbiased_copy,
            0.0,
            x_normalized_grad,
        )  # copy + scalar prod
        sum_fiber_async(
            -1.0 / self.numel_in_channel, x_normalized_grad, 0.0, mean_grad, 1, 0
        )
        add_fiber_async(1.0, mean_grad, 1.0, x_normalized_grad, 1, 0)
        self.tmp_buff_channels.invalidate_submit()

        # d(variance)/d(inv_std) = (xvar_grad*x_normalized_grad)
        prod_fiber_async(xvar_grad, 1.0, x_normalized_grad, 1)
        self.inv_std.invalidate_submit()

    def _normalize_backward(self):
        # Nominator part
        nominator_grad = self.x_grad_tmp  # zeros(self.x.value.shape)

        # nominator_grad = self.grad*inv_denominator
        copy_async(self.y.grad, nominator_grad)
        inv_denominator_ref = self.inv_std
        prod_fiber_async(inv_denominator_ref, 1.0, nominator_grad, 1)

        nominator_grad_x = nominator_grad
        self._compute_grad_normalized_input_over_x(nominator_grad_x)

        # Inversed denominator part
        inv_denominator_grad = self.tmp_buff_full  # zeros(self.x.value.shape)

        # denominator_grad = self.grad*nominator
        nominator_ref = self.x_unbiased_copy
        copy_async(nominator_ref, inv_denominator_grad)
        prod_async(self.y.grad, inv_denominator_grad)

        inv_denominator_grad_x = inv_denominator_grad
        self._compute_grad_inv_std_over_x(inv_denominator_grad_x)

        self.x_unbiased_copy.invalidate_submit()

        # grad_x = nominator_grad_x + inv_denominator_grad_x
        add_async(1.0, inv_denominator_grad_x, 1.0, nominator_grad_x)
        self.tmp_buff_full.invalidate_submit()

        # accumulate calculated gradient
        add_async(1.0, nominator_grad_x, 1.0, self.x.grad)
        self.x_grad_tmp.invalidate_submit()

    def _learnable_transform_backward(self):
        # bias_grad = grad
        sum_fiber_async(1.0, self.y.grad, 0.0, self.bias.grad, 1, 0)
        self.bias.grad.wont_use()

        # weight_grad = x_normalized * grad
        sumprod_fiber_async(
            1.0,
            self.y.grad,
            self.x_normalized,
            1.0,
            self.weight.grad,
            1,
            redux=self.redux,
        )
        self.weight.grad.wont_use()
        self.x_normalized.invalidate_submit()

        # norm_grad = grad * weight
        prod_fiber_async(self.weight.value, 1.0, self.y.grad, 1)
        self.weight.value.wont_use()
