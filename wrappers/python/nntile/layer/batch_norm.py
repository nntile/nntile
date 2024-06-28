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

from operator import mul
from functools import reduce

from nntile.tensor import (
    Tensor, TensorMoments,
    prod_async, prod_slice_async, pow_async, sum_slice_async, sum_fiber_async, norm_slice_async, 
    hypot_scalar_inverse_async, fill_async, add_slice_async, add_slice3_async,
    add_fiber_async, add_scalar_async,
    hypot_scalar_inverse_async,
    prod_fiber_async, prod_fiber3_async,
    add_async,

    zeros, ones
)

from nntile.layer.base_layer import BaseLayer

class BatchNorm2d(BaseLayer):
    def __init__(self, x: TensorMoments, weight: Tensor, bias: Tensor, x_res_tm: TensorMoments, x_mean: Tensor, inv_std: Tensor, eps: float = 1e-05, redux: bool = False):
        self.x_tm = x
        self.x = x.value
        self.eps = eps**0.5 # we will square it in forward (implementation constraints)

        # TODO: add usage of learnable transforms 
        self.weight = weight
        self.bias = bias

        self.x_res_tm = x_res_tm
        self.x_res = self.x_res_tm.value

        self.x_normalized_copy = zeros(self.x_res.shape, dtype=type(x.value))
        self.x_mean = x_mean
        self.inv_std = inv_std
        self.std = zeros(inv_std.shape, dtype=type(x.value))
        
        self.std_tmp_3dim = zeros(self.x.shape[1:], dtype=type(x.value))
        self.std_tmp_2dim = zeros([self.x.shape[1]]+self.x.shape[3:], dtype=type(x.value))

        self.n_channels = self.x.shape[1]
        self.numel_in_channel = reduce(mul, self.x.shape)/self.n_channels

        self.grad = None

        self.redux = False

        self.tmp_buff_full = zeros(self.x.shape, dtype=type(x.value))

        self.grad_x_inv_denominator = zeros(self.x.shape, dtype=type(x.value))
        
        self.tmp_buff_channels = zeros([self.x.shape[1]], dtype=type(x.value))
        self.tmp_buff_channels_internal = zeros([self.x.shape[1]], dtype=type(x.value))

    @staticmethod
    def generate_simple(x, eps: float = 1e-05, redux=False):
        x_res = TensorMoments(zeros(x.value.shape, dtype=type(x.value)), zeros(x.value.shape, dtype=type(x.value)), grad_required=True)
        n_channels = x.value.shape[1]
        x_mean = zeros([n_channels], dtype=type(x.value))
        inv_std = zeros([n_channels], dtype=type(x.value))

        weight = ones([n_channels], dtype=type(x.value))
        bias = zeros([n_channels], dtype=type(x.value))
 
        return BatchNorm2d(x, weight, bias, x_res, x_mean, inv_std, eps=eps, redux=redux)

    def forward_async(self):
        #copy tensor 
        #TODO: add add_fiber3_async and remove copy
        add_async(1.0, self.x, 0.0, self.x_res) 

        # mean
        sum_fiber_async(1.0/self.numel_in_channel, self.x, 0.0, self.x_mean, 1, 0)

        # X_res = X - mean
        add_fiber_async(-1.0, self.x_mean, 1.0, self.x_res, 1, 0)
        # copy for backward
        add_async(1.0, self.x_res, 0.0, self.x_normalized_copy)

        # inverse std
        norm_slice_async(1.0, self.x_res, 0.0, self.std_tmp_3dim, 0, redux=self.redux)
        norm_slice_async(1.0, self.std_tmp_3dim, 0.0, self.std_tmp_2dim, 2, redux=self.redux)
        norm_slice_async(1.0/self.numel_in_channel**0.5, self.std_tmp_2dim, 0.0, self.std, 1, redux=self.redux)

        add_async(1.0, self.std, 0.0, self.inv_std) # copy
        hypot_scalar_inverse_async(self.eps, 1.0, self.inv_std)

        # X_res = X_res/std
        prod_fiber_async(self.inv_std, 1.0, self.x_res, 1)

    def _compute_grad_normalized_input_over_x(self, A_grad_nnt):    
        grad_mean = self.tmp_buff_channels # zeros([self.n_channels])
        sum_fiber_async(1.0/self.numel_in_channel, A_grad_nnt, 0.0, grad_mean, 1, 0)
        
        grad_x_nnt = self.x_tm.grad # zeros(A_grad_nnt.shape)
        add_async(1.0, A_grad_nnt, 0.0, grad_x_nnt) # copy
        
        # grad computing over mean
        add_fiber_async(-1.0, grad_mean, 1.0, grad_x_nnt, 1, 0)
        return grad_x_nnt

    def _compute_grad_inv_std_over_x(self, A_nnt, A_grad_nnt):
        # Compute grad d(out)/d(variance)
        inv_denom_grad = self.tmp_buff_channels # zeros([self.n_channels])
        sum_fiber_async(1.0, A_grad_nnt, 0.0, inv_denom_grad, 1, 0)

        # x_var_eps = self.inv_std**-2
        # xvar_grad = denom_grad[None, :, None, None] * -0.5*(xvar_eps)**(-1.5)
        x_var_eps_ref = self.inv_std
        xvar_grad = x_var_eps_ref # inplace for reuse. Invalidates self.inv_std
        
        pow_async(-0.5, -2.0*-1.5, xvar_grad) # fuse two powers from above
        prod_async(inv_denom_grad, xvar_grad)

        # Compute grad d(variance)/d(x)
        x_normalized_grad = self.grad_x_inv_denominator # zeros(A_nnt.shape)
        mean_grad = self.tmp_buff_channels_internal #  zeros([self.n_channels])
        
        # x_normalized_grad = 1.0/self.numel_in_channel*2*(A-xmean[None, :, None, None])
        # x_grad = (-1*x_normalized_grad.sum([0,2,3])[None, :, None,None]/self.numel_in_channel + x_normalized_grad)
        add_async(1.0/self.numel_in_channel*2, self.x_normalized_copy, 0.0, x_normalized_grad) # copy + scalar prod
        sum_fiber_async(-1.0/self.numel_in_channel, x_normalized_grad, 0.0, mean_grad, 1, 0)
        add_fiber_async(1.0, mean_grad, 1.0, x_normalized_grad, 1, 0)
        
        # d(variance)/d(inv_std) = (xvar_grad*x_normalized_grad)
        prod_fiber_async(xvar_grad, 1.0, x_normalized_grad, 1)
        return x_normalized_grad
        
    
    def backward_async(self):
        # Nominator part
        nominator_grad = self.tmp_buff_full # zeros(self.x.shape)
        
        # nominator_grad = self.grad*inv_denominator
        add_async(1.0, self.grad, 0.0, nominator_grad) # copy
        inv_denominator_ref = self.inv_std
        prod_fiber_async(inv_denominator_ref, 1.0, nominator_grad, 1)

        nominator_grad_x = self._compute_grad_normalized_input_over_x(nominator_grad)

        # Inversed denominator part
        inv_denominator_grad = self.tmp_buff_full # zeros(self.x.shape)
        
        # denominator_grad = self.grad*nominator
        nominator_ref = self.x_normalized_copy 
        add_async(1.0, nominator_ref, 0.0, inv_denominator_grad) # copy
        prod_async(self.grad, inv_denominator_grad)
        
        inv_denominator_grad_x = self._compute_grad_inv_std_over_x(self.x, inv_denominator_grad)

        # grad_x = nominator_grad_x + inv_denominator_grad_x
        add_async(1.0, inv_denominator_grad_x, 1.0, nominator_grad_x)
