/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor.hh
 * Header for Tensor<T> class with corresponding operations
 *
 * @version 1.1.0
 * */

#pragma once

// Get Tensor<T> class
#include <nntile/core/tensor/tensor.hh>
// MPI distributions
#include <nntile/core/tensor/distributions.hh>

// Tensor operations
#include <nntile/core/tensor/add_slice_inplace.hh>
#include <nntile/core/tensor/add_slice.hh>
#include <nntile/core/tensor/add_fiber_inplace.hh>
#include <nntile/core/tensor/add_fiber.hh>
#include <nntile/core/tensor/scale_fiber.hh>
#include <nntile/core/tensor/multiply_slice.hh>
#include <nntile/core/tensor/multiply_fiber_inplace.hh>
#include <nntile/core/tensor/multiply_fiber.hh>
#include <nntile/core/tensor/clear.hh>
#include <nntile/core/tensor/copy.hh>
#include <nntile/core/tensor/copy_intersection.hh>
#include <nntile/core/tensor/gather.hh>
#include <nntile/core/tensor/gelu.hh>
#include <nntile/core/tensor/gelu_inplace.hh>
#include <nntile/core/tensor/gelutanh.hh>
#include <nntile/core/tensor/gelutanh_inplace.hh>
#include <nntile/core/tensor/gemm.hh>
#include <nntile/core/tensor/multiply.hh>
#include <nntile/core/tensor/multiply_inplace.hh>
#include <nntile/core/tensor/randn.hh>
#include <nntile/core/tensor/relu_inplace.hh>
#include <nntile/core/tensor/relu.hh>
#include <nntile/core/tensor/relu_backward.hh>
#include <nntile/core/tensor/scatter.hh>
#include <nntile/core/tensor/fill.hh>
#include <nntile/core/tensor/sum_slice.hh>
#include <nntile/core/tensor/sum_fiber.hh>
#include <nntile/core/tensor/sum.hh>
#include <nntile/core/tensor/norm_slice_inplace.hh>
#include <nntile/core/tensor/norm_slice.hh>
#include <nntile/core/tensor/pow.hh>
#include <nntile/core/tensor/maxsumexp.hh>
#include <nntile/core/tensor/softmax.hh>
#include <nntile/core/tensor/softmax_inplace.hh>
#include <nntile/core/tensor/sqrt.hh>
#include <nntile/core/tensor/sqrt_inplace.hh>
#include <nntile/core/tensor/sumprod_slice.hh>
#include <nntile/core/tensor/sumprod_fiber.hh>
#include <nntile/core/tensor/logsumexp.hh>
#include <nntile/core/tensor/total_sum_accum.hh>
#include <nntile/core/tensor/subtract_indexed_outputs.hh>
#include <nntile/core/tensor/scale.hh>
#include <nntile/core/tensor/scale_inplace.hh>
#include <nntile/core/tensor/scale_slice.hh>
#include <nntile/core/tensor/gelu_backward.hh>
#include <nntile/core/tensor/gelutanh_backward.hh>
#include <nntile/core/tensor/add.hh>
#include <nntile/core/tensor/add_inplace.hh>
#include <nntile/core/tensor/embedding.hh>
#include <nntile/core/tensor/embedding_backward.hh>
#include <nntile/core/tensor/mask_scalar.hh>
#include <nntile/core/tensor/hypot.hh>
#include <nntile/core/tensor/hypot_inplace.hh>
#include <nntile/core/tensor/hypot_scalar_inverse.hh>
#include <nntile/core/tensor/adam_step.hh>
#include <nntile/core/tensor/adamw_step.hh>
#include <nntile/core/tensor/sgd_step.hh>
#include <nntile/core/tensor/transpose.hh>
#include <nntile/core/tensor/silu.hh>
#include <nntile/core/tensor/silu_backward.hh>
#include <nntile/core/tensor/silu_inplace.hh>
#include <nntile/core/tensor/conv2d_inplace.hh>
#include <nntile/core/tensor/conv2d_bwd_input_inplace.hh>
#include <nntile/core/tensor/conv2d_bwd_weight_inplace.hh>
#include <nntile/core/tensor/rope.hh>
#include <nntile/core/tensor/rope_backward.hh>
#include <nntile/core/tensor/norm_fiber_inplace.hh>
#include <nntile/core/tensor/norm_fiber.hh>
#include <nntile/core/tensor/norm.hh>
#include <nntile/core/tensor/log_scalar.hh>
#include <nntile/core/tensor/flash_sdpa_fwd_cudnn.hh>
#include <nntile/core/tensor/flash_sdpa_bwd_cudnn.hh>
#include <nntile/core/tensor/isfinite.hh>

//! @namespace nntile::core::tensor
/*! This namespace holds high-level routines for Tensor<T>
 * */
namespace nntile::core::tensor
{

} // namespace nntile::core::tensor
