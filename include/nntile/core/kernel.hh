/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel.hh
 * General info about namespace nntile::core::kernel
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/core/kernel/accumulate_maxsumexp.hh>
#include <nntile/core/kernel/accumulate_attn_output.hh>
#include <nntile/core/kernel/add_slice_inplace.hh>
#include <nntile/core/kernel/add_slice.hh>
#include <nntile/core/kernel/add_fiber_inplace.hh>
#include <nntile/core/kernel/add_fiber.hh>
#include <nntile/core/kernel/scale_fiber.hh>
#include <nntile/core/kernel/multiply_slice.hh>
#include <nntile/core/kernel/multiply_fiber_inplace.hh>
#include <nntile/core/kernel/multiply_fiber.hh>
#include <nntile/core/kernel/gelu.hh>
#include <nntile/core/kernel/gelu_inplace.hh>
#include <nntile/core/kernel/gelutanh.hh>
#include <nntile/core/kernel/gelutanh_inplace.hh>
#include <nntile/core/kernel/hypot.hh>
#include <nntile/core/kernel/hypot_inplace.hh>
#include <nntile/core/kernel/hypot_scalar_inverse.hh>
#include <nntile/core/kernel/multiply.hh>
#include <nntile/core/kernel/multiply_inplace.hh>
#include <nntile/core/kernel/randn.hh>
#include <nntile/core/kernel/relu_inplace.hh>
#include <nntile/core/kernel/relu.hh>
#include <nntile/core/kernel/relu_backward.hh>
#include <nntile/core/kernel/subcopy.hh>
#include <nntile/core/kernel/fill.hh>
#include <nntile/core/kernel/sum_slice.hh>
#include <nntile/core/kernel/sum_fiber.hh>
#include <nntile/core/kernel/sum.hh>
#include <nntile/core/kernel/norm_slice_inplace.hh>
#include <nntile/core/kernel/norm_slice.hh>
#include <nntile/core/kernel/norm.hh>
#include <nntile/core/kernel/pow.hh>
#include <nntile/core/kernel/maxsumexp.hh>
#include <nntile/core/kernel/softmax.hh>
#include <nntile/core/kernel/softmax_inplace.hh>
#include <nntile/core/kernel/sqrt.hh>
#include <nntile/core/kernel/sqrt_inplace.hh>
#include <nntile/core/kernel/sumprod_slice.hh>
#include <nntile/core/kernel/sumprod_fiber.hh>
#include <nntile/core/kernel/logsumexp.hh>
#include <nntile/core/kernel/total_sum_accum.hh>
#include <nntile/core/kernel/subtract_indexed_outputs.hh>
#include <nntile/core/kernel/gelu_backward.hh>
#include <nntile/core/kernel/gelutanh_backward.hh>
#include <nntile/core/kernel/add.hh>
#include <nntile/core/kernel/add_inplace.hh>
#include <nntile/core/kernel/embedding.hh>
#include <nntile/core/kernel/embedding_backward.hh>
#include <nntile/core/kernel/mask_scalar.hh>
#include <nntile/core/kernel/scale.hh>
#include <nntile/core/kernel/scale_inplace.hh>
#include <nntile/core/kernel/scale_slice.hh>
#include <nntile/core/kernel/adam_step.hh>
#include <nntile/core/kernel/adamw_step.hh>
#include <nntile/core/kernel/sgd_step.hh>
#include <nntile/core/kernel/transpose.hh>
#include <nntile/core/kernel/silu.hh>
#include <nntile/core/kernel/silu_backward.hh>
#include <nntile/core/kernel/conv2d_inplace.hh>
#include <nntile/core/kernel/conv2d_bwd_input_inplace.hh>
#include <nntile/core/kernel/conv2d_bwd_weight_inplace.hh>
#include <nntile/core/kernel/rope.hh>
#include <nntile/core/kernel/rope_backward.hh>
#include <nntile/core/kernel/norm_fiber_inplace.hh>
#include <nntile/core/kernel/norm_fiber.hh>
#include <nntile/core/kernel/flash_sdpa_fwd_cudnn.hh>
#include <nntile/core/kernel/flash_sdpa_bwd_cudnn.hh>
#include <nntile/core/kernel/isfinite.hh>


//! @namespace nntile::core::kernel
/*! This namespace holds low-level routines for codelets
 * */
namespace nntile::core::kernel
{

} // namespace nntile::core::kernel
