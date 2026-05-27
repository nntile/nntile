/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu.hh
 * StarPU wrappers for data handles and low-level kernels
 *
 * @version 1.1.0
 * */

#pragma once

// Some definitions, that will be refactored later
#include <nntile/core/starpu/config.hh>

// StarPU wrappers for low-level kernels
#include <nntile/core/starpu/accumulate.hh>
#include <nntile/core/starpu/accumulate_hypot.hh>
#include <nntile/core/starpu/accumulate_maxsumexp.hh>
#include <nntile/core/starpu/adam_step.hh>
#include <nntile/core/starpu/adamw_step.hh>
#include <nntile/core/starpu/add.hh>
#include <nntile/core/starpu/add_fiber.hh>
#include <nntile/core/starpu/add_fiber_inplace.hh>
#include <nntile/core/starpu/scale_fiber.hh>
#include <nntile/core/starpu/add_inplace.hh>
#include <nntile/core/starpu/add_slice.hh>
#include <nntile/core/starpu/add_slice_inplace.hh>
#include <nntile/core/starpu/clear.hh>
#include <nntile/core/starpu/conv2d_bwd_input_inplace.hh>
#include <nntile/core/starpu/conv2d_bwd_weight_inplace.hh>
#include <nntile/core/starpu/conv2d_inplace.hh>
#include <nntile/core/starpu/copy.hh>
#include <nntile/core/starpu/embedding.hh>
#include <nntile/core/starpu/embedding_backward.hh>
#include <nntile/core/starpu/fill.hh>
#include <nntile/core/starpu/gelu.hh>
#include <nntile/core/starpu/gelu_inplace.hh>
#include <nntile/core/starpu/gelu_backward.hh>
#include <nntile/core/starpu/gelutanh.hh>
#include <nntile/core/starpu/gelutanh_backward.hh>
#include <nntile/core/starpu/gelutanh_inplace.hh>
#include <nntile/core/starpu/gemm.hh>
#include <nntile/core/starpu/hypot.hh>
#include <nntile/core/starpu/hypot_inplace.hh>
#include <nntile/core/starpu/hypot_scalar_inverse.hh>
#include <nntile/core/starpu/log_scalar.hh>
#include <nntile/core/starpu/logsumexp.hh>
#include <nntile/core/starpu/mask_scalar.hh>
#include <nntile/core/starpu/maxsumexp.hh>
#include <nntile/core/starpu/norm_fiber.hh>
#include <nntile/core/starpu/norm_fiber_inplace.hh>
#include <nntile/core/starpu/norm_slice_inplace.hh>
#include <nntile/core/starpu/norm_slice.hh>
#include <nntile/core/starpu/pow.hh>
#include <nntile/core/starpu/multiply.hh>
#include <nntile/core/starpu/multiply_fiber_inplace.hh>
#include <nntile/core/starpu/multiply_fiber.hh>
#include <nntile/core/starpu/multiply_inplace.hh>
#include <nntile/core/starpu/multiply_slice.hh>
#include <nntile/core/starpu/randn.hh>
#include <nntile/core/starpu/relu_inplace.hh>
#include <nntile/core/starpu/relu_backward.hh>
#include <nntile/core/starpu/relu.hh>
#include <nntile/core/starpu/rope.hh>
#include <nntile/core/starpu/rope_backward.hh>
#include <nntile/core/starpu/scale.hh>
#include <nntile/core/starpu/scale_inplace.hh>
#include <nntile/core/starpu/scale_slice.hh>
#include <nntile/core/starpu/silu_backward.hh>
#include <nntile/core/starpu/silu.hh>
#include <nntile/core/starpu/silu_inplace.hh>
#include <nntile/core/starpu/softmax.hh>
#include <nntile/core/starpu/softmax_inplace.hh>
#include <nntile/core/starpu/sqrt.hh>
#include <nntile/core/starpu/sqrt_inplace.hh>
#include <nntile/core/starpu/subcopy.hh>
#include <nntile/core/starpu/subtract_indexed_outputs.hh>
#include <nntile/core/starpu/sum_fiber.hh>
#include <nntile/core/starpu/sum_slice.hh>
#include <nntile/core/starpu/sum.hh>
#include <nntile/core/starpu/sumprod_fiber.hh>
#include <nntile/core/starpu/sumprod_slice.hh>
#include <nntile/core/starpu/total_sum_accum.hh>
#include <nntile/core/starpu/transpose.hh>
#include <nntile/core/starpu/flash_sdpa_fwd_cudnn.hh>
#include <nntile/core/starpu/flash_sdpa_bwd_cudnn.hh>
#include <nntile/core/starpu/isfinite.hh>


//! @namespace nntile::core::starpu
/*! This namespace holds StarPU wrappers
 * */
namespace nntile::core::starpu
{

} // namespace nntile::core::starpu
