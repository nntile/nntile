/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile.hh
 * Header for Tile<T> class with corresponding operations
 *
 * @version 1.1.0
 * */

#pragma once

// Get Tile<T> class
#include <nntile/core/tile.hh>

// Tile<T> operations
#include <nntile/core/add_slice_inplace.hh>
#include <nntile/core/add_slice.hh>
#include <nntile/core/add_fiber_inplace.hh>
#include <nntile/core/add_fiber.hh>
#include <nntile/core/scale_fiber.hh>
#include <nntile/core/multiply_slice.hh>
#include <nntile/core/multiply_fiber_inplace.hh>
#include <nntile/core/multiply_fiber.hh>
#include <nntile/core/clear.hh>
#include <nntile/core/copy.hh>
#include <nntile/core/copy_intersection.hh>
#include <nntile/core/conv2d_inplace.hh>
#include <nntile/core/conv2d_bwd_input_inplace.hh>
#include <nntile/core/conv2d_bwd_weight_inplace.hh>
#include <nntile/core/embedding.hh>
#include <nntile/core/embedding_backward.hh>
#include <nntile/core/gelu.hh>
#include <nntile/core/gelu_inplace.hh>
#include <nntile/core/gelutanh.hh>
#include <nntile/core/gelutanh_inplace.hh>
#include <nntile/core/gemm.hh>
#include <nntile/core/multiply.hh>
#include <nntile/core/add_inplace.hh>
#include <nntile/core/multiply_inplace.hh>
#include <nntile/core/randn.hh>
#include <nntile/core/relu_inplace.hh>
#include <nntile/core/relu.hh>
#include <nntile/core/relu_backward.hh>
#include <nntile/core/fill.hh>
#include <nntile/core/sum_slice.hh>
#include <nntile/core/sum_fiber.hh>
#include <nntile/core/sum.hh>
#include <nntile/core/norm_slice_inplace.hh>
#include <nntile/core/norm_slice.hh>
#include <nntile/core/pow.hh>
#include <nntile/core/maxsumexp.hh>
#include <nntile/core/softmax.hh>
#include <nntile/core/softmax_inplace.hh>
#include <nntile/core/sqrt.hh>
#include <nntile/core/sqrt_inplace.hh>
#include <nntile/core/sumprod_slice.hh>
#include <nntile/core/sumprod_fiber.hh>
#include <nntile/core/logsumexp.hh>
#include <nntile/core/total_sum_accum.hh>
#include <nntile/core/subtract_indexed_outputs.hh>
#include <nntile/core/scale.hh>
#include <nntile/core/scale_inplace.hh>
#include <nntile/core/scale_slice.hh>
#include <nntile/core/gelu_backward.hh>
#include <nntile/core/gelutanh_backward.hh>
#include <nntile/core/add.hh>
#include <nntile/core/mask_scalar.hh>
#include <nntile/core/hypot.hh>
#include <nntile/core/hypot_inplace.hh>
#include <nntile/core/adam_step.hh>
#include <nntile/core/adamw_step.hh>
#include <nntile/core/silu.hh>
#include <nntile/core/silu_backward.hh>
#include <nntile/core/rope.hh>
#include <nntile/core/rope_backward.hh>
#include <nntile/core/norm_fiber.hh>
#include <nntile/core/norm_fiber_inplace.hh>
#include <nntile/core/flash_sdpa_fwd_cudnn.hh>
#include <nntile/core/flash_sdpa_bwd_cudnn.hh>
#include <nntile/core/norm.hh>
#include <nntile/core/transpose.hh>
#include <nntile/core/hypot_scalar_inverse.hh>
#include <nntile/core/log_scalar.hh>
#include <nntile/core/isfinite.hh>

//! @namespace nntile::core
/*! This namespace holds high-level routines for Tile<T>
 * */
namespace nntile::core
{

} // namespace nntile::core
