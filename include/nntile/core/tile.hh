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
#include <nntile/core/tile/tile.hh>

// Tile<T> operations
#include <nntile/core/tile/add_slice_inplace.hh>
#include <nntile/core/tile/add_slice.hh>
#include <nntile/core/tile/add_fiber_inplace.hh>
#include <nntile/core/tile/add_fiber.hh>
#include <nntile/core/tile/scale_fiber.hh>
#include <nntile/core/tile/multiply_slice.hh>
#include <nntile/core/tile/multiply_fiber_inplace.hh>
#include <nntile/core/tile/multiply_fiber.hh>
#include <nntile/core/tile/clear.hh>
#include <nntile/core/tile/copy.hh>
#include <nntile/core/tile/copy_intersection.hh>
#include <nntile/core/tile/conv2d_inplace.hh>
#include <nntile/core/tile/conv2d_bwd_input_inplace.hh>
#include <nntile/core/tile/conv2d_bwd_weight_inplace.hh>
#include <nntile/core/tile/embedding.hh>
#include <nntile/core/tile/embedding_backward.hh>
#include <nntile/core/tile/gelu.hh>
#include <nntile/core/tile/gelu_inplace.hh>
#include <nntile/core/tile/gelutanh.hh>
#include <nntile/core/tile/gelutanh_inplace.hh>
#include <nntile/core/tile/gemm.hh>
#include <nntile/core/tile/multiply.hh>
#include <nntile/core/tile/add_inplace.hh>
#include <nntile/core/tile/multiply_inplace.hh>
#include <nntile/core/tile/randn.hh>
#include <nntile/core/tile/relu_inplace.hh>
#include <nntile/core/tile/relu.hh>
#include <nntile/core/tile/relu_backward.hh>
#include <nntile/core/tile/fill.hh>
#include <nntile/core/tile/sum_slice.hh>
#include <nntile/core/tile/sum_fiber.hh>
#include <nntile/core/tile/sum.hh>
#include <nntile/core/tile/norm_slice_inplace.hh>
#include <nntile/core/tile/norm_slice.hh>
#include <nntile/core/tile/pow.hh>
#include <nntile/core/tile/maxsumexp.hh>
#include <nntile/core/tile/softmax.hh>
#include <nntile/core/tile/softmax_inplace.hh>
#include <nntile/core/tile/sqrt.hh>
#include <nntile/core/tile/sqrt_inplace.hh>
#include <nntile/core/tile/sumprod_slice.hh>
#include <nntile/core/tile/sumprod_fiber.hh>
#include <nntile/core/tile/logsumexp.hh>
#include <nntile/core/tile/total_sum_accum.hh>
#include <nntile/core/tile/subtract_indexed_outputs.hh>
#include <nntile/core/tile/scale.hh>
#include <nntile/core/tile/scale_inplace.hh>
#include <nntile/core/tile/scale_slice.hh>
#include <nntile/core/tile/gelu_backward.hh>
#include <nntile/core/tile/gelutanh_backward.hh>
#include <nntile/core/tile/add.hh>
#include <nntile/core/tile/mask_scalar.hh>
#include <nntile/core/tile/hypot.hh>
#include <nntile/core/tile/hypot_inplace.hh>
#include <nntile/core/tile/adam_step.hh>
#include <nntile/core/tile/adamw_step.hh>
#include <nntile/core/tile/silu.hh>
#include <nntile/core/tile/silu_backward.hh>
#include <nntile/core/tile/rope.hh>
#include <nntile/core/tile/rope_backward.hh>
#include <nntile/core/tile/norm_fiber.hh>
#include <nntile/core/tile/norm_fiber_inplace.hh>
#include <nntile/core/tile/flash_sdpa_fwd_cudnn.hh>
#include <nntile/core/tile/flash_sdpa_bwd_cudnn.hh>
#include <nntile/core/tile/norm.hh>
#include <nntile/core/tile/transpose.hh>
#include <nntile/core/tile/hypot_scalar_inverse.hh>
#include <nntile/core/tile/log_scalar.hh>
#include <nntile/core/tile/isfinite.hh>

//! @namespace nntile::core::tile
/*! This namespace holds high-level routines for Tile<T>
 * */
namespace nntile::core::tile
{

} // namespace nntile::core::tile
