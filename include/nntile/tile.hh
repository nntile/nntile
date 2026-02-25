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
#include <nntile/tile/tile.hh>

// Tile<T> operations
#include <nntile/tile/add_slice_inplace.hh>
#include <nntile/tile/add_slice.hh>
#include <nntile/tile/add_fiber_inplace.hh>
#include <nntile/tile/add_fiber.hh>
#include <nntile/tile/scale_fiber.hh>
#include <nntile/tile/multiply_slice.hh>
#include <nntile/tile/multiply_fiber_inplace.hh>
#include <nntile/tile/multiply_fiber.hh>
#include <nntile/tile/clear.hh>
#include <nntile/tile/copy.hh>
#include <nntile/tile/copy_intersection.hh>
#include <nntile/tile/gelu.hh>
#include <nntile/tile/gelu_inplace.hh>
#include <nntile/tile/gelutanh.hh>
#include <nntile/tile/gelutanh_inplace.hh>
#include <nntile/tile/gemm.hh>
#include <nntile/tile/multiply.hh>
#include <nntile/tile/add_inplace.hh>
#include <nntile/tile/multiply_inplace.hh>
#include <nntile/tile/randn.hh>
#include <nntile/tile/relu_inplace.hh>
#include <nntile/tile/relu.hh>
#include <nntile/tile/relu_backward.hh>
#include <nntile/tile/fill.hh>
#include <nntile/tile/sum_slice.hh>
#include <nntile/tile/sum_fiber.hh>
#include <nntile/tile/sum.hh>
#include <nntile/tile/norm_slice_inplace.hh>
#include <nntile/tile/norm_slice.hh>
#include <nntile/tile/pow.hh>
#include <nntile/tile/maxsumexp.hh>
#include <nntile/tile/softmax.hh>
#include <nntile/tile/softmax_inplace.hh>
#include <nntile/tile/sqrt.hh>
#include <nntile/tile/sqrt_inplace.hh>
#include <nntile/tile/sumprod_slice.hh>
#include <nntile/tile/sumprod_fiber.hh>
#include <nntile/tile/logsumexp.hh>
#include <nntile/tile/total_sum_accum.hh>
#include <nntile/tile/subtract_indexed_outputs.hh>
#include <nntile/tile/scale.hh>
#include <nntile/tile/scale_inplace.hh>
#include <nntile/tile/scale_slice.hh>
#include <nntile/tile/gelu_backward.hh>
#include <nntile/tile/gelutanh_backward.hh>
#include <nntile/tile/add.hh>
#include <nntile/tile/mask_scalar.hh>
#include <nntile/tile/hypot.hh>
#include <nntile/tile/hypot_inplace.hh>
#include <nntile/tile/adam_step.hh>
#include <nntile/tile/adamw_step.hh>
#include <nntile/tile/silu.hh>
#include <nntile/tile/silu_backward.hh>
#include <nntile/tile/rope.hh>
#include <nntile/tile/rope_backward.hh>
#include <nntile/tile/norm_fiber.hh>
#include <nntile/tile/norm_fiber_inplace.hh>
#include <nntile/tile/flash_sdpa_fwd_cudnn.hh>
#include <nntile/tile/flash_sdpa_bwd_cudnn.hh>
#include <nntile/tile/norm.hh>
#include <nntile/tile/transpose.hh>
#include <nntile/tile/hypot_scalar_inverse.hh>
#include <nntile/tile/log_scalar.hh>

//! @namespace nntile::tile
/*! This namespace holds high-level routines for Tile<T>
 * */
namespace nntile::tile
{

} // namespace nntile::tile
