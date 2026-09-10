/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/graph_ops.hh
 * TileGraph operations.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/defs.h>

// NNTile headers
#include <nntile/tile/ops/add.hh>
#include <nntile/tile/ops/add_fiber.hh>
#include <nntile/tile/ops/add_fiber_inplace.hh>
#include <nntile/tile/ops/add_inplace.hh>
#include <nntile/tile/ops/add_slice.hh>
#include <nntile/tile/ops/add_slice_inplace.hh>
#include <nntile/tile/ops/clear.hh>
#include <nntile/tile/ops/invalidate.hh>
#include <nntile/tile/ops/unregister.hh>
#include <nntile/tile/ops/conv2d_bwd_input_inplace.hh>
#include <nntile/tile/ops/conv2d_bwd_weight_inplace.hh>
#include <nntile/tile/ops/conv2d_inplace.hh>
#include <nntile/tile/ops/copy.hh>
#include <nntile/tile/ops/copy_intersection.hh>
#include <nntile/tile/ops/embedding.hh>
#include <nntile/tile/ops/embedding_backward.hh>
#include <nntile/tile/ops/fill.hh>
#ifdef NNTILE_USE_FLASH_SDPA
#include <nntile/tile/ops/flash_sdpa_bwd_cudnn.hh>
#include <nntile/tile/ops/flash_sdpa_fwd_cudnn.hh>
#endif
#include <nntile/tile/ops/gelu.hh>
#include <nntile/tile/ops/gelu_backward.hh>
#include <nntile/tile/ops/gelu_inplace.hh>
#include <nntile/tile/ops/gelutanh.hh>
#include <nntile/tile/ops/gelutanh_backward.hh>
#include <nntile/tile/ops/gelutanh_inplace.hh>
#include <nntile/tile/ops/gemm.hh>
#include <nntile/tile/ops/hypot.hh>
#include <nntile/tile/ops/hypot_inplace.hh>
#include <nntile/tile/ops/hypot_scalar_inverse.hh>
#include <nntile/tile/ops/log_scalar.hh>
#include <nntile/tile/ops/logsumexp.hh>
#include <nntile/tile/ops/mask_scalar.hh>
#include <nntile/tile/ops/maxsumexp.hh>
#include <nntile/tile/ops/multiply.hh>
#include <nntile/tile/ops/multiply_fiber.hh>
#include <nntile/tile/ops/multiply_fiber_inplace.hh>
#include <nntile/tile/ops/multiply_inplace.hh>
#include <nntile/tile/ops/multiply_slice.hh>
#include <nntile/tile/ops/norm.hh>
#include <nntile/tile/ops/norm_fiber.hh>
#include <nntile/tile/ops/norm_fiber_inplace.hh>
#include <nntile/tile/ops/norm_slice.hh>
#include <nntile/tile/ops/norm_slice_inplace.hh>
#include <nntile/tile/ops/pow.hh>
#include <nntile/tile/ops/randn.hh>
#include <nntile/tile/ops/relu.hh>
#include <nntile/tile/ops/sgd_step.hh>
#include <nntile/tile/ops/adam_step.hh>
#include <nntile/tile/ops/adamw_step.hh>
#include <nntile/tile/ops/relu_backward.hh>
#include <nntile/tile/ops/relu_inplace.hh>
#include <nntile/tile/ops/rope.hh>
#include <nntile/tile/ops/rope_backward.hh>
#include <nntile/tile/ops/scale.hh>
#include <nntile/tile/ops/scale_fiber.hh>
#include <nntile/tile/ops/scale_inplace.hh>
#include <nntile/tile/ops/scale_slice.hh>
#include <nntile/tile/ops/silu.hh>
#include <nntile/tile/ops/silu_backward.hh>
#include <nntile/tile/ops/silu_inplace.hh>
#include <nntile/tile/ops/softmax.hh>
#include <nntile/tile/ops/softmax_inplace.hh>
#include <nntile/tile/ops/sqrt.hh>
#include <nntile/tile/ops/sqrt_inplace.hh>
#include <nntile/tile/ops/sum.hh>
#include <nntile/tile/ops/sum_fiber.hh>
#include <nntile/tile/ops/sum_slice.hh>
#include <nntile/tile/ops/subtract_indexed_outputs.hh>
#include <nntile/tile/ops/sumprod_fiber.hh>
#include <nntile/tile/ops/sumprod_slice.hh>
#include <nntile/tile/ops/total_sum_accum.hh>
#include <nntile/tile/ops/transpose.hh>
