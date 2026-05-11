/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/graph_ops.hh
 * TileGraph operations.
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/graph/tile/ops/add.hh>
#include <nntile/graph/tile/ops/add_fiber.hh>
#include <nntile/graph/tile/ops/add_fiber_inplace.hh>
#include <nntile/graph/tile/ops/add_inplace.hh>
#include <nntile/graph/tile/ops/add_slice.hh>
#include <nntile/graph/tile/ops/add_slice_inplace.hh>
#include <nntile/graph/tile/ops/clear.hh>
#include <nntile/graph/tile/ops/conv2d_bwd_input_inplace.hh>
#include <nntile/graph/tile/ops/conv2d_bwd_weight_inplace.hh>
#include <nntile/graph/tile/ops/conv2d_inplace.hh>
#include <nntile/graph/tile/ops/copy.hh>
#include <nntile/graph/tile/ops/copy_intersection.hh>
#include <nntile/graph/tile/ops/embedding.hh>
#include <nntile/graph/tile/ops/embedding_backward.hh>
#include <nntile/graph/tile/ops/fill.hh>
#include <nntile/graph/tile/ops/flash_sdpa_bwd_cudnn.hh>
#include <nntile/graph/tile/ops/flash_sdpa_fwd_cudnn.hh>
#include <nntile/graph/tile/ops/gelu.hh>
#include <nntile/graph/tile/ops/gelu_backward.hh>
#include <nntile/graph/tile/ops/gelu_inplace.hh>
#include <nntile/graph/tile/ops/gelutanh.hh>
#include <nntile/graph/tile/ops/gelutanh_backward.hh>
#include <nntile/graph/tile/ops/gelutanh_inplace.hh>
#include <nntile/graph/tile/ops/gemm.hh>
#include <nntile/graph/tile/ops/hypot.hh>
#include <nntile/graph/tile/ops/hypot_inplace.hh>
#include <nntile/graph/tile/ops/hypot_scalar_inverse.hh>
#include <nntile/graph/tile/ops/log_scalar.hh>
#include <nntile/graph/tile/ops/logsumexp.hh>
#include <nntile/graph/tile/ops/mask_scalar.hh>
#include <nntile/graph/tile/ops/maxsumexp.hh>
#include <nntile/graph/tile/ops/multiply.hh>
#include <nntile/graph/tile/ops/multiply_fiber.hh>
#include <nntile/graph/tile/ops/multiply_fiber_inplace.hh>
#include <nntile/graph/tile/ops/multiply_inplace.hh>
#include <nntile/graph/tile/ops/multiply_slice.hh>
#include <nntile/graph/tile/ops/norm.hh>
#include <nntile/graph/tile/ops/norm_fiber.hh>
#include <nntile/graph/tile/ops/norm_fiber_inplace.hh>
#include <nntile/graph/tile/ops/norm_slice.hh>
#include <nntile/graph/tile/ops/norm_slice_inplace.hh>
#include <nntile/graph/tile/ops/pow.hh>
#include <nntile/graph/tile/ops/randn.hh>
#include <nntile/graph/tile/ops/relu.hh>
#include <nntile/graph/tile/ops/sgd_step.hh>
#include <nntile/graph/tile/ops/adam_step.hh>
#include <nntile/graph/tile/ops/adamw_step.hh>
#include <nntile/graph/tile/ops/relu_backward.hh>
#include <nntile/graph/tile/ops/relu_inplace.hh>
#include <nntile/graph/tile/ops/rope.hh>
#include <nntile/graph/tile/ops/rope_backward.hh>
#include <nntile/graph/tile/ops/scale.hh>
#include <nntile/graph/tile/ops/scale_fiber.hh>
#include <nntile/graph/tile/ops/scale_inplace.hh>
#include <nntile/graph/tile/ops/scale_slice.hh>
#include <nntile/graph/tile/ops/silu.hh>
#include <nntile/graph/tile/ops/silu_backward.hh>
#include <nntile/graph/tile/ops/silu_inplace.hh>
#include <nntile/graph/tile/ops/softmax.hh>
#include <nntile/graph/tile/ops/softmax_inplace.hh>
#include <nntile/graph/tile/ops/sqrt.hh>
#include <nntile/graph/tile/ops/sqrt_inplace.hh>
#include <nntile/graph/tile/ops/sum.hh>
#include <nntile/graph/tile/ops/sum_fiber.hh>
#include <nntile/graph/tile/ops/sum_slice.hh>
#include <nntile/graph/tile/ops/subtract_indexed_outputs.hh>
#include <nntile/graph/tile/ops/sumprod_fiber.hh>
#include <nntile/graph/tile/ops/sumprod_slice.hh>
#include <nntile/graph/tile/ops/total_sum_accum.hh>
#include <nntile/graph/tile/ops/transpose.hh>
