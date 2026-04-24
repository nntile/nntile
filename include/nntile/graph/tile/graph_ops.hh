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
#include <nntile/graph/tile/add.hh>
#include <nntile/graph/tile/add_fiber.hh>
#include <nntile/graph/tile/add_fiber_inplace.hh>
#include <nntile/graph/tile/add_inplace.hh>
#include <nntile/graph/tile/add_slice.hh>
#include <nntile/graph/tile/add_slice_inplace.hh>
#include <nntile/graph/tile/clear.hh>
#include <nntile/graph/tile/conv2d_bwd_input_inplace.hh>
#include <nntile/graph/tile/conv2d_bwd_weight_inplace.hh>
#include <nntile/graph/tile/conv2d_inplace.hh>
#include <nntile/graph/tile/copy.hh>
#include <nntile/graph/tile/copy_intersection.hh>
#include <nntile/graph/tile/embedding.hh>
#include <nntile/graph/tile/embedding_backward.hh>
#include <nntile/graph/tile/fill.hh>
#include <nntile/graph/tile/flash_sdpa_bwd_cudnn.hh>
#include <nntile/graph/tile/flash_sdpa_fwd_cudnn.hh>
#include <nntile/graph/tile/gelu.hh>
#include <nntile/graph/tile/gelu_backward.hh>
#include <nntile/graph/tile/gelu_inplace.hh>
#include <nntile/graph/tile/gelutanh.hh>
#include <nntile/graph/tile/gelutanh_backward.hh>
#include <nntile/graph/tile/gelutanh_inplace.hh>
#include <nntile/graph/tile/gemm.hh>
#include <nntile/graph/tile/hypot.hh>
#include <nntile/graph/tile/hypot_inplace.hh>
#include <nntile/graph/tile/hypot_scalar_inverse.hh>
#include <nntile/graph/tile/log_scalar.hh>
#include <nntile/graph/tile/logsumexp.hh>
#include <nntile/graph/tile/mask_scalar.hh>
#include <nntile/graph/tile/maxsumexp.hh>
#include <nntile/graph/tile/multiply.hh>
#include <nntile/graph/tile/multiply_fiber.hh>
#include <nntile/graph/tile/multiply_fiber_inplace.hh>
#include <nntile/graph/tile/multiply_inplace.hh>
#include <nntile/graph/tile/multiply_slice.hh>
#include <nntile/graph/tile/norm.hh>
#include <nntile/graph/tile/norm_fiber.hh>
#include <nntile/graph/tile/norm_fiber_inplace.hh>
#include <nntile/graph/tile/norm_slice.hh>
#include <nntile/graph/tile/norm_slice_inplace.hh>
#include <nntile/graph/tile/pow.hh>
#include <nntile/graph/tile/randn.hh>
#include <nntile/graph/tile/relu.hh>
#include <nntile/graph/tile/sgd_step.hh>
#include <nntile/graph/tile/adam_step.hh>
#include <nntile/graph/tile/adamw_step.hh>
#include <nntile/graph/tile/relu_backward.hh>
#include <nntile/graph/tile/relu_inplace.hh>
#include <nntile/graph/tile/rope.hh>
#include <nntile/graph/tile/rope_backward.hh>
#include <nntile/graph/tile/scale.hh>
#include <nntile/graph/tile/scale_fiber.hh>
#include <nntile/graph/tile/scale_inplace.hh>
#include <nntile/graph/tile/scale_slice.hh>
#include <nntile/graph/tile/silu.hh>
#include <nntile/graph/tile/silu_backward.hh>
#include <nntile/graph/tile/silu_inplace.hh>
#include <nntile/graph/tile/softmax.hh>
#include <nntile/graph/tile/softmax_inplace.hh>
#include <nntile/graph/tile/sqrt.hh>
#include <nntile/graph/tile/sqrt_inplace.hh>
#include <nntile/graph/tile/sum.hh>
#include <nntile/graph/tile/sum_fiber.hh>
#include <nntile/graph/tile/sum_slice.hh>
#include <nntile/graph/tile/subtract_indexed_outputs.hh>
#include <nntile/graph/tile/sumprod_fiber.hh>
#include <nntile/graph/tile/sumprod_slice.hh>
#include <nntile/graph/tile/total_sum_accum.hh>
#include <nntile/graph/tile/transpose.hh>
