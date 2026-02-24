/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical_graph_ops.hh
 * Logical graph operations.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>
#include <nntile/graph/logical/gelu.hh>
#include <nntile/graph/logical/gelu_inplace.hh>
#include <nntile/graph/logical/gelu_backward.hh>
#include <nntile/graph/logical/gelutanh.hh>
#include <nntile/graph/logical/gelutanh_inplace.hh>
#include <nntile/graph/logical/gelutanh_backward.hh>
#include <nntile/graph/logical/relu.hh>
#include <nntile/graph/logical/relu_inplace.hh>
#include <nntile/graph/logical/relu_backward.hh>
#include <nntile/graph/logical/silu.hh>
#include <nntile/graph/logical/silu_inplace.hh>
#include <nntile/graph/logical/silu_backward.hh>
#include <nntile/graph/logical/sqrt.hh>
#include <nntile/graph/logical/add.hh>
#include <nntile/graph/logical/add_inplace.hh>
#include <nntile/graph/logical/multiply.hh>
#include <nntile/graph/logical/multiply_inplace.hh>
#include <nntile/graph/logical/multiply_slice.hh>
#include <nntile/graph/logical/clear.hh>
#include <nntile/graph/logical/softmax.hh>
#include <nntile/graph/logical/softmax_inplace.hh>
#include <nntile/graph/logical/pow.hh>
#include <nntile/graph/logical/pow_inplace.hh>
#include <nntile/graph/logical/hypot.hh>
#include <nntile/graph/logical/hypot_inplace.hh>
#include <nntile/graph/logical/hypot_scalar_inverse.hh>
#include <nntile/graph/logical/fill.hh>
#include <nntile/graph/logical/embedding.hh>
#include <nntile/graph/logical/embedding_backward.hh>
#include <nntile/graph/logical/copy.hh>
#include <nntile/graph/logical/transpose.hh>
// #include <nntile/graph/logical/scatter.hh>
// #include <nntile/graph/logical/copy_intersection.hh>
#include <nntile/graph/logical/log_scalar.hh>
#include <nntile/graph/logical/mask_scalar.hh>
#include <nntile/graph/logical/subtract_indexed_outputs.hh>
// #include <nntile/graph/logical/gather.hh>
#include <nntile/graph/logical/gemm.hh>
#include <nntile/graph/logical/scale_fiber.hh>
#include <nntile/graph/logical/scale_slice.hh>
#include <nntile/graph/logical/scale.hh>
#include <nntile/graph/logical/scale_inplace.hh>
#include <nntile/graph/logical/randn.hh>
#include <nntile/graph/logical/sum.hh>
#include <nntile/graph/logical/sum_fiber.hh>
#include <nntile/graph/logical/sum_slice.hh>
#include <nntile/graph/logical/norm.hh>
#include <nntile/graph/logical/logsumexp.hh>
#include <nntile/graph/logical/maxsumexp.hh>
#include <nntile/graph/logical/total_sum_accum.hh>
#include <nntile/graph/logical/sumprod_fiber.hh>
#include <nntile/graph/logical/sumprod_slice.hh>
#include <nntile/graph/logical/norm_fiber.hh>
#include <nntile/graph/logical/norm_fiber_inplace.hh>
#include <nntile/graph/logical/norm_slice.hh>
#include <nntile/graph/logical/norm_slice_inplace.hh>
#include <nntile/graph/logical/sgd_step.hh>
#include <nntile/graph/logical/adam_step.hh>
#include <nntile/graph/logical/adamw_step.hh>
#include <nntile/graph/logical/flash_sdpa_fwd_cudnn.hh>
#include <nntile/graph/logical/flash_sdpa_bwd_cudnn.hh>
#include <nntile/graph/logical/rope.hh>
#include <nntile/graph/logical/rope_backward.hh>
#include <nntile/graph/logical/conv2d_inplace.hh>
#include <nntile/graph/logical/conv2d_bwd_input_inplace.hh>
#include <nntile/graph/logical/conv2d_bwd_weight_inplace.hh>
#include <nntile/graph/logical/add_fiber.hh>
#include <nntile/graph/logical/add_fiber_inplace.hh>
#include <nntile/graph/logical/add_slice.hh>
#include <nntile/graph/logical/add_slice_inplace.hh>
#include <nntile/graph/logical/multiply_fiber.hh>
#include <nntile/graph/logical/multiply_fiber_inplace.hh>
#include <nntile/graph/logical/sqrt_inplace.hh>

namespace nntile::graph
{
} // namespace nntile::graph
