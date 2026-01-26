/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/compiled_graph_ops.hh
 * Compiled graph operations.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/compiled_graph.hh>

#include <nntile/graph/compiled/add.hh>
#include <nntile/graph/compiled/add_fiber.hh>
#include <nntile/graph/compiled/add_fiber_inplace.hh>
#include <nntile/graph/compiled/add_inplace.hh>
#include <nntile/graph/compiled/add_slice.hh>
#include <nntile/graph/compiled/add_slice_inplace.hh>
#include <nntile/graph/compiled/adam_step.hh>
#include <nntile/graph/compiled/adamw_step.hh>
#include <nntile/graph/compiled/clear.hh>
#include <nntile/graph/compiled/conv2d_bwd_input_inplace.hh>
#include <nntile/graph/compiled/conv2d_bwd_weight_inplace.hh>
#include <nntile/graph/compiled/conv2d_inplace.hh>
#include <nntile/graph/compiled/copy.hh>
#include <nntile/graph/compiled/copy_intersection.hh>
#include <nntile/graph/compiled/embedding.hh>
#include <nntile/graph/compiled/embedding_backward.hh>
#include <nntile/graph/compiled/fill.hh>
#include <nntile/graph/compiled/flash_sdpa_bwd_cudnn.hh>
#include <nntile/graph/compiled/flash_sdpa_fwd_cudnn.hh>
#include <nntile/graph/compiled/gather.hh>
#include <nntile/graph/compiled/gelu.hh>
#include <nntile/graph/compiled/gelu_backward.hh>
#include <nntile/graph/compiled/gelu_inplace.hh>
#include <nntile/graph/compiled/gelutanh.hh>
#include <nntile/graph/compiled/gelutanh_backward.hh>
#include <nntile/graph/compiled/gelutanh_inplace.hh>
#include <nntile/graph/compiled/gemm.hh>
#include <nntile/graph/compiled/hypot.hh>
#include <nntile/graph/compiled/hypot_inplace.hh>
#include <nntile/graph/compiled/hypot_scalar_inverse.hh>
#include <nntile/graph/compiled/log_scalar.hh>
#include <nntile/graph/compiled/logsumexp.hh>
#include <nntile/graph/compiled/mask_scalar.hh>
#include <nntile/graph/compiled/maxsumexp.hh>
#include <nntile/graph/compiled/multiply.hh>
#include <nntile/graph/compiled/multiply_fiber.hh>
#include <nntile/graph/compiled/multiply_fiber_inplace.hh>
#include <nntile/graph/compiled/multiply_inplace.hh>
#include <nntile/graph/compiled/multiply_slice.hh>
#include <nntile/graph/compiled/norm.hh>
#include <nntile/graph/compiled/norm_fiber.hh>
#include <nntile/graph/compiled/norm_fiber_inplace.hh>
#include <nntile/graph/compiled/norm_slice.hh>
#include <nntile/graph/compiled/norm_slice_inplace.hh>
#include <nntile/graph/compiled/pow.hh>
#include <nntile/graph/compiled/pow_inplace.hh>
#include <nntile/graph/compiled/randn.hh>
#include <nntile/graph/compiled/relu.hh>
#include <nntile/graph/compiled/relu_backward.hh>
#include <nntile/graph/compiled/relu_inplace.hh>
#include <nntile/graph/compiled/rope.hh>
#include <nntile/graph/compiled/rope_backward.hh>
#include <nntile/graph/compiled/scale.hh>
#include <nntile/graph/compiled/scale_fiber.hh>
#include <nntile/graph/compiled/scale_inplace.hh>
#include <nntile/graph/compiled/scale_slice.hh>
#include <nntile/graph/compiled/scatter.hh>
#include <nntile/graph/compiled/sgd_step.hh>
#include <nntile/graph/compiled/silu.hh>
#include <nntile/graph/compiled/silu_backward.hh>
#include <nntile/graph/compiled/silu_inplace.hh>
#include <nntile/graph/compiled/softmax.hh>
#include <nntile/graph/compiled/softmax_inplace.hh>
#include <nntile/graph/compiled/sqrt.hh>
#include <nntile/graph/compiled/sqrt_inplace.hh>
#include <nntile/graph/compiled/subtract_indexed_outputs.hh>
#include <nntile/graph/compiled/sum.hh>
#include <nntile/graph/compiled/sum_fiber.hh>
#include <nntile/graph/compiled/sum_slice.hh>
#include <nntile/graph/compiled/sumprod_fiber.hh>
#include <nntile/graph/compiled/sumprod_slice.hh>
#include <nntile/graph/compiled/total_sum_accum.hh>
#include <nntile/graph/compiled/transpose.hh>
