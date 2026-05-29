/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor_graph/graph_ops.hh
 * TensorGraph operations.
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/tensor_graph/ops/add.hh>
#include <nntile/tensor_graph/ops/add_fiber.hh>
#include <nntile/tensor_graph/ops/add_fiber_inplace.hh>
#include <nntile/tensor_graph/ops/add_inplace.hh>
#include <nntile/tensor_graph/ops/add_slice.hh>
#include <nntile/tensor_graph/ops/add_slice_inplace.hh>
#include <nntile/tensor_graph/ops/clear.hh>
#include <nntile/tensor_graph/ops/conv2d_bwd_input_inplace.hh>
#include <nntile/tensor_graph/ops/conv2d_bwd_weight_inplace.hh>
#include <nntile/tensor_graph/ops/conv2d_inplace.hh>
#include <nntile/tensor_graph/ops/concat.hh>
#include <nntile/tensor_graph/ops/copy.hh>
#include <nntile/tensor_graph/ops/copy_intersection.hh>
#include <nntile/tensor_graph/ops/embedding.hh>
#include <nntile/tensor_graph/ops/embedding_backward.hh>
#include <nntile/tensor_graph/ops/fill.hh>
#include <nntile/tensor_graph/ops/flash_sdpa_bwd_cudnn.hh>
#include <nntile/tensor_graph/ops/flash_sdpa_fwd_cudnn.hh>
#include <nntile/tensor_graph/ops/gather.hh>
#include <nntile/tensor_graph/ops/gelu.hh>
#include <nntile/tensor_graph/ops/gelu_backward.hh>
#include <nntile/tensor_graph/ops/gelu_inplace.hh>
#include <nntile/tensor_graph/ops/gelutanh.hh>
#include <nntile/tensor_graph/ops/gelutanh_backward.hh>
#include <nntile/tensor_graph/ops/gelutanh_inplace.hh>
#include <nntile/tensor_graph/ops/gemm.hh>
#include <nntile/tensor_graph/ops/hypot.hh>
#include <nntile/tensor_graph/ops/hypot_inplace.hh>
#include <nntile/tensor_graph/ops/hypot_scalar_inverse.hh>
#include <nntile/tensor_graph/ops/log_scalar.hh>
#include <nntile/tensor_graph/ops/logsumexp.hh>
#include <nntile/tensor_graph/ops/mask_scalar.hh>
#include <nntile/tensor_graph/ops/maxsumexp.hh>
#include <nntile/tensor_graph/ops/multiply.hh>
#include <nntile/tensor_graph/ops/multiply_fiber.hh>
#include <nntile/tensor_graph/ops/multiply_fiber_inplace.hh>
#include <nntile/tensor_graph/ops/multiply_inplace.hh>
#include <nntile/tensor_graph/ops/multiply_slice.hh>
#include <nntile/tensor_graph/ops/norm.hh>
#include <nntile/tensor_graph/ops/norm_fiber.hh>
#include <nntile/tensor_graph/ops/norm_fiber_inplace.hh>
#include <nntile/tensor_graph/ops/norm_slice.hh>
#include <nntile/tensor_graph/ops/norm_slice_inplace.hh>
#include <nntile/tensor_graph/ops/pow.hh>
#include <nntile/tensor_graph/ops/randn.hh>
#include <nntile/tensor_graph/ops/relu.hh>
#include <nntile/tensor_graph/ops/sgd_step.hh>
#include <nntile/tensor_graph/ops/adam_step.hh>
#include <nntile/tensor_graph/ops/adamw_step.hh>
#include <nntile/tensor_graph/ops/relu_backward.hh>
#include <nntile/tensor_graph/ops/relu_inplace.hh>
#include <nntile/tensor_graph/ops/rope.hh>
#include <nntile/tensor_graph/ops/rope_backward.hh>
#include <nntile/tensor_graph/ops/scale.hh>
#include <nntile/tensor_graph/ops/scale_fiber.hh>
#include <nntile/tensor_graph/ops/scale_inplace.hh>
#include <nntile/tensor_graph/ops/scale_slice.hh>
#include <nntile/tensor_graph/ops/scatter.hh>
#include <nntile/tensor_graph/ops/silu.hh>
#include <nntile/tensor_graph/ops/silu_backward.hh>
#include <nntile/tensor_graph/ops/silu_inplace.hh>
#include <nntile/tensor_graph/ops/softmax.hh>
#include <nntile/tensor_graph/ops/softmax_inplace.hh>
#include <nntile/tensor_graph/ops/sqrt.hh>
#include <nntile/tensor_graph/ops/sqrt_inplace.hh>
#include <nntile/tensor_graph/ops/sum.hh>
#include <nntile/tensor_graph/ops/sum_fiber.hh>
#include <nntile/tensor_graph/ops/sum_slice.hh>
#include <nntile/tensor_graph/ops/subtract_indexed_outputs.hh>
#include <nntile/tensor_graph/ops/sumprod_fiber.hh>
#include <nntile/tensor_graph/ops/sumprod_slice.hh>
#include <nntile/tensor_graph/ops/total_sum_accum.hh>
#include <nntile/tensor_graph/ops/transpose.hh>
