/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/graph_ops.hh
 * TensorGraph operations.
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/graph/tensor/ops/add.hh>
#include <nntile/graph/tensor/ops/add_fiber.hh>
#include <nntile/graph/tensor/ops/add_fiber_inplace.hh>
#include <nntile/graph/tensor/ops/add_inplace.hh>
#include <nntile/graph/tensor/ops/add_slice.hh>
#include <nntile/graph/tensor/ops/add_slice_inplace.hh>
#include <nntile/graph/tensor/ops/clear.hh>
#include <nntile/graph/tensor/ops/conv2d_bwd_input_inplace.hh>
#include <nntile/graph/tensor/ops/conv2d_bwd_weight_inplace.hh>
#include <nntile/graph/tensor/ops/conv2d_inplace.hh>
#include <nntile/graph/tensor/ops/concat.hh>
#include <nntile/graph/tensor/ops/copy.hh>
#include <nntile/graph/tensor/ops/copy_intersection.hh>
#include <nntile/graph/tensor/ops/embedding.hh>
#include <nntile/graph/tensor/ops/embedding_backward.hh>
#include <nntile/graph/tensor/ops/fill.hh>
#include <nntile/graph/tensor/ops/flash_sdpa_bwd_cudnn.hh>
#include <nntile/graph/tensor/ops/flash_sdpa_fwd_cudnn.hh>
#include <nntile/graph/tensor/ops/gather.hh>
#include <nntile/graph/tensor/ops/gelu.hh>
#include <nntile/graph/tensor/ops/gelu_backward.hh>
#include <nntile/graph/tensor/ops/gelu_inplace.hh>
#include <nntile/graph/tensor/ops/gelutanh.hh>
#include <nntile/graph/tensor/ops/gelutanh_backward.hh>
#include <nntile/graph/tensor/ops/gelutanh_inplace.hh>
#include <nntile/graph/tensor/ops/gemm.hh>
#include <nntile/graph/tensor/ops/hypot.hh>
#include <nntile/graph/tensor/ops/hypot_inplace.hh>
#include <nntile/graph/tensor/ops/hypot_scalar_inverse.hh>
#include <nntile/graph/tensor/ops/log_scalar.hh>
#include <nntile/graph/tensor/ops/logsumexp.hh>
#include <nntile/graph/tensor/ops/mask_scalar.hh>
#include <nntile/graph/tensor/ops/maxsumexp.hh>
#include <nntile/graph/tensor/ops/multiply.hh>
#include <nntile/graph/tensor/ops/multiply_fiber.hh>
#include <nntile/graph/tensor/ops/multiply_fiber_inplace.hh>
#include <nntile/graph/tensor/ops/multiply_inplace.hh>
#include <nntile/graph/tensor/ops/multiply_slice.hh>
#include <nntile/graph/tensor/ops/norm.hh>
#include <nntile/graph/tensor/ops/norm_fiber.hh>
#include <nntile/graph/tensor/ops/norm_fiber_inplace.hh>
#include <nntile/graph/tensor/ops/norm_slice.hh>
#include <nntile/graph/tensor/ops/norm_slice_inplace.hh>
#include <nntile/graph/tensor/ops/pow.hh>
#include <nntile/graph/tensor/ops/randn.hh>
#include <nntile/graph/tensor/ops/relu.hh>
#include <nntile/graph/tensor/ops/sgd_step.hh>
#include <nntile/graph/tensor/ops/adam_step.hh>
#include <nntile/graph/tensor/ops/adamw_step.hh>
#include <nntile/graph/tensor/ops/relu_backward.hh>
#include <nntile/graph/tensor/ops/relu_inplace.hh>
#include <nntile/graph/tensor/ops/rope.hh>
#include <nntile/graph/tensor/ops/rope_backward.hh>
#include <nntile/graph/tensor/ops/scale.hh>
#include <nntile/graph/tensor/ops/scale_fiber.hh>
#include <nntile/graph/tensor/ops/scale_inplace.hh>
#include <nntile/graph/tensor/ops/scale_slice.hh>
#include <nntile/graph/tensor/ops/scatter.hh>
#include <nntile/graph/tensor/ops/silu.hh>
#include <nntile/graph/tensor/ops/silu_backward.hh>
#include <nntile/graph/tensor/ops/silu_inplace.hh>
#include <nntile/graph/tensor/ops/softmax.hh>
#include <nntile/graph/tensor/ops/softmax_inplace.hh>
#include <nntile/graph/tensor/ops/sqrt.hh>
#include <nntile/graph/tensor/ops/sqrt_inplace.hh>
#include <nntile/graph/tensor/ops/sum.hh>
#include <nntile/graph/tensor/ops/sum_fiber.hh>
#include <nntile/graph/tensor/ops/sum_slice.hh>
#include <nntile/graph/tensor/ops/subtract_indexed_outputs.hh>
#include <nntile/graph/tensor/ops/sumprod_fiber.hh>
#include <nntile/graph/tensor/ops/sumprod_slice.hh>
#include <nntile/graph/tensor/ops/total_sum_accum.hh>
#include <nntile/graph/tensor/ops/transpose.hh>
