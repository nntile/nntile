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
#include <nntile/graph/tensor/add.hh>
#include <nntile/graph/tensor/add_fiber.hh>
#include <nntile/graph/tensor/add_fiber_inplace.hh>
#include <nntile/graph/tensor/add_inplace.hh>
#include <nntile/graph/tensor/add_scalar_scaled_inplace.hh>
#include <nntile/graph/tensor/add_slice.hh>
#include <nntile/graph/tensor/add_slice_inplace.hh>
#include <nntile/graph/tensor/clear.hh>
#include <nntile/graph/tensor/conv2d_bwd_input_inplace.hh>
#include <nntile/graph/tensor/conv2d_bwd_weight_inplace.hh>
#include <nntile/graph/tensor/conv2d_inplace.hh>
#include <nntile/graph/tensor/copy.hh>
#include <nntile/graph/tensor/copy_intersection.hh>
#include <nntile/graph/tensor/embedding.hh>
#include <nntile/graph/tensor/embedding_backward.hh>
#include <nntile/graph/tensor/fill.hh>
#include <nntile/graph/tensor/flash_sdpa_bwd_cudnn.hh>
#include <nntile/graph/tensor/flash_sdpa_fwd_cudnn.hh>
#include <nntile/graph/tensor/gather.hh>
#include <nntile/graph/tensor/gelu.hh>
#include <nntile/graph/tensor/gelu_backward.hh>
#include <nntile/graph/tensor/gelu_inplace.hh>
#include <nntile/graph/tensor/gelutanh.hh>
#include <nntile/graph/tensor/gelutanh_backward.hh>
#include <nntile/graph/tensor/gelutanh_inplace.hh>
#include <nntile/graph/tensor/gemm.hh>
#include <nntile/graph/tensor/hypot.hh>
#include <nntile/graph/tensor/hypot_inplace.hh>
#include <nntile/graph/tensor/hypot_scalar_inverse.hh>
#include <nntile/graph/tensor/log_scalar.hh>
#include <nntile/graph/tensor/logsumexp.hh>
#include <nntile/graph/tensor/mask_scalar.hh>
#include <nntile/graph/tensor/maxsumexp.hh>
#include <nntile/graph/tensor/multiply.hh>
#include <nntile/graph/tensor/multiply_fiber.hh>
#include <nntile/graph/tensor/multiply_fiber_inplace.hh>
#include <nntile/graph/tensor/multiply_inplace.hh>
#include <nntile/graph/tensor/multiply_slice.hh>
#include <nntile/graph/tensor/norm.hh>
#include <nntile/graph/tensor/norm_fiber.hh>
#include <nntile/graph/tensor/norm_fiber_inplace.hh>
#include <nntile/graph/tensor/norm_slice.hh>
#include <nntile/graph/tensor/norm_slice_inplace.hh>
#include <nntile/graph/tensor/pow.hh>
#include <nntile/graph/tensor/randn.hh>
#include <nntile/graph/tensor/relu.hh>
#include <nntile/graph/tensor/sgd_step.hh>
#include <nntile/graph/tensor/adam_step.hh>
#include <nntile/graph/tensor/adamw_step.hh>
#include <nntile/graph/tensor/relu_backward.hh>
#include <nntile/graph/tensor/relu_inplace.hh>
#include <nntile/graph/tensor/rope.hh>
#include <nntile/graph/tensor/rope_backward.hh>
#include <nntile/graph/tensor/scale.hh>
#include <nntile/graph/tensor/scale_fiber.hh>
#include <nntile/graph/tensor/scale_inplace.hh>
#include <nntile/graph/tensor/scale_slice.hh>
#include <nntile/graph/tensor/scatter.hh>
#include <nntile/graph/tensor/silu.hh>
#include <nntile/graph/tensor/silu_backward.hh>
#include <nntile/graph/tensor/silu_inplace.hh>
#include <nntile/graph/tensor/softmax.hh>
#include <nntile/graph/tensor/softmax_inplace.hh>
#include <nntile/graph/tensor/sqrt.hh>
#include <nntile/graph/tensor/sqrt_inplace.hh>
#include <nntile/graph/tensor/sum.hh>
#include <nntile/graph/tensor/sum_fiber.hh>
#include <nntile/graph/tensor/sum_slice.hh>
#include <nntile/graph/tensor/subtract_indexed_outputs.hh>
#include <nntile/graph/tensor/sumprod_fiber.hh>
#include <nntile/graph/tensor/sumprod_slice.hh>
#include <nntile/graph/tensor/total_sum_accum.hh>
#include <nntile/graph/tensor/transpose.hh>
