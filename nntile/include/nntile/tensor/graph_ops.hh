/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/graph_ops.hh
 * TensorGraph operations.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/defs.h>

// NNTile headers
#include <nntile/tensor/ops/add.hh>
#include <nntile/tensor/ops/add_fiber.hh>
#include <nntile/tensor/ops/add_fiber_inplace.hh>
#include <nntile/tensor/ops/add_inplace.hh>
#include <nntile/tensor/ops/add_slice.hh>
#include <nntile/tensor/ops/add_slice_inplace.hh>
#include <nntile/tensor/ops/clear.hh>
#include <nntile/tensor/ops/invalidate.hh>
#include <nntile/tensor/ops/conv2d_bwd_input_inplace.hh>
#include <nntile/tensor/ops/conv2d_bwd_weight_inplace.hh>
#include <nntile/tensor/ops/conv2d_inplace.hh>
#include <nntile/tensor/ops/concat.hh>
#include <nntile/tensor/ops/copy.hh>
#include <nntile/tensor/ops/copy_intersection.hh>
#include <nntile/tensor/ops/embedding.hh>
#include <nntile/tensor/ops/embedding_backward.hh>
#include <nntile/tensor/ops/fill.hh>
#ifdef NNTILE_USE_FLASH_SDPA
#include <nntile/tensor/ops/flash_sdpa_bwd_cudnn.hh>
#include <nntile/tensor/ops/flash_sdpa_fwd_cudnn.hh>
#endif
#include <nntile/tensor/ops/gather.hh>
#include <nntile/tensor/ops/gelu.hh>
#include <nntile/tensor/ops/gelu_backward.hh>
#include <nntile/tensor/ops/gelu_inplace.hh>
#include <nntile/tensor/ops/gelutanh.hh>
#include <nntile/tensor/ops/gelutanh_backward.hh>
#include <nntile/tensor/ops/gelutanh_inplace.hh>
#include <nntile/tensor/ops/gemm.hh>
#include <nntile/tensor/ops/hypot.hh>
#include <nntile/tensor/ops/hypot_inplace.hh>
#include <nntile/tensor/ops/hypot_scalar_inverse.hh>
#include <nntile/tensor/ops/log_scalar.hh>
#include <nntile/tensor/ops/logsumexp.hh>
#include <nntile/tensor/ops/mask_scalar.hh>
#include <nntile/tensor/ops/maxsumexp.hh>
#include <nntile/tensor/ops/multiply.hh>
#include <nntile/tensor/ops/multiply_fiber.hh>
#include <nntile/tensor/ops/multiply_fiber_inplace.hh>
#include <nntile/tensor/ops/multiply_inplace.hh>
#include <nntile/tensor/ops/multiply_slice.hh>
#include <nntile/tensor/ops/norm.hh>
#include <nntile/tensor/ops/norm_fiber.hh>
#include <nntile/tensor/ops/norm_fiber_inplace.hh>
#include <nntile/tensor/ops/norm_slice.hh>
#include <nntile/tensor/ops/norm_slice_inplace.hh>
#include <nntile/tensor/ops/pow.hh>
#include <nntile/tensor/ops/randn.hh>
#include <nntile/tensor/ops/relu.hh>
#include <nntile/tensor/ops/sgd_step.hh>
#include <nntile/tensor/ops/adam_step.hh>
#include <nntile/tensor/ops/adamw_step.hh>
#include <nntile/tensor/ops/relu_backward.hh>
#include <nntile/tensor/ops/relu_inplace.hh>
#include <nntile/tensor/ops/rope.hh>
#include <nntile/tensor/ops/rope_backward.hh>
#include <nntile/tensor/ops/scale.hh>
#include <nntile/tensor/ops/scale_fiber.hh>
#include <nntile/tensor/ops/scale_inplace.hh>
#include <nntile/tensor/ops/scale_slice.hh>
#include <nntile/tensor/ops/scatter.hh>
#include <nntile/tensor/ops/silu.hh>
#include <nntile/tensor/ops/silu_backward.hh>
#include <nntile/tensor/ops/silu_inplace.hh>
#include <nntile/tensor/ops/softmax.hh>
#include <nntile/tensor/ops/softmax_inplace.hh>
#include <nntile/tensor/ops/sqrt.hh>
#include <nntile/tensor/ops/sqrt_inplace.hh>
#include <nntile/tensor/ops/sum.hh>
#include <nntile/tensor/ops/sum_fiber.hh>
#include <nntile/tensor/ops/sum_slice.hh>
#include <nntile/tensor/ops/subtract_indexed_outputs.hh>
#include <nntile/tensor/ops/sumprod_fiber.hh>
#include <nntile/tensor/ops/sumprod_slice.hh>
#include <nntile/tensor/ops/total_sum_accum.hh>
#include <nntile/tensor/ops/transpose.hh>
