/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu.hh
 * StarPU wrappers for data handles and low-level kernels
 *
 * @version 1.1.0
 * */

#pragma once

// Some definitions, that will be refactored later
#include <nntile/starpu/config.hh>

// StarPU wrappers for low-level kernels
#include <nntile/starpu/accumulate.hh>
#include <nntile/starpu/accumulate_hypot.hh>
#include <nntile/starpu/accumulate_maxsumexp.hh>
#include <nntile/starpu/adam_step.hh>
#include <nntile/starpu/adamw_step.hh>
#include <nntile/starpu/add.hh>
#include <nntile/starpu/add_fiber.hh>
#include <nntile/starpu/add_fiber_inplace.hh>
#include <nntile/starpu/scale_fiber.hh>
#include <nntile/starpu/add_inplace.hh>
#include <nntile/starpu/add_slice.hh>
#include <nntile/starpu/add_slice_inplace.hh>
#include <nntile/starpu/clear.hh>
#include <nntile/starpu/conv2d_bwd_input_inplace.hh>
#include <nntile/starpu/conv2d_bwd_weight_inplace.hh>
#include <nntile/starpu/conv2d_inplace.hh>
#include <nntile/starpu/copy.hh>
#include <nntile/starpu/embedding.hh>
#include <nntile/starpu/embedding_backward.hh>
#include <nntile/starpu/fill.hh>
#include <nntile/starpu/gelu.hh>
#include <nntile/starpu/gelu_inplace.hh>
#include <nntile/starpu/gelu_backward.hh>
#include <nntile/starpu/gelutanh.hh>
#include <nntile/starpu/gelutanh_backward.hh>
#include <nntile/starpu/gelutanh_inplace.hh>
#include <nntile/starpu/gemm.hh>
#include <nntile/starpu/hypot.hh>
#include <nntile/starpu/hypot_inplace.hh>
#include <nntile/starpu/hypot_scalar_inverse.hh>
#include <nntile/starpu/log_scalar.hh>
#include <nntile/starpu/logsumexp.hh>
#include <nntile/starpu/mask_scalar.hh>
#include <nntile/starpu/maxsumexp.hh>
#include <nntile/starpu/norm_fiber.hh>
#include <nntile/starpu/norm_fiber_inplace.hh>
#include <nntile/starpu/norm_slice_inplace.hh>
#include <nntile/starpu/norm_slice.hh>
#include <nntile/starpu/pow.hh>
#include <nntile/starpu/multiply.hh>
#include <nntile/starpu/multiply_fiber_inplace.hh>
#include <nntile/starpu/multiply_fiber.hh>
#include <nntile/starpu/multiply_inplace.hh>
#include <nntile/starpu/multiply_slice.hh>
#include <nntile/starpu/randn.hh>
#include <nntile/starpu/relu_inplace.hh>
#include <nntile/starpu/relu_backward.hh>
#include <nntile/starpu/relu.hh>
#include <nntile/starpu/rope.hh>
#include <nntile/starpu/rope_backward.hh>
#include <nntile/starpu/scale.hh>
#include <nntile/starpu/scale_inplace.hh>
#include <nntile/starpu/silu_backward.hh>
#include <nntile/starpu/silu.hh>
#include <nntile/starpu/softmax.hh>
#include <nntile/starpu/softmax_inplace.hh>
#include <nntile/starpu/sqrt.hh>
#include <nntile/starpu/sqrt_inplace.hh>
#include <nntile/starpu/subcopy.hh>
#include <nntile/starpu/subtract_indexed_outputs.hh>
#include <nntile/starpu/sum_fiber.hh>
#include <nntile/starpu/sum_slice.hh>
#include <nntile/starpu/sumprod_fiber.hh>
#include <nntile/starpu/sumprod_slice.hh>
#include <nntile/starpu/total_sum_accum.hh>
#include <nntile/starpu/transpose.hh>


//! @namespace nntile::starpu
/*! This namespace holds StarPU wrappers
 * */
namespace nntile::starpu
{

} // namespace nntile::starpu
