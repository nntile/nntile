/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor.hh
 * Header for Tensor<T> class with corresponding operations
 *
 * @version 1.1.0
 * */

#pragma once

// Get Tensor<T> class
#include <nntile/tensor/tensor.hh>
// MPI distributions
#include <nntile/tensor/distributions.hh>

// Tensor operations
#include <nntile/tensor/axpy.hh>
#include <nntile/tensor/add_slice.hh>
#include <nntile/tensor/add_slice3.hh>
#include <nntile/tensor/add_fiber.hh>
#include <nntile/tensor/prod_slice.hh>
#include <nntile/tensor/prod_fiber.hh>
#include <nntile/tensor/prod_fiber3.hh>
#include <nntile/tensor/clear.hh>
#include <nntile/tensor/copy.hh>
#include <nntile/tensor/copy_intersection.hh>
#include <nntile/tensor/gather.hh>
#include <nntile/tensor/gelu.hh>
#include <nntile/tensor/gelutanh.hh>
#include <nntile/tensor/gelutanh_inplace.hh>
#include <nntile/tensor/dgelu.hh>
#include <nntile/tensor/dgelutanh.hh>
#include <nntile/tensor/drelu.hh>
#include <nntile/tensor/gemm.hh>
#include <nntile/tensor/nrm2.hh>
#include <nntile/tensor/normalize.hh>
#include <nntile/tensor/prod.hh>
#include <nntile/tensor/prod_inplace.hh>
#include <nntile/tensor/randn.hh>
#include <nntile/tensor/relu.hh>
#include <nntile/tensor/relu_forward.hh>
#include <nntile/tensor/relu_backward.hh>
#include <nntile/tensor/scatter.hh>
#include <nntile/tensor/fill.hh>
#include <nntile/tensor/sum_slice.hh>
#include <nntile/tensor/sum_fiber.hh>
#include <nntile/tensor/norm_slice.hh>
#include <nntile/tensor/pow.hh>
#include <nntile/tensor/sumnorm.hh>
#include <nntile/tensor/flash_maxsumexp.hh>
#include <nntile/tensor/maxsumexp.hh>
#include <nntile/tensor/flash_softmax_gemm.hh>
#include <nntile/tensor/flash_softmax_gemm_backward.hh>
#include <nntile/tensor/softmax.hh>
#include <nntile/tensor/softmax_inplace.hh>
#include <nntile/tensor/sqrt.hh>
#include <nntile/tensor/sqrt_inplace.hh>
#include <nntile/tensor/maximum.hh>
#include <nntile/tensor/addcdiv.hh>
#include <nntile/tensor/sumprod_slice.hh>
#include <nntile/tensor/sumprod_fiber.hh>
#include <nntile/tensor/logsumexp.hh>
#include <nntile/tensor/total_sum_accum.hh>
#include <nntile/tensor/subtract_indexed_outputs.hh>
#include <nntile/tensor/scal.hh>
#include <nntile/tensor/scal_inplace.hh>
#include <nntile/tensor/gelu_backward.hh>
#include <nntile/tensor/gelutanh_backward.hh>
#include <nntile/tensor/add.hh>
#include <nntile/tensor/add_scalar.hh>
#include <nntile/tensor/embedding.hh>
#include <nntile/tensor/embedding_backward.hh>
//#include <nntile/tensor/fp32_to_fp16.hh>
//#include <nntile/tensor/fp16_to_fp32.hh>
#include <nntile/tensor/mask_scalar.hh>
#include <nntile/tensor/hypot.hh>
#include <nntile/tensor/hypot_scalar_inverse.hh>
#include <nntile/tensor/adam_step.hh>
#include <nntile/tensor/adamw_step.hh>
#include <nntile/tensor/transpose.hh>
#include <nntile/tensor/silu_forward.hh>
#include <nntile/tensor/silu_backward.hh>
#include <nntile/tensor/conv2d_inplace.hh>
#include <nntile/tensor/conv2d_bwd_input_inplace.hh>
#include <nntile/tensor/conv2d_bwd_weight_inplace.hh>
#include <nntile/tensor/rope.hh>
#include <nntile/tensor/rope_backward.hh>
#include <nntile/tensor/norm_fiber.hh>

//! @namespace nntile::tensor
/*! This namespace holds high-level routines for Tensor<T>
 * */
namespace nntile::tensor
{

} // namespace nntile::tensor
