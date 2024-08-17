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

// StarPU wrappers for data handles and config
#include <nntile/starpu/config.hh>

// StarPU wrappers for low-level kernels
#include <nntile/starpu/accumulate.hh>
#include <nntile/starpu/accumulate_hypot.hh>
#include <nntile/starpu/accumulate_maxsumexp.hh>
#include <nntile/starpu/axpy.hh>
#include <nntile/starpu/add_slice.hh>
#include <nntile/starpu/add_slice3.hh>
#include <nntile/starpu/add_fiber.hh>
#include <nntile/starpu/prod_slice.hh>
#include <nntile/starpu/prod_fiber.hh>
#include <nntile/starpu/prod_fiber3.hh>
#include <nntile/starpu/clear.hh>
#include <nntile/starpu/copy.hh>
#include <nntile/starpu/gelu.hh>
#include <nntile/starpu/gelutanh.hh>
#include <nntile/starpu/gelutanh_inplace.hh>
#include <nntile/starpu/dgelu.hh>
#include <nntile/starpu/dgelutanh.hh>
#include <nntile/starpu/drelu.hh>
#include <nntile/starpu/gemm.hh>
#include <nntile/starpu/hypot.hh>
#include <nntile/starpu/hypot_scalar_inverse.hh>
#include <nntile/starpu/nrm2.hh>
#include <nntile/starpu/normalize.hh>
#include <nntile/starpu/prod.hh>
#include <nntile/starpu/prod_inplace.hh>
#include <nntile/starpu/randn.hh>
#include <nntile/starpu/relu.hh>
#include <nntile/starpu/relu_forward.hh>
#include <nntile/starpu/relu_backward.hh>
#include <nntile/starpu/subcopy.hh>
#include <nntile/starpu/sumnorm.hh>
#include <nntile/starpu/fill.hh>
#include <nntile/starpu/sum_slice.hh>
#include <nntile/starpu/sum_fiber.hh>
#include <nntile/starpu/norm_slice.hh>
#include <nntile/starpu/pow.hh>
#include <nntile/starpu/flash_maxsumexp.hh>
#include <nntile/starpu/maxsumexp.hh>
#include <nntile/starpu/softmax.hh>
#include <nntile/starpu/flash_softmax_gemm.hh>
#include <nntile/starpu/flash_softmax_gemm_backward_sumprod_slice.hh>
#include <nntile/starpu/flash_softmax_gemm_backward_dq_dk.hh>
#include <nntile/starpu/softmax_inplace.hh>
#include <nntile/starpu/sqrt.hh>
#include <nntile/starpu/sqrt_inplace.hh>
#include <nntile/starpu/maximum.hh>
#include <nntile/starpu/addcdiv.hh>
#include <nntile/starpu/sumprod_slice.hh>
#include <nntile/starpu/sumprod_fiber.hh>
#include <nntile/starpu/logsumexp.hh>
#include <nntile/starpu/total_sum_accum.hh>
#include <nntile/starpu/subtract_indexed_outputs.hh>
#include <nntile/starpu/scal.hh>
#include <nntile/starpu/scal_inplace.hh>
#include <nntile/starpu/gelu_backward.hh>
#include <nntile/starpu/gelutanh_backward.hh>
#include <nntile/starpu/add.hh>
#include <nntile/starpu/add_scalar.hh>
#include <nntile/starpu/embedding.hh>
#include <nntile/starpu/embedding_backward.hh>
//#include <nntile/starpu/fp32_to_fp16.hh>
//#include <nntile/starpu/fp16_to_fp32.hh>
#include <nntile/starpu/mask_scalar.hh>
#include <nntile/starpu/adam_step.hh>
#include <nntile/starpu/adamw_step.hh>
#include <nntile/starpu/transpose.hh>
#include <nntile/starpu/silu_forward.hh>
#include <nntile/starpu/silu_backward.hh>
#include <nntile/starpu/conv2d_inplace.hh>
#include <nntile/starpu/conv2d_bwd_input_inplace.hh>
#include <nntile/starpu/conv2d_bwd_weight_inplace.hh>
#include <nntile/starpu/rope.hh>
#include <nntile/starpu/rope_backward.hh>
#include <nntile/starpu/norm_fiber.hh>

//! @namespace nntile::starpu
/*! This namespace holds StarPU wrappers
 * */
namespace nntile::starpu
{

// Init all codelets
void init()
{
    accumulate::init();
    accumulate_hypot::init();
    accumulate_maxsumexp::init();
    axpy::init();
    add_slice::init();
    add_slice3::init();
    add_fiber::init();
    prod_slice::init();
    prod_fiber::init();
    prod_fiber3::init();
    clear::init();
    copy::init();
    gelu::init();
    gelutanh::init();
    gelutanh_inplace::init();
    dgelu::init();
    dgelutanh::init();
    drelu::init();
    gemm::init();
    hypot::init();
    hypot_scalar_inverse::init();
    nrm2::init();
    normalize::init();
    randn::init();
    relu::init();
    relu_forward::init();
    relu_backward::init();
    prod::init();
    prod_inplace::init();
    subcopy::init();
    sumnorm::init();
    fill::init();
    sum_slice::init();
    sum_fiber::init();
    norm_slice::init();
    norm_fiber::init();
    pow::init();
    softmax::init();
    softmax_inplace::init();
    flash_softmax_gemm::init();
    flash_softmax_gemm_backward_sumprod_slice::init();
    flash_softmax_gemm_backward_dq_dk::init();
    flash_maxsumexp::init();
    maxsumexp::init();
    sqrt::init();
    sqrt_inplace::init();
    maximum::init();
    addcdiv::init();
    sumprod_slice::init();
    sumprod_fiber::init();
    logsumexp::init();
    total_sum_accum::init();
    subtract_indexed_outputs::init();
    scal::init();
    scal_inplace::init();
    gelu_backward::init();
    gelutanh_backward::init();
    add::init();
    add_scalar::init();
    embedding::init();
    embedding_backward::init();
    //fp32_to_fp16::init();
    //fp16_to_fp32::init();
    mask_scalar::init();
    adam_step::init();
    adamw_step::init();
    transpose::init();
    silu_forward::init();
    silu_backward::init();
    conv2d_inplace::init();
    conv2d_bwd_input_inplace::init();
    conv2d_bwd_weight_inplace::init();
    rope::init();
    rope_backward::init();
}

// Restrict StarPU codelets to certain computational units
void restrict_where(uint32_t where)
{
    accumulate::restrict_where(where);
    accumulate_hypot::restrict_where(where);
    accumulate_maxsumexp::restrict_where(where);
    axpy::restrict_where(where);
    add_slice::restrict_where(where);
    add_slice3::restrict_where(where);
    add_fiber::restrict_where(where);
    prod_slice::restrict_where(where);
    prod_fiber::restrict_where(where);
    prod_fiber3::restrict_where(where);
    clear::restrict_where(where);
    copy::restrict_where(where);
    gelu::restrict_where(where);
    gelutanh::restrict_where(where);
    gelutanh_inplace::restrict_where(where);
    dgelu::restrict_where(where);
    dgelutanh::restrict_where(where);
    drelu::restrict_where(where);
    gemm::restrict_where(where);
    hypot::restrict_where(where);
    hypot_scalar_inverse::restrict_where(where);
    nrm2::restrict_where(where);
    normalize::restrict_where(where);
    prod::restrict_where(where);
    prod_inplace::restrict_where(where);
    randn::restrict_where(where);
    relu::restrict_where(where);
    relu_forward::restrict_where(where);
    relu_backward::restrict_where(where);
    subcopy::restrict_where(where);
    sumnorm::restrict_where(where);
    fill::restrict_where(where);
    sum_slice::restrict_where(where);
    sum_fiber::restrict_where(where);
    norm_slice::restrict_where(where);
    norm_fiber::restrict_where(where);
    pow::restrict_where(where);
    softmax::restrict_where(where);
    softmax_inplace::restrict_where(where);
    flash_softmax_gemm::restrict_where(where);
    flash_softmax_gemm_backward_sumprod_slice::restrict_where(where);
    flash_softmax_gemm_backward_dq_dk::restrict_where(where);
    flash_maxsumexp::restrict_where(where);
    maxsumexp::restrict_where(where);
    sqrt::restrict_where(where);
    sqrt_inplace::restrict_where(where);
    maximum::restrict_where(where);
    addcdiv::restrict_where(where);
    sumprod_slice::restrict_where(where);
    sumprod_fiber::restrict_where(where);
    logsumexp::restrict_where(where);
    total_sum_accum::restrict_where(where);
    subtract_indexed_outputs::restrict_where(where);
    scal::restrict_where(where);
    scal_inplace::restrict_where(where);
    gelu_backward::restrict_where(where);
    gelutanh_backward::restrict_where(where);
    add::restrict_where(where);
    add_scalar::restrict_where(where);
    embedding::restrict_where(where);
    embedding_backward::restrict_where(where);
    //fp32_to_fp16::restrict_where(where);
    //fp16_to_fp32::restrict_where(where);
    mask_scalar::restrict_where(where);
    adam_step::restrict_where(where);
    adamw_step::restrict_where(where);
    transpose::restrict_where(where);
    silu_forward::restrict_where(where);
    silu_backward::restrict_where(where);
    conv2d_inplace::restrict_where(where);
    conv2d_bwd_input_inplace::restrict_where(where);
    conv2d_bwd_weight_inplace::restrict_where(where);
    rope::restrict_where(where);
    rope_backward::restrict_where(where);
}

// Restore computational units for StarPU codelets
void restore_where()
{
    accumulate::restore_where();
    accumulate_hypot::restore_where();
    accumulate_maxsumexp::restore_where();
    axpy::restore_where();
    add_slice::restore_where();
    add_slice3::restore_where();
    add_fiber::restore_where();
    prod_slice::restore_where();
    prod_fiber::restore_where();
    prod_fiber3::restore_where();
    clear::restore_where();
    copy::restore_where();
    gelu::restore_where();
    gelutanh::restore_where();
    gelutanh_inplace::restore_where();
    dgelu::restore_where();
    dgelutanh::restore_where();
    drelu::restore_where();
    gemm::restore_where();
    hypot::restore_where();
    hypot_scalar_inverse::restore_where();
    nrm2::restore_where();
    normalize::restore_where();
    prod::restore_where();
    prod_inplace::restore_where();
    randn::restore_where();
    relu::restore_where();
    relu_forward::restore_where();
    relu_backward::restore_where();
    subcopy::restore_where();
    sumnorm::restore_where();
    fill::restore_where();
    sum_slice::restore_where();
    sum_fiber::restore_where();
    norm_slice::restore_where();
    norm_fiber::restore_where();
    pow::restore_where();
    softmax::restore_where();
    softmax_inplace::restore_where();
    flash_softmax_gemm::restore_where();
    flash_softmax_gemm_backward_sumprod_slice::restore_where();
    flash_softmax_gemm_backward_dq_dk::restore_where();
    flash_maxsumexp::restore_where();
    maxsumexp::restore_where();
    sqrt::restore_where();
    sqrt_inplace::restore_where();
    maximum::restore_where();
    addcdiv::restore_where();
    sumprod_slice::restore_where();
    sumprod_fiber::restore_where();
    logsumexp::restore_where();
    total_sum_accum::restore_where();
    subtract_indexed_outputs::restore_where();
    scal::restore_where();
    scal_inplace::restore_where();
    gelu_backward::restore_where();
    gelutanh_backward::restore_where();
    add::restore_where();
    add_scalar::restore_where();
    embedding::restore_where();
    embedding_backward::restore_where();
    //fp32_to_fp16::restore_where();
    //fp16_to_fp32::restore_where();
    mask_scalar::restore_where();
    adam_step::restore_where();
    adamw_step::restore_where();
    transpose::restore_where();
    silu_forward::restore_where();
    silu_backward::restore_where();
    conv2d_inplace::restore_where();
    conv2d_bwd_input_inplace::restore_where();
    conv2d_bwd_weight_inplace::restore_where();
    rope::restore_where();
    rope_backward::restore_where();
}

} // namespace nntile::starpu
