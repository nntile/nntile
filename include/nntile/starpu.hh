/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu.hh
 * StarPU wrappers for data handles and low-level kernels
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Aleksandr Katrutsa
 * @author Konstantin Sozykin
 * @date 2023-04-26
 * */

#pragma once

// StarPU wrappers for data handles and config
#include <nntile/starpu/config.hh>

// StarPU wrappers for low-level kernels
#include <nntile/starpu/axpy.hh>
#include <nntile/starpu/bias_slice.hh>
#include <nntile/starpu/bias_fiber.hh>
#include <nntile/starpu/biasprod.hh>
#include <nntile/starpu/biasprod_outer.hh>
#include <nntile/starpu/clear.hh>
#include <nntile/starpu/gelu.hh>
#include <nntile/starpu/gelutanh.hh>
#include <nntile/starpu/dgelu.hh>
#include <nntile/starpu/dgelutanh.hh>
#include <nntile/starpu/drelu.hh>
#include <nntile/starpu/gemm.hh>
#include <nntile/starpu/hypot.hh>
#include <nntile/starpu/nrm2.hh>
#include <nntile/starpu/normalize.hh>
#include <nntile/starpu/prod.hh>
#include <nntile/starpu/randn.hh>
#include <nntile/starpu/relu.hh>
#include <nntile/starpu/relu_backward.hh>
#include <nntile/starpu/subcopy.hh>
#include <nntile/starpu/sumnorm.hh>
#include <nntile/starpu/fill.hh>
#include <nntile/starpu/sum_slice.hh>
#include <nntile/starpu/sum_fiber.hh>
#include <nntile/starpu/norm.hh>
#include <nntile/starpu/pow.hh>
#include <nntile/starpu/maxsumexp.hh>
#include <nntile/starpu/softmax.hh>
#include <nntile/starpu/sqrt.hh>
#include <nntile/starpu/maximum.hh>
#include <nntile/starpu/addcdiv.hh>
#include <nntile/starpu/sumprod_slice.hh>
#include <nntile/starpu/scalprod_outer.hh>
#include <nntile/starpu/logsumexp.hh>
#include <nntile/starpu/total_sum_accum.hh>
#include <nntile/starpu/subtract_indexed_column.hh>
#include <nntile/starpu/scal.hh>
#include <nntile/starpu/gelu_backward.hh>
#include <nntile/starpu/gelutanh_backward.hh>

namespace nntile
{
//! @namespace nntile::starpu
/*! This namespace holds StarPU wrappers
 * */
namespace starpu
{

// Init all codelets
void init()
{
    axpy::init();
    bias_slice::init();
    bias_fiber::init();
    biasprod::init();
    biasprod_outer::init();
    clear::init();
    gelu::init();
    gelutanh::init();
    dgelu::init();
    dgelutanh::init();
    drelu::init();
    gemm::init();
    hypot::init();
    nrm2::init();
    normalize::init();
    randn::init();
    relu::init();
    relu_backward::init();
    prod::init();
    subcopy::init();
    sumnorm::init();
    fill::init();
    sum_slice::init();
    sum_fiber::init();
    norm::init();
    pow::init();
    softmax::init();
    maxsumexp::init();
    sqrt::init();
    maximum::init();
    addcdiv::init();
    sumprod_slice::init();
    scalprod_outer::init();
    logsumexp::init();
    total_sum_accum::init();
    subtract_indexed_column::init();
    scal::init();
    gelu_backward::init();
    gelutanh_backward::init();
}

// Restrict StarPU codelets to certain computational units
void restrict_where(uint32_t where)
{
    axpy::restrict_where(where);
    bias_slice::restrict_where(where);
    bias_fiber::restrict_where(where);
    biasprod::restrict_where(where);
    biasprod_outer::restrict_where(where);
    clear::restrict_where(where);
    gelu::restrict_where(where);
    gelutanh::restrict_where(where);
    dgelu::restrict_where(where);
    dgelutanh::restrict_where(where);
    drelu::restrict_where(where);
    gemm::restrict_where(where);
    hypot::restrict_where(where);
    nrm2::restrict_where(where);
    normalize::restrict_where(where);
    prod::restrict_where(where);
    randn::restrict_where(where);
    relu::restrict_where(where);
    relu_backward::restrict_where(where);
    subcopy::restrict_where(where);
    sumnorm::restrict_where(where);
    fill::restrict_where(where);
    sum_slice::restrict_where(where);
    sum_fiber::restrict_where(where);
    norm::restrict_where(where);
    pow::restrict_where(where);
    softmax::restrict_where(where);
    maxsumexp::restrict_where(where);
    sqrt::restrict_where(where);
    maximum::restrict_where(where);
    addcdiv::restrict_where(where);
    sumprod_slice::restrict_where(where);
    scalprod_outer::restrict_where(where);
    logsumexp::restrict_where(where);
    total_sum_accum::restrict_where(where);
    subtract_indexed_column::restrict_where(where);
    scal::restrict_where(where);
    gelu_backward::restrict_where(where);
    gelutanh_backward::restrict_where(where);
}

// Restore computational units for StarPU codelets
void restore_where()
{
    axpy::restore_where();
    bias_slice::restore_where();
    bias_fiber::restore_where();
    biasprod::restore_where();
    biasprod_outer::restore_where();
    clear::restore_where();
    gelu::restore_where();
    gelutanh::restore_where();
    dgelu::restore_where();
    dgelutanh::restore_where();
    drelu::restore_where();
    gemm::restore_where();
    hypot::restore_where();
    nrm2::restore_where();
    normalize::restore_where();
    prod::restore_where();
    randn::restore_where();
    relu::restore_where();
    relu_backward::restore_where();
    subcopy::restore_where();
    sumnorm::restore_where();
    fill::restore_where();
    sum_slice::restore_where();
    sum_fiber::restore_where();
    norm::restore_where();
    pow::restore_where();
    softmax::restore_where();
    maxsumexp::restore_where();
    sqrt::restore_where();
    maximum::restore_where();
    addcdiv::restore_where();
    sumprod_slice::restore_where();
    scalprod_outer::restore_where();
    logsumexp::restore_where();
    total_sum_accum::restore_where();
    subtract_indexed_column::restore_where();
    scal::restore_where();
    gelu_backward::restore_where();
    gelutanh_backward::restore_where();
}

} // namespace starpu
} // namespace nntile

