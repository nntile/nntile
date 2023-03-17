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
 * @date 2023-02-10
 * */

#pragma once

// StarPU wrappers for data handles and config
#include <nntile/starpu/config.hh>

// StarPU wrappers for low-level kernels
#include <nntile/starpu/axpy.hh>
#include <nntile/starpu/bias.hh>
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
#include <nntile/starpu/subcopy.hh>
#include <nntile/starpu/sumnorm.hh>
#include <nntile/starpu/maxsumexp.hh>
#include <nntile/starpu/softmax.hh>
#include <nntile/starpu/sqrt.hh>
#include <nntile/starpu/maximum.hh>
#include <nntile/starpu/addcdiv.hh>
#include <nntile/starpu/logsumexp.hh>
#include <nntile/starpu/total_sum_accum.hh>


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
    bias::init();
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
    prod::init();
    subcopy::init();
    sumnorm::init();
    softmax::init();
    maxsumexp::init();
    sqrt::init();
    maximum::init();
    addcdiv::init();
    logsumexp::init();
    total_sum_accum::init();
}

// Restrict StarPU codelets to certain computational units
void restrict_where(uint32_t where)
{
    axpy::restrict_where(where);
    bias::restrict_where(where);
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
    subcopy::restrict_where(where);
    sumnorm::restrict_where(where);
    softmax::restrict_where(where);
    maxsumexp::restrict_where(where);
    sqrt::restrict_where(where);
    maximum::restrict_where(where);
    addcdiv::restrict_where(where);
    logsumexp::restrict_where(where);
    total_sum_accum::restrict_where(where);
}

// Restore computational units for StarPU codelets
void restore_where()
{
    axpy::restore_where();
    bias::restore_where();
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
    subcopy::restore_where();
    sumnorm::restore_where();
    softmax::restore_where();
    maxsumexp::restore_where();
    sqrt::restore_where();
    maximum::restore_where();
    addcdiv::restore_where();
    logsumexp::restore_where();
    total_sum_accum::restore_where();
}

} // namespace starpu
} // namespace nntile

