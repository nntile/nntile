/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
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
 * @date 2022-09-29
 * */

#pragma once

// StarPU wrappers for data handles and config
#include <nntile/starpu/config.hh>

// StarPU wrappers for low-level kernels
#include <nntile/starpu/bias.hh>
#include <nntile/starpu/clear.hh>
#include <nntile/starpu/gelu.hh>
#include <nntile/starpu/gelutanh.hh>
#include <nntile/starpu/gemm.hh>
#include <nntile/starpu/normalize.hh>
#include <nntile/starpu/randn.hh>
#include <nntile/starpu/relu.hh>
#include <nntile/starpu/subcopy.hh>
#include <nntile/starpu/sumnorm.hh>

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
    bias::init();
    clear::init();
    gelu::init();
    gelutanh::init();
    gemm::init();
    normalize::init();
    randn::init();
    relu::init();
    subcopy::init();
    sumnorm::init();
}

// Restrict StarPU codelets to certain computational units
void restrict_where(uint32_t where)
{
    bias::restrict_where(where);
    clear::restrict_where(where);
    gelu::restrict_where(where);
    gelutanh::restrict_where(where);
    gemm::restrict_where(where);
    normalize::restrict_where(where);
    randn::restrict_where(where);
    relu::restrict_where(where);
    subcopy::restrict_where(where);
    sumnorm::restrict_where(where);
}

// Restore computational units for StarPU codelets
void restore_where()
{
    bias::restore_where();
    clear::restore_where();
    gelu::restore_where();
    gelutanh::restore_where();
    gemm::restore_where();
    normalize::restore_where();
    randn::restore_where();
    relu::restore_where();
    subcopy::restore_where();
    sumnorm::restore_where();
}

} // namespace starpu
} // namespace nntile

