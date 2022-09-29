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

} // namespace starpu
} // namespace nntile

