/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/gelutanh_inplace.hh
 * Approximate GeLU low-level kernels
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-07-01
 * */

#pragma once

#include <nntile/kernel/gelutanh_inplace/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/gelutanh_inplace/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::gelutanh_inplace
/*! Low-level implementations of Approximate GeLU operation
 * */
namespace gelutanh_inplace
{

} // namespace gelutanh_inplace
} // namespace kernel
} // namespace nntile

