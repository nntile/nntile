/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/subcopy.hh
 * Subtensor copy low-level kernels
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-27
 * */

#pragma once

#include <nntile/kernel/subcopy/cpu.hh>
// No support for subcopy on CUDA
//#include <nntile/defs.h>
//#ifdef NNTILE_USE_CUDA
//#include <nntile/kernel/subcopy/cuda.hh>
//#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::subcopy
/*! Low-level implementations of copying subtensor
 * */
namespace subcopy
{

} // namespace subcopy
} // namespace kernel
} // namespace nntile

