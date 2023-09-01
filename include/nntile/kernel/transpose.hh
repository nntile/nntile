/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/transpose.hh
 * Transpose low-level kernel
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-07-20
 * */

#pragma once

#include <nntile/kernel/transpose/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/transpose/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::transpose
/*! Low-level implementations of transpose operation
 * */
namespace transpose
{

} // namespace transpose
} // namespace kernel
} // namespace nntile

