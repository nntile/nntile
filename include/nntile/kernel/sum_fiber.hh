/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/sum_fiber.hh
 * Sums over slices into a fiber of a buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-25
 * */

#pragma once

#include <nntile/kernel/sum_fiber/cpu.hh>
//#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
//#include <nntile/kernel/sum_fiber/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::sum_fiber
/*! Low-level implementations of computing sum_fiber operation
 * */
namespace sum_fiber
{

} // namespace sum_fiber
} // namespace kernel
} // namespace nntile

