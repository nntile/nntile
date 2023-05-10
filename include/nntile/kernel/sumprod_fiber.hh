/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/sumprod_fiber.hh
 * Sums over slices into a fiber of a product of buffers
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-05-02
 * */

#pragma once

#include <nntile/kernel/sumprod_fiber/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
//#include <nntile/kernel/sumprod_fiber/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::sumprod_fiber
/*! Low-level implementations of sumprod_fiber operation
 * axed
 * */
namespace sumprod_fiber
{

} // namespace sumprod_fiber
} // namespace kernel
} // namespace nntile

