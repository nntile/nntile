/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/add_fiber.hh
 * Bias operation over slices from a fiber of a buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-26
 * */

#pragma once

#include <nntile/kernel/add_fiber/cpu.hh>
//#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
//#include <nntile/kernel/add_fiber/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::add_fiber
/*! Low-level implementations of add_fiber operation
 * */
namespace add_fiber
{

} // namespace add_fiber
} // namespace kernel
} // namespace nntile

