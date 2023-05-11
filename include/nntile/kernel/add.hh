/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/add.hh
 * Add low-level kernel
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Aleksandr Katrutsa
 * @date 2023-05-08
 * */

#pragma once

#include <nntile/kernel/add/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
// #include <nntile/kernel/add/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::add
/*! Low-level implementations of add operation
 * */
namespace add
{

} // namespace add
} // namespace kernel
} // namespace nntile

