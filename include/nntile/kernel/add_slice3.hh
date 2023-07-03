/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/add_slice3.hh
 * Per-element addition of a tensor and a broadcasted slice
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-07-03
 * */

#pragma once

#include <nntile/kernel/add_slice3/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/add_slice3/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::add_slice3
/*! Low-level implementations of add_slice3 operation
 * */
namespace add_slice3
{

} // namespace add_slice3
} // namespace kernel
} // namespace nntile

