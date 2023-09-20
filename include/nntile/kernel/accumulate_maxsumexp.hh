/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/accumulate_maxsumexp.hh
 * Accumulate maxsumexp buffers low-level kernel
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-09-20
 * */

#pragma once

#include <nntile/kernel/accumulate_maxsumexp/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/accumulate_maxsumexp/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::accumulate_maxsumexp
/*! Low-level implementations of accumulate maxsumexp buffers operation
 * */
namespace accumulate_maxsumexp
{

} // namespace accumulate_maxsumexp
} // namespace kernel
} // namespace nntile

