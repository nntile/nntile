/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/logsumexp.hh
 * Low-level kernels to compute logsumexp based on the result of maxsumexp operation
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-03-15
 * */

#pragma once

#include <nntile/kernel/logsumexp/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/maxsumexp/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::logsumexp
/*! Low-level implementations of computing logsumexp from the result of maxsumexp operation
 * */
namespace logsumexp
{

} // namespace logsumexp
} // namespace kernel
} // namespace nntile

