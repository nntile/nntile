/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/maxsumexp.hh
 * Low-level kernels to compute maximums and sums of exponents along axis
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-12-07
 * */

#pragma once

#include <nntile/kernel/maxsumexp/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/maxsumexp/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::maxsumexp
/*! Low-level implementations of computing maximums and sums of exponents
 * */
namespace maxsumexp
{

} // namespace maxsumexp
} // namespace kernel
} // namespace nntile

