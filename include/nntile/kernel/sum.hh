/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/sum.hh
 * Low-level kernels to compute sum along axis
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author K. Sozykin
 * @date 2022-02-20
 * */

#pragma once

#include <nntile/kernel/sum/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/sum/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::sum
/*! Low-level implementations of computing sum and norm operation
 * */
namespace sum
{

} // namespace sum
} // namespace kernel
} // namespace nntile

