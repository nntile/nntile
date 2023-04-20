/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/sum_outer.hh
 * Low-level kernels to compute sum along outer axes
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-19
 * */

#pragma once

#include <nntile/kernel/sum_outer/cpu.hh>
//#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
//#include <nntile/kernel/sum_outer/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::sum_outer
/*! Low-level implementations of computing sum_outer operation
 * */
namespace sum_outer
{

} // namespace sum_outer
} // namespace kernel
} // namespace nntile

