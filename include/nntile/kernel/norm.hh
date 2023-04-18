/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/norm.hh
 * Low-level kernels to compute norm along axis
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-18
 * */

#pragma once

#include <nntile/kernel/norm/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/norm/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::norm
/*! Low-level implementations of computing norm operation
 * */
namespace norm
{

} // namespace norm
} // namespace kernel
} // namespace nntile

