/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/drelu.hh
 * Derivative of ReLU low-level kernels
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-01-31
 * */

#pragma once

#include <nntile/kernel/drelu/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/drelu/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::drelu
/*! Low-level implementations of derivative of ReLU operation
 * */
namespace drelu
{

} // namespace drelu
} // namespace kernel
} // namespace nntile

