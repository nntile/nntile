/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/dgelu.hh
 * Derivative GeLU low-level kernels
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-10-24
 * */

#pragma once

#include <nntile/kernel/dgelu/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/dgelu/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::dgelu
/*! Low-level implementations of derivative of GeLU operation
 * */
namespace dgelu
{

} // namespace dgelu
} // namespace kernel
} // namespace nntile

