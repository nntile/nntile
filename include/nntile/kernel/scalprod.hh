/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/sumnorm.hh
 * Low-level kernels to compute scalar product of slices of two buffers
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-03-26
 * */

#pragma once

#include <nntile/kernel/scalprod/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/scalprod/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::scalprod
/*! Low-level implementations of computing scalar product of slices
 * */
namespace scalprod
{

} // namespace scalprod
} // namespace kernel
} // namespace nntile

