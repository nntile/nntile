/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/scalprod_outer.hh
 * Low-level kernels to compute scalar product of two buffers along outer axes
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-20
 * */

#pragma once

#include <nntile/kernel/scalprod_outer/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
//#include <nntile/kernel/scalprod_outer/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::scalprod_outer
/*! Low-level implementations of computing scalar product of slices along outer
 * axed
 * */
namespace scalprod_outer
{

} // namespace scalprod_outer
} // namespace kernel
} // namespace nntile

