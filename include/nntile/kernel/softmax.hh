/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/softmax.hh
 * Low-level kernels to softmax along axis
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-12-08
 * */

#pragma once

#include <nntile/kernel/softmax/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/softmax/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::softmax
/*! Low-level implementations of softmax operation
 * */
namespace softmax
{

} // namespace softmax
} // namespace kernel
} // namespace nntile

