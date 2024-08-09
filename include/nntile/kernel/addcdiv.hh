/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/addcdiv.hh
 * Per-element addcdiv operation on the given buffers. low-level kernels
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/kernel/addcdiv/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/addcdiv/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::addcdiv
/*! Low-level implementations of addcdiv operation
 * */
namespace nntile::kernel::addcdiv
{

} // namespace nntile::kernel::addcdiv
