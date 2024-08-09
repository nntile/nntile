/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/maximum.hh
 * Per-element maximum of two buffers. low-level kernels
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/kernel/maximum/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/maximum/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::maximum
/*! Low-level implementations of maximum operation
 * */
namespace nntile::kernel::maximum
{

} // namespace nntile::kernel::maximum
