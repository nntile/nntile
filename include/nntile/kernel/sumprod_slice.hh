/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/sumprod_slice.hh
 * Sums over fibers into a slice of a product of buffers
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/kernel/sumprod_slice/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/sumprod_slice/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::sumprod_slice
/*! Low-level implementations of computing sumprod_slice operation
 * */
namespace nntile::kernel::sumprod_slice
{

} // namespace nntile::kernel::sumprod_slice
