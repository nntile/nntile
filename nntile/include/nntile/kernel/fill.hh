/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/fill.hh
 * Low-level kernel to fill data with a provided value
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/kernel/fill/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/fill/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::fill
/*! Low-level implementations of fill operation
 * */
namespace nntile::kernel::fill
{

} // namespace nntile::kernel::fill
