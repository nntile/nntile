/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/sum.hh
 * Sum all elements of a buffer
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/kernel/sum/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/sum/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::sum
/*! Low-level implementations of sum operation
 * */
namespace nntile::kernel::sum
{

} // namespace nntile::kernel::sum
