/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/total_sum_accum.hh
 * Low-level kernels to compute total_sum_accum
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/kernel/total_sum_accum/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/total_sum_accum/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::total_sum_accum
/*! Low-level implementations of computing accumulated total sum
 * */
namespace nntile::kernel::total_sum_accum
{

} // namespace nntile::kernel::total_sum_accum
