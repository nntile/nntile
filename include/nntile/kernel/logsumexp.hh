/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/logsumexp.hh
 * Logsumexp based on the result of maxsumexp operation
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/kernel/logsumexp/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/logsumexp/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::logsumexp
/*! Low-level implementations of computing logsumexp from the result of maxsumexp operation
 * */
namespace nntile::kernel::logsumexp
{

} // namespace nntile::kernel::logsumexp
