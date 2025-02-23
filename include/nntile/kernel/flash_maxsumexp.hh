/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/flash_maxsumexp.hh
 * Low-level kernel to compute maxsumexp(alpha*QK')
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/flash_maxsumexp/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::flash_maxsumexp
/*! Low-level implementations to compute maxsumexp(alpha*QK')
 * */
namespace nntile::kernel::flash_maxsumexp
{

} // namespace nntile::kernel::flash_maxsumexp
