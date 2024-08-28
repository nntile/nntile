/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/normalize.hh
 * Low-level kernels to normalize along axis
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/kernel/normalize/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/normalize/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::normalize
/*! Low-level implementations of normalization operation
 * */
namespace nntile::kernel::normalize
{

} // namespace nntile::kernel::normalize
