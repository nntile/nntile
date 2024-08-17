/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/prod_inplace.hh
 * Per-element product of two buffers. low-level kernels
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/kernel/prod_inplace/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/prod_inplace/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::prod_inplace
/*! Low-level implementations of prod operation
 * */
namespace nntile::kernel::prod_inplace
{

} // namespace nntile::kernel::prod_inplace
