/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/subcopy.hh
 * Subtensor copy low-level kernels
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/kernel/subcopy/cpu.hh>
// No support for subcopy on CUDA
//#include <nntile/defs.h>
//#ifdef NNTILE_USE_CUDA
//#include <nntile/kernel/subcopy/cuda.hh>
//#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::subcopy
/*! Low-level implementations of copying subtensor
 * */
namespace nntile::kernel::subcopy
{

} // namespace nntile::kernel::subcopy
