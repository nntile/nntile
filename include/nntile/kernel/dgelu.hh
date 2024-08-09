/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/dgelu.hh
 * Derivative of GeLU low-level kernels
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/kernel/dgelu/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/dgelu/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::dgelu
/*! Low-level implementations of derivative of GeLU operation
 * */
namespace nntile::kernel::dgelu
{

} // namespace nntile::kernel::dgelu
