/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/gelu.hh
 * GeLU operation on buffers
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/core/kernel/gelu/cpu.hh>
#include <nntile/core/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/core/kernel/gelu/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::core::kernel::gelu
/*! Low-level implementations of GeLU operation
 * */
namespace nntile::core::kernel::gelu
{

} // namespace nntile::core::kernel::gelu
