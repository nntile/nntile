/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/kernel/hypot_inplace.hh
 * hypot_inplace low-level kernel
 *
 * @version 2.0.0
 * */

#pragma once

#include <nntile/kernel/hypot_inplace/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/hypot_inplace/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::hypot_inplace
/*! Low-level implementations of hypot_inplace operation
 * */
namespace nntile::kernel::hypot_inplace
{

} // namespace nntile::kernel::hypot_inplace
