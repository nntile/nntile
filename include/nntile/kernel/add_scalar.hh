/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/add_scalar.hh
 * Add_ scalar low-level kernel
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/kernel/add_scalar/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/add_scalar/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::add_scalar
/*! Low-level implementations of add_scalar operation
 * */
namespace nntile::kernel::add_scalar
{

} // namespace nntile::kernel::add_scalar
