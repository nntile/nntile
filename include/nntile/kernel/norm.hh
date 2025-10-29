/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/norm.hh
 * Euclidean norm of all elements in a buffer
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/kernel/norm/cpu.hh>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/norm/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::norm
/*! Low-level implementations of computing norm operation
 * */
namespace nntile::kernel::norm
{

} // namespace nntile::kernel::norm
