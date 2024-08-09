/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/scal.hh
 * Scal low-level kernel
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/kernel/scal/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/scal/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::scal
/*! Low-level implementations of scal operation
 * */
namespace nntile::kernel::scal
{

} // namespace nntile::kernel::scal
