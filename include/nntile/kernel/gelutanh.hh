/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/gelutanh.hh
 * Approximate GeLU low-level kernels
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/kernel/gelutanh/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/gelutanh/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::gelutanh
/*! Low-level implementations of Approximate GeLU operation
 * */
namespace nntile::kernel::gelutanh
{

} // namespace nntile::kernel::gelutanh
