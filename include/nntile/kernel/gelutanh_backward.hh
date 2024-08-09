/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/gelutanh_backward.hh
 * Backward approximate GeLU low-level kernels
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/kernel/gelutanh_backward/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/gelutanh_backward/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::gelutanh_backward
/*! Low-level implementations of backward approximate GeLU operation
 * */
namespace nntile::kernel::gelutanh_backward
{

} // namespace nntile::kernel::gelutanh_backward
