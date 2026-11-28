/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/kernel/multiply_inplace.hh
 * Per-element product of two buffers. low-level kernels
 *
 * @version 2.0.0
 * */

#pragma once

#include <nntile/kernel/multiply_inplace/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/multiply_inplace/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::multiply_inplace
/*! Low-level implementations of prod operation
 * */
namespace nntile::kernel::multiply_inplace
{

} // namespace nntile::kernel::multiply_inplace
