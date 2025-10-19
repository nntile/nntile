/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/multiply_slice.hh
 * Per-element multiplication of a tensor by a broadcasted slice
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/kernel/multiply_slice/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/multiply_slice/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::multiply_slice
/*! Low-level implementations of multiply_slice operation
 * */
namespace nntile::kernel::multiply_slice
{

} // namespace nntile::kernel::prod_slice
