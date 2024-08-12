/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/prod_fiber3.hh
 * Bias-like product along outer axes low-level kernels
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/kernel/prod_fiber3/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/prod_fiber3/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::prod_fiber3
/*! Low-level implementations of prod_fiber3 operation
 * */
namespace nntile::kernel::prod_fiber3
{

} // namespace nntile::kernel::prod_fiber3
