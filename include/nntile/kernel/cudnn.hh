/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/cudnn.hh
 * cuDNN helper utilities for NNTile kernels
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/defs.h>

#ifdef NNTILE_USE_CUDA

#include <cudnn.h>
#include <starpu.h>
#include <stdexcept>
#include <string>

namespace nntile::kernel
{

//! Get cuDNN handle for current StarPU worker
/*! This function accesses the global cuDNN handles array that is initialized
 * in context.cc during StarPU initialization.
 */
cudnnHandle_t get_cudnn_handle();

} // namespace nntile::kernel

#endif // NNTILE_USE_CUDA
