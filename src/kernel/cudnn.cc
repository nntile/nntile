/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/cudnn.cc
 * cuDNN helper utilities implementation
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/cudnn.hh"

#ifdef NNTILE_USE_CUDA

#include <starpu.h>

// External declaration of global cuDNN handles from context.cc
extern cudnnHandle_t nntile_cudnn_handles[STARPU_NMAXWORKERS];

namespace nntile::kernel
{

//! Get cuDNN handle for current StarPU worker
cudnnHandle_t get_cudnn_handle()
{
    int worker_id = starpu_worker_get_id();
    return nntile_cudnn_handles[worker_id];
}

} // namespace nntile::kernel

#endif // NNTILE_USE_CUDA
