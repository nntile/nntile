/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/lars_step/cuda.hh
 * Fused Lars step on CUDA buffers
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::lars_step
{
    // Index num_iter, Index num_elems, Index num_steps, Scalar gamma_0, Scalar momentum,
    // Scalar weight_decay, Scalar lars_coefficient, const T *grad, T *momentum_buffer, T *p
template<typename T>
void cuda(cudaStream_t stream, Index num_iter, Index num_elems, Index num_steps,
    Scalar gamma_0, Scalar momentum, Scalar weight_decay, Scalar lars_coefficient,
    const T *grad, T *momentum_buffer, T *p)
    noexcept;

} // namespace nntile::kernel::lars_step
