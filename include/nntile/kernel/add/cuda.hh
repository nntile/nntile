/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/add/cuda.hh
 * Add operation on buffers on CUDA
 *
 * @version 1.0.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::add
{

// Apply add for buffers on CUDA
template<typename T>
void cuda(cudaStream_t stream, Index nelems, T alpha, const T* src, T beta,
        T* dst)
    noexcept;

// specific specialization for fp16_t, temporary solution
void cuda16(cudaStream_t stream, Index nelems, fp32_t alpha, const fp16_t *src, fp32_t beta,
        fp16_t *dst)
    noexcept;

} // namespace nntile::kernel::add

