/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/norm_fiber/cuda.hh
 * Euclidean norms over slices into a fiber of a product of a buffer on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::norm_fiber
{

// Euclidean norms over slices into a fiber of a product of buffers on GPU
template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Index batch, Scalar alpha,
        const T *src, Scalar beta, T *dst)
    noexcept;

} // namespace nntile::kernel::norm_fiber
