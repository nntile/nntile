/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/kernel/silu_backward/cuda.hh
 * Backward SiLU operation on CUDA
 *
 * @version 2.0.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/kernel/cuda.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::silu_backward
{

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha, const T *x, const T *dy, Scalar beta, T *dx)
    noexcept;

template<typename T>
void chunked(cudaStream_t stream, Index nelems, Scalar alpha,
        const T* const* x_chunks, Index nx, Index elems_x,
        const T* const* dy_chunks, Index ndy, Index elems_dy,
        Scalar beta,
        T* const* dx_chunks, Index ndx, Index elems_dx,
        ChunkClass cls = chunk_class())
    noexcept;

} // namespace nntile::kernel::silu_backward
