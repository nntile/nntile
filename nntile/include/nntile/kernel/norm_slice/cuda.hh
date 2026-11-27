/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/kernel/norm_slice/cuda.hh
 * Euclidean norms of fibers into a slice of a buffer on CUDA (out-of-place version)
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/kernel/cuda.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::norm_slice
{

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T *src1, Scalar beta, const T *src2, T *dst)
    noexcept;

//! Chunked Euclidean norms of fibers in ONE launch.
template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T* const* src1_chunks, Index nsrc1,
        Scalar beta, const T* const* src2_chunks, Index nsrc2,
        T* const* dst_chunks, Index ndst,
        Index elems_src1, Index elems_src2, Index elems_dst,
        ChunkClass cls = chunk_class())
    noexcept;

} // namespace nntile::kernel::norm_slice
