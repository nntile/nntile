/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/kernel/rope/cuda.hh
 * RoPE operation on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/kernel/cuda.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::rope
{

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, const T *sin, const T *cos,
        const T *src, T *dst)
    noexcept;

//! Chunked RoPE in ONE launch.
template<typename T>
void chunked(cudaStream_t stream, Index m, Index n,
        const T* const* sin_chunks, Index nsin,
        const T* const* cos_chunks, Index ncos,
        const T* const* src_chunks, Index nsrc,
        T* const* dst_chunks, Index ndst,
        Index elems_sin, Index elems_cos, Index elems_src, Index elems_dst,
        ChunkClass cls = chunk_class())
    noexcept;

} // namespace nntile::kernel::rope
