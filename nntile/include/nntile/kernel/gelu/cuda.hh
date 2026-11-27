/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/kernel/gelu/cuda.hh
 * GeLU operation on a buffer on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/kernel/cuda.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::gelu
{

// GeLU operation on a buffer on CUDA
template<typename T>
void cuda(cudaStream_t stream, Index nelems, const T *src, T *dst)
    noexcept;

//! GeLU over chunked buffers on CUDA in ONE launch.
/*! Same per-element expression as cuda(), but each operand is a chain of
 * equal-size chunks. The chunk base-pointer tables travel as device-table
 * views; every element resolves its chunk from its flat offset with the
 * runtime class shift/mask (a single-chunk operand uses the sentinel shift).
 * */
template<typename T>
void chunked(cudaStream_t stream, Index nelems,
        const T* const* src_chunks, Index nsrc,
        T* const* dst_chunks, Index ndst,
        Index elems_src, Index elems_dst,
        ChunkClass cls = chunk_class())
    noexcept;

} // namespace nntile::kernel::gelu
