/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/kernel/subcopy/cuda.hh
 * Copy subarray based on contiguous indices
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/kernel/cuda.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::subcopy
{

// Complex copying of one multidimensional array into another
template<typename T>
void cuda(cudaStream_t stream, Index ndim, const Index *src_start,
        const Index *src_stride, const Index *copy_shape, const T *src_,
        const Index *dst_start, const Index *dst_stride, T *dst_)
    noexcept;

//! Chunked complex copying in ONE launch.
template<typename T>
void chunked(cudaStream_t stream, Index ndim, const Index *src_start,
        const Index *src_stride, const Index *copy_shape,
        const T* const* src_chunks, Index nsrc,
        const Index *dst_start, const Index *dst_stride,
        T* const* dst_chunks, Index ndst,
        Index elems_src, Index elems_dst,
        ChunkClass cls = chunk_class())
    noexcept;

} // namespace nntile::kernel::subcopy
