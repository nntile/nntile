/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/kernel/hypot_inplace/cuda.hh
 * hypot_inplace operation on buffers on CUDA
 *
 * @version 2.0.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/kernel/cuda.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::hypot_inplace
{

// Apply hypot_inplace for buffers on CUDA
template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha, const T* src, Scalar beta,
        T* dst)
    noexcept;

template<typename T>
void chunked(cudaStream_t stream, Index nelems,
        Scalar alpha,
        const T* const* src_chunks, Index nsrc,
        Scalar beta,
        T* const* dst_chunks, Index ndst,
        Index elems_src, Index elems_dst,
        ChunkClass cls = chunk_class())
    noexcept;

} // namespace nntile::kernel::hypot_inplace
