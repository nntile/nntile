/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/scale/cuda.hh
 * Scale operation on buffers on CUDA
 *
 * @version 1.2.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::scale
{

// Apply scale for buffers on CUDA
template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha, const T* src, T* dst)
    noexcept;

//! Scale chunked buffers on CUDA (ONE launch, chunk tables via UVA)
/*! Same per-element operation and thread/block mapping as cuda(). */
template<typename T>
void chunked(cudaStream_t stream, Index nelems, Scalar alpha,
        const T* const* src_chunks, Index nsrc, T* const* dst_chunks,
        Index ndst, Index elems_src, Index elems_dst)
    noexcept;

} // namespace nntile::kernel::scale
