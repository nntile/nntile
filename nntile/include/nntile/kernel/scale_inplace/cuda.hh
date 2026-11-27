/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/kernel/scale_inplace/cuda.hh
 * Scale inplace operation on buffers on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/kernel/cuda.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::scale_inplace
{

// Apply scale inplace for buffers on CUDA
template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha, T* data)
    noexcept;

//! Scale a chunked buffer in place on CUDA in ONE launch.
template<typename T>
void chunked(cudaStream_t stream, Index nelems, Scalar alpha,
        T* const* data_chunks, Index ndata, Index elems_data,
        ChunkClass cls = chunk_class())
    noexcept;

} // namespace nntile::kernel::scale_inplace
