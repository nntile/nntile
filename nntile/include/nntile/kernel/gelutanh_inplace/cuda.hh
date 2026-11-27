/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/kernel/gelutanh_inplace/cuda.hh
 * Approximate GeLU operation on CUDA based on tanh function
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/kernel/cuda.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::gelutanh_inplace
{

template<typename T>
void cuda(cudaStream_t stream, Index nelems, T *data)
    noexcept;

//! Approximate GeLU over a chunked buffer on CUDA in ONE launch.
template<typename T>
void chunked(cudaStream_t stream, Index nelems,
        T* const* data_chunks, Index ndata, Index elems_data,
        ChunkClass cls = chunk_class())
    noexcept;

} // namespace nntile::kernel::gelutanh_inplace
