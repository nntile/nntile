/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/kernel/hypot_scalar_inverse/cuda.hh
 * Inverse of a hypot operation of a buffer and a scalar
 *
 * @version 2.0.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>
#include <nntile/kernel/cuda.hh>

namespace nntile::kernel::hypot_scalar_inverse
{

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar eps, Scalar alpha, T* dst)
    noexcept;

//! Inverse hypot of a scalar and chunked buffers on CUDA (ONE launch)
template<typename T>
void chunked(cudaStream_t stream, Index nelems, Scalar eps, Scalar alpha,
        T* const* dst_chunks, Index ndst, Index elems_dst,
        ChunkClass cls = chunk_class())
    noexcept;

} // namespace nntile::kernel::hypot_scalar_inverse
