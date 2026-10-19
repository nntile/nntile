/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/gelutanh_backward/cuda.hh
 * Backward approximate GeLU operation on CUDA
 *
 * @version 1.2.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::gelutanh_backward
{

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha, const T *x, const T *dy, Scalar beta, T *dx)
    noexcept;

//! Backward approximate GeLU over chunked buffers on CUDA (ONE launch)
template<typename T>
void chunked(cudaStream_t stream, Index nelems, Scalar alpha,
        const T* const* x_chunks, Index nx, const T* const* dy_chunks,
        Index ndy, Scalar beta, T* const* dx_chunks, Index ndx,
        Index elems_x, Index elems_dy, Index elems_dx)
    noexcept;

} // namespace nntile::kernel::gelutanh_backward
