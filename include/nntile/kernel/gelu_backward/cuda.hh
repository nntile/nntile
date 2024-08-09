/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/gelu_backward/cuda.hh
 * Backward GeLU operation on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::gelu_backward
{

template<typename T>
void cuda(cudaStream_t stream, Index nelems, const T *x, const T *dy, T *dx)
    noexcept;

} // namespace nntile::kernel::gelu_backward
