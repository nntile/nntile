/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/softmax_inplace/cuda.hh
 * softmax_inplace operation on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::softmax_inplace
{

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, const T *maxsumexp,
        Scalar alpha, T *dst)
    noexcept;

} // namespace nntile::kernel::softmax_inplace
