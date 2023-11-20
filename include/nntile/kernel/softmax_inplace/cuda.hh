/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/softmax_inplace/cuda.hh
 * softmax_inplace operation on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-11-20
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile
{
namespace kernel
{
namespace softmax_inplace
{

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, const T *maxsumexp,
        T alpha, T *dst)
    noexcept;

} // namespace softmax_inplace
} // namespace kernel
} // namespace nntile

