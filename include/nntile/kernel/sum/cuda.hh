/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/sum/cuda.hh
 * Sum of a buffer on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Konstantin Sozykin
 * @date 2023-04-13
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile
{
namespace kernel
{
namespace sum
{

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, T alpha,
        const T *src, T beta, T *sum_dst)
    noexcept;

} // namespace sum
} // namespace kernel
} // namespace nntile

