/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/maxsumexp/cuda.hh
 * Max and sum of exponents of a buffer on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-12-07
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile
{
namespace kernel
{
namespace maxsumexp
{

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, const T *src,
        T *maxsumexp)
    noexcept;

} // namespace maxsumexp
} // namespace kernel
} // namespace nntile

