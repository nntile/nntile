/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/logsumexp/cuda.hh
 * Logsumexp of buffer on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-06-29
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile
{
namespace kernel
{
namespace logsumexp
{

template<typename T>
void cuda(cudaStream_t stream, Index m, const T *maxsumexp, T *logsumexp)
    noexcept;

} // namespace logsumexp
} // namespace kernel
} // namespace nntile

