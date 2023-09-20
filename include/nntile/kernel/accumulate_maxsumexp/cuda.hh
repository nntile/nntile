/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/accumulate_maxsumexp/cuda.hh
 * Accumulate maxsumexp buffers on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-09-20
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile
{
namespace kernel
{
namespace accumulate_maxsumexp
{

// Accumulate maxsumexp buffers on CUDA
template<typename T>
void cuda(cudaStream_t stream, Index nelems, const T* src, T* dst)
    noexcept;

} // namespace accumulate_maxsumexp
} // namespace kernel
} // namespace nntile

